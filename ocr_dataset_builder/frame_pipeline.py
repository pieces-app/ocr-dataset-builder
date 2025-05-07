import concurrent.futures  # Added for parallelism
import logging
import os  # Added for os.cpu_count() for default max_workers
import shutil  # Added for file copying
import sys
import time
from pathlib import Path

import fire  # Import fire for CLI

# from rich import print # Commented out, using RichHandler for logging
from rich.logging import RichHandler  # Added for rich logging
from tqdm import tqdm


from ocr_dataset_builder.video_processing import extract_frames

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles its own formatting
    datefmt="[%X]",  # Optional: time format for non-RichHandler (if any)
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".mkv",
    ".avi",
    ".mov",
]  # Add more if needed


def find_video_file(directory: Path) -> Path | None:
    """Finds the first video file in a directory based on common extensions."""
    for ext in VIDEO_EXTENSIONS:
        videos = list(directory.glob(f"*{ext}"))
        if videos:
            if len(videos) > 1:
                logging.warning(
                    f"Multiple videos ({len(videos)}) found in {directory.name}, "
                    f"using first: {videos[0].name}"
                )
            return videos[0]  # Return the first one found
    return None


def find_metadata_file(directory: Path) -> Path | None:
    """Finds the first .info.json file in a directory."""
    metadata_files = list(directory.glob("*.info.json"))
    if metadata_files:
        if len(metadata_files) > 1:
            logging.warning(
                f"Multiple .info.json files found in {directory.name}, "
                f"using first: {metadata_files[0].name}"
            )
        return metadata_files[0]
    return None


def _process_single_video_dir(
    video_dir_absolute_path: Path,  # Changed from video_dir
    video_dir_relative_path: str,  # Added for checkpointing
    input_root: Path,
    output_root: Path,
    target_fps: int,
    max_dimension: int | None,
    max_frames_per_video: int | None,
) -> tuple[str, int | None, str]:  # First element is now relative_path
    """Helper function to process a single video directory."""
    # dir_name = video_dir_absolute_path.name # Use relative_path for external reporting
    metadata_copied = False
    logging.debug(f"Worker: Entering _process_single_video_dir for {video_dir_relative_path}")
    try:
        logging.debug(f"Worker: Attempting to find video file in {video_dir_absolute_path}")
        video_file = find_video_file(video_dir_absolute_path)
        logging.debug(f"Worker: Attempting to find metadata file in {video_dir_absolute_path}")
        metadata_file = find_metadata_file(video_dir_absolute_path)

        if not video_file:
            logging.warning(f"Worker: No video file found for {video_dir_relative_path}")
            return (video_dir_relative_path, None, "No video file found")

        # Determine mirrored output directory path using absolute input_root and relative_path
        # relative_path_obj = video_dir_absolute_path.relative_to(input_root) # This was how it was done before
        output_video_frames_dir = (
            output_root / video_dir_relative_path
        )  # Simpler now
        logging.debug(f"Worker: Output directory for frames: {output_video_frames_dir}")
        # extract_frames will create this directory

        if metadata_file:
            try:
                logging.debug(f"Worker: Creating output directory (if not exists): {output_video_frames_dir}")
                output_video_frames_dir.mkdir(parents=True, exist_ok=True)
                target_metadata_path = (
                    output_video_frames_dir / metadata_file.name
                )
                logging.debug(f"Worker: Copying metadata {metadata_file.name} to {target_metadata_path}")
                shutil.copy2(metadata_file, target_metadata_path)
                metadata_copied = True
                logging.debug(  # Reduced log level for less noise
                    f"Worker: Copied metadata {metadata_file.name} for {video_dir_relative_path}"
                )
            except Exception as meta_e:
                logging.error(
                    f"Worker: Failed to copy metadata {metadata_file.name} "
                    f"for {video_dir_relative_path}: {meta_e}", exc_info=True
                )
        else:
            logging.warning(
                f"  No .info.json metadata file found for {video_dir_relative_path}"
            )

        logging.info(f"Worker: Starting frame extraction for {video_file.name} (Dir: {video_dir_relative_path})")
        extracted_paths = extract_frames(
            str(video_file),
            str(output_video_frames_dir),
            target_fps=target_fps,
            max_dimension=max_dimension,
            max_frames_per_video=max_frames_per_video,
        )
        logging.info(f"Worker: Finished frame extraction for {video_file.name}, {len(extracted_paths) if extracted_paths else 0} frames. (Dir: {video_dir_relative_path})")

        status_suffix = (
            " (Metadata OK)"
            if metadata_copied
            else " (Metadata MISSING/ERROR)"
        )
        if extracted_paths:
            return (
                video_dir_relative_path,
                len(extracted_paths),
                "Success" + status_suffix,
            )
        else:
            return (
                video_dir_relative_path,
                0,
                "Extraction/Sampling resulted in 0 frames" + status_suffix,
            )

    except Exception as e:
        logging.error(
            f"Worker: Unhandled error in _process_single_video_dir for {video_dir_relative_path}: {e}",
            exc_info=True,  # Log full traceback
        )
        # Return relative path for consistency
        return (video_dir_relative_path, None, f"Error: {e}")


def _perform_processing_pass(
    input_root: Path,
    output_root: Path,
    target_fps: int,
    max_dimension: int | None,
    max_frames_per_video: int | None,
    effective_start_index: int | None,
    effective_end_index: int | None,
    max_workers: int | None,
    checkpoint_file_name: str,
    dry_run: bool,
) -> tuple[int, int, int, int, int]: # attempted_this_pass, successful_this_pass, failed_this_pass, no_video_this_pass, total_frames_this_pass
    """
    Performs a single pass of discovering and processing videos.
    Returns statistics for the pass.
    """
    checkpoint_path = output_root / checkpoint_file_name
    processed_dirs_from_checkpoint = set()
    malformed_checkpoint_entries = 0
    max_path_len_checkpoint = 1024 # Arbitrary reasonable limit

    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line_num, line in enumerate(f):
                    path_str = line.strip()
                    # Add check for malformed/long lines
                    if not path_str: # Skip empty lines
                        continue
                    if '\\n' in path_str or '\r' in path_str: # Check for embedded newlines
                        logging.warning(
                            f"Skipping malformed entry in Frame checkpoint {checkpoint_path} (line ~{line_num+1}): Contains newline chars."
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    if len(path_str) > max_path_len_checkpoint:
                        logging.warning(
                            f"Skipping malformed entry in Frame checkpoint {checkpoint_path} (line ~{line_num+1}): Exceeds max length {max_path_len_checkpoint}. Entry: \"{path_str[:80]}...\""
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    
                    processed_dirs_from_checkpoint.add(path_str)
            
            logging.info(
                f"Loaded {len(processed_dirs_from_checkpoint)} valid processed directory paths from Frame checkpoint."
            )
            if malformed_checkpoint_entries > 0:
                 logging.warning(f"Skipped {malformed_checkpoint_entries} potentially malformed entries during Frame checkpoint load.")

        except Exception as e:
            logging.error(
                f"Could not read Frame checkpoint file {checkpoint_path}: {e}. Processing all (or as per slice)."
            )
            processed_dirs_from_checkpoint.clear()
            malformed_checkpoint_entries = 0 # Reset counter

    all_potential_dirs_absolute = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    all_potential_dirs_relative = [
        d.relative_to(input_root).as_posix()
        for d in all_potential_dirs_absolute
    ]

    if dry_run:
        logging.debug(f"[DRY RUN] All potential absolute dirs: {all_potential_dirs_absolute}")
        logging.debug(f"[DRY RUN] All potential relative dirs: {all_potential_dirs_relative}")

    logging.info(
        f"Found {len(all_potential_dirs_relative)} total potential video directories in input."
    )

    valid_processed_relative_paths = {
        path_str
        for path_str in processed_dirs_from_checkpoint
        if (input_root / path_str).is_dir()
    }
    if len(valid_processed_relative_paths) < len(
        processed_dirs_from_checkpoint
    ):
        logging.info(
            f"{len(processed_dirs_from_checkpoint) - len(valid_processed_relative_paths)} "
            f"stale directory paths removed from checkpoint list."
        )
        if dry_run:
            logging.debug(f"[DRY RUN] Valid processed relative paths (after checking existence): {valid_processed_relative_paths}")

    relative_dirs_to_consider = sorted(
        [
            rel_path
            for rel_path in all_potential_dirs_relative
            if rel_path not in valid_processed_relative_paths
        ]
    )
    if dry_run:
        logging.debug(f"[DRY RUN] Relative dirs to consider (after checkpoint filtering): {relative_dirs_to_consider}")
    
    total_dirs_to_consider = len(relative_dirs_to_consider)
    
    actual_start_index = effective_start_index if effective_start_index is not None else 0
    actual_end_index = (
        effective_end_index if effective_end_index is not None else total_dirs_to_consider
    )

    if not (0 <= actual_start_index <= total_dirs_to_consider):
        logging.error(
            f"Invalid actual_start_index: {actual_start_index}. For list of {total_dirs_to_consider} items, must be 0 <= index <= {total_dirs_to_consider}."
        )
        return 0, 0, 0, 0, 0
    if not (actual_start_index <= actual_end_index <= total_dirs_to_consider):
        logging.error(
            f"Invalid actual_end_index: {actual_end_index}. For list of {total_dirs_to_consider} items, must be start_index <= index <= {total_dirs_to_consider}."
        )
        return 0, 0, 0, 0, 0

    sliced_relative_dirs_for_this_run = relative_dirs_to_consider[
        actual_start_index:actual_end_index
    ]
    absolute_dirs_for_this_run_map = {
        rel_path: (input_root / rel_path)
        for rel_path in sliced_relative_dirs_for_this_run
    }

    target_count_this_pass = len(sliced_relative_dirs_for_this_run)

    if target_count_this_pass == 0:
        logging.info(
            "No new directories selected for processing in this pass."
        )
        return 0, 0, 0, 0, 0

    logging.info(
        f"Targeting {target_count_this_pass} directories for processing this pass "
        f"(slice: index {actual_start_index} to {actual_end_index -1} of remaining dirs) "
        f"using up to {max_workers if max_workers else os.cpu_count()} workers."
    )

    pass_attempted_count = target_count_this_pass
    pass_successful_run_count = 0
    pass_failed_run_count = 0
    pass_no_video_run_count = 0
    pass_total_frames_saved = 0

    if dry_run:
        logging.info(f"[DRY RUN] Would attempt to process {target_count_this_pass} directories in this pass:")
        for i, rel_path in enumerate(sliced_relative_dirs_for_this_run):
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            logging.info(f"[DRY RUN]   {i+1}. Relative: '{rel_path}', Absolute: '{abs_path}'")
        logging.info("[DRY RUN] Skipping actual ProcessPoolExecutor and video processing for this pass.")
        # For dry run, simulate that all targeted items would be "attempted" and "successful" for pass stats
        return target_count_this_pass, target_count_this_pass, 0, 0, 0

    future_to_rel_path = {} # Map Future objects to the relative path they process
    current_max_workers = max_workers if max_workers else os.cpu_count()
    if not current_max_workers or current_max_workers < 1: 
        current_max_workers = 1 

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=current_max_workers
    ) as executor:
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            future = executor.submit(
                _process_single_video_dir,
                abs_path,
                rel_path,
                input_root,
                output_root,
                target_fps,
                max_dimension,
                max_frames_per_video,
            )
            future_to_rel_path[future] = rel_path # Store the mapping

        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_rel_path.keys()), # Iterate over futures
            total=target_count_this_pass,
            unit="dir",
            desc="Processing Dirs (Pass)",
            ncols=100, 
        )

        for completed_future in progress_bar:
            rel_path_for_future = future_to_rel_path[completed_future] # Get path associated with this future
            try:
                returned_relative_path, frame_count_or_none, status = (
                    completed_future.result() # Get result
                )
                # Sanity check: the returned path should match the input path
                if returned_relative_path != rel_path_for_future:
                     logging.error(f"Mismatch in returned path ('{returned_relative_path}') and expected path ('{rel_path_for_future}') for future. Check worker logic.")
                     # Treat this as a failure for safety, don't checkpoint
                     pass_failed_run_count += 1
                     continue # Skip to next future

                # Process result based on status from worker
                if status.startswith("Success"):
                    pass_successful_run_count += 1
                    if frame_count_or_none is not None:
                        pass_total_frames_saved += frame_count_or_none
                    try:
                        with open(checkpoint_path, "a") as cp_file:
                            cp_file.write(f"{returned_relative_path}\\n")
                        logging.debug(
                            f"Checkpointed: {returned_relative_path}"
                        )
                    except Exception as cp_e:
                        logging.error(
                            f"Failed to write to checkpoint file {checkpoint_path} for {returned_relative_path}: {cp_e}"
                        )
                elif status == "No video file found":
                    pass_no_video_run_count += 1
                    # Optionally checkpoint no-video dirs if you want to permanently skip them
                    # try:
                    #     with open(checkpoint_path, "a") as cp_file:
                    #         cp_file.write(f"{returned_relative_path}\\n")
                    # except Exception as cp_e:
                    #     logging.error(f"Failed to checkpoint no-video dir {returned_relative_path}: {cp_e}")
                else:
                    pass_failed_run_count += 1
                    logging.warning(
                        f"Directory {returned_relative_path} failed with status: {status}"
                    )
            except concurrent.futures.process.BrokenProcessPool:
                # Specific handling for BrokenProcessPool
                pass_failed_run_count += 1
                logging.error(
                    f"Worker process for directory '{rel_path_for_future}' died unexpectedly (BrokenProcessPool). "
                    f"Possible memory issue or problematic video file. Directory will not be checkpointed."
                )
            except Exception as exc:
                # Handle other exceptions from future.result() or the block above
                pass_failed_run_count += 1
                logging.error(
                    f"An unexpected error occurred while processing result for directory '{rel_path_for_future}': {exc}",
                    exc_info=True
                )
    
    return pass_attempted_count, pass_successful_run_count, pass_failed_run_count, pass_no_video_run_count, pass_total_frames_saved

def process_dataset_videos(
    input_root_dir: str | Path,
    output_root_dir: str | Path,
    target_fps: int = 1,
    max_dimension: int | None = 1024,
    max_frames_per_video: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    max_workers: int | None = None,
    checkpoint_file_name: str = ".processed_video_dirs.log",
    dry_run: bool = False,
    daemon_mode: bool = False,
    watch_interval_seconds: int = 300,
):
    """
    Processes video directories. Can run once or in daemon mode to continuously monitor.
    """
    input_root = Path(input_root_dir).resolve()
    output_root = Path(output_root_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        logging.error(f"Input dataset directory not found: {input_root}")
        return

    logging.info(f"Starting video processing from: {input_root}")
    logging.info(f"Outputting frames to: {output_root}")
    logging.info(f"Checkpoint file: {output_root / checkpoint_file_name}")
    logging.info(f"Target FPS: {target_fps}")
    if max_dimension and max_dimension > 0:
        logging.info(f"Max frame dimension: {max_dimension}px")
    else:
        logging.info("Frame resizing disabled.")
    if max_frames_per_video and max_frames_per_video > 0:
        logging.info(
            f"Max frames per video (sampling): {max_frames_per_video}"
        )
    else:
        logging.info("Frame sampling disabled.")

    if dry_run:
        logging.info("[bold yellow]*** DRY RUN ACTIVATED ***[/bold yellow] No actual video processing will occur.")
    
    if daemon_mode:
        logging.info(
            f"[DAEMON MODE] Activated. Watch interval: {watch_interval_seconds}s. "
            f"Initial slice for first pass: start_index={start_index}, end_index={end_index}. "
            "Subsequent passes will process all new directories."
        )
    else:
        logging.info(f"Single run mode. Slice to consider: start_index={start_index}, end_index={end_index}")

    run_iteration = 0
    overall_attempted_count = 0
    overall_successful_count = 0
    overall_failed_count = 0
    overall_no_video_count = 0
    overall_total_frames_saved = 0
    
    first_pass_daemon = True

    try:
        while True:
            run_iteration += 1
            logging.info(f"--- Starting Processing Pass #{run_iteration} ---")
            pass_start_time = time.time()

            current_start_idx_for_pass = None
            current_end_idx_for_pass = None

            if daemon_mode:
                if first_pass_daemon:
                    current_start_idx_for_pass = start_index
                    current_end_idx_for_pass = end_index
                    first_pass_daemon = False
                else: # Subsequent daemon passes process all new
                    current_start_idx_for_pass = 0
                    current_end_idx_for_pass = None # Process all available
            else: # Single run mode
                current_start_idx_for_pass = start_index
                current_end_idx_for_pass = end_index
            
            pass_attempted, pass_successful, pass_failed, pass_no_video, pass_frames = _perform_processing_pass(
                input_root=input_root,
                output_root=output_root,
                target_fps=target_fps,
                max_dimension=max_dimension,
                max_frames_per_video=max_frames_per_video,
                effective_start_index=current_start_idx_for_pass,
                effective_end_index=current_end_idx_for_pass,
                max_workers=max_workers,
                checkpoint_file_name=checkpoint_file_name,
                dry_run=dry_run,
            )
            
            pass_duration = time.time() - pass_start_time

            logging.info(f"--- Pass #{run_iteration} Summary ---")
            logging.info(f"Directories targeted in this pass: {pass_attempted}")
            logging.info(f"Successfully processed in this pass: {pass_successful}")
            logging.info(f"Skipped (no video) in this pass: {pass_no_video}")
            logging.info(f"Failed in this pass: {pass_failed}")
            logging.info(f"Frames saved in this pass: {pass_frames}")
            logging.info(f"Pass #{run_iteration} duration: {pass_duration:.2f} seconds")

            overall_attempted_count += pass_attempted # This is more like "targeted"
            overall_successful_count += pass_successful
            overall_failed_count += pass_failed
            overall_no_video_count += pass_no_video
            overall_total_frames_saved += pass_frames
            
            if not daemon_mode:
                break 

            logging.info(f"[DAEMON MODE] Next scan in {watch_interval_seconds} seconds. (Ctrl+C to stop)")
            time.sleep(watch_interval_seconds)
            
    except KeyboardInterrupt:
        logging.info("[bold red]KeyboardInterrupt received. Shutting down gracefully...[/bold red]")
    finally:
        logging.info("--- Overall Summary ---")
        if daemon_mode:
            logging.info(f"Total passes executed: {run_iteration}")
        logging.info(f"Total directories targeted across all passes: {overall_attempted_count}")
        logging.info(f"Total successfully processed: {overall_successful_count}")
        logging.info(f"Total skipped (no video file found): {overall_no_video_count}")
        logging.info(f"Total failed during processing: {overall_failed_count}")
        logging.info(f"Total frames saved: {overall_total_frames_saved}")
        logging.info(f"See {output_root / checkpoint_file_name} for a list of successfully completed directories.")

class PipelineCLI:
    """CLI for the OCR Dataset Builder Frame Extraction Pipeline."""

    def process_videos(
        self,
        dataset_path: str,
        output_path: str,
        target_fps: int = 1,
        max_dimension: int | None = 1024,
        max_frames_per_video: int | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        max_workers: int | None = None,
        checkpoint_log: str = ".processed_video_dirs.log",
        dry_run: bool = False,
        daemon_mode: bool = False,
        watch_interval_seconds: int = 300,
    ):
        """
        Processes videos from subdirectories for frame extraction.
        Can run once or in daemon mode to continuously monitor for new videos.

        Args:
            dataset_path: Root directory of the input video dataset.
            output_path: Root directory for saving extracted frames.
            target_fps: Target frames per second for extraction. Defaults to 1.
            max_dimension: Max dimension (px) for frame resizing. Defaults to 1024.
                           Set to 0 or None to disable resizing.
            max_frames_per_video: Max frames to randomly sample per video.
                                  Defaults to None (all frames).
            start_index: 0-based index of the first video directory to process
                         (after checkpointing). Applies to single run or
                         the first pass of daemon mode. Defaults to 0.
            end_index: 0-based index *after* the last directory to process.
                       Applies to single run or the first pass of daemon mode.
                       Defaults to processing all available.
            max_workers: Max number of parallel processes. Defaults to CPU count.
            checkpoint_log: Name for the checkpoint file. Defaults to
                            '.processed_video_dirs.log'.
            dry_run: Simulate run, log actions, but do not process videos.
                     Defaults to False.
            daemon_mode: If True, run continuously, scanning for new videos
                         at specified intervals. Defaults to False.
            watch_interval_seconds: Interval in seconds to wait between scans
                                    in daemon mode. Defaults to 300 (5 minutes).
        """
        if max_dimension == 0:
            max_dimension = None

        process_dataset_videos(
            input_root_dir=dataset_path,
            output_root_dir=output_path,
            target_fps=target_fps,
            max_dimension=max_dimension,
            max_frames_per_video=max_frames_per_video,
            start_index=start_index,
            end_index=end_index,
            max_workers=max_workers,
            checkpoint_file_name=checkpoint_log,
            dry_run=dry_run,
            daemon_mode=daemon_mode,
            watch_interval_seconds=watch_interval_seconds,
        )


if __name__ == "__main__":
    fire.Fire(PipelineCLI)
