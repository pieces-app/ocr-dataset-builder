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
    try:
        video_file = find_video_file(video_dir_absolute_path)
        metadata_file = find_metadata_file(video_dir_absolute_path)

        if not video_file:
            # Return relative path for consistency in reporting
            return (video_dir_relative_path, None, "No video file found")

        # Determine mirrored output directory path using absolute input_root and relative_path
        # relative_path_obj = video_dir_absolute_path.relative_to(input_root) # This was how it was done before
        output_video_frames_dir = (
            output_root / video_dir_relative_path
        )  # Simpler now
        # extract_frames will create this directory

        if metadata_file:
            try:
                output_video_frames_dir.mkdir(parents=True, exist_ok=True)
                target_metadata_path = (
                    output_video_frames_dir / metadata_file.name
                )
                shutil.copy2(metadata_file, target_metadata_path)
                metadata_copied = True
                logging.debug(  # Reduced log level for less noise
                    f"  Copied metadata {metadata_file.name} to {output_video_frames_dir.name}"
                )
            except Exception as meta_e:
                logging.error(
                    f"  Failed to copy metadata {metadata_file.name} "
                    f"for {video_dir_relative_path}: {meta_e}"
                )
        else:
            logging.warning(
                f"  No .info.json metadata file found for {video_dir_relative_path}"
            )

        extracted_paths = extract_frames(
            str(video_file),
            str(output_video_frames_dir),
            target_fps=target_fps,
            max_dimension=max_dimension,
            max_frames_per_video=max_frames_per_video,
        )

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
            f"Error in worker for {video_dir_relative_path}: {e}",
            exc_info=False,
        )
        # Return relative path for consistency
        return (video_dir_relative_path, None, f"Error: {e}")


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
):
    """
    Processes a slice of video directories concurrently, with checkpointing.
    New directories added to input_root_dir will be considered for processing.
    Previously successfully processed directories are skipped.

    Args:
        input_root_dir: Root directory containing video subdirectories.
        output_root_dir: Root directory for saving extracted frames & checkpoints.
        target_fps: Target frames per second for extraction.
        max_dimension: Max dimension (width or height) for frame resizing.
        max_frames_per_video: Max frames to randomly sample per video.
        start_index: 0-based index of the first video directory *to consider processing*
                     (after accounting for checkpoints). Defaults to 0.
        end_index: 0-based index after the last directory *to consider processing*.
                   Defaults to processing all (after accounting for checkpoints).
        max_workers: Max number of processes. Defaults to os.cpu_count().
        checkpoint_file_name: Name of the file to store/load processed directory paths.
                              Stored in output_root_dir.
        dry_run: If True, simulate the run, log actions, but do not process videos.
    """
    input_root = Path(input_root_dir).resolve()
    output_root = Path(output_root_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)  # Ensure output root exists

    if not input_root.is_dir():
        logging.error(f"Input dataset directory not found: {input_root}")
        return

    logging.info(f"Starting video processing from: {input_root}")
    logging.info(f"Outputting frames to: {output_root}")
    logging.info(f"Checkpoint file: {output_root / checkpoint_file_name}")
    logging.info(f"Target FPS: {target_fps}")  # Log target FPS
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
        # Optionally, force debug logging in dry_run mode if not already set
        # current_log_level = logging.getLogger().getEffectiveLevel()
        # if current_log_level > logging.DEBUG:
        #     logging.getLogger().setLevel(logging.DEBUG)
        #     logging.debug("[DRY RUN] Temporarily set log level to DEBUG for detailed dry run output.")

    # --- Checkpoint Loading ---
    checkpoint_path = output_root / checkpoint_file_name
    processed_dirs_from_checkpoint = set()
    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line in f:
                    processed_dirs_from_checkpoint.add(line.strip())
            logging.info(
                f"Loaded {len(processed_dirs_from_checkpoint)} processed directory paths from checkpoint."
            )
            if dry_run:
                logging.debug(f"[DRY RUN] Checkpoint content: {processed_dirs_from_checkpoint}")
        except Exception as e:
            logging.error(
                f"Could not read checkpoint file {checkpoint_path}: {e}. Processing all (or as per slice)."
            )
            processed_dirs_from_checkpoint.clear()  # Ensure it's empty on error

    # --- Re-indexing and Filtering ---
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

    # Filter out dirs from checkpoint that no longer exist in input
    # (though with relative paths, this mostly handles if checkpoint has stale entries for deleted source dirs)
    valid_processed_relative_paths = {
        path_str
        for path_str in processed_dirs_from_checkpoint
        if (input_root / path_str).is_dir()  # Check existence using full path
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

    # Determine the slice based on the list of directories to consider
    total_dirs_to_consider = len(relative_dirs_to_consider)
    actual_start_index = start_index if start_index is not None else 0
    actual_end_index = (
        end_index if end_index is not None else total_dirs_to_consider
    )

    # Validate indices against the list to consider
    if not (
        0 <= actual_start_index <= total_dirs_to_consider
    ):  # Can be equal if list is empty
        logging.error(
            f"Invalid start_index: {start_index}. For list of {total_dirs_to_consider} items, must be 0 <= index <= {total_dirs_to_consider}."
        )
        if dry_run:
            logging.debug(f"[DRY RUN] Start index validation failed: start={actual_start_index}, total={total_dirs_to_consider}")
        return
    if not (actual_start_index <= actual_end_index <= total_dirs_to_consider):
        logging.error(
            f"Invalid end_index: {end_index}. For list of {total_dirs_to_consider} items, must be start_index <= index <= {total_dirs_to_consider}."
        )
        if dry_run:
            logging.debug(f"[DRY RUN] End index validation failed: start={actual_start_index}, end={actual_end_index}, total={total_dirs_to_consider}")
        return

    # Get the final list of relative paths for this run's slice
    sliced_relative_dirs_for_this_run = relative_dirs_to_consider[
        actual_start_index:actual_end_index
    ]

    # Convert back to absolute paths for processing, but keep relative for checkpointing
    absolute_dirs_for_this_run_map = {
        rel_path: (input_root / rel_path)
        for rel_path in sliced_relative_dirs_for_this_run
    }
    if dry_run:
        logging.debug(f"[DRY RUN] Sliced relative dirs for this run: {sliced_relative_dirs_for_this_run}")
        logging.debug(f"[DRY RUN] Absolute dirs map for this run: {absolute_dirs_for_this_run_map}")

    target_count = len(sliced_relative_dirs_for_this_run)

    if target_count == 0:
        logging.warning(
            "No directories selected for processing in this run (after checkpointing and slicing)."
        )
        return

    logging.info(
        f"Targeting {target_count} directories for actual processing this run "
        f"(original slice: index {actual_start_index} to {actual_end_index -1} of remaining dirs) "
        f"using up to {max_workers if max_workers else os.cpu_count()} workers."
    )

    processed_count = 0
    successful_run_count = (
        0  # Renamed from successful_count to avoid confusion
    )
    failed_run_count = 0  # Renamed
    no_video_run_count = 0  # Renamed
    total_frames_saved_this_run = 0  # Renamed
    start_time = time.time()

    if dry_run:
        logging.info(f"[DRY RUN] Would attempt to process {target_count} directories:")
        for i, rel_path in enumerate(sliced_relative_dirs_for_this_run):
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            logging.info(f"[DRY RUN]   {i+1}. Relative: '{rel_path}', Absolute: '{abs_path}'")
        logging.info("[DRY RUN] Skipping actual ProcessPoolExecutor and video processing.")
        # Log summary for dry run
        duration = time.time() - start_time
        logging.info(f"--- Dry Run Summary ---")
        logging.info(f"Input root: {input_root}")
        logging.info(f"Output root: {output_root}")
        logging.info(f"Checkpoint file: {checkpoint_path}")
        logging.info(f"Target FPS: {target_fps}, Max Dimension: {max_dimension}, Max Frames/Video: {max_frames_per_video}")
        logging.info(f"Max workers: {max_workers if max_workers else os.cpu_count()}")
        logging.info(f"Original slice indices: start={start_index}, end={end_index}")
        logging.info(f"Dirs found in input: {len(all_potential_dirs_relative)}")
        logging.info(f"Dirs loaded from checkpoint: {len(processed_dirs_from_checkpoint)}")
        logging.info(f"Dirs to consider after checkpoint: {total_dirs_to_consider}")
        logging.info(f"Actual slice for this run: {actual_start_index} to {actual_end_index-1}")
        logging.info(f"Number of directories that WOULD BE processed: {target_count}")
        logging.info(f"Dry run setup duration: {duration:.2f} seconds")
        return # End execution for dry run

    futures = []
    if not max_workers:  # Handle default for max_workers if None
        max_workers = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Iterate over the relative paths, get absolute path from map for submission
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            future = executor.submit(
                _process_single_video_dir,
                abs_path,  # Absolute path for processing
                rel_path,  # Relative path for checkpointing/reporting
                input_root,
                output_root,
                target_fps,
                max_dimension,
                max_frames_per_video,
            )
            futures.append(future)

        progress_bar = tqdm(
            concurrent.futures.as_completed(futures),
            total=target_count,
            unit="dir",
            desc="Processing Dirs",
            ncols=100,
        )

        for future in progress_bar:
            try:
                # dir_name here is the relative_path from _process_single_video_dir
                returned_relative_path, frame_count_or_none, status = (
                    future.result()
                )
                processed_count += 1

                if status.startswith("Success"):
                    successful_run_count += 1
                    if frame_count_or_none is not None:
                        total_frames_saved_this_run += frame_count_or_none
                    # --- Checkpoint Saving ---
                    try:
                        with open(checkpoint_path, "a") as cp_file:
                            cp_file.write(f"{returned_relative_path}\n")
                        logging.debug(
                            f"Checkpointed: {returned_relative_path}"
                        )
                    except Exception as cp_e:
                        logging.error(
                            f"Failed to write to checkpoint file {checkpoint_path} for {returned_relative_path}: {cp_e}"
                        )

                elif status == "No video file found":
                    no_video_run_count += 1
                else:  # Other errors from _process_single_video_dir
                    failed_run_count += 1
                    logging.warning(
                        f"Directory {returned_relative_path} failed with status: {status}"
                    )

            except Exception as exc:  # Errors from future.result() itself
                failed_run_count += 1
                # How to get dir_name if future failed before returning? This is tricky.
                # For now, just log the exception. The dir won't be checkpointed.
                logging.error(
                    f"A task in the executor failed: {exc}", exc_info=True
                )

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"--- Processing Summary for This Run ---")
    logging.info(f"Attempted to process: {target_count} directories")
    logging.info(
        f"Successfully processed & saved frames for: {successful_run_count} directories"
    )
    logging.info(
        f"Skipped (no video file found): {no_video_run_count} directories"
    )
    logging.info(f"Failed during processing: {failed_run_count} directories")
    logging.info(
        f"Total frames saved in this run: {total_frames_saved_this_run}"
    )
    logging.info(f"Total processing time: {duration:.2f} seconds")
    logging.info(
        f"See {checkpoint_path} for a list of successfully completed directories."
    )


class PipelineCLI:
    """CLI for the OCR Dataset Builder Frame Extraction Pipeline."""

    def process_videos(
        self,
        dataset_path: str,
        output_path: str,
        target_fps: int = 1,
        max_dimension: int | None = 1024,
        max_frames_per_video: int | None = None,  # Added this missing param
        start_index: int | None = None,
        end_index: int | None = None,
        max_workers: int | None = None,
        checkpoint_log: str = ".processed_video_dirs.log",
        dry_run: bool = False,
    ):
        """
        Processes videos from subdirectories in dataset_path and saves extracted frames.

        Args:
            dataset_path: Path to the root directory of the input dataset.
                          Each subdirectory is expected to contain one video.
            output_path: Path to the directory where extracted frames and copied
                         metadata will be saved. A mirrored structure will be created.
            target_fps (int, optional): Target frames per second for extraction.
                                        Defaults to 1.
            max_dimension (int | None, optional): Max dimension (width or height)
                                                  for frame resizing. If None or 0,
                                                  original size is kept. Defaults to 1024.
            max_frames_per_video (int | None, optional): Maximum number of frames to
                                                         randomly sample and save per video.
                                                         If None, all extracted frames
                                                         (after FPS and resizing) are saved.
                                                         Defaults to None.
            start_index (int | None, optional): 0-based index of the first video
                                                directory to process from the list of
                                                pending directories (after checkpointing).
                                                Defaults to 0.
            end_index (int | None, optional): 0-based index *after* the last video
                                              directory to process from the list of
                                              pending directories. Defaults to processing all.
            max_workers (int | None, optional): Maximum number of parallel processes
                                                to use. Defaults to the number of CPU cores.
            checkpoint_log (str, optional): Name for the checkpoint log file.
                                            Stored in the output_path.
                                            Defaults to '.processed_video_dirs.log'.
            dry_run (bool, optional): If True, simulates the run, logs actions,
                                      but does not actually process video files.
                                      Defaults to False.
        """
        if max_dimension == 0:  # Allow 0 to mean None for convenience from CLI
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
        )


if __name__ == "__main__":
    fire.Fire(PipelineCLI)
