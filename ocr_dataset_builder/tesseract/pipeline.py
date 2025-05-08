import concurrent.futures
import json
import logging
import os  # For os.cpu_count()
import time
from pathlib import Path

import fire
from rich.logging import RichHandler  # Added for rich logging
from tqdm import tqdm

# Local package imports (assuming package is installed/dev mode)
from .processing import (
    check_tesseract_install,
    process_image_with_tesseract,
)

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles its own formatting
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)


def _process_tesseract_directory(
    frame_dir_absolute_path: Path,
    frame_dir_relative_path: str,
    output_root: Path, # Removed input_root as frame_dir_relative_path is used
    language: str = "eng",
) -> tuple[str, int | None, str]:
    """
    Processes all frames in a single frame directory using Tesseract.

    Args:
        frame_dir_absolute_path: Path to the specific directory containing frames.
        frame_dir_relative_path: Relative path of the frame directory (for checkpointing & output structure).
        output_root: Root path to save the output JSON.
        language: Language code for Tesseract.

    Returns:
        Tuple: (frame_dir_relative_path, processed_frame_count | None, status_message)
    """
    output_subdir = output_root / frame_dir_relative_path
    output_json_path = output_subdir / "tesseract_ocr.json"

    try:
        frames = sorted(
            frame_dir_absolute_path.glob("frame_*.png")
        )
        if not frames:
            frames = sorted(
                frame_dir_absolute_path.glob("frame_*.jpg")
            )

        if not frames:
            return (
                frame_dir_relative_path,
                0,
                "No frame_*.png or frame_*.jpg files found",
            )

        logging.info(
            f"[{frame_dir_relative_path}] Found {len(frames)} frames. Processing..."
        )
        output_subdir.mkdir(parents=True, exist_ok=True)

        ocr_results = {}
        processed_count = 0
        fail_count = 0

        for frame_path in tqdm(
            frames,
            desc=f"{frame_dir_relative_path[:25]} Frames", # Keep consistent desc
            leave=False,
            ncols=100,
        ):
            try: # Add try-except around individual frame processing
                ocr_text = process_image_with_tesseract(
                    frame_path, language=language
                )
                if ocr_text is not None:
                    ocr_results[frame_path.name] = ocr_text
                    processed_count += 1
                else:
                    ocr_results[frame_path.name] = "<<< OCR_FAILED_EMPTY_TEXT >>>" # More specific
                    fail_count += 1
            except Exception as frame_e:
                logging.error(f"Error processing frame {frame_path.name} in {frame_dir_relative_path}: {frame_e}", exc_info=True)
                ocr_results[frame_path.name] = f"<<< OCR_EXCEPTION: {frame_e} >>>"
                fail_count += 1


        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)

        status_msg = f"Success ({processed_count} OK, {fail_count} Failed)"
        logging.info(f"[{frame_dir_relative_path}] {status_msg}. Saved JSON: {output_json_path}")
        return (frame_dir_relative_path, processed_count, status_msg)

    except Exception as e:
        logging.error(
            f"Error in worker for {frame_dir_relative_path}: {e}",
            exc_info=True, # Ensure full traceback
        )
        return (frame_dir_relative_path, None, f"Error: {e}")


def _perform_tesseract_processing_pass(
    input_root: Path,
    output_root: Path,
    language: str,
    max_workers: int | None,
    effective_start_index: int | None,
    effective_end_index: int | None,
    checkpoint_file_name: str,
    # dry_run: bool, # Not implementing dry_run for tesseract for now to keep focus
) -> tuple[int, int, int, int]: # attempted_this_pass, successful_this_pass, failed_this_pass, total_frames_this_pass
    """
    Performs a single pass of discovering and processing frame directories with Tesseract.
    Returns statistics for the pass.
    """
    checkpoint_path = output_root / checkpoint_file_name
    processed_dirs_from_checkpoint = set()
    malformed_checkpoint_entries = 0
    max_path_len_checkpoint = 1024 # Arbitrary reasonable limit for a single path

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
                            f"Skipping malformed entry in Tesseract checkpoint {checkpoint_path} (line ~{line_num+1}): Contains newline chars."
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    if len(path_str) > max_path_len_checkpoint:
                        logging.warning(
                            f"Skipping malformed entry in Tesseract checkpoint {checkpoint_path} (line ~{line_num+1}): Exceeds max length {max_path_len_checkpoint}. Entry: \"{path_str[:80]}...\""
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    
                    processed_dirs_from_checkpoint.add(path_str)

            logging.info(
                f"Loaded {len(processed_dirs_from_checkpoint)} valid processed dirs from Tesseract checkpoint: {checkpoint_path}"
            )
            if malformed_checkpoint_entries > 0:
                logging.warning(f"Skipped {malformed_checkpoint_entries} potentially malformed entries during Tesseract checkpoint load.")

        except Exception as e:
            logging.error(f"Could not read Tesseract checkpoint {checkpoint_path}: {e}")
            processed_dirs_from_checkpoint.clear()
            malformed_checkpoint_entries = 0 # Reset counter on read error

    all_potential_frame_dirs_abs = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    all_potential_frame_dirs_rel = [
        d.relative_to(input_root).as_posix()
        for d in all_potential_frame_dirs_abs
    ]
    logging.info(
        f"Found {len(all_potential_frame_dirs_rel)} potential frame directories in {input_root}."
    )

    valid_processed_relative_paths = {
        path_str
        for path_str in processed_dirs_from_checkpoint
        if (input_root / path_str).is_dir() # Check existence
    }
    if len(valid_processed_relative_paths) < len(
        processed_dirs_from_checkpoint
    ):
        logging.info(
            f"{len(processed_dirs_from_checkpoint) - len(valid_processed_relative_paths)} "
            f"stale directory paths removed from checkpoint list."
        )

    relative_dirs_to_consider = sorted(
        [
            rel_path
            for rel_path in all_potential_frame_dirs_rel
            if rel_path not in valid_processed_relative_paths
        ]
    )
    logging.info(
        f"{len(relative_dirs_to_consider)} frame directories to consider after checkpoint."
    )

    total_dirs_to_consider = len(relative_dirs_to_consider)
    actual_start_index = effective_start_index if effective_start_index is not None else 0
    actual_end_index = (
        effective_end_index if effective_end_index is not None else total_dirs_to_consider
    )
    # Ensure indices are valid
    actual_start_index = max(0, actual_start_index)
    actual_end_index = min(total_dirs_to_consider, actual_end_index)


    if actual_start_index >= actual_end_index:
        logging.info(
            f"No new directories selected for processing in this pass (Start: {actual_start_index}, End: {actual_end_index}, Total considered: {total_dirs_to_consider})."
        )
        return 0, 0, 0, 0 # attempted, successful, failed, frames

    sliced_relative_dirs_for_this_run = relative_dirs_to_consider[
        actual_start_index:actual_end_index
    ]
    absolute_dirs_for_this_run_map = {
        rel_path: (input_root / rel_path)
        for rel_path in sliced_relative_dirs_for_this_run
    }
    target_count_this_pass = len(sliced_relative_dirs_for_this_run)

    if target_count_this_pass == 0:
        # This case should ideally be caught by the check above, but as a safeguard:
        logging.info("No frame directories selected for processing in this pass.")
        return 0, 0, 0, 0

    logging.info(
        f"Targeting {target_count_this_pass} frame directories for processing this pass "
        f"(slice index {actual_start_index} to {actual_end_index -1} of remaining dirs)."
    )

    pass_attempted_count = target_count_this_pass
    pass_successful_dir_count = 0
    pass_failed_dir_count = 0
    pass_total_frames_processed = 0

    current_max_workers = max_workers if max_workers else os.cpu_count()
    if not current_max_workers or current_max_workers < 1:
        current_max_workers = 1
    
    logging.info(f"Submitting {target_count_this_pass} Tesseract tasks to ProcessPoolExecutor with {current_max_workers} workers...")

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=current_max_workers) as executor:
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            future = executor.submit(
                _process_tesseract_directory,
                abs_path,
                rel_path,
                output_root,
                language,
            )
            futures.append(future)

        dir_progress_desc = "Processing Dirs (Tesseract Pass)"
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=target_count_this_pass,
            desc=dir_progress_desc,
            ncols=100,
            leave=True # Keep bar after completion for pass summary
        ):
            try:
                returned_relative_path, frame_count_or_none, status = future.result()
                if frame_count_or_none is not None:
                    pass_total_frames_processed += frame_count_or_none
                
                if status.startswith("Success") or status.startswith("No frame_*.png") : # Treat "No frames" as a kind of success for checkpointing
                    pass_successful_dir_count +=1
                    try:
                        with open(checkpoint_path, "a") as cp_f:
                            cp_f.write(f"{returned_relative_path}\n")
                        logging.debug(f"Tesseract Checkpointed: {returned_relative_path}")
                    except Exception as cp_e:
                        logging.error(f"Failed to write to Tesseract checkpoint {checkpoint_path} for {returned_relative_path}: {cp_e}")
                else:
                    pass_failed_dir_count += 1
                    logging.warning(
                        f"Tesseract processing for directory {returned_relative_path} failed with status: {status}"
                    )
            except concurrent.futures.process.BrokenProcessPool as bpp_exc:
                # Need to retrieve which future (and thus which path) caused this if possible
                # This is complex; for now, log generally and mark one failure.
                # To get the specific path, you'd need to map futures to paths *before* as_completed.
                pass_failed_dir_count +=1 # The directory processing failed
                logging.error(f"A Tesseract worker process died (BrokenProcessPool): {bpp_exc}. This directory will be marked as failed.", exc_info=True)

            except Exception as exc:
                pass_failed_dir_count += 1
                # Again, identifying the specific directory is tricky here without prior mapping
                logging.error(
                    f"An unexpected error occurred while processing a Tesseract directory future: {exc}",
                    exc_info=True
                )

    return pass_attempted_count, pass_successful_dir_count, pass_failed_dir_count, pass_total_frames_processed


def run_tesseract_pipeline(
    input_dir: str = "./extracted_frames",
    output_dir: str = "./tesseract_output",
    language: str = "eng",
    max_workers: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    checkpoint_file_name: str = ".processed_tesseract_dirs.log",
    daemon_mode: bool = False,
    watch_interval_seconds: int = 300,
):
    """
    Runs the Tesseract OCR pipeline on directories of extracted frames.
    Can run once or in daemon mode to continuously monitor for new directories.
    """
    from rich import print # Local import for this main orchestrator

    print("--- Initializing Tesseract OCR Pipeline --- ")
    if not check_tesseract_install():
        print(
            "[bold red]Tesseract installation check failed. Please ensure Tesseract is installed and accessible.[/bold red]"
        )
        return
    print("[green]Tesseract installation check successful.[/green]")

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.is_dir():
        print(f"[red]Input directory for Tesseract not found: {input_root}[/red]")
        return

    logging.info(f"Input Frame Directory (for Tesseract): {input_root}")
    logging.info(f"Output Tesseract JSON Directory: {output_root}")
    checkpoint_full_path = output_root / checkpoint_file_name
    logging.info(f"Tesseract Checkpoint file path: {str(checkpoint_full_path)}")
    logging.info(f"Tesseract Language: {language}")
    logging.info(f"Max Workers (Tesseract Dirs): {max_workers if max_workers else 'Default (CPU Count)'}")

    if daemon_mode:
        logging.info(
            f"[DAEMON MODE] Tesseract Pipeline activated. Watch interval: {watch_interval_seconds}s. "
            f"Initial slice for first pass: start_index={start_index}, end_index={end_index}. "
            "Subsequent passes will process all new directories."
        )
    else:
        logging.info(f"Single run mode for Tesseract. Slice to consider: start_index={start_index}, end_index={end_index}")

    run_iteration = 0
    overall_dirs_attempted = 0
    overall_dirs_successful = 0
    overall_dirs_failed = 0
    overall_total_frames_processed = 0
    
    first_pass_daemon = True
    pipeline_start_time = time.time()

    try:
        while True:
            run_iteration += 1
            logging.info(f"--- Starting Tesseract Processing Pass #{run_iteration} ---")
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
            
            pass_attempted, pass_successful, pass_failed, pass_frames = _perform_tesseract_processing_pass(
                input_root=input_root,
                output_root=output_root,
                language=language,
                max_workers=max_workers,
                effective_start_index=current_start_idx_for_pass,
                effective_end_index=current_end_idx_for_pass,
                checkpoint_file_name=checkpoint_file_name,
            )
            
            pass_duration = time.time() - pass_start_time

            logging.info(f"--- Tesseract Pass #{run_iteration} Summary ---")
            logging.info(f"Frame Dirs targeted in this Tesseract pass: {pass_attempted}")
            logging.info(f"Successfully processed (dirs) in this Tesseract pass: {pass_successful}")
            logging.info(f"Failed (dirs) in this Tesseract pass: {pass_failed}")
            logging.info(f"Total frames submitted to Tesseract in this pass: {pass_frames}")
            logging.info(f"Tesseract Pass #{run_iteration} duration: {pass_duration:.2f} seconds")

            overall_dirs_attempted += pass_attempted
            overall_dirs_successful += pass_successful
            overall_dirs_failed += pass_failed
            overall_total_frames_processed += pass_frames
            
            if not daemon_mode:
                break # Exit loop if not in daemon mode

            logging.info(f"[DAEMON MODE] Next Tesseract scan in {watch_interval_seconds} seconds. (Ctrl+C to stop)")
            time.sleep(watch_interval_seconds)
            
    except KeyboardInterrupt:
        print("[bold red]Tesseract Pipeline KeyboardInterrupt. Shutting down gracefully...[/bold red]")
    finally:
        pipeline_duration = time.time() - pipeline_start_time
        print("--- Overall Tesseract Processing Summary --- ")
        if daemon_mode:
            print(f"Total Tesseract passes executed: {run_iteration}")
        print(f"Total Frame Directories Targeted: {overall_dirs_attempted}")
        print(f"Total Successfully Processed Dirs: {overall_dirs_successful}")
        print(f"Total Failed Dirs: {overall_dirs_failed}")
        print(f"Total Frames Submitted for OCR: {overall_total_frames_processed}")
        print(f"Total Tesseract Pipeline Duration: {pipeline_duration:.2f} seconds")
        print(f"See {checkpoint_full_path} for completed Tesseract input directories.")

class TesseractPipelineCLI:
    """CLI for the Tesseract OCR Pipeline."""

    def run(
        self,
        input_dir: str = "./extracted_frames",
        output_dir: str = "./tesseract_output",
        language: str = "eng",
        max_workers: int | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        checkpoint_log: str = ".processed_tesseract_dirs.log", # Renamed from checkpoint_file_name
        daemon_mode: bool = False,
        watch_interval_seconds: int = 300,
    ):
        """
        Runs the Tesseract OCR pipeline on extracted frame directories.

        Args:
            input_dir: Root directory of extracted frames (from frame_pipeline.py).
            output_dir: Root directory to save Tesseract OCR JSON outputs.
            language: Language code for Tesseract (e.g., 'eng').
            max_workers: Max number of parallel directory processing. Defaults to os.cpu_count().
            start_index: 0-based index of the first directory to process (after checkpointing).
                         Applies to single run or the first pass of daemon mode.
            end_index: 0-based index *after* the last directory to process (after checkpointing).
                       Applies to single run or the first pass of daemon mode.
            checkpoint_log: Name for the checkpoint file within the output_dir.
            daemon_mode: If True, run continuously, scanning for new directories at intervals.
            watch_interval_seconds: Interval in seconds for daemon mode scans.
        """
        # Simple validation for worker count to avoid issues with ProcessPoolExecutor
        if max_workers is not None and max_workers < 1:
            logging.warning("max_workers for Tesseract cannot be less than 1. Setting to 1.")
            max_workers = 1

        run_tesseract_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            language=language,
            max_workers=max_workers,
            start_index=start_index,
            end_index=end_index,
            checkpoint_file_name=checkpoint_log, # Pass CLI's checkpoint_log as checkpoint_file_name
            daemon_mode=daemon_mode,
            watch_interval_seconds=watch_interval_seconds,
        )

if __name__ == "__main__":
    fire.Fire(TesseractPipelineCLI) 