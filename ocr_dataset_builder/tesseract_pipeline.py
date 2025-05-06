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
from ocr_dataset_builder.tesseract_processing import (
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
    frame_dir_absolute_path: Path,  # Renamed from video_dir
    frame_dir_relative_path: str,  # Added for checkpointing
    input_root: Path,  # Used for relative path in output, but frame_dir_relative_path is primary now
    output_root: Path,
    language: str = "eng",
) -> tuple[str, int | None, str]:  # First element is now relative_path
    """
    Processes all frames in a single frame directory using Tesseract.

    Args:
        frame_dir_absolute_path: Path to the specific directory containing frames.
        frame_dir_relative_path: Relative path of the frame directory (for checkpointing).
        input_root: Root path of the input frames directory (mostly for context, not direct use if rel_path is good).
        output_root: Root path to save the output JSON.
        language: Language code for Tesseract.

    Returns:
        Tuple: (frame_dir_relative_path, processed_frame_count | None, status_message)
    """
    # dir_name = frame_dir_absolute_path.name # Use relative_path for logging/reporting
    output_subdir = output_root / frame_dir_relative_path
    output_json_path = output_subdir / "tesseract_ocr.json"

    try:
        frames = sorted(
            frame_dir_absolute_path.glob("frame_*.png")
        )  # Assuming .png from frame_pipeline
        if not frames:
            frames = sorted(
                frame_dir_absolute_path.glob("frame_*.jpg")
            )  # Fallback for jpg

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
            desc=f"{frame_dir_relative_path[:25]} Frames",
            leave=False,
            ncols=100,
        ):
            ocr_text = process_image_with_tesseract(frame_path, language=language)
            if ocr_text is not None:
                ocr_results[frame_path.name] = ocr_text
                processed_count += 1
            else:
                ocr_results[frame_path.name] = "<<< OCR_FAILED >>>"
                fail_count += 1

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)

        status_msg = f"Success ({processed_count} OK, {fail_count} Failed)"
        logging.info(f"[{frame_dir_relative_path}] {status_msg}. Saved JSON.")
        return (frame_dir_relative_path, processed_count, status_msg)

    except Exception as e:
        logging.error(
            f"Error in worker for {frame_dir_relative_path}: {e}", exc_info=False
        )
        return (frame_dir_relative_path, None, f"Error: {e}")


def run_tesseract_pipeline(
    input_dir: str = "./extracted_frames",
    output_dir: str = "./tesseract_output",
    language: str = "eng",
    max_workers: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    checkpoint_file_name: str = ".processed_tesseract_dirs.log",
):
    """
    Runs Tesseract OCR processing pipeline on extracted frame directories.

    Args:
        input_dir: Root directory containing extracted frame subdirectories.
        output_dir: Root directory to save the Tesseract OCR results & checkpoint.
        language: Tesseract language code (e.g., 'eng').
        max_workers: Max parallel processes for directories. Default: CPU count.
        start_index: 0-based index of the first frame directory to process (after checkpoint).
        end_index: 0-based index of the directory *after* the last one (after checkpoint).
        checkpoint_file_name: Log file for processed directories.
    """
    from rich import print  # Keep import here due to conditional Tesseract check

    print("--- Starting Tesseract OCR Pipeline --- ")
    if not check_tesseract_install():
        print(
            "[red]Tesseract check failed. Please install Tesseract OCR "
            "and ensure it's in PATH.[/red]"
        )
        return
    print("[green]Tesseract installation check successful.[/green]")

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input Frame Directory: {input_root}")
    logging.info(f"Output JSON Directory: {output_root}")
    logging.info(f"Checkpoint file: {output_root / checkpoint_file_name}")
    logging.info(f"Tesseract Language: {language}")
    logging.info(
        f"Max Workers (Dirs): {max_workers if max_workers else os.cpu_count()}"
    )

    # --- Checkpoint Loading ---
    checkpoint_path = output_root / checkpoint_file_name
    processed_dirs_from_checkpoint = set()
    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line in f:
                    processed_dirs_from_checkpoint.add(line.strip())
            logging.info(
                f"Loaded {len(processed_dirs_from_checkpoint)} processed dirs from checkpoint."
            )
        except Exception as e:
            logging.error(
                f"Could not read checkpoint {checkpoint_path}: {e}. Processing all."
            )
            processed_dirs_from_checkpoint.clear()

    # --- Re-indexing and Filtering ---
    all_potential_frame_dirs_abs = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    all_potential_frame_dirs_rel = [
        d.relative_to(input_root).as_posix() for d in all_potential_frame_dirs_abs
    ]
    logging.info(
        f"Found {len(all_potential_frame_dirs_rel)} potential frame directories in input."
    )

    valid_processed_relative_paths = {
        path_str
        for path_str in processed_dirs_from_checkpoint
        if (input_root / path_str).is_dir()
    }
    if len(valid_processed_relative_paths) < len(processed_dirs_from_checkpoint):
        logging.info(
            f"{len(processed_dirs_from_checkpoint) - len(valid_processed_relative_paths)} "
            f"stale directory paths removed from checkpoint."
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
    actual_start_index = start_index if start_index is not None else 0
    actual_end_index = end_index if end_index is not None else total_dirs_to_consider
    actual_start_index = max(0, actual_start_index)
    actual_end_index = min(total_dirs_to_consider, actual_end_index)

    if actual_start_index >= actual_end_index:
        logging.warning(
            f"Start index ({actual_start_index}) >= end index ({actual_end_index}). No directories to process."
        )
        return

    sliced_relative_dirs_for_this_run = relative_dirs_to_consider[
        actual_start_index:actual_end_index
    ]
    absolute_dirs_for_this_run_map = {
        rel_path: (input_root / rel_path)
        for rel_path in sliced_relative_dirs_for_this_run
    }
    target_count = len(sliced_relative_dirs_for_this_run)

    if target_count == 0:
        logging.info("No frame directories selected for processing in this run.")
        return

    logging.info(
        f"Targeting {target_count} frame directories for actual processing this run "
        f"(slice index {actual_start_index} to {actual_end_index -1} of remaining dirs)"
    )

    start_time = time.time()
    processed_dir_count_this_run = 0
    successful_dir_count_this_run = 0
    failed_dir_count_this_run = 0
    total_frames_processed_this_run = 0

    if not max_workers:
        max_workers = os.cpu_count()

    logging.info(f"Submitting {target_count} tasks to ProcessPoolExecutor...")
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            future = executor.submit(
                _process_tesseract_directory,
                abs_path,  # Absolute path for processing
                rel_path,  # Relative path for checkpointing
                input_root,
                output_root,
                language,
            )
            futures.append(future)

        progress_bar = tqdm(
            concurrent.futures.as_completed(futures),
            total=target_count,
            unit="dir",
            desc="Processing Dirs (Tesseract)",
            ncols=100,
        )

        for future in progress_bar:
            processed_dir_count_this_run += 1
            try:
                returned_relative_path, frame_count_or_none, status = future.result()
                is_success = frame_count_or_none is not None and status.startswith(
                    "Success"
                )

                if is_success:
                    successful_dir_count_this_run += 1
                    if frame_count_or_none is not None:  # Should be true if success
                        total_frames_processed_this_run += frame_count_or_none
                    # --- Checkpoint Saving ---
                    try:
                        with open(checkpoint_path, "a") as cp_file:
                            cp_file.write(f"{returned_relative_path}\n")
                        logging.debug(
                            f"Checkpointed Tesseract dir: {returned_relative_path}"
                        )
                    except Exception as cp_e:
                        logging.error(
                            f"Failed to write Tesseract checkpoint for {returned_relative_path}: {cp_e}"
                        )
                else:
                    failed_dir_count_this_run += 1
                    logging.warning(
                        f"Tesseract dir '{returned_relative_path}' failed/issues: {status}"
                    )
            except Exception as exc:
                failed_dir_count_this_run += 1
                logging.error(
                    f"Error retrieving Tesseract dir result: {exc}", exc_info=True
                )

    end_time = time.time()
    duration = end_time - start_time
    print("--- Tesseract Pipeline Finished ---")
    logging.info(f"Total Dirs Targeted this run: {target_count}")
    logging.info(f"Total Dirs Attempted this run: {processed_dir_count_this_run}")
    logging.info(f"Successful Dirs this run: {successful_dir_count_this_run}")
    logging.info(f"Failed/Partial Dirs this run: {failed_dir_count_this_run}")
    logging.info(
        f"Total Frames Processed (successful dirs, this run): {total_frames_processed_this_run}"
    )
    logging.info(f"Total processing time: {duration:.2f} seconds.")
    logging.info(
        f"See {checkpoint_path} for successfully Tesseract-processed directories."
    )


if __name__ == "__main__":
    fire.Fire(run_tesseract_pipeline)
