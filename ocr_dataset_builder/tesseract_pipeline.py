import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path

import fire
from tqdm import tqdm

# Ensure the package modules can be found
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import necessary functions AFTER potentially modifying sys.path
from ocr_dataset_builder.tesseract_processing import (
    process_image_with_tesseract,
    check_tesseract_install,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _process_tesseract_directory(
    video_dir: Path,
    input_root: Path,
    output_root: Path,
    language: str = "eng",
) -> tuple[str, int | None, str]:
    """
    Processes all frames in a single video directory using Tesseract.

    Args:
        video_dir: Path to the specific video directory containing frames.
        input_root: Root path of the input frames directory (for relative path calculation).
        output_root: Root path to save the output JSON.
        language: Language code for Tesseract.

    Returns:
        Tuple: (directory_name, processed_frame_count | None, status_message)
    """
    dir_name = video_dir.name
    output_subdir = output_root / video_dir.relative_to(input_root)
    output_json_path = output_subdir / "tesseract_ocr.json"

    try:
        frames = sorted(video_dir.glob("frame_*.jpg"))
        if not frames:
            return (dir_name, 0, "No frame_*.jpg files found")

        # Shortened log message
        logging.info(f"[{dir_name}] Found {len(frames)} frames. Processing...")
        output_subdir.mkdir(parents=True, exist_ok=True)

        ocr_results = {}
        processed_count = 0
        fail_count = 0

        # Process frames sequentially within a directory
        for frame_path in tqdm(
            frames, desc=f"{dir_name[:25]} Frames", leave=False, ncols=100
        ):
            ocr_text = process_image_with_tesseract(frame_path, language=language)
            if ocr_text is not None:
                ocr_results[frame_path.name] = ocr_text
                processed_count += 1
            else:
                # Placeholder for failed OCR
                ocr_results[frame_path.name] = "<<< OCR_FAILED >>>"
                fail_count += 1

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)

        status_msg = f"Success ({processed_count} OK, {fail_count} Failed)"
        # Shortened log message
        logging.info(f"[{dir_name}] {status_msg}. Saved JSON.")
        return (dir_name, processed_count, status_msg)

    except Exception as e:
        # Shortened log message
        logging.error(f"Error in worker for {dir_name}: {e}", exc_info=False)
        return (dir_name, None, f"Error: {e}")


def run_tesseract_pipeline(
    input_dir: str = "./extracted_frames",
    output_dir: str = "./tesseract_output",
    language: str = "eng",
    max_workers: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
):
    """
    Runs Tesseract OCR processing pipeline on extracted frame directories.

    Args:
        input_dir: Root directory containing extracted frame subdirectories.
        output_dir: Root directory to save the Tesseract OCR results.
        language: Tesseract language code (e.g., 'eng').
        max_workers: Max parallel processes for directories. Default: CPU count.
        start_index: 0-based index of the first video subdirectory to process.
        end_index: 0-based index of the subdirectory *after* the last one.
    """
    # Use rich print directly since it's imported conditionally later
    from rich import print

    print("--- Starting Tesseract OCR Pipeline --- ")
    if not check_tesseract_install():
        # Wrapped long line
        print(
            "[red]Tesseract check failed. Please install Tesseract OCR "
            "and ensure it's in PATH.[/red]"
        )
        return
    print("[green]Tesseract installation check successful.[/green]")

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input Frame Directory: {input_root.resolve()}")
    logging.info(f"Output JSON Directory: {output_root.resolve()}")
    logging.info(f"Tesseract Language: {language}")
    # Wrapped long line
    logging.info(f"Max Workers (Dirs): {max_workers if max_workers else 'CPU Count'}")

    all_video_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not all_video_dirs:
        # Wrapped long line
        logging.error(f"No video subdirectories found in {input_root}. Exiting.")
        return

    total_dirs = len(all_video_dirs)
    logging.info(f"Found {total_dirs} potential video directories.")

    actual_start_index = start_index if start_index is not None else 0
    actual_end_index = end_index if end_index is not None else total_dirs
    actual_start_index = max(0, actual_start_index)
    actual_end_index = min(total_dirs, actual_end_index)

    if actual_start_index >= actual_end_index:
        # Wrapped long line
        logging.warning(
            f"Start index ({actual_start_index}) >= end index "
            f"({actual_end_index}). No directories."
        )
        return

    directories_to_process = all_video_dirs[actual_start_index:actual_end_index]
    target_count = len(directories_to_process)

    # Wrapped long line
    logging.info(
        f"Targeting {target_count} directories "
        f"(index {actual_start_index} to {actual_end_index - 1})"
    )

    start_time = time.time()
    processed_dir_count = 0
    successful_dir_count = 0
    failed_dir_count = 0
    total_frames_processed = 0

    # Wrapped long line
    logging.info(f"Submitting {target_count} tasks to ProcessPoolExecutor...")
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for video_dir in directories_to_process:
            future = executor.submit(
                _process_tesseract_directory,
                video_dir,
                input_root,
                output_root,
                language,
            )
            futures.append(future)

        results_iterator = concurrent.futures.as_completed(futures)
        progress_bar = tqdm(
            results_iterator,
            total=target_count,
            unit="dir",
            desc="Processing Dirs (Tesseract)",
            ncols=100,
        )

        for future in progress_bar:
            processed_dir_count += 1
            try:
                # Added type hint for clarity
                result: tuple[str, int | None, str] = future.result()
                dir_name, frame_count_or_none, status = result
                is_success = frame_count_or_none is not None and status.startswith(
                    "Success"
                )
                if is_success:
                    successful_dir_count += 1
                    total_frames_processed += frame_count_or_none
                else:
                    failed_dir_count += 1
                    # Wrapped long line
                    logging.warning(f"Dir '{dir_name}' failed/issues: {status}")
            except Exception as exc:
                failed_dir_count += 1
                # Wrapped long line
                logging.error(f"Error retrieving dir result: {exc}", exc_info=True)

    end_time = time.time()
    duration = end_time - start_time
    print("--- Tesseract Pipeline Finished ---")
    logging.info(f"Total Directories Targeted: {target_count}")
    logging.info(f"Total Directories Attempted: {processed_dir_count}")
    logging.info(f"Successful Directories: {successful_dir_count}")
    logging.info(f"Failed/Partial Directories: {failed_dir_count}")
    # Wrapped long line
    logging.info(f"Total Frames Processed (successful dirs): {total_frames_processed}")
    logging.info(f"Total processing time: {duration:.2f} seconds.")


if __name__ == "__main__":
    fire.Fire(run_tesseract_pipeline)
