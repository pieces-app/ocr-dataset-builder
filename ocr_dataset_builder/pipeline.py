import logging
from pathlib import Path
import sys
import time
import concurrent.futures  # Added for parallelism
import shutil  # Added for file copying
from tqdm import tqdm

# Ensure the package modules can be found
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ocr_dataset_builder.video_processing import extract_frames

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

VIDEO_EXTENSIONS = [".mp4", ".webm", ".mkv", ".avi", ".mov"]  # Add more if needed


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
    video_dir: Path,
    input_root: Path,
    output_root: Path,
    target_fps: int,
) -> tuple[str, int | None, str]:
    """Helper function to process a single video directory."""
    dir_name = video_dir.name
    metadata_copied = False
    try:
        video_file = find_video_file(video_dir)
        metadata_file = find_metadata_file(video_dir)

        if not video_file:
            return (dir_name, None, "No video file found")

        # Determine mirrored output directory path
        relative_path = video_dir.relative_to(input_root)
        output_video_frames_dir = output_root / relative_path
        # extract_frames will create this directory

        # Copy metadata file first
        if metadata_file:
            try:
                # Ensure output dir exists *before* copying metadata
                output_video_frames_dir.mkdir(parents=True, exist_ok=True)
                target_metadata_path = output_video_frames_dir / metadata_file.name
                shutil.copy2(
                    metadata_file, target_metadata_path
                )  # copy2 preserves metadata
                metadata_copied = True
                logging.info(
                    f"  Copied metadata {metadata_file.name} to {output_video_frames_dir.name}"
                )
            except Exception as meta_e:
                logging.error(
                    f"  Failed to copy metadata {metadata_file.name} "
                    f"for {dir_name}: {meta_e}"
                )
        else:
            logging.warning(f"  No .info.json metadata file found for {dir_name}")

        # Now extract frames
        extracted_paths = extract_frames(
            str(video_file),
            str(output_video_frames_dir),
            target_fps=target_fps,
        )

        if extracted_paths:
            status_suffix = (
                " (Metadata OK)" if metadata_copied else " (Metadata MISSING/ERROR)"
            )
            return (dir_name, len(extracted_paths), "Success" + status_suffix)
        else:
            status_suffix = (
                " (Metadata OK)" if metadata_copied else " (Metadata MISSING/ERROR)"
            )
            return (dir_name, 0, "Extraction failed/0 frames" + status_suffix)

    except Exception as e:
        logging.error(f"Error in worker for {dir_name}: {e}", exc_info=False)
        return (dir_name, None, f"Error: {e}")


def process_dataset_videos(
    input_root_dir: str | Path,
    output_root_dir: str | Path,
    target_fps: int = 1,
    start_index: int | None = None,
    end_index: int | None = None,
    max_workers: int | None = None,  # Number of parallel processes
):
    """
    Processes a slice of video directories concurrently using ProcessPoolExecutor.

    Args:
        input_root_dir: Root directory containing video subdirectories.
        output_root_dir: Root directory for saving extracted frames.
        target_fps: Target frames per second for extraction.
        start_index: 0-based index of the first video directory to process.
                     Defaults to 0 if None.
        end_index: 0-based index after the last directory to process.
                   Defaults to processing all if None.
        max_workers: Max number of processes to use. Defaults to os.cpu_count().
    """
    input_root = Path(input_root_dir)
    output_root = Path(output_root_dir)

    if not input_root.is_dir():
        logging.error(f"Input dataset directory not found: {input_root}")
        return

    logging.info(f"Starting video processing from: {input_root}")
    logging.info(f"Outputting frames to: {output_root}")

    # Get sorted list of directories
    video_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    total_dirs = len(video_dirs)
    logging.info(f"Found {total_dirs} potential video directories.")

    # Determine the slice
    actual_start_index = start_index if start_index is not None else 0
    actual_end_index = end_index if end_index is not None else total_dirs

    # Validate indices
    if not (0 <= actual_start_index <= total_dirs):
        logging.error(
            f"Invalid start_index: {start_index}. Must be 0 <= index <= {total_dirs}."
        )
        return
    if not (actual_start_index <= actual_end_index <= total_dirs):
        logging.error(
            f"Invalid end_index: {end_index}. Must be start_index <= index <= {total_dirs}."
        )
        return

    directories_to_process = video_dirs[actual_start_index:actual_end_index]
    target_count = len(directories_to_process)

    if target_count == 0:
        logging.warning("No directories selected based on indices.")
        return

    logging.info(
        f"Targeting {target_count} directories (index {actual_start_index} to {actual_end_index - 1}) "
        f"using up to {max_workers if max_workers else 'all available'} workers."
    )
    # Confirm the number of directories being submitted
    logging.info(f"Submitting {target_count} tasks to the executor.")

    processed_count = 0
    successful_count = 0
    failed_count = 0
    no_video_count = 0
    total_frames_extracted = 0
    start_time = time.time()

    futures = []
    # Use ProcessPoolExecutor for CPU-bound tasks like video decoding
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for video_dir in directories_to_process:
            future = executor.submit(
                _process_single_video_dir,
                video_dir,
                input_root,
                output_root,
                target_fps,
            )
            futures.append(future)

        # Use tqdm to show progress as futures complete
        results_iterator = concurrent.futures.as_completed(futures)
        progress_bar = tqdm(
            results_iterator,
            total=target_count,
            unit="dir",
            desc="Processing Dirs",
            ncols=100,
        )

        for future in progress_bar:
            try:
                dir_name, frame_count_or_none, status = future.result()
                processed_count += 1
                if status.startswith("Success") and frame_count_or_none is not None:
                    successful_count += 1
                    total_frames_extracted += frame_count_or_none
                    # Temporarily disable per-dir logging to avoid tqdm interference
                    # logging.info(
                    #     f"Dir {dir_name}: OK ({frame_count_or_none} frames)"
                    # )
                elif status == "No video file found":
                    no_video_count += 1
                    # logging.warning(f"Dir {dir_name}: SKIPPED (no video found)")
                else:  # Handles errors and extraction failures
                    failed_count += 1
                    # logging.error(f"Dir {dir_name}: FAILED ({status})")

                # Optionally update tqdm description
                # progress_bar.set_description(f"Processed {dir_name[:20]}...")
            except Exception as exc:
                # This catches errors in future.result() itself, though unlikely
                # if _process_single_video_dir handles its exceptions.
                failed_count += 1
                logging.error(f"Error retrieving result for a task: {exc}")

    end_time = time.time()
    duration = end_time - start_time
    logging.info("===")
    logging.info("Finished processing dataset slice.")
    logging.info(f"  Targeted: {target_count} directories")
    logging.info(f"  Attempted: {processed_count} directories")
    logging.info(f"  Successful: {successful_count} directories")
    logging.info(f"  Skipped (No Video): {no_video_count} directories")
    logging.info(f"  Failed: {failed_count} directories")
    logging.info(f"  Total Frames Extracted: {total_frames_extracted}")
    logging.info(f"  Total processing time: {duration:.2f} seconds.")


if __name__ == "__main__":
    # --- Configuration for example run ---
    INPUT_DATASET_PATH = "/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/"
    OUTPUT_FRAMES_PATH = "./extracted_frames"
    TARGET_FPS = 1
    START_INDEX = 0
    END_INDEX = 10  # Process first 10 dirs (0-9)
    MAX_WORKERS = 4  # Adjust based on your CPU cores (e.g., os.cpu_count())
    # Set to None to use default (usually core count)

    print("Running pipeline example (concurrently):")
    print(f" Input Dir:   {INPUT_DATASET_PATH}")
    print(f" Output Dir:  {OUTPUT_FRAMES_PATH}")
    print(f" Target FPS:  {TARGET_FPS}")
    print(f" Start Index: {START_INDEX if START_INDEX is not None else '0'}")
    print(f" End Index:   {END_INDEX if END_INDEX is not None else 'All'}")
    print(f" Max Workers: {MAX_WORKERS if MAX_WORKERS is not None else 'Default'}")
    print("---")

    if not Path(INPUT_DATASET_PATH).is_dir():
        print(f"ERROR: Input dataset dir not found: {INPUT_DATASET_PATH}")
        print("Please update INPUT_DATASET_PATH.")
    else:
        process_dataset_videos(
            INPUT_DATASET_PATH,
            OUTPUT_FRAMES_PATH,
            target_fps=TARGET_FPS,
            start_index=START_INDEX,
            end_index=END_INDEX,
            max_workers=MAX_WORKERS,  # Pass max_workers
        )
