import cv2
import logging
import math
from pathlib import Path
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_frames(video_path: str, output_dir: str, target_fps: int = 1) -> list[str]:
    """
    Extracts frames from a video at a specified target FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted frames.
        target_fps: Target FPS for extraction (default: 1).

    Returns:
        List of paths to extracted frames, or empty list on error.
    """
    video_path_obj = Path(video_path)
    output_dir_obj = Path(output_dir)

    if not video_path_obj.is_file():
        logging.error(f"Video file not found: {video_path}")
        return []

    # Ensure output directory exists
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir_obj.resolve()}")

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        logging.warning(
            f"Could not read native FPS from {video_path_obj.name}. Assuming 30."
        )
        native_fps = 30

    logging.info(f"Video Native FPS: {native_fps:.2f}, Target: {target_fps} FPS")
    if total_frames_estimate > 0:
        logging.info(f"Estimated total frames: {total_frames_estimate}")
    else:
        # Cannot estimate total frames. Progress bar will not show percentage.
        logging.warning("Could not get total frame count for progress bar.")

    frame_interval = 1  # Read every frame by default
    if target_fps > 0 and target_fps < native_fps:
        frame_interval = int(round(native_fps / target_fps))
    elif target_fps <= 0:
        logging.warning(
            "Target FPS must be positive. Defaulting to extracting all frames."
        )
        frame_interval = 1  # Extract all frames if target_fps is invalid
    # If target_fps >= native_fps, frame_interval remains 1
    # (extract all available frames)

    logging.info(f"Extracting every {frame_interval}-th frame.")

    frame_count = 0
    saved_frame_count = 0
    extracted_frame_paths = []

    # Initialize tqdm progress bar
    progress_bar = tqdm(
        total=total_frames_estimate if total_frames_estimate > 0 else None,
        unit="frame",
        desc=f"Extracting {video_path_obj.name[:35]}",  # Truncate name
        ncols=100,
        leave=False,  # Make inner bar disappear after completion
    )

    loop_start_time = time.time()
    while True:
        read_start_time = time.time()
        ret, frame = cap.read()
        read_duration = time.time() - read_start_time
        # Add more verbose logging, potentially conditional on a debug flag later
        # logging.debug(f"Frame {frame_count}: cap.read() took {read_duration:.4f}s")

        if not ret:
            logging.info(f"End of video reached or read error at frame {frame_count}.")
            break

        progress_bar.update(1)

        if frame_count % frame_interval == 0:
            current_second = int(round(frame_count / native_fps))
            frame_filename = f"frame_{current_second:06d}.jpg"
            frame_path = output_dir_obj / frame_filename

            save_start_time = time.time()
            try:
                if frame is not None and frame.size > 0:
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frame_paths.append(str(frame_path))
                    saved_frame_count += 1
                else:
                    # Log potentially empty frame
                    logging.warning(f"Frame {frame_count} empty/invalid, skip write.")
                # logging.debug(f"Frame {frame_count}: Saved to {frame_path}")
            except Exception as e:
                # Log write error
                logging.error(
                    f"Failed write frame {frame_count} to {frame_path.name}: {e}"
                )
                # Decide whether to continue or stop on write error

            save_duration = time.time() - save_start_time
            # logging.debug(f"Frame {frame_count}: cv2.imwrite() took {save_duration:.4f}s")

        frame_count += 1

        loop_end_time = time.time()
        if loop_end_time - loop_start_time > 10:
            logging.info(f"  ... still processing frame {frame_count}")
            loop_start_time = loop_end_time

    progress_bar.close()
    cap.release()
    logging.info(f"Finished video: {video_path_obj.name}")
    logging.info(f"Total frames read: {frame_count}, Saved frames: {saved_frame_count}")

    if not extracted_frame_paths:
        logging.warning(
            f"No frames extracted from {video_path_obj.name}. Output: {output_dir}"
        )

    return extracted_frame_paths


def get_human_readable_size(size_bytes: int) -> str:
    """Converts bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# Example usage (for testing when the script is run directly)
if __name__ == "__main__":
    # !!! IMPORTANT: Verify this path points to the actual video file !!!
    # Assumes video file is in the same directory as .info.json/.vtt
    # and has a common extension like .mp4 or .webm.
    # Please ADJUST the filename/extension if needed.
    test_video_path = (
        "/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/"
        "Databases_ Index optimization with dates [I0qXl321kjE]/"
        "Databases： Index optimization with dates [I0qXl321kjE].mp4"
    )  # <-- VERIFY THIS PATH & EXTENSION

    test_output_dir = "./temp_test_frames"  # <-- You can change this too

    print("--- Running Frame Extraction Test ---")
    print(f"--- Input Video: {test_video_path}")
    print(f"--- Output Dir:  {test_output_dir}")
    print("--- Ensure input video path is correct before proceeding. ---")

    video_file = Path(test_video_path)
    if video_file.is_file():
        video_size_bytes = video_file.stat().st_size
        video_size_readable = get_human_readable_size(video_size_bytes)
        print(f"--- Original Video Size: {video_size_readable}")

        extracted_paths = extract_frames(str(video_file), test_output_dir, target_fps=1)
        if extracted_paths:
            # Calculate output directory size
            output_dir_path = Path(test_output_dir)
            total_frames_size_bytes = sum(
                f.stat().st_size for f in output_dir_path.glob("**/*") if f.is_file()
            )
            frames_size_readable = get_human_readable_size(total_frames_size_bytes)

            print(
                f"\n✅ Extracted {len(extracted_paths)} frames to {output_dir_path.name}"
            )
            print(f"--- Total frames size: {frames_size_readable}")
            # Uncomment to see the first few paths:
            # print(f"   First 5 paths: {extracted_paths[:5]}")
        else:
            print("\n❌ Frame extraction failed or produced no frames.")
    else:
        print(f"\n❌ Test video file not found: {test_video_path}")
        print(f"   Please update the 'test_video_path' variable in {__file__}")
