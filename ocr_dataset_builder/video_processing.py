import logging
import math
import random
from pathlib import Path

import cv2
from tqdm import tqdm
from rich import print

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 1,
    max_dimension: int | None = 1024,
    max_frames_per_video: int | None = None,
) -> list[str]:
    """📸 Extracts, processes, and saves frames from a video file.

    This function performs a multi-step process:
    1.  **Initial Extraction**: Reads the video and extracts frames based on
        the `target_fps`. If `target_fps` is 1 (default), it aims to get
        one frame per second of video.
    2.  **Resizing (Optional)**: If `max_dimension` is specified, each
        extracted frame is resized so that its largest dimension (width or
        height) does not exceed `max_dimension`, maintaining aspect ratio.
    3.  **Sampling (Optional)**: If `max_frames_per_video` is specified and
        the number of extracted (and resized) frames exceeds this value,
        a random subset of `max_frames_per_video` frames is selected.
    4.  **Saving**: The selected frames are saved as JPEG images
        (`frame_XXXXXX.jpg`, where XXXXXX is the second mark of the frame
        in the video) in the specified `output_dir`.

    The function includes progress bars for frame extraction and saving,
    and logs its operations for monitoring.

    Args:
        video_path (str): The absolute or relative path to the input video
                          file.
        output_dir (str): The directory where extracted frames will be saved.
                          It will be created if it doesn't exist.
        target_fps (int, optional): The desired frames per second to extract.
                                    Defaults to 1. If set to 0 or a
                                    negative value, it defaults to extracting
                                    every frame.
        max_dimension (int | None, optional): The maximum size (in pixels)
                                              for the largest dimension of
                                              the extracted frames. If None,
                                              no resizing is performed.
                                              Defaults to 1024.
        max_frames_per_video (int | None, optional): The maximum number of
                                                     frames to randomly
                                                     sample and save from the
                                                     video. If None, all
                                                     frames meeting the FPS and
                                                     resizing criteria are
                                                     saved. Defaults to None.

    Returns:
        list[str]: A list of strings, where each string is the absolute path
                   to a saved frame. Returns an empty list if the video
                   cannot be opened or no frames are saved.

    Raises:
        Logs errors for issues like file not found, inability to open video,
        or frame writing failures, but aims to not hard crash, returning an
        empty list instead.
    """
    video_path_obj = Path(video_path)
    output_dir_obj = Path(output_dir)

    if not video_path_obj.is_file():
        logging.error(f"Video file not found: {video_path}")
        return []

    output_dir_obj.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory set to: {output_dir_obj.resolve()}")
    if max_dimension:
        logging.info(f"Resizing frames to max dimension: {max_dimension}px")
    else:
        logging.info("Resizing disabled (max_dimension=None)")
    if max_frames_per_video:
        logging.info(f"Sampling enabled: Max {max_frames_per_video} frames per video.")
    else:
        logging.info("Sampling disabled (max_frames_per_video=None)")

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
        logging.warning("Could not get total frame count for progress bar.")

    frame_interval = 1
    if target_fps > 0 and target_fps < native_fps:
        frame_interval = int(round(native_fps / target_fps))
    elif target_fps <= 0:
        logging.warning("Target FPS must be positive. Defaulting to every frame.")
        frame_interval = 1

    logging.info(f"Extracting every {frame_interval}-th frame.")

    frame_count = 0
    candidate_frames = []

    progress_bar = tqdm(
        total=total_frames_estimate if total_frames_estimate > 0 else None,
        unit="frame",
        desc=f"Extracting {video_path_obj.name[:35]}",
        ncols=100,
        leave=False,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        progress_bar.update(1)

        if frame_count % frame_interval == 0:
            if frame is None or frame.size == 0:
                logging.warning(f"Frame {frame_count} empty/invalid, skipping.")
                frame_count += 1
                continue

            current_second = int(round(frame_count / native_fps))
            frame_filename = f"frame_{current_second:06d}.jpg"
            frame_path = output_dir_obj / frame_filename

            frame_to_process = frame

            # --- Resizing Logic --- (Applied before adding to candidates)
            if max_dimension and max_dimension > 0:
                height, width = frame.shape[:2]
                current_max_dim = max(height, width)
                if current_max_dim > max_dimension:
                    scale = max_dimension / current_max_dim
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    try:
                        resized_frame = cv2.resize(
                            frame, (new_width, new_height), interpolation=interpolation
                        )
                        frame_to_process = resized_frame
                    except Exception as resize_e:
                        logging.error(
                            f"Failed resizing frame {frame_count}: {resize_e}. Using original."
                        )
                        frame_to_process = frame

            # Add candidate frame (path and data) to list
            candidate_frames.append((frame_path, frame_to_process))

        frame_count += 1

    progress_bar.close()
    cap.release()
    logging.info(
        f"Finished reading video: {video_path_obj.name}. Found {len(candidate_frames)} candidate frames."
    )

    # --- Sampling Logic --- Start
    frames_to_save = []
    if (
        max_frames_per_video
        and max_frames_per_video > 0
        and len(candidate_frames) > max_frames_per_video
    ):
        logging.info(
            f"Sampling {max_frames_per_video} frames from {len(candidate_frames)} candidates..."
        )
        selected_indices = sorted(
            random.sample(range(len(candidate_frames)), max_frames_per_video)
        )
        frames_to_save = [candidate_frames[i] for i in selected_indices]
        logging.info(f"Selected {len(frames_to_save)} frames after sampling.")
    else:
        # No sampling needed or not enough candidates
        frames_to_save = candidate_frames
        if max_frames_per_video and max_frames_per_video > 0:
            logging.info(
                f"Skipping sampling: Found {len(candidate_frames)} candidates, which is <= max {max_frames_per_video}."
            )
        # Else: Sampling was disabled (max_frames_per_video is None)
    # --- Sampling Logic --- End

    # --- Saving Logic --- Start
    saved_frame_count = 0
    extracted_frame_paths = []
    if frames_to_save:
        logging.info(f"Saving {len(frames_to_save)} final frames...")
        save_progress_bar = tqdm(
            frames_to_save,
            total=len(frames_to_save),
            desc="Saving Frames",
            ncols=100,
            leave=False,
        )
        for frame_path, frame_data in save_progress_bar:
            try:
                cv2.imwrite(str(frame_path), frame_data)
                extracted_frame_paths.append(str(frame_path))
                saved_frame_count += 1
            except Exception as e:
                logging.error(f"Failed write frame to {frame_path.name}: {e}")
        save_progress_bar.close()
    else:
        logging.warning("No frames selected or available to save.")
    # --- Saving Logic --- End

    logging.info(f"Final saved frames: {saved_frame_count}")

    if not extracted_frame_paths:
        logging.warning(
            f"No frames saved for {video_path_obj.name}. Output: {output_dir}"
        )

    return extracted_frame_paths


def get_human_readable_size(size_bytes: int) -> str:
    """💾 Converts a size in bytes to a human-readable string format.

    Supports B, KB, MB, GB, TB.

    Args:
        size_bytes (int): The size in bytes.

    Returns:
        str: A string representing the size in a human-readable unit
             (e.g., "1.23 MB"). Returns "0 B" if input is non-positive.
    """
    if size_bytes <= 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


# Example usage - Now includes max_frames_per_video
if __name__ == "__main__":
    test_video_path = (
        "/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/"
        "Databases_ Index optimization with dates [I0qXl321kjE]/"
        "Databases： Index optimization with dates [I0qXl321kjE].mp4"
    )

    test_output_dir = "./temp_test_frames_sampled"
    test_max_dimension = 768
    test_max_frames = 15  # Example: Sample down to 15 frames

    print("--- Running Frame Extraction Test (Resize + Sample) ---")
    print(f"--- Input Video: {test_video_path}")
    print(f"--- Output Dir:  {test_output_dir}")
    print(
        f"--- Max Dimension: {test_max_dimension if test_max_dimension else 'Original'}"
    )
    print(f"--- Max Frames/Vid: {test_max_frames if test_max_frames else 'All'}")
    print("--- Ensure input video path is correct before proceeding. ---")

    video_file = Path(test_video_path)
    if video_file.is_file():
        video_size_bytes = video_file.stat().st_size
        video_size_readable = get_human_readable_size(video_size_bytes)
        print(f"--- Original Video Size: {video_size_readable}")

        extracted_paths = extract_frames(
            str(video_file),
            test_output_dir,
            target_fps=1,
            max_dimension=test_max_dimension,
            max_frames_per_video=test_max_frames,  # Pass sampling limit
        )
        if extracted_paths:
            output_dir_path = Path(test_output_dir)
            total_frames_size_bytes = sum(
                f.stat().st_size for f in output_dir_path.glob("**/*") if f.is_file()
            )
            frames_size_readable = get_human_readable_size(total_frames_size_bytes)

            print(f"\n✅ Saved {len(extracted_paths)} frames to {output_dir_path.name}")
            print(f"--- Total frames size: {frames_size_readable}")
        else:
            print("\n❌ Frame extraction/saving failed or produced no frames.")
    else:
        print(f"\n❌ Test video file not found: {test_video_path}")
        print(f"   Please update the 'test_video_path' variable in {__file__}")
