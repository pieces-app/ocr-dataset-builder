import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil # For dummy data creation

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms # For example transform

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OcrMultimodalDataset(Dataset):
    """
    PyTorch Dataset for combining frame images, LLM task outputs, and Tesseract OCR text.

    This dataset is designed to work with a specific directory structure where frame images,
    LLM (Large Language Model) outputs, and Tesseract OCR outputs are organized per video.

    **Expected Directory Structure:**

    1.  `frames_root_dir/`
        ├── VIDEO_ID_1/
        │   ├── frame_0000.png
        │   ├── frame_0001.png
        │   └── ...
        └── VIDEO_ID_2/
            ├── frame_0000.png
            └── ...

    2.  `llm_outputs_root_dir/`
        ├── VIDEO_ID_1/
        │   ├── llm_output_batch_0001.json
        │   ├── llm_output_batch_0002.json
        │   └── ... (one or more batch files per video)
        └── VIDEO_ID_2/
            └── ...

        Each `llm_output_batch_xxxx.json` file is expected to contain:
        {
            "task1_raw_ocr": ["text_for_frame1_in_batch", "text_for_frame2_in_batch", ...],
            "task2_augmented": ["text_for_frame1_in_batch", ...], 
            "task3_cleaned": ["text_for_frame1_in_batch", ...],
            "task4_markdown": ["markdown_for_frame1_in_batch", ...],
            "task5_summary": "Narrative summary applicable to all frames in this batch.",
            "frame_ids": ["frame_xxxxxx.png", ...] (Optional, not currently used by loader but good practice)
        }
        Note: For "task2", the loader also checks for an alternate key specified by `alternate_task2_key`.

    3.  `tesseract_outputs_root_dir/`
        ├── VIDEO_ID_1/
        │   └── tesseract_ocr.json
        └── VIDEO_ID_2/
            └── ...

        Each `tesseract_ocr.json` file is expected to be a dictionary mapping frame filenames
        to their Tesseract OCR text:
        {
            "frame_0000.png": "Tesseract text for frame 0",
            "frame_0001.png": "Tesseract text for frame 1",
            ...
        }

    **Data Aggregation and Mapping:**
    - For each video, LLM task data (Task 1-4) from all its sorted batch files are concatenated.
    - The `task5_summary` from each LLM batch file is associated with all frames processed in that batch.
    - These concatenated LLM task outputs are then mapped 1-to-1 with the sorted frame image files for that video.
    - Tesseract OCR data is mapped directly using frame filenames.

    **F:i-1 Notation:**
    - The dataset handles a special notation `F:index` (e.g., `F:0`, `F:15`) within the LLM task strings 
      (Tasks 1-4). This indicates that the content for the current frame is the same as the content of the
      referenced frame (0-indexed within the video's frames for that task).
    - If text follows `F:index`, it's appended to the reconstructed content of the referenced frame.
    - This reconstruction happens on-the-fly during `__getitem__`.

    **Output per Sample (`__getitem__`):**
    Returns a dictionary with the following keys:
    - `"frame_path"` (str): Absolute path to the frame image.
    - `"image"` (torch.Tensor or PIL.Image): The loaded frame image (transformed if `image_transform` is set).
    - `"task1_raw_ocr"` (str): Reconstructed LLM output for Task 1.
    - `"task2_augmented"` (str): Reconstructed LLM output for Task 2.
    - `"task3_cleaned"` (str): Reconstructed LLM output for Task 3.
    - `"task4_markdown"` (str): Reconstructed LLM output for Task 4.
    - `"task5_summary"` (str): Narrative summary for the frame (from its LLM batch).
    - `"tesseract_ocr"` (str): Tesseract OCR text for the frame.
    """

    def __init__(
        self,
        frames_root_dir: Union[str, Path],
        llm_outputs_root_dir: Union[str, Path],
        tesseract_outputs_root_dir: Union[str, Path],
        image_transform: Optional[Callable] = None,
        # List of specific video_ids to load, if None, load all found
        video_ids_to_load: Optional[List[str]] = None,
        # Expected task keys from LLM output
        task_keys: List[str] = [
            "task1_raw_ocr",
            # task2 can be 'task2_augmented' or 'task2_augmented_imperfections'
            "task2_augmented", # Preferred key, will check alternate
            "task3_cleaned",
            "task4_markdown",
            "task5_summary", # This is the narrative, special handling
        ],
        alternate_task2_key: str = "task2_augmented_imperfections"
    ):
        """
        Initializes the OcrMultimodalDataset.

        Args:
            frames_root_dir (Union[str, Path]): Path to the root directory containing frame images,
                organized by video ID subdirectories.
            llm_outputs_root_dir (Union[str, Path]): Path to the root directory containing LLM JSON outputs,
                organized by video ID subdirectories, with each containing `llm_output_batch_*.json` files.
            tesseract_outputs_root_dir (Union[str, Path]): Path to the root directory containing Tesseract JSON
                outputs, organized by video ID subdirectories, with each containing a `tesseract_ocr.json` file.
            image_transform (Optional[Callable], optional): A function/transform to apply to the PIL Image.
                Defaults to None.
            video_ids_to_load (Optional[List[str]], optional): A specific list of video IDs to load.
                If None, all video IDs found in `frames_root_dir` will be attempted. Defaults to None.
            task_keys (List[str], optional): A list of keys expected in the LLM JSON batch files corresponding
                to different processing tasks. `task5_summary` is handled specially as a narrative for the batch.
                Defaults to ["task1_raw_ocr", "task2_augmented", "task3_cleaned", "task4_markdown", "task5_summary"].
            alternate_task2_key (str, optional): An alternate key to check for 'task2' data if the primary
                key (e.g., "task2_augmented") is not found in an LLM batch file.
                Defaults to "task2_augmented_imperfections".
        """
        self.frames_root_dir = Path(frames_root_dir)
        self.llm_outputs_root_dir = Path(llm_outputs_root_dir)
        self.tesseract_outputs_root_dir = Path(tesseract_outputs_root_dir)
        self.image_transform = image_transform
        self.task_keys = task_keys
        self.alternate_task2_key = alternate_task2_key

        self.samples = []  # Will store (frame_path, video_id, frame_stem, frame_idx_in_video)
        self.llm_data_for_video: Dict[str, Dict[str, List[str]]] = {}
        self.tesseract_data_for_video: Dict[str, Dict[str, str]] = {}

        self._load_data(video_ids_to_load)

    def _reconstruct_llm_output(
        self,
        task_data_list: List[str],
        current_frame_idx_in_video: int,
        reconstruction_cache: Dict[int, str],
    ) -> str:
        """
        Recursively reconstructs the LLM output for a specific frame for a given task,
        handling the `F:index` notation.

        The `F:index` notation means the content is the same as frame `index` for that task.
        If text follows `F:index`, it's appended to the reconstructed content of frame `index`.
        The `reconstruction_cache` is used to memoize results for a single `__getitem__` call
        for a specific task to avoid redundant computations within that call.

        Args:
            task_data_list (List[str]): The concatenated list of raw LLM outputs for a specific task
                across all frames of a single video.
            current_frame_idx_in_video (int): The 0-based index of the current frame within the video
                for which to reconstruct the output.
            reconstruction_cache (Dict[int, str]): A dictionary used as a cache for already reconstructed
                outputs for frames of the current task within the current `__getitem__` call.

        Returns:
            str: The fully reconstructed LLM output string for the specified frame and task.
        """
        if current_frame_idx_in_video in reconstruction_cache:
            return reconstruction_cache[current_frame_idx_in_video]

        current_output = task_data_list[current_frame_idx_in_video]

        if current_output.startswith("F:"):
            try:
                parts = current_output.split(":", 1)
                ref_idx_str = parts[1].split(" ", 1)[0] # Get index part before any appended string
                ref_idx = int(ref_idx_str)

                # Ensure the reference is not to itself or a future frame to avoid infinite loops
                if ref_idx >= current_frame_idx_in_video :
                    logging.warning(
                        f"Invalid frame reference F:{ref_idx} for frame {current_frame_idx_in_video}. Using raw value."
                    )
                    reconstruction_cache[current_frame_idx_in_video] = current_output
                    return current_output

                # Recursively reconstruct the referenced frame's output for this task
                reconstructed_ref_output = self._reconstruct_llm_output(
                    task_data_list, ref_idx, reconstruction_cache
                )
                
                # Check for appended content
                appended_content = ""
                if " " in parts[1]:
                    appended_content = parts[1].split(" ", 1)[1]
                
                final_output = reconstructed_ref_output + appended_content
                reconstruction_cache[current_frame_idx_in_video] = final_output
                return final_output
            except (ValueError, IndexError) as e:
                logging.warning(
                    f"Error parsing F:i-1 notation '{current_output}' for frame {current_frame_idx_in_video}: {e}. Using raw value."
                )
                reconstruction_cache[current_frame_idx_in_video] = current_output
                return current_output
        else:
            reconstruction_cache[current_frame_idx_in_video] = current_output
            return current_output

    def _load_data(self, video_ids_to_load: Optional[List[str]] = None):
        """
        Scans the data directories, loads and preprocesses LLM and Tesseract data.

        For each video:
        1. Finds all frame image files and sorts them.
        2. Finds all `llm_output_batch_*.json` files, sorts them, and concatenates their task data.
           - `task5_summary` is expanded to apply to all frames within its original batch.
        3. Loads `tesseract_ocr.json`.
        4. Validates frame counts between image files and loaded LLM data.
        5. Populates `self.samples` with tuples of (frame_path, video_id, frame_stem, frame_idx_in_video).
        6. Stores aggregated LLM data in `self.llm_data_for_video` and Tesseract data in `self.tesseract_data_for_video`.

        Args:
            video_ids_to_load (Optional[List[str]]): Specific video IDs to load. If None, loads all.
        """
        logging.info("Scanning dataset...")
        possible_video_ids = sorted(
            [d.name for d in self.frames_root_dir.iterdir() if d.is_dir()]
        )

        target_video_ids = possible_video_ids
        if video_ids_to_load:
            target_video_ids = [vid for vid in video_ids_to_load if vid in possible_video_ids]
            if not target_video_ids:
                logging.warning("None of the specified video_ids_to_load were found. Loading all videos.")
                target_video_ids = possible_video_ids
            elif len(target_video_ids) < len(video_ids_to_load):
                omitted = set(video_ids_to_load) - set(target_video_ids)
                logging.warning(f"Some specified video_ids not found in frames_root_dir: {omitted}")


        for video_id in target_video_ids:
            video_frames_dir = self.frames_root_dir / video_id
            video_llm_dir = self.llm_outputs_root_dir / video_id
            video_tesseract_dir = self.tesseract_outputs_root_dir / video_id

            if not video_llm_dir.is_dir():
                logging.warning(f"LLM output directory not found for {video_id}, skipping.")
                continue
            if not video_tesseract_dir.is_dir():
                logging.warning(f"Tesseract output directory not found for {video_id}, skipping.")
                continue

            frame_files = sorted(
                [f for f in video_frames_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            )
            if not frame_files:
                logging.warning(f"No frame images found in {video_frames_dir}, skipping {video_id}.")
                continue

            # Initialize per-video LLM data storage
            self.llm_data_for_video[video_id] = {key: [] for key in self.task_keys if key != "task5_summary"}
            self.llm_data_for_video[video_id]["task5_summary_list"] = [] # Special list for narratives

            llm_batch_files = sorted(video_llm_dir.glob("llm_output_batch_*.json"))
            if not llm_batch_files:
                logging.warning(f"No LLM batch files found for {video_id} in {video_llm_dir}, skipping.")
                continue
            
            # Concatenate LLM task data from all batch files for this video
            current_video_frame_count_from_llm = 0
            for batch_file_path in llm_batch_files:
                try:
                    with open(batch_file_path, "r") as f:
                        batch_data = json.load(f)
                    
                    frames_in_this_batch = 0
                    # Check Task 1 to get frame count for this batch
                    if self.task_keys[0] in batch_data and isinstance(batch_data[self.task_keys[0]], list):
                       frames_in_this_batch = len(batch_data[self.task_keys[0]])
                    
                    if frames_in_this_batch == 0:
                        logging.warning(f"LLM batch file {batch_file_path} has no frames in task1 list. Skipping this batch.")
                        continue
                    
                    current_video_frame_count_from_llm += frames_in_this_batch

                    for task_idx, task_key in enumerate(self.task_keys):
                        if task_key == "task5_summary":
                            summary = batch_data.get(task_key, "")
                            self.llm_data_for_video[video_id]["task5_summary_list"].extend(
                                [summary] * frames_in_this_batch
                            )
                        else:
                            actual_task_key_in_file = task_key
                            if task_key == "task2_augmented" and task_key not in batch_data:
                                if self.alternate_task2_key in batch_data:
                                    actual_task_key_in_file = self.alternate_task2_key
                                else:
                                    logging.warning(f"Neither '{task_key}' nor '{self.alternate_task2_key}' found in {batch_file_path} for {video_id}. Filling with empty strings.")
                                    self.llm_data_for_video[video_id][task_key].extend([""] * frames_in_this_batch)
                                    continue
                            
                            task_content_list = batch_data.get(actual_task_key_in_file)
                            if isinstance(task_content_list, list) and len(task_content_list) == frames_in_this_batch:
                                self.llm_data_for_video[video_id][task_key].extend(task_content_list)
                            else:
                                logging.warning(
                                    f"Task '{actual_task_key_in_file}' in {batch_file_path} for {video_id} is missing, not a list, or length mismatch. "
                                    f"Expected {frames_in_this_batch}, got {len(task_content_list) if isinstance(task_content_list, list) else 'N/A'}. "
                                    f"Filling with empty strings."
                                )
                                self.llm_data_for_video[video_id][task_key].extend([""] * frames_in_this_batch)

                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON from {batch_file_path} for {video_id}. Skipping this batch.")
                except Exception as e:
                    logging.error(f"Unexpected error loading batch {batch_file_path} for {video_id}: {e}. Skipping this batch.")
            
            # Validate frame counts
            if len(frame_files) != current_video_frame_count_from_llm:
                logging.warning(
                    f"Mismatch in frame count for {video_id}: "
                    f"{len(frame_files)} image files found, but LLM data covers {current_video_frame_count_from_llm} frames. "
                    f"Dataset will be truncated to the minimum of these. This might indicate missing/corrupt LLM batch files."
                )
            
            num_frames_for_video = min(len(frame_files), current_video_frame_count_from_llm)
            if num_frames_for_video == 0:
                logging.warning(f"Zero usable frames for video {video_id} after LLM data reconciliation. Skipping.")
                # Clean up potentially partially filled data for this video_id
                if video_id in self.llm_data_for_video: del self.llm_data_for_video[video_id]
                continue


            # Truncate LLM data if necessary due to frame count mismatch
            for task_key in self.llm_data_for_video[video_id]:
                self.llm_data_for_video[video_id][task_key] = self.llm_data_for_video[video_id][task_key][:num_frames_for_video]


            # Load Tesseract data
            tesseract_file = video_tesseract_dir / "tesseract_ocr.json"
            if tesseract_file.is_file():
                try:
                    with open(tesseract_file, "r") as f:
                        self.tesseract_data_for_video[video_id] = json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding Tesseract JSON for {video_id}. Tesseract data will be missing for this video.")
                    self.tesseract_data_for_video[video_id] = {} # Ensure key exists
            else:
                logging.warning(f"Tesseract file {tesseract_file} not found for {video_id}. Tesseract data will be missing.")
                self.tesseract_data_for_video[video_id] = {}


            # Create samples for this video
            for i in range(num_frames_for_video):
                frame_path = frame_files[i]
                frame_stem = frame_path.stem
                self.samples.append((frame_path, video_id, frame_stem, i))
            
            logging.info(f"Loaded {num_frames_for_video} samples for video: {video_id}")

        if not self.samples:
            logging.error(
                "No samples loaded. Check dataset paths and structure. "
                "Expected structure: frames_root/VID_ID/frame.png, "
                "llm_outputs_root/VID_ID/llm_output_batch_*.json, "
                "tesseract_outputs_root/VID_ID/tesseract_ocr.json"
            )
        else:
            logging.info(f"Dataset loaded successfully with {len(self.samples)} total samples from {len(self.llm_data_for_video)} videos.")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single data sample corresponding to the given index.

        Each sample is a dictionary containing the frame image (or path), reconstructed LLM task outputs,
        the narrative summary for that frame's batch, and Tesseract OCR text.
        The `F:i-1` notation in LLM tasks is resolved on-the-fly.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary representing the multimodal data sample.
                Keys include 'frame_path', 'image', 'task1_raw_ocr', 'task2_augmented',
                'task3_cleaned', 'task4_markdown', 'task5_summary', 'tesseract_ocr'.

        Raises:
            IndexError: If the idx is out of bounds.
        """
        if not 0 <= idx < len(self.samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.samples)}.")

        frame_path, video_id, frame_stem, frame_idx_in_video = self.samples[idx]

        try:
            image = Image.open(frame_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            logging.error(f"Error loading image {frame_path}: {e}. Returning empty tensor.")
            image = torch.empty(0) # Or handle more gracefully

        item: Dict[str, Any] = {"frame_path": str(frame_path), "image": image}

        # LLM data reconstruction
        # Cache for F:i-1 reconstruction, specific to this __getitem__ call and video
        llm_reconstruction_cache_for_item: Dict[str, Dict[int, str]] = {task: {} for task in self.task_keys if task != "task5_summary"}

        video_llm_data = self.llm_data_for_video.get(video_id, {})
        
        for task_key_idx, task_key_template in enumerate(self.task_keys):
            if task_key_template == "task5_summary":
                # task5_summary is handled via the "task5_summary_list" which is already per-frame
                item[task_key_template] = video_llm_data.get("task5_summary_list", [])[frame_idx_in_video] if frame_idx_in_video < len(video_llm_data.get("task5_summary_list", [])) else ""

            elif task_key_template == "task2_augmented": # Handle alternate key name
                task_data_list_aug = video_llm_data.get("task2_augmented", [])
                task_data_list_alt = video_llm_data.get(self.alternate_task2_key, [])
                
                # Prefer 'task2_augmented', use alternate if primary is empty or not present for this frame
                # This logic assumes one of them should have the data if the key exists for the video
                chosen_task_list = []
                if frame_idx_in_video < len(task_data_list_aug) and task_data_list_aug[frame_idx_in_video]:
                    chosen_task_list = task_data_list_aug
                elif frame_idx_in_video < len(task_data_list_alt) and task_data_list_alt[frame_idx_in_video]:
                     chosen_task_list = task_data_list_alt
                elif frame_idx_in_video < len(task_data_list_aug): # fallback to primary even if empty, to allow reconstruction
                    chosen_task_list = task_data_list_aug
                else: # fallback to alternate even if empty
                    chosen_task_list = task_data_list_alt

                if frame_idx_in_video < len(chosen_task_list):
                    item[task_key_template] = self._reconstruct_llm_output(
                        chosen_task_list,
                        frame_idx_in_video,
                        llm_reconstruction_cache_for_item[task_key_template],
                    )
                else:
                    item[task_key_template] = "" # Data missing for this frame
            else:
                task_data_list = video_llm_data.get(task_key_template, [])
                if frame_idx_in_video < len(task_data_list):
                    item[task_key_template] = self._reconstruct_llm_output(
                        task_data_list,
                        frame_idx_in_video,
                        llm_reconstruction_cache_for_item[task_key_template],
                    )
                else:
                    item[task_key_template] = "" # Data missing for this frame

        # Tesseract data
        video_tesseract_data = self.tesseract_data_for_video.get(video_id, {})
        # Tesseract keys might have .png, .jpg etc.
        tesseract_text = ""
        for ext in ['.png', '.jpg', '.jpeg']: # Check common extensions
            tesseract_text = video_tesseract_data.get(frame_stem + ext, "")
            if tesseract_text:
                break
        item["tesseract_ocr"] = tesseract_text
        
        return item

# --- Example Usage ---
def _create_dummy_llm_batch_file(
    dir_path: Path, batch_num: int, num_frames: int, video_id: str, task_keys: list, alternate_task2_key: str
):
    data: Dict[str, Any] = {key: [] for key in task_keys if key != "task5_summary"}
    data["task5_summary"] = f"This is the narrative for {video_id}, batch {batch_num}."
    
    frame_ids = [] # Not strictly used by current loader but good practice for external tools

    for i in range(num_frames):
        frame_idx_in_batch = i
        frame_ids.append(f"frame_{batch_num * num_frames + i:06d}.png")
        
        data["task1_raw_ocr"].append(f"Raw OCR for frame {frame_idx_in_batch} of batch {batch_num} in {video_id}.")
        
        # Use the alternate key for task2 in some batches for testing
        task2_key_to_use = "task2_augmented" if batch_num % 2 == 0 else alternate_task2_key
        if task2_key_to_use not in data: data[task2_key_to_use] = [] # Ensure list exists if using alternate

        if i > 0 and i % 3 == 0 : # Use F:i-1 notation for some
             data[task2_key_to_use].append(f"F:{i-1} ... plus augmented content for frame {frame_idx_in_batch}, batch {batch_num}.")
        else:
             data[task2_key_to_use].append(f"Augmented OCR for frame {frame_idx_in_batch}, batch {batch_num}.")
        
        data["task3_cleaned"].append(f"Cleaned OCR for frame {frame_idx_in_batch}, batch {batch_num}.")
        data["task4_markdown"].append(f"```markdown\n### Frame Content Analysis: {frame_idx_in_batch}\n- Detail for batch {batch_num}\n```")

    # Ensure all task lists are filled if alternate was used for task2
    if "task2_augmented" not in data and "task2_augmented" in task_keys:
        data["task2_augmented"] = [""] * num_frames 
    if alternate_task2_key not in data and alternate_task2_key in task_keys: # Should not happen if alternate_task2_key is in self.task_keys
         pass # already handled or primary used.

    data["frame_ids"] = frame_ids # Include for completeness, though not used by this loader version

    with open(dir_path / f"llm_output_batch_{batch_num:04d}.json", "w") as f:
        json.dump(data, f, indent=4)

def _create_dummy_tesseract_file(dir_path: Path, num_frames: int, video_id: str):
    data = {}
    for i in range(num_frames):
        data[f"frame_{i:06d}.png"] = f"Tesseract text for frame {i} of {video_id}."
    with open(dir_path / "tesseract_ocr.json", "w") as f:
        json.dump(data, f, indent=4)

def _create_dummy_frame_images(dir_path: Path, num_frames: int):
    for i in range(num_frames):
        try:
            img = Image.new("RGB", (100, 100), color=f"hsl({(i*20)%360}, 50%, 50%)")
            img.save(dir_path / f"frame_{i:06d}.png")
        except Exception as e:
            logging.error(f"Failed to create dummy image frame_{i:06d}.png: {e}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO) # Ensure info logs are visible for example
    
    dummy_base_dir = Path("./dummy_ocr_multimodal_dataset")
    if dummy_base_dir.exists():
        shutil.rmtree(dummy_base_dir)
    dummy_base_dir.mkdir(parents=True, exist_ok=True)

    frames_root = dummy_base_dir / "frames"
    llm_root = dummy_base_dir / "llm_outputs"
    tesseract_root = dummy_base_dir / "tesseract_outputs"

    frames_root.mkdir()
    llm_root.mkdir()
    tesseract_root.mkdir()

    video_ids = ["video1", "video2"]
    frames_per_video_map = {"video1": 7, "video2": 12} # Total frames for each video
    frames_per_llm_batch = 5 # How many frames per llm_output_batch_*.json file

    # Define task keys as expected by the Dataset class
    # Note: "task5_summary" is handled internally by the dataset based on batch structure
    dataset_task_keys = [
        "task1_raw_ocr",
        "task2_augmented", # This is the primary key the Dataset will look for
        "task3_cleaned",
        "task4_markdown",
        "task5_summary" # This is used by the dummy data generator to place it in batch files
    ]
    dataset_alternate_task2_key = "task2_augmented_imperfections"


    for vid in video_ids:
        vid_frames_dir = frames_root / vid
        vid_llm_dir = llm_root / vid
        vid_tesseract_dir = tesseract_root / vid

        vid_frames_dir.mkdir()
        vid_llm_dir.mkdir()
        vid_tesseract_dir.mkdir()

        total_frames_for_this_video = frames_per_video_map[vid]
        _create_dummy_frame_images(vid_frames_dir, total_frames_for_this_video)
        _create_dummy_tesseract_file(vid_tesseract_dir, total_frames_for_this_video, vid)

        num_batches = (total_frames_for_this_video + frames_per_llm_batch - 1) // frames_per_llm_batch
        frames_processed_count = 0
        for i in range(num_batches):
            frames_in_this_batch = min(frames_per_llm_batch, total_frames_for_this_video - frames_processed_count)
            if frames_in_this_batch <= 0: break
            # Pass the full list of task keys the generator expects (including alternate)
            # The generator will use the alternate key for task2 based on batch_num
            _create_dummy_llm_batch_file(vid_llm_dir, i + 1, frames_in_this_batch, vid, dataset_task_keys, dataset_alternate_task2_key)
            frames_processed_count += frames_in_this_batch
            
    logging.info(f"Dummy dataset created at {dummy_base_dir.resolve()}")

    # Example usage
    simple_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = OcrMultimodalDataset(
        frames_root_dir=frames_root,
        llm_outputs_root_dir=llm_root,
        tesseract_outputs_root_dir=tesseract_root,
        image_transform=simple_transform,
        task_keys=dataset_task_keys, # Pass the primary task keys
        alternate_task2_key=dataset_alternate_task2_key
    )

    if len(dataset) > 0:
        logging.info(f"Successfully created dataset with {len(dataset)} samples.")
        
        # Test a few samples
        indices_to_test = [0, frames_per_video_map["video1"] -1 , len(dataset) -1 ]
        if len(dataset) <= 5: # if dataset is small, test all
            indices_to_test = list(range(len(dataset)))

        for i in indices_to_test:
            if i < len(dataset) :
                logging.info(f"--- Sample {i} ---")
                sample = dataset[i]
                logging.info(f"  Frame Path: {sample['frame_path']}")
                logging.info(f"  Image Tensor Shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else 'N/A'}")
                logging.info(f"  Task 1 (Raw OCR): '{sample.get('task1_raw_ocr', '')[:50]}...'")
                logging.info(f"  Task 2 (Augmented): '{sample.get('task2_augmented', '')[:50]}...'")
                logging.info(f"  Task 3 (Cleaned): '{sample.get('task3_cleaned', '')[:50]}...'")
                logging.info(f"  Task 4 (Markdown): '{sample.get('task4_markdown', '')[:50]}...'")
                logging.info(f"  Task 5 (Narrative): '{sample.get('task5_summary', '')[:50]}...'")
                logging.info(f"  Tesseract OCR: '{sample.get('tesseract_ocr', '')[:50]}...'")
            else:
                logging.warning(f"Index {i} out of bounds for dataset of length {len(dataset)}")
        
        # Test with DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        try:
            batch_sample = next(iter(dataloader))
            logging.info("--- Batch Sample from DataLoader ---")
            logging.info(f"  Batch Image Shape: {batch_sample['image'].shape}")
            logging.info(f"  Batch Task 1 (first item): {batch_sample['task1_raw_ocr'][0][:50]}...")
            logging.info(f"  Batch Task 5 (first item): {batch_sample['task5_summary'][0][:50]}...")
            logging.info("Successfully loaded a batch.")
        except Exception as e:
            logging.error(f"Error loading batch from DataLoader: {e}")

    else:
        logging.error("Dataset creation resulted in 0 samples. Please check logs.")

    # Consider cleaning up the dummy directory after testing if not needed
    # shutil.rmtree(dummy_base_dir)
    # logging.info(f"Cleaned up dummy dataset at {dummy_base_dir.resolve()}") 