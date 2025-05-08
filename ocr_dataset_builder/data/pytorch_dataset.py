import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil # For dummy data creation
import random # Added for selecting augmentation function

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms # For example transform

from rich.logging import RichHandler

# Import the new augmentation functions
from .ocr_augmentations import (
    setting_slight_stutter,
    setting_gappy_and_fragmented,
    setting_overly_eager_diff,
    setting_line_boundary_chaos,
    setting_classic_bad_ocr,
    setting_the_echo_chamber,
    setting_telegraphic_transmission,
    setting_jittery_frame_capture,
    setting_minimalist_diff_max_omission,
    setting_comprehensive_degradation
)

# Import the new cleaning function
from ocr_dataset_builder.tesseract.ocr_utils import clean_tesseract_ocr

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)


class OcrMultimodalDataset(Dataset):
    """
    PyTorch Dataset for combining frame images, LLM task outputs, and Tesseract OCR text.

    This dataset is designed to work with a specific directory structure where frame images,
    LLM (Large Language Model) outputs, and Tesseract OCR outputs are organized per video.
    It also includes filepaths to original video metadata and subtitle files.

    **Expected Directory Structure:**

    1.  `frames_root_dir/` (Processed frame images)
        ├── VIDEO_ID_1/
        │   ├── frame_0000.png
        │   └── ...
        └── VIDEO_ID_2/
            └── ...

    2.  `llm_outputs_root_dir/` (LLM JSON outputs)
        ├── VIDEO_ID_1/
        │   ├── llm_output_batch_*.json
        │   └── ...
        └── VIDEO_ID_2/
            └── ...
        (See `__init__` docstring for LLM batch JSON structure)

    3.  `tesseract_outputs_root_dir/` (Tesseract JSON outputs)
        ├── VIDEO_ID_1/
        │   └── tesseract_ocr.json
        └── VIDEO_ID_2/
            └── ...
        (Maps frame filenames to Tesseract text)

    4.  `original_video_data_root_dir/` (Original video data before frame extraction)
        ├── VIDEO_ID_1/
        │   ├── video_file.mp4  (Actual video file, not directly used by dataset but illustrative)
        │   ├── some_name.info.json (Video metadata file)
        │   ├── english_subs.srt (Subtitle file)
        │   └── french_subs.vtt (Another subtitle file)
        └── VIDEO_ID_2/
            └── ...

    **Output per Sample (`__getitem__`):**
    Returns a dictionary with the following keys:
    - `"frame_path"` (str): Absolute path to the frame image.
    - `"image"` (torch.Tensor or PIL.Image): The loaded frame image.
    - `"task1_raw_ocr"` (str): Reconstructed LLM output for Task 1.
    - `"task2_augmented"` (str): Reconstructed LLM output for Task 2.
    - `"task3_cleaned"` (str): Reconstructed LLM output for Task 3.
    - `"task4_markdown"` (str): Reconstructed LLM output for Task 4.
    - `"task5_summary"` (str): Narrative summary for the frame.
    - `"tesseract_ocr"` (str): Tesseract OCR text for the frame.
    - `"metadata_filepath"` (Optional[str]): Path to the `.info.json` metadata file for the video, or None.
    - `"subtitle_filepaths"` (List[str]): List of paths to found subtitle files (.srt, .vtt) for the video.
    """

    SUBTITLE_EXTENSIONS = [".srt", ".vtt"]
    METADATA_EXTENSION = ".info.json"


    def __init__(
        self,
        frames_root_dir: Union[str, Path],
        llm_outputs_root_dir: Union[str, Path],
        tesseract_outputs_root_dir: Union[str, Path],
        original_video_data_root_dir: Union[str, Path], # New parameter
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
        alternate_task2_key: str = "task2_augmented_imperfections",
        custom_augmentation_funcs: Optional[List[Callable[[str], str]]] = None # New parameter for custom augmentations
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
            original_video_data_root_dir (Union[str, Path]): Path to the root directory containing original video data,
                where metadata (.info.json) and subtitle (.srt, .vtt) files are located
                within their respective video ID subdirectories.
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
            custom_augmentation_funcs (Optional[List[Callable[[str], str]]], optional): A list of functions
                that take a string (clean OCR) and return an augmented string. If provided, one function
                will be randomly applied to the 'task3_cleaned' output. Defaults to None.
        """
        self.frames_root_dir = Path(frames_root_dir)
        self.llm_outputs_root_dir = Path(llm_outputs_root_dir)
        self.tesseract_outputs_root_dir = Path(tesseract_outputs_root_dir)
        self.original_video_data_root_dir = Path(original_video_data_root_dir)
        self.image_transform = image_transform
        self.task_keys = task_keys
        self.alternate_task2_key = alternate_task2_key
        self.custom_augmentation_funcs = custom_augmentation_funcs # Store the custom augmentation functions

        self.samples = []  # Will store (frame_path, video_id, frame_stem, frame_idx_in_video)
        self.llm_data_for_video: Dict[str, Dict[str, List[str]]] = {}
        self.tesseract_data_for_video: Dict[str, Dict[str, str]] = {}
        self.auxiliary_file_paths: Dict[str, Dict[str, Any]] = {} # For metadata/subtitle paths

        self._load_data(video_ids_to_load)

    @staticmethod
    def _find_metadata_file(directory: Path) -> Optional[Path]:
        """Finds the first .info.json file in a directory."""
        if not directory.is_dir():
            return None
        metadata_files = list(directory.glob(f"*{OcrMultimodalDataset.METADATA_EXTENSION}"))
        if metadata_files:
            if len(metadata_files) > 1:
                logging.debug( # Changed to debug
                    f"Multiple {OcrMultimodalDataset.METADATA_EXTENSION} files found in {directory.name}, "
                    f"using first: {metadata_files[0].name}"
                )
            return metadata_files[0]
        return None

    @staticmethod
    def _find_subtitle_files(directory: Path) -> List[Path]:
        """Finds all subtitle files (.srt, .vtt) in a directory."""
        if not directory.is_dir():
            return []
        subtitle_files = []
        for ext in OcrMultimodalDataset.SUBTITLE_EXTENSIONS:
            subtitle_files.extend(list(directory.glob(f"*{ext}")))
        return sorted(subtitle_files) # Sort for consistent ordering

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
            content_after_F = current_output[2:]  # Remove "F:" prefix
            
            ref_idx_str = ""
            appended_content_starts_at = 0
            for char_idx, char_val in enumerate(content_after_F):
                if char_val.isdigit():
                    ref_idx_str += char_val
                    appended_content_starts_at = char_idx + 1
                else:
                    break  # First non-digit found, marks end of index

            appended_content = content_after_F[appended_content_starts_at:]

            if not ref_idx_str: # Should ideally not happen if "F:" was found and content follows
                logging.warning(
                    f"Could not extract numeric index from F: notation '{current_output}' for frame {current_frame_idx_in_video}. Using raw value."
                )
                reconstruction_cache[current_frame_idx_in_video] = current_output
                return current_output
            
            try:
                ref_idx = int(ref_idx_str)

                # Ensure the reference is not to itself or a future frame to avoid infinite loops
                if ref_idx >= current_frame_idx_in_video :
                    logging.warning(
                        f"Invalid frame reference F:{ref_idx_str} (parsed as {ref_idx}) for frame {current_frame_idx_in_video}. Reference must be to a previous frame. Using raw value '{current_output}'."
                    )
                    reconstruction_cache[current_frame_idx_in_video] = current_output
                    return current_output

                # Recursively reconstruct the referenced frame's output for this task
                reconstructed_ref_output = self._reconstruct_llm_output(
                    task_data_list, ref_idx, reconstruction_cache
                )
                
                final_output = reconstructed_ref_output + appended_content
                reconstruction_cache[current_frame_idx_in_video] = final_output
                return final_output
            except ValueError: # Catches if ref_idx_str is not a valid integer after all
                logging.warning(
                    f"Error converting extracted index '{ref_idx_str}' to int from F: notation '{current_output}' for frame {current_frame_idx_in_video}. Using raw value."
                )
                reconstruction_cache[current_frame_idx_in_video] = current_output
                return current_output
        else:
            reconstruction_cache[current_frame_idx_in_video] = current_output
            return current_output

    def _load_data(self, video_ids_to_load: Optional[List[str]] = None):
        """
        Scans the data directories, loads, validates, and preprocesses LLM and Tesseract data.
        Also finds and stores paths to original video metadata and subtitle files.
        Only includes samples if frame images, structurally complete LLM data (from valid batches),
        and Tesseract OCR text are all present.

        Validation Steps:
        1. Video Level: Checks for existence of frame, LLM, and Tesseract directories/files.
        2. LLM Batch Level: Validates each LLM batch file for JSON format and ensures all core task
           lists (Task 1-4) are present and have consistent lengths. Task5_summary must exist.
           Invalid batches are skipped.
        3. Frame Level (Post Aggregation): Ensures each potential sample (frame image + aggregated LLM data)
           also has a corresponding entry in the Tesseract OCR JSON file for that video.

        Args:
            video_ids_to_load (Optional[List[str]]): Specific video IDs to load. If None, loads all.
        """
        logging.info("Starting dataset scan with STRICT filtering...")
        possible_video_ids = sorted(
            [d.name for d in self.frames_root_dir.iterdir() if d.is_dir()]
        )

        target_video_ids = possible_video_ids
        if video_ids_to_load:
            target_video_ids = [vid for vid in video_ids_to_load if vid in possible_video_ids]
            if not target_video_ids:
                logging.warning("None of the specified video_ids_to_load were found. Attempting to load all videos.")
                target_video_ids = possible_video_ids
            elif len(target_video_ids) < len(video_ids_to_load):
                omitted = set(video_ids_to_load) - set(target_video_ids)
                logging.warning(f"Some specified video_ids not found in frames_root_dir: {omitted}")

        for video_id in target_video_ids:
            logging.debug(f"Processing video_id: {video_id}")
            
            # Paths for processed data
            video_frames_dir = self.frames_root_dir / video_id
            video_llm_dir = self.llm_outputs_root_dir / video_id
            video_tesseract_file = self.tesseract_outputs_root_dir / video_id / "tesseract_ocr.json"

            # Path for original video data (metadata, subtitles)
            original_video_dir = self.original_video_data_root_dir / video_id
            
            # Find auxiliary files
            metadata_file_path = self._find_metadata_file(original_video_dir)
            subtitle_file_paths = self._find_subtitle_files(original_video_dir)
            self.auxiliary_file_paths[video_id] = {
                "metadata_filepath": str(metadata_file_path) if metadata_file_path else None,
                "subtitle_filepaths": [str(sf) for sf in subtitle_file_paths]
            }
            if not original_video_dir.is_dir():
                 logging.warning(f"Original video data directory not found for {video_id} at {original_video_dir}. Metadata/subtitles will be missing.")


            if not video_frames_dir.is_dir():
                logging.warning(f"Frames directory not found for {video_id} at {video_frames_dir}. Skipping video.")
                continue
            if not video_llm_dir.is_dir():
                logging.warning(f"LLM output directory not found for {video_id} at {video_llm_dir}. Skipping video.")
                continue
            if not video_tesseract_file.is_file():
                logging.warning(f"Tesseract OCR JSON file not found for {video_id} at {video_tesseract_file}. Skipping video.")
                continue

            frame_paths_for_video = sorted(
                [f for f in video_frames_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            )
            if not frame_paths_for_video:
                logging.warning(f"No frame images found in {video_frames_dir}. Skipping video {video_id}.")
                continue

            try:
                with open(video_tesseract_file, "r") as f:
                    tesseract_data_for_this_video = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding Tesseract JSON {video_tesseract_file} for {video_id}: {e}. Skipping video.")
                continue
            except Exception as e:
                logging.error(f"Unexpected error loading Tesseract JSON {video_tesseract_file} for {video_id}: {e}. Skipping video.")
                continue

            llm_batch_files = sorted(video_llm_dir.glob("llm_output_batch_*.json"))
            if not llm_batch_files:
                logging.warning(f"No LLM batch files found for {video_id} in {video_llm_dir}. Skipping video.")
                continue

            # Aggregate LLM data from all VALID batches for this video
            aggregated_llm_tasks_raw = {key: [] for key in self.task_keys if key != "task5_summary"}
            aggregated_llm_summaries_raw = []
            llm_processed_frame_count_for_video = 0

            for batch_file_path in llm_batch_files:
                try:
                    with open(batch_file_path, "r") as f:
                        batch_data = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding LLM batch JSON {batch_file_path} for {video_id}: {e}. Skipping batch.")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error loading LLM batch {batch_file_path} for {video_id}: {e}. Skipping batch.")
                    continue

                # Determine number of frames in this batch (e.g., from task1_raw_ocr)
                primary_task_key = self.task_keys[0]
                num_frames_in_batch = len(batch_data.get(primary_task_key, []))
                if num_frames_in_batch == 0:
                    logging.warning(f"LLM batch {batch_file_path} for {video_id} has 0 frames based on '{primary_task_key}'. Skipping batch.")
                    continue

                is_batch_structurally_valid = True
                current_batch_task_data_temp = {}

                # Validate tasks 1-4 structure
                for task_key_template in self.task_keys:
                    if task_key_template == "task5_summary":
                        if task_key_template not in batch_data:
                            logging.warning(f"LLM batch {batch_file_path} for {video_id} is missing '{task_key_template}'. Skipping batch.")
                            is_batch_structurally_valid = False
                            break
                        current_batch_task_data_temp[task_key_template] = batch_data[task_key_template]
                        continue # Handled separately for aggregation

                    actual_key_in_file = task_key_template
                    task_list_from_batch = batch_data.get(actual_key_in_file)

                    if task_key_template == "task2_augmented" and not task_list_from_batch:
                        if self.alternate_task2_key in batch_data:
                            actual_key_in_file = self.alternate_task2_key
                            task_list_from_batch = batch_data.get(actual_key_in_file)
                    
                    if not isinstance(task_list_from_batch, list) or len(task_list_from_batch) != num_frames_in_batch:
                        logging.warning(
                            f"LLM batch {batch_file_path} for {video_id}: Task '{actual_key_in_file}' is missing, not a list, or length mismatch. "
                            f"Expected {num_frames_in_batch} items, found {len(task_list_from_batch) if isinstance(task_list_from_batch, list) else 'N/A'}. Skipping batch."
                        )
                        is_batch_structurally_valid = False
                        break
                    current_batch_task_data_temp[actual_key_in_file] = task_list_from_batch
                
                if not is_batch_structurally_valid:
                    continue # Skip this batch

                # Batch is valid, append its data
                for task_key_template in self.task_keys:
                    if task_key_template == "task5_summary":
                        aggregated_llm_summaries_raw.extend([current_batch_task_data_temp[task_key_template]] * num_frames_in_batch)
                    elif task_key_template == "task2_augmented":
                        # Append from primary key if it existed, else from alternate if that was used
                        if task_key_template in current_batch_task_data_temp: # Primary was found and valid
                            aggregated_llm_tasks_raw[task_key_template].extend(current_batch_task_data_temp[task_key_template])
                        elif self.alternate_task2_key in current_batch_task_data_temp: # Alternate was found and valid
                            aggregated_llm_tasks_raw[task_key_template].extend(current_batch_task_data_temp[self.alternate_task2_key])
                        # This case should ideally not be hit if batch validation was thorough for task2 presence
                    elif task_key_template in current_batch_task_data_temp: # For task1, task3, task4
                         aggregated_llm_tasks_raw[task_key_template].extend(current_batch_task_data_temp[task_key_template])
                
                llm_processed_frame_count_for_video += num_frames_in_batch
            # End of LLM batch processing for this video

            if llm_processed_frame_count_for_video == 0:
                logging.warning(f"No valid LLM data aggregated for {video_id} after processing all batches. Skipping video.")
                continue

            # Now filter frames based on tesseract availability and create final samples
            num_potential_samples = min(len(frame_paths_for_video), llm_processed_frame_count_for_video)
            
            video_specific_samples = []
            # These will store only data for frames that pass all checks for this video
            final_video_llm_tasks = {key: [] for key in self.task_keys if key != "task5_summary"}
            final_video_llm_summaries = []
            final_video_tesseract = {}

            for frame_idx in range(num_potential_samples):
                current_frame_path = frame_paths_for_video[frame_idx]
                current_frame_name = current_frame_path.name
                current_frame_stem = current_frame_path.stem

                tesseract_text_for_frame = tesseract_data_for_this_video.get(current_frame_name)
                if tesseract_text_for_frame is None:
                    # Try with stem and common extensions as a fallback, though direct name match is preferred
                    for ext in ['.png', '.jpg', '.jpeg']:
                        tesseract_text_for_frame = tesseract_data_for_this_video.get(current_frame_stem + ext)
                        if tesseract_text_for_frame is not None: break
                
                if tesseract_text_for_frame is None:
                    logging.debug(f"Tesseract data missing for frame {current_frame_name} in {video_id}. Skipping frame.")
                    continue # Skip this frame, it doesn't have Tesseract data
                
                # Frame has image, LLM data (from valid batch), and Tesseract data. Add it.
                # The `frame_idx_in_video_for_storage` is the index within the *filtered* lists for this video.
                frame_idx_in_video_for_storage = len(final_video_llm_tasks[self.task_keys[0]]) 
                video_specific_samples.append((current_frame_path, video_id, current_frame_stem, frame_idx_in_video_for_storage))

                for task_key_template in self.task_keys:
                    if task_key_template == "task5_summary":
                        final_video_llm_summaries.append(aggregated_llm_summaries_raw[frame_idx])
                    else:
                        final_video_llm_tasks[task_key_template].append(aggregated_llm_tasks_raw[task_key_template][frame_idx])
                
                final_video_tesseract[current_frame_name] = tesseract_text_for_frame

            if video_specific_samples:
                self.samples.extend(video_specific_samples)
                # Store the filtered LLM and Tesseract data for this video
                self.llm_data_for_video[video_id] = final_video_llm_tasks
                self.llm_data_for_video[video_id]["task5_summary_list"] = final_video_llm_summaries # Add summaries list
                self.tesseract_data_for_video[video_id] = final_video_tesseract
                logging.info(f"Successfully loaded {len(video_specific_samples)} validated samples for video: {video_id}.")
            else:
                logging.warning(f"No validated samples found for video {video_id} after all checks.")

        # Final summary log
        if not self.samples:
            logging.error(
                "STRICT MODE: No samples loaded. Check dataset paths, file structures, LLM batch validity, and Tesseract data presence for all frames."
            )
        else:
            logging.info(f"STRICT MODE: Dataset loaded successfully with {len(self.samples)} total validated samples from {len(self.llm_data_for_video)} videos.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single data sample corresponding to the given index.

        Each sample is a dictionary containing the frame image (or path), 
        tesseract_ocr, llm_clean_ocr, augmented_llm_clean_ocr, markdown, and summary.
        The `F:i-1` notation in LLM tasks is resolved on-the-fly from the source files.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary representing the multimodal data sample.
                Keys include 'frame_path', 'image', 'tesseract_ocr', 'llm_clean_ocr',
                'augmented_llm_clean_ocr', 'markdown', 'summary', 'metadata_filepath', 'subtitle_filepaths'.

        Raises:
            IndexError: If the idx is out of bounds.
        """
        if not 0 <= idx < len(self.samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.samples)}.")

        frame_path, video_id, frame_stem, frame_idx_in_video = self.samples[idx]

        try:
            pil_image = Image.open(frame_path).convert("RGB")
            image_tensor = self.image_transform(pil_image) if self.image_transform else pil_image
        except Exception as e:
            logging.error(f"Error loading image {frame_path}: {e}. Returning empty tensor or placeholder.")
            # Depending on strictness, could return None or a placeholder tensor
            image_tensor = torch.empty(0) # Placeholder

        # Initialize the dictionary for the final selected data
        final_item: Dict[str, Any] = {
            "frame_path": str(frame_path),
            "video_id": video_id,
            "frame_stem": frame_stem,
            "image": image_tensor,
            "tesseract_ocr": "",
            "llm_clean_ocr": "",
            "augmented_llm_clean_ocr": "",
            "markdown": "",
            "summary": "",
        }

        # --- Load ALL relevant data from source files first ---
        # This part remains similar, loading based on self.task_keys from the JSON structure
        raw_loaded_llm_tasks: Dict[str, str] = {}
        llm_reconstruction_cache_for_item: Dict[str, Dict[int, str]] = {task: {} for task in self.task_keys if task != "task5_summary"}
        video_llm_data = self.llm_data_for_video.get(video_id, {})

        for task_key_template in self.task_keys: # self.task_keys are from old 5-task prompt for now
            if task_key_template == "task5_summary":
                raw_loaded_llm_tasks[task_key_template] = video_llm_data.get("task5_summary_list", [])[frame_idx_in_video] if frame_idx_in_video < len(video_llm_data.get("task5_summary_list", [])) else ""
            elif task_key_template == "task2_augmented":
                task_data_list_aug = video_llm_data.get("task2_augmented", [])
                task_data_list_alt = video_llm_data.get(self.alternate_task2_key, [])
                chosen_task_list = []
                if frame_idx_in_video < len(task_data_list_aug) and task_data_list_aug[frame_idx_in_video]:
                    chosen_task_list = task_data_list_aug
                elif frame_idx_in_video < len(task_data_list_alt) and task_data_list_alt[frame_idx_in_video]:
                     chosen_task_list = task_data_list_alt
                elif frame_idx_in_video < len(task_data_list_aug):
                    chosen_task_list = task_data_list_aug
                else:
                    chosen_task_list = task_data_list_alt
                if frame_idx_in_video < len(chosen_task_list):
                    raw_loaded_llm_tasks[task_key_template] = self._reconstruct_llm_output(
                        chosen_task_list, frame_idx_in_video, llm_reconstruction_cache_for_item[task_key_template]
                    )
                else:
                    raw_loaded_llm_tasks[task_key_template] = ""
            else:
                task_data_list = video_llm_data.get(task_key_template, [])
                if frame_idx_in_video < len(task_data_list):
                    raw_loaded_llm_tasks[task_key_template] = self._reconstruct_llm_output(
                        task_data_list, frame_idx_in_video, llm_reconstruction_cache_for_item.get(task_key_template, {})
                    )
                else:
                    raw_loaded_llm_tasks[task_key_template] = ""
        
        # --- Populate final_item with prioritized mapping ---

        # 1. Tesseract OCR
        video_tesseract_data = self.tesseract_data_for_video.get(video_id, {})
        tesseract_text = ""
        for ext in ['.png', '.jpg', '.jpeg']:
            tesseract_text = video_tesseract_data.get(frame_stem + ext, "")
            if tesseract_text: break

        # Clean the Tesseract text using the utility function
        cleaned_tesseract_text = clean_tesseract_ocr(tesseract_text)
        final_item["tesseract_ocr"] = cleaned_tesseract_text

        # 2. LLM Clean OCR (Source for our augmentation)
        #    Priority: new key "cleaned_ocr" (or similar from new prompt) -> old "task3_cleaned"
        #    For llm_processing.py, "==== TASK 1: Cleaned OCR Text ====" might become "cleaned_ocr_text"
        #    Let's assume a potential new key could be "task1_cleaned_ocr" if llm_pipeline keeps taskN_ prefix
        clean_ocr_source_text = raw_loaded_llm_tasks.get("task1_cleaned_ocr", 
                                        raw_loaded_llm_tasks.get("cleaned_ocr", 
                                                               raw_loaded_llm_tasks.get("task3_cleaned", "")))
        final_item["llm_clean_ocr"] = clean_ocr_source_text

        # 3. Augmented LLM Clean OCR (Our Python augmentations)
        if self.custom_augmentation_funcs and clean_ocr_source_text:
            chosen_augmentation_func = random.choice(self.custom_augmentation_funcs)
            final_item["augmented_llm_clean_ocr"] = chosen_augmentation_func(clean_ocr_source_text)
        else:
            final_item["augmented_llm_clean_ocr"] = "" # or clean_ocr_source_text if no augmentation desired when funcs are missing

        # 4. Markdown
        #    Priority: new key "structured_markdown" (or similar) -> old "task4_markdown"
        #    Potential new key: "task2_structured_markdown"
        final_item["markdown"] = raw_loaded_llm_tasks.get("task2_structured_markdown",
                                         raw_loaded_llm_tasks.get("structured_markdown",
                                                                raw_loaded_llm_tasks.get("task4_markdown", "")))

        # 5. Summary
        #    Priority: new key "narrative_summary" (or similar) -> old "task5_summary"
        #    Potential new key: "task3_narrative_summary"
        final_item["summary"] = raw_loaded_llm_tasks.get("task3_narrative_summary",
                                       raw_loaded_llm_tasks.get("narrative_summary",
                                                              raw_loaded_llm_tasks.get("task5_summary", "")))
        
        # Add auxiliary file paths
        aux_paths = self.auxiliary_file_paths.get(video_id, {"metadata_filepath": None, "subtitle_filepaths": []})
        final_item["metadata_filepath"] = aux_paths["metadata_filepath"]
        final_item["subtitle_filepaths"] = aux_paths["subtitle_filepaths"]
        
        # item["custom_augmented_ocr"] is now final_item["augmented_llm_clean_ocr"]
        # The old raw_loaded_llm_tasks are not directly returned unless mapped above.

        return final_item

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

def _create_dummy_original_video_files(original_video_dir: Path, video_id: str):
    """Creates dummy .info.json and .srt files."""
    original_video_dir.mkdir(parents=True, exist_ok=True)
    # Create dummy .info.json
    with open(original_video_dir / f"{video_id}.info.json", "w") as f:
        json.dump({"title": f"Dummy Title for {video_id}", "uploader": "Dummy Uploader"}, f, indent=4)
    # Create dummy .srt file
    with open(original_video_dir / f"{video_id}_en.srt", "w") as f:
        f.write("1\n00:00:01,000 --> 00:00:02,000\nDummy Subtitle 1\n\n")
        f.write("2\n00:00:03,000 --> 00:00:04,000\nDummy Subtitle 2\n")
    # Create a dummy .vtt as well
    with open(original_video_dir / f"{video_id}_fr.vtt", "w") as f:
        f.write("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nDummy VTT Subtitle 1\n\n")
        f.write("00:00:03.000 --> 00:00:04.000\nDummy VTT Subtitle 2\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO) # Ensure info logs are visible

    # --- Comment out or remove dummy data creation ---
    # dummy_base_dir = Path("./dummy_ocr_multimodal_dataset_with_originals")
    # if dummy_base_dir.exists():
    #     shutil.rmtree(dummy_base_dir)
    # dummy_base_dir.mkdir(parents=True, exist_ok=True)
    # 
    # frames_root_dummy = dummy_base_dir / "frames"
    # llm_root_dummy = dummy_base_dir / "llm_outputs"
    # tesseract_root_dummy = dummy_base_dir / "tesseract_outputs"
    # 
    # frames_root_dummy.mkdir()
    # llm_root_dummy.mkdir()
    # tesseract_root_dummy.mkdir()
    # 
    # video_ids_dummy = ["video1", "video2"]
    # frames_per_video_map_dummy = {"video1": 7, "video2": 12}
    # frames_per_llm_batch_dummy = 3
    # 
    # dataset_task_keys_for_generator = [
    #     "task1_raw_ocr",
    #     "task2_augmented",
    #     "task2_augmented_imperfections",
    #     "task3_cleaned",
    #     "task4_markdown",
    #     "task5_summary"
    # ]
    # 
    # for vid in video_ids_dummy:
    #     vid_frames_dir = frames_root_dummy / vid
    #     vid_llm_dir = llm_root_dummy / vid
    #     vid_tesseract_dir = tesseract_root_dummy / vid
    # 
    #     vid_frames_dir.mkdir()
    #     vid_llm_dir.mkdir()
    #     vid_tesseract_dir.mkdir()
    # 
    #     total_frames_for_this_video = frames_per_video_map_dummy[vid]
    #     _create_dummy_frame_images(vid_frames_dir, total_frames_for_this_video)
    #     _create_dummy_tesseract_file(vid_tesseract_dir, total_frames_for_this_video, vid)
    # 
    #     num_batches = (total_frames_for_this_video + frames_per_llm_batch_dummy - 1) // frames_per_llm_batch_dummy
    #     frames_processed_in_video = 0
    #     for i in range(num_batches):
    #         frames_in_this_batch = min(frames_per_llm_batch_dummy, total_frames_for_this_video - frames_processed_in_video)
    #         if frames_in_this_batch <= 0: break
    #         _create_dummy_llm_batch_file(
    #             vid_llm_dir, 
    #             i + 1, 
    #             frames_in_this_batch, 
    #             vid, 
    #             dataset_task_keys_for_generator, 
    #             "task2_augmented_imperfections"
    #         )
    #         frames_processed_in_video += frames_in_this_batch
    # 
    # logging.info(f"Dummy dataset generation skipped for real data test.")

    # --- Use Real Dataset Paths ---
    frames_root = Path("/mnt/data-store/pieces-ocr-v-0-1-0-frames/")
    llm_root = Path("/mnt/data-store/pieces-ocr-v-0-1-0-llm_output/")
    tesseract_root = Path("/mnt/data-store/pieces-ocr-v-0-1-0-tesseract_output/")
    original_video_data_root = Path("/mnt/data-store/pieces-ocr-v-0-1-0/") # Path to original videos and their metadata/subs

    logging.info(f"Attempting to load real dataset from:")
    logging.info(f"  Frames root (processed): {frames_root}")
    logging.info(f"  LLM root (processed): {llm_root}")
    logging.info(f"  Tesseract root (processed): {tesseract_root}")
    logging.info(f"  Original Video Data root: {original_video_data_root}")

    # Standard image transform
    simple_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Resize to a common size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the task keys expected by the Dataset class loader
    # These should match the keys used in your LLM output JSONs
    dataset_task_keys_for_loader = [
        "task1_raw_ocr",
        "task2_augmented",
        "task3_cleaned",
        "task4_markdown",
        "task5_summary",
        # Add potential new keys here if you want them to be loaded by _reconstruct_llm_output directly,
        # though the mapping logic in __getitem__ tries to handle this flexibly.
        # e.g., "task1_cleaned_ocr", "task2_structured_markdown", "task3_narrative_summary"
    ]
    dataset_alternate_task2_key_for_loader = "task2_augmented_imperfections"

    # Optional: Load only a specific subset of video IDs for faster testing
    # video_ids_to_test = ["#038 Asynchronous Programming in C# [ شرح بالعربي ]  #async #await #thread #task [kDUDX3VJFEc]"] 
    video_ids_to_test = None # Load all videos

    # List of our new augmentation functions
    all_custom_augmentations = [
        setting_slight_stutter,
        setting_gappy_and_fragmented,
        setting_overly_eager_diff,
        setting_line_boundary_chaos,
        setting_classic_bad_ocr,
        setting_the_echo_chamber,
        setting_telegraphic_transmission,
        setting_jittery_frame_capture,
        setting_minimalist_diff_max_omission,
        setting_comprehensive_degradation
    ]

    dataset = OcrMultimodalDataset(
        frames_root_dir=frames_root,
        llm_outputs_root_dir=llm_root,
        tesseract_outputs_root_dir=tesseract_root,
        original_video_data_root_dir=original_video_data_root,
        image_transform=simple_transform,
        task_keys=dataset_task_keys_for_loader,
        alternate_task2_key=dataset_alternate_task2_key_for_loader,
        custom_augmentation_funcs=all_custom_augmentations,
        video_ids_to_load=video_ids_to_test
    )

    if len(dataset) > 0:
        logging.info(f"Successfully created dataset with {len(dataset)} samples.")
        
        # Test a few samples
        # num_samples_to_show = min(5, len(dataset))
        # indices_to_test = list(range(num_samples_to_show))
        # if len(dataset) > num_samples_to_show: # Add a few from the end if dataset is large enough
        #    indices_to_test.extend(list(range(len(dataset) - min(3, len(dataset)//2), len(dataset))))
        # indices_to_test = sorted(list(set(indices_to_test))) # Ensure unique and sorted

        # Simplified: Test first 3, and last 2 if dataset is large enough
        indices_to_test = []
        if len(dataset) > 0: indices_to_test.extend(range(min(3, len(dataset))))
        if len(dataset) > 5: indices_to_test.extend(range(len(dataset)-2, len(dataset)))
        indices_to_test = sorted(list(set(indices_to_test))) # Unique sorted indices

        for i in indices_to_test:
            if i < len(dataset): # Double check index before access
                logging.info(f"--- Sample {i} ---")
                try:
                    sample = dataset[i]
                    logging.info(f"  Frame Path: {sample.get('frame_path', 'N/A')}")
                    logging.info(f"  Image Tensor Shape: {sample.get('image').shape if hasattr(sample.get('image'), 'shape') else 'N/A'}")
                    
                    # Print the new desired fields
                    logging.info(f"  Tesseract OCR (Cleaned): '{str(sample.get('tesseract_ocr', 'N/A'))[:200]}...'" )
                    logging.info(f"  LLM Clean OCR: '{str(sample.get('llm_clean_ocr', 'N/A'))[:200]}...'" )
                    logging.info(f"  Augmented LLM Clean OCR: '{str(sample.get('augmented_llm_clean_ocr', 'N/A'))[:200]}...'" )
                    logging.info(f"  Markdown: '{str(sample.get('markdown', 'N/A'))[:200]}...'" )
                    logging.info(f"  Summary: '{str(sample.get('summary', 'N/A'))[:200]}...'" )

                    # logging.info(f"  Raw Loaded Task1 (for debug): '{str(raw_loaded_llm_tasks.get('task1_raw_ocr', ''))[:100]}...'") 
                    # The above line would error as raw_loaded_llm_tasks is local to __getitem__

                    logging.info(f"  metadata_filepath: {sample.get('metadata_filepath', 'N/A')}")
                    logging.info(f"  subtitle_filepaths: {sample.get('subtitle_filepaths', [])}")
                except Exception as e:
                    logging.error(f"Error accessing or printing sample at index {i}: {e}")
            else:
                logging.warning(f"Test index {i} out of bounds for dataset of length {len(dataset)}")
        
        # # Test with DataLoader
        # if len(dataset) > 0:
        #     from torch.utils.data import DataLoader
        #     # Ensure batch_size is not greater than dataset size if dataset is very small
        #     effective_batch_size = min(4, len(dataset))
        #     if effective_batch_size > 0:
        #         dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        #         try:
        #             batch_sample = next(iter(dataloader))
        #             logging.info("--- Batch Sample from DataLoader ---")
        #             logging.info(f"  Batch Image Shape: {batch_sample['image'].shape}")
        #             logging.info(f"  Batch Task 1 (first item): {str(batch_sample['task1_raw_ocr'][0])}...")
        #             logging.info(f"  Batch Task 5 (first item): {str(batch_sample['task5_summary'][0])}...")
        #             logging.info(f"  Batch Metadata (first item): {batch_sample['metadata_filepath'][0]}")
        #             logging.info(f"  Batch Subtitles (first item): {batch_sample['subtitle_filepaths'][0]}") # This will be a list of lists
        #             logging.info("Successfully loaded a batch.")
        #         except StopIteration:
        #             logging.warning("DataLoader was empty. This can happen if the dataset size is smaller than the batch size or 0.")
        #         except Exception as e:
        #             logging.error(f"Error loading batch from DataLoader: {e}")
        #     else:
        #         logging.info("Dataset too small or empty to create a DataLoader batch.")
        # else:
        #     logging.info("Dataset is empty, skipping DataLoader test.")

    else:
        logging.error("Dataset creation resulted in 0 samples. Please check paths and data structure of the real dataset.")

    # The dummy data cleanup is already commented out.
    # shutil.rmtree(dummy_base_dir)
    # logging.info(f"Cleaned up dummy dataset at {dummy_base_dir.resolve()}") 