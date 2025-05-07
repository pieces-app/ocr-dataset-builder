import argparse
import json
import logging
import sys
from pathlib import Path


try:
    from ocr_dataset_builder.pytorch_dataset import OcrMultimodalDataset
except ImportError:
    print("Error: Could not import OcrMultimodalDataset. Make sure the ocr_dataset_builder package is installed and in your PYTHONPATH.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="[%X]",
)

def extract_text_data(
    frames_root_dir: str,
    llm_outputs_root_dir: str,
    tesseract_outputs_root_dir: str,
    original_video_data_root_dir: str,
    output_file_path: str,
    requested_llm_task_keys: list[str],
    extraction_mode: str,
    video_ids_to_process: list[str] | None = None,
):
    """
    Extracts specified text fields from the OcrMultimodalDataset and saves them
    to a JSON Lines file, supporting different extraction modes.
    """
    logging.info(f"Initializing OcrMultimodalDataset...")
    logging.info(f"  Frames root: {frames_root_dir}")
    logging.info(f"  LLM root: {llm_outputs_root_dir}")
    logging.info(f"  Tesseract root: {tesseract_outputs_root_dir}")
    logging.info(f"  Original Video Data root: {original_video_data_root_dir}")
    logging.info(f"  Extraction Mode: {extraction_mode}")
    if extraction_mode == "standard":
        logging.info(f"  Requested LLM tasks for standard mode: {requested_llm_task_keys}")
    if video_ids_to_process:
        logging.info(f"  Processing specific video IDs: {len(video_ids_to_process)}")

    try:
        dataset = OcrMultimodalDataset(
            frames_root_dir=Path(frames_root_dir),
            llm_outputs_root_dir=Path(llm_outputs_root_dir),
            tesseract_outputs_root_dir=Path(tesseract_outputs_root_dir),
            original_video_data_root_dir=Path(original_video_data_root_dir),
            video_ids_to_load=video_ids_to_process,
            image_transform=None,
        )
    except Exception as e:
        logging.error(f"Failed to initialize OcrMultimodalDataset: {e}", exc_info=True)
        return

    if not dataset or len(dataset) == 0:
        logging.warning("Dataset is empty or failed to load. No data to extract.")
        return

    logging.info(f"Dataset loaded with {len(dataset)} samples. Starting extraction to {output_file_path}...")

    processed_source_samples = 0
    output_records_count = 0
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if sample is None:
                    logging.warning(f"Sample {i} is None, skipping.")
                    continue
                
                processed_source_samples +=1
                frame_path_str = str(sample.get("frame_path", ""))

                if extraction_mode == "standard":
                    record = {"frame_path": frame_path_str}
                    for task_key in requested_llm_task_keys:
                        if task_key == "task2_augmented":
                            llm_text = sample.get("task2_augmented", sample.get("task2_augmented_imperfections"))
                        else:
                            llm_text = sample.get(task_key)
                        record[task_key] = llm_text if llm_text is not None else ""
                    record["tesseract_ocr"] = sample.get("tesseract_ocr", "")
                    outfile.write(json.dumps(record) + "\n")
                    output_records_count += 1

                elif extraction_mode == "cleaning_pairs":
                    tesseract_text = sample.get("tesseract_ocr", "")
                    llm_task1 = sample.get("task1_raw_ocr", "")
                    llm_task2 = sample.get("task2_augmented", sample.get("task2_augmented_imperfections", ""))
                    llm_task3_cleaned = sample.get("task3_cleaned", "")

                    # Record 1: Tesseract vs. LLM Task 3
                    record1 = {
                        "frame_path": frame_path_str,
                        "raw_ocr": tesseract_text,
                        "clean_ocr": llm_task3_cleaned
                    }
                    outfile.write(json.dumps(record1) + "\n")

                    # Record 2: LLM Task 1 vs. LLM Task 3
                    record2 = {
                        "frame_path": frame_path_str,
                        "raw_ocr": llm_task1,
                        "clean_ocr": llm_task3_cleaned
                    }
                    outfile.write(json.dumps(record2) + "\n")

                    # Record 3: LLM Task 2 vs. LLM Task 3
                    record3 = {
                        "frame_path": frame_path_str,
                        "raw_ocr": llm_task2,
                        "clean_ocr": llm_task3_cleaned
                    }
                    outfile.write(json.dumps(record3) + "\n")
                    output_records_count += 3
                
                else:
                    logging.error(f"Unknown extraction_mode: {extraction_mode}")
                    return # Stop if mode is unknown

                if processed_source_samples % 1000 == 0:
                    logging.info(f"Processed {processed_source_samples} source samples, generated {output_records_count} output records...")

            except Exception as e:
                logging.error(f"Error processing source sample {i} (Frame: {sample.get('frame_path', 'Unknown')}): {e}", exc_info=False)

    logging.info(f"Extraction complete. Processed {processed_source_samples} source samples.")
    logging.info(f"Saved {output_records_count} text records to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract text-only data from the OcrMultimodalDataset.")
    parser.add_argument("--frames_root", type=str, required=True, help="Root directory of processed frames.")
    parser.add_argument("--llm_root", type=str, required=True, help="Root directory of LLM outputs.")
    parser.add_argument("--tesseract_root", type=str, required=True, help="Root directory of Tesseract outputs.")
    parser.add_argument("--video_data_root", type=str, required=True, help="Root directory of original video data (e.g., for metadata, subtitles).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON Lines file.")
    
    parser.add_argument(
        "--extraction_mode",
        type=str,
        default="standard",
        choices=["standard", "cleaning_pairs"],
        help="Extraction mode: 'standard' for one line per frame, 'cleaning_pairs' for multiple raw/clean pairs per frame."
    )

    # llm_tasks is still relevant for ensuring OcrMultimodalDataset loads these tasks
    # In 'cleaning_pairs' mode, the script specifically uses task1, task2, and task3.
    default_llm_tasks = "task1_raw_ocr,task2_augmented,task3_cleaned" 
    parser.add_argument(
        "--llm_tasks", 
        type=str, 
        default=default_llm_tasks,
        help=f"Comma-separated list of LLM task keys to extract or ensure are loaded by the dataset (e.g., task1_raw_ocr,task2_augmented,task3_cleaned). Defaults to: {default_llm_tasks}"
    )
    parser.add_argument(
        "--video_ids", 
        type=str, 
        default=None,
        help="Comma-separated list of specific video IDs to process. Processes all if not specified."
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.")

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    # These tasks are needed by OcrMultimodalDataset to load the data correctly,
    # especially if 'cleaning_pairs' mode will specifically look for them.
    # required_tasks_for_cleaning_mode = ["task1_raw_ocr", "task2_augmented", "task3_cleaned"] # This logic is still useful for user feedback/validation
    
    requested_llm_task_keys_from_args = [key.strip() for key in args.llm_tasks.split(",") if key.strip()]
    
    # The OcrMultimodalDataset will load all tasks it finds. 
    # The `requested_llm_task_keys_from_args` is used by this script to determine what to extract.
    # For cleaning_pairs mode, we ensure the log reflects what tasks are critical.
    if args.extraction_mode == "cleaning_pairs":
        critical_tasks_for_cleaning = ["task1_raw_ocr", "task2_augmented", "task2_augmented_imperfections", "task3_cleaned"]
        logging.info(f"For 'cleaning_pairs' mode, the script will attempt to use these tasks from the loaded samples: {critical_tasks_for_cleaning}")
        # We don't need to modify loaded_llm_task_keys for the dataset constructor anymore.
        # The OcrMultimodalDataset loads all available tasks by default.


    if not requested_llm_task_keys_from_args and args.extraction_mode == "standard":
        logging.error("No LLM task keys specified for 'standard' extraction. Please use --llm_tasks.")
        sys.exit(1)
        
    video_ids_list = [vid.strip() for vid in args.video_ids.split(",")] if args.video_ids else None

    extract_text_data(
        frames_root_dir=args.frames_root,
        llm_outputs_root_dir=args.llm_root,
        tesseract_outputs_root_dir=args.tesseract_root,
        original_video_data_root_dir=args.video_data_root,
        output_file_path=args.output_file,
        requested_llm_task_keys=requested_llm_task_keys_from_args, # Used to select fields from sample
        extraction_mode=args.extraction_mode,
        video_ids_to_process=video_ids_list
    )

if __name__ == "__main__":
    main() 