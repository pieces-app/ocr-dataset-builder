import argparse
import json
import logging
import sys
from pathlib import Path


try:
    from ocr_dataset_builder.data.pytorch_dataset import OcrMultimodalDataset
    # Import all augmentation functions
    from ocr_dataset_builder.data.ocr_augmentations import (
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
except ImportError:
    print("Error: Could not import OcrMultimodalDataset or augmentation functions. Make sure the ocr_dataset_builder package is installed and in your PYTHONPATH.")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="[%X]",
)

# Define the list of all augmentation functions
ALL_CUSTOM_AUGMENTATIONS = [
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

def extract_text_data(
    frames_root_dir: str,
    llm_outputs_root_dir: str,
    tesseract_outputs_root_dir: str,
    original_video_data_root_dir: str,
    output_file_path: str,
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
            custom_augmentation_funcs=ALL_CUSTOM_AUGMENTATIONS,
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
                    record = {
                        "frame_path": frame_path_str,
                        "tesseract_ocr": sample.get("tesseract_ocr", ""),
                        "llm_clean_ocr": sample.get("llm_clean_ocr", ""),
                        "augmented_llm_clean_ocr": sample.get("augmented_llm_clean_ocr", ""),
                        "markdown": sample.get("markdown", ""),
                        "summary": sample.get("summary", "")
                    }
                    outfile.write(json.dumps(record) + "\n")
                    output_records_count += 1

                elif extraction_mode == "cleaning_pairs":
                    tesseract_text = sample.get("tesseract_ocr", "")
                    llm_clean_text = sample.get("llm_clean_ocr", "")

                    record = {
                        "frame_path": frame_path_str,
                        "raw_ocr": tesseract_text,
                        "clean_ocr": llm_clean_text
                    }
                    outfile.write(json.dumps(record) + "\n")
                    output_records_count += 1
                
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
    parser.add_argument("--llm_root", type=str, required=True, help="Root directory of LLM outputs (used as source by OcrMultimodalDataset).")
    parser.add_argument("--tesseract_root", type=str, required=True, help="Root directory of Tesseract outputs.")
    parser.add_argument("--video_data_root", type=str, required=True, help="Root directory of original video data (e.g., for metadata, subtitles).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON Lines file.")
    
    parser.add_argument(
        "--extraction_mode",
        type=str,
        default="standard",
        choices=["standard", "cleaning_pairs"],
        help="Extraction mode: 'standard' for one line per frame with all key text fields, 'cleaning_pairs' for Tesseract OCR vs. LLM Clean OCR pairs."
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
    
    if args.extraction_mode == "cleaning_pairs":
        logging.info("For 'cleaning_pairs' mode, the script will generate pairs of Tesseract OCR (cleaned by dataset) vs. LLM Clean OCR.")
    elif args.extraction_mode == "standard":
        logging.info("For 'standard' mode, the script will extract: tesseract_ocr, llm_clean_ocr, augmented_llm_clean_ocr, markdown, summary.")
        
    video_ids_list = [vid.strip() for vid in args.video_ids.split(",")] if args.video_ids else None

    extract_text_data(
        frames_root_dir=args.frames_root,
        llm_outputs_root_dir=args.llm_root,
        tesseract_outputs_root_dir=args.tesseract_root,
        original_video_data_root_dir=args.video_data_root,
        output_file_path=args.output_file,
        extraction_mode=args.extraction_mode,
        video_ids_to_process=video_ids_list
    )

if __name__ == "__main__":
    main() 