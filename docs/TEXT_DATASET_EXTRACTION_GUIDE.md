# Guide: Text Dataset Extraction Script (`extract_text_dataset.py`)

## 1. Purpose

The `extract_text_dataset.py` script is a utility designed to process the multimodal OCR dataset (built using `OcrMultimodalDataset`) and extract specific text-based information into a structured, text-only format. This is particularly useful for preparing datasets for training Natural Language Processing (NLP) models, such as models for OCR text correction, text summarization, or other text-based analysis tasks.

The script can generate datasets in two primary modes:
1.  **Standard Mode**: Extracts selected LLM task outputs and Tesseract OCR for each frame.
2.  **Cleaning Pairs Mode**: Generates pairs of "raw" OCR (from Tesseract, LLM Task 1, and LLM Task 2) and "clean" OCR (from LLM Task 3) for each frame, specifically tailored for training text-cleaning models.

## 2. Core Functionality

The script leverages the `OcrMultimodalDataset` class to:
- Load frame image paths.
- Load and reconstruct LLM task outputs from batch JSON files (handling the `F:i-1` notation).
- Load Tesseract OCR outputs.

It then iterates through the loaded samples and, based on the chosen extraction mode, selects and formats the required text data.

## 3. Output Format

The script outputs data in the **JSON Lines (JSONL)** format. Each line in the output file is a valid JSON object representing one data record.

## 4. Extraction Modes

### 4.1. Standard Mode (`--extraction_mode standard`)

In this mode, each line in the output JSONL file corresponds to one frame from the dataset. The record includes:
- `frame_path`: The path to the source frame image.
- LLM task outputs: Specified by the `--llm_tasks` argument (e.g., `task1_raw_ocr`, `task2_augmented`, etc.).
- `tesseract_ocr`: The OCR text from Tesseract for that frame.

**Example JSONL Record (Standard Mode):**
```json
{"frame_path": "/path/to/video1/frame_0001.png", "task1_raw_ocr": "Raw OCR text...", "task3_cleaned": "Cleaned OCR text...", "tesseract_ocr": "Tesseract text..."}
```

### 4.2. Cleaning Pairs Mode (`--extraction_mode cleaning_pairs`)

This mode is specifically designed for training models to clean or improve OCR text. For each source frame, it generates **three** JSONL records:
1.  Tesseract OCR vs. LLM Task 3 (Cleaned OCR)
2.  LLM Task 1 (Raw OCR) vs. LLM Task 3 (Cleaned OCR)
3.  LLM Task 2 (Augmented/Imperfections OCR) vs. LLM Task 3 (Cleaned OCR)

Each record contains:
- `frame_path`: The path to the source frame image.
- `raw_ocr`: The "noisier" version of the OCR text.
- `clean_ocr`: The "cleaner" version (LLM Task 3).

**Example JSONL Records (Cleaning Pairs Mode - for one source frame):**
```json
{"frame_path": "/path/to/video1/frame_0001.png", "raw_ocr": "Tesseract text for frame 1", "clean_ocr": "LLM Task 3 cleaned text for frame 1"}
{"frame_path": "/path/to/video1/frame_0001.png", "raw_ocr": "LLM Task 1 raw OCR for frame 1", "clean_ocr": "LLM Task 3 cleaned text for frame 1"}
{"frame_path": "/path/to/video1/frame_0001.png", "raw_ocr": "LLM Task 2 augmented OCR for frame 1", "clean_ocr": "LLM Task 3 cleaned text for frame 1"}
```

## 5. Command-Line Arguments

The script accepts the following arguments:

-   `--frames_root` (required): Path to the root directory of processed frame images.
-   `--llm_root` (required): Path to the root directory of LLM output JSON files.
-   `--tesseract_root` (required): Path to the root directory of Tesseract OCR JSON files.
-   `--video_data_root` (required): Path to the root directory of original video data (used by `OcrMultimodalDataset` for metadata/subtitles, though not directly output by this script).
-   `--output_file` (required): Path where the output JSON Lines file will be saved.
-   `--extraction_mode` (optional): Specifies the extraction logic.
    -   `standard` (default): One record per frame with selected tasks.
    -   `cleaning_pairs`: Three raw/clean OCR pairs per frame.
-   `--llm_tasks` (optional): Comma-separated list of LLM task keys to include if `extraction_mode` is `standard` (e.g., `task1_raw_ocr,task3_cleaned`). If `extraction_mode` is `cleaning_pairs`, the script hardcodes the use of `task1_raw_ocr`, `task2_augmented` (or `task2_augmented_imperfections`), and `task3_cleaned`. This argument still influences which tasks `OcrMultimodalDataset` attempts to load. Default: `task1_raw_ocr,task2_augmented,task3_cleaned`.
-   `--video_ids` (optional): Comma-separated list of specific video IDs to process. If not provided, all videos found will be processed.
-   `--log_level` (optional): Sets the logging level (e.g., `DEBUG`, `INFO`, `WARNING`). Default: `INFO`.

## 6. Example Usage

### 6.1. Standard Extraction

Extract `task1_raw_ocr`, `task3_cleaned`, and `tesseract_ocr` for all videos:
```bash
python ocr_dataset_builder/extract_text_dataset.py \
    --frames_root "/mnt/data-store/pieces-ocr-v-0-1-0-frames/" \
    --llm_root "/mnt/data-store/pieces-ocr-v-0-1-0-llm_output/" \
    --tesseract_root "/mnt/data-store/pieces-ocr-v-0-1-0-tesseract_output/" \
    --video_data_root "/mnt/data-store/pieces-ocr-v-0-1-0/" \
    --output_file "./standard_text_data.jsonl" \
    --extraction_mode standard \
    --llm_tasks "task1_raw_ocr,task3_cleaned" \
    --log_level INFO
```

### 6.2. Cleaning Pairs Extraction

Generate raw/clean OCR pairs for training a text cleaning model, for a specific video:
```bash
python ocr_dataset_builder/extract_text_dataset.py \
    --frames_root "/mnt/data-store/pieces-ocr-v-0-1-0-frames/" \
    --llm_root "/mnt/data-store/pieces-ocr-v-0-1-0-llm_output/" \
    --tesseract_root "/mnt/data-store/pieces-ocr-v-0-1-0-tesseract_output/" \
    --video_data_root "/mnt/data-store/pieces-ocr-v-0-1-0/" \
    --output_file "./text_cleaning_pairs_dataset.jsonl" \
    --extraction_mode cleaning_pairs \
    --video_ids "VIDEO_ID_TO_PROCESS" \
    --log_level DEBUG
```

Make sure to replace placeholder paths and `VIDEO_ID_TO_PROCESS` with your actual data locations and desired video ID. 