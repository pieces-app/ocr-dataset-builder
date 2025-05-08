# Pipeline Guide: OCR Dataset Builder

This guide provides detailed instructions on how to run the various pipelines included in the OCR Dataset Builder project.

## Table of Contents

1.  [Frame Extraction Pipeline (`ocr_dataset_builder/video/frame_pipeline.py`)](#1-frame-extraction-pipeline)
2.  [Tesseract OCR Pipeline (`ocr_dataset_builder/tesseract/pipeline.py`)](#2-tesseract-ocr-pipeline)
3.  [LLM Multimodal Analysis Pipeline (`ocr_dataset_builder/llm/pipeline.py`)](#3-llm-multimodal-analysis-pipeline)

---

## 1. Frame Extraction Pipeline

**Script:** `ocr_dataset_builder/video/frame_pipeline.py`

This pipeline is responsible for processing raw video files from a dataset, extracting frames at a specified rate, resizing them, sampling a maximum number per video, and saving them to an output directory. It also copies associated metadata files.

### Purpose

The purpose of this pipeline is to extract frames from video files and save them to an output directory.

### CLI Usage

Run the pipeline using `python -m ocr_dataset_builder.video.frame_pipeline process_videos ...`

### Arguments

*   `dataset_path` (str):
    *   Description: Path to the root directory of the input dataset. Each subdirectory is expected to contain one video and optionally a `.info.json` metadata file.
    *   Required: Yes
*   `output_path` (str):
    *   Description: Path to the directory where extracted frames and copied metadata will be saved. A mirrored structure of the input dataset will be created here. The checkpoint file will also be stored in this directory.
    *   Required: Yes
*   `target_fps` (int, optional):
    *   Description: Target frames per second for extraction.
    *   Default: `1`
*   `max_dimension` (int | None, optional):
    *   Description: Maximum dimension (width or height) for frame resizing. If `None` or `0`, original size is kept.
    *   Default: `1024`
*   `max_frames_per_video` (int | None, optional):
    *   Description: Maximum number of frames to randomly sample and save per video. If `None`, all extracted frames (after FPS and resizing) are saved.
    *   Default: `None`
*   `start_index` (int | None, optional):
    *   Description: 0-based index of the first video directory to process from the list of pending directories (after accounting for checkpointing).
    *   Default: `0` (process from the beginning of pending directories)
*   `end_index` (int | None, optional):
    *   Description: 0-based index *after* the last video directory to process from the list of pending directories. If `None`, processes all pending directories until the end.
    *   Default: `None`
*   `max_workers` (int | None, optional):
    *   Description: Maximum number of parallel processes to use for processing video directories.
    *   Default: Number of CPU cores (`os.cpu_count()`)
*   `checkpoint_log` (str, optional):
    *   Description: Name for the checkpoint log file that tracks successfully processed video directories. Stored in the `output_path`.
    *   Default: `.processed_video_dirs.log`

### Example

```bash
python -m ocr_dataset_builder.video.frame_pipeline process_videos \
    --dataset_path "/path/to/video_dataset" \
    --output_path "./output/extracted_frames" \
    --target_fps 1 \
    --max_dimension 768 \
    --max_frames_per_video 100 \
    --max_workers 4 \
    --checkpoint_log ".frame_pipeline_checkpoint.log"
```

### Outputs

*   Extracted frames (e.g., `frame_XXXXXX.png`) saved in subdirectories under `output_path`, mirroring the `dataset_path` structure.
*   Copied metadata files (e.g., `.info.json`) alongside the frames.
*   A checkpoint file (e.g., `.frame_pipeline_checkpoint.log`) in `output_path`.

---

## 2. Tesseract OCR Pipeline

**Script:** `ocr_dataset_builder/tesseract/pipeline.py`

This pipeline (optional) processes directories of extracted frames (typically the output of the Frame Extraction Pipeline) using the Tesseract OCR engine. It generates JSON files containing the OCR text for each frame within a directory.

### Purpose

The purpose of this pipeline is to process extracted frames using the Tesseract OCR engine and generate JSON files containing the OCR text for each frame.

### CLI Usage

Run the pipeline using `python -m ocr_dataset_builder.tesseract.pipeline run ...`

### Arguments

*   `input_dir` (str, optional):
    *   Description: Root directory containing extracted frame subdirectories (output from `frame_pipeline.py`).
    *   Default: `./extracted_frames`
*   `output_dir` (str, optional):
    *   Description: Root directory to save the Tesseract OCR results (JSON files per input subdirectory) and the checkpoint file.
    *   Default: `./tesseract_output`
*   `language` (str, optional):
    *   Description: Tesseract language code (e.g., 'eng', 'deu').
    *   Default: `eng`
*   `max_workers` (int | None, optional):
    *   Description: Maximum number of parallel processes to use for processing frame directories.
    *   Default: Number of CPU cores (`os.cpu_count()`)
*   `start_index` (int | None, optional):
    *   Description: 0-based index of the first frame directory to process (after checkpointing).
    *   Default: `0`
*   `end_index` (int | None, optional):
    *   Description: 0-based index *after* the last frame directory to process (after checkpointing).
    *   Default: `None`
*   `checkpoint_file_name` (str, optional):
    *   Description: Name for the checkpoint log file that tracks successfully processed frame directories. Stored in the `output_dir`.
    *   Default: `.processed_tesseract_dirs.log`

### Example

```bash
python -m ocr_dataset_builder.tesseract.pipeline run \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/tesseract_output" \
    --language "eng" \
    --max_workers 4
```

### Outputs

*   JSON files (e.g., `tesseract_ocr.json`) in subdirectories under `output_dir`, containing OCR text per frame.
*   A checkpoint file (e.g., `.processed_tesseract_dirs.log`) in `output_dir`.

---

## 3. LLM Multimodal Analysis Pipeline

**Script:** `ocr_dataset_builder/llm/pipeline.py`

This pipeline processes sequences of extracted frames using a multi-modal Large Language Model (LLM) like Gemini, based on a detailed prompt. It generates structured analytical data for each frame sequence.

### Purpose

The purpose of this pipeline is to process extracted frames using a multi-modal Large Language Model (LLM) and generate structured analytical data for each frame sequence.

### CLI Usage

Run the pipeline using `python -m ocr_dataset_builder.llm.pipeline run ...`

### Arguments

*   `input_dir` (str, optional):
    *   Description: Root directory containing the extracted frame subdirectories (typically output from `frame_pipeline.py`).
    *   Default: `./extracted_frames`
*   `output_dir` (str, optional):
    *   Description: Root directory to save the LLM processing results (JSON files per batch/sequence) and the checkpoint file.
    *   Default: `./llm_output`
*   `batch_size` (int, optional):
    *   Description: Number of frames to process in each LLM call (per batch/sequence).
    *   Default: `60`
*   `max_workers_dirs` (int | None, optional):
    *   Description: Maximum number of parallel video directories to process.
    *   Default: `2`
*   `max_workers_batches_per_dir` (int | None, optional):
    *   Description: Maximum number of parallel batches of frames to process for LLM calls within a single video directory.
    *   Default: `2`
*   `start_index` (int | None, optional):
    *   Description: 0-based index of the first video directory to process (after checkpointing).
    *   Default: `0`
*   `end_index` (int | None, optional):
    *   Description: 0-based index *after* the last video directory to process (after checkpointing).
    *   Default: `None`
*   `prompt_path` (str, optional):
    *   Description: Path to the LLM prompt file.
    *   Default: `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`
*   `model_name` (str, optional):
    *   Description: Gemini model name to be used (e.g., "gemini-1.5-pro-latest").
    *   Default: Value of `GEMINI_MODEL_NAME` environment variable, or `gemini-1.5-pro-latest`.
*   `checkpoint_log` (str, optional):
    *   Description: Name for the checkpoint log file that tracks successfully processed video directories. Stored in the `output_dir`.
    *   Default: `.processed_llm_video_dirs.log`

### Example

```bash
python -m ocr_dataset_builder.llm.pipeline run \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/llm_analysis_results" \
    --batch_size 30 \
    --max_workers_dirs 2 \
    --max_workers_batches_per_dir 1 \
    --model_name "gemini-1.5-flash-latest"
```

### Outputs

*   JSON files (e.g., `llm_output_batch_XXXX.json`) in subdirectories under `output_dir`, containing structured LLM analysis for each N-frame sequence.
*   A checkpoint file (e.g., `.processed_llm_video_dirs.log`) in `output_dir`.

---
*This guide should be updated as pipeline parameters or behaviors change.* 