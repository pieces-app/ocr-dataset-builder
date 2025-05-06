# Design Document: OCR Training Dataset Builder

## 1. Introduction

This document outlines the design and architecture of the OCR Training Dataset Builder. The primary goal is to process YouTube video content (frames and metadata) to generate a high-quality dataset for training multi-modal OCR and visual understanding models, leveraging a sophisticated LLM prompt for analysis. The system is architected as a series of interconnected pipelines for frame extraction, optional Tesseract-based OCR, and LLM-based multimodal analysis.

## 2. High-Level Architecture

The system operates in sequential stages, typically involving these core pipelines:

1. **Frame Extraction Pipeline (`frame_pipeline.py`)**: Ingests raw video datasets, extracts, processes (resizes, samples), and saves individual frames along with associated video metadata.
2. **Tesseract OCR Pipeline (`tesseract_pipeline.py`) (Optional)**: Processes extracted frames using Tesseract to generate baseline raw OCR text. This can serve as an input for comparison or potentially for the LLM at a later stage.
3. **LLM Analysis Pipeline (`llm_pipeline.py`)**: Takes sequences of extracted frames (and optionally, Tesseract OCR output), sends them to a multi-modal LLM (e.g., Gemini) using the detailed prompt, and processes the structured analytical output.
4. **Output Generation**: The final rich, structured data from the LLM pipeline is saved, linking back to the source video and frames.

## 3. Core Modules and Components

### 3.1. Frame Extraction & Processing (`ocr_dataset_builder/video_processing.py`)

* **Purpose**: To efficiently extract, resize, sample, and save frames from individual video files.
* **Key Function**: `extract_frames()`
  * **Input**: Video file path, output directory, target FPS, max dimension for resizing, max frames to sample.
  * **Process**: Uses OpenCV (`cv2`) for video operations. Detailed in its docstrings.
  * **Output**: List of paths to saved frames (e.g., `frame_XXXXXX.png`).
* **Utilities**: `get_human_readable_size()`.

### 3.2. Frame Extraction Pipeline Orchestration (`ocr_dataset_builder/frame_pipeline.py`)

* **Purpose**: To manage the frame extraction process for an entire dataset of videos.
* **Responsibilities**:
  * Traverses input dataset directory structure.
  * For each video subdirectory, invokes `video_processing.py`'s `extract_frames` function.
  * Handles parallel processing of video directories using `concurrent.futures.ProcessPoolExecutor`.
  * Copies metadata files (e.g., `.info.json`) to the mirrored output subdirectories.
  * Provides a CLI using `fire` (e.g., `process_videos` command) for specifying dataset paths, processing parameters (FPS, workers, slicing indices).
  * Includes `tqdm` progress bars.

### 3.3. Tesseract OCR Processing (`ocr_dataset_builder/tesseract_processing.py`) (Optional)

* **Purpose**: To perform OCR on a single image frame using Tesseract.
* **Key Function**: (e.g., `ocr_frame_with_tesseract()`)
  * **Input**: Path to an image frame.
  * **Process**: Uses `pytesseract` to extract text from the image.
  * **Output**: Raw extracted text string.

### 3.4. Tesseract OCR Pipeline Orchestration (`ocr_dataset_builder/tesseract_pipeline.py`) (Optional)

* **Purpose**: To apply Tesseract OCR across a dataset of extracted frames.
* **Responsibilities**:
  * Traverses directories containing frames (typically output from `frame_pipeline.py`).
  * For each frame, invokes `tesseract_processing.py`.
  * Saves the Tesseract output (e.g., as `.txt` files corresponding to each frame or a consolidated JSON).
  * Provides a CLI for specifying input frame directory, output directory, and Tesseract configurations.
  * May include parallel processing for frames.

### 3.5. LLM Prompt (`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`)

* **Purpose**: To guide the LLM in analyzing frame sequences and producing structured data.
* **Design for Frame Sequences**: Tasks 1-4 per frame, Task 5 per sequence. Includes detailed speaker attribution and redundancy/appending rules.

### 3.6. LLM Interaction Module (`ocr_dataset_builder/llm_processing.py`)

* **Purpose**: To handle direct communication with the LLM API (e.g., Google Vertex AI for Gemini) for a single sequence of frames.
* **Responsibilities**:
  * Takes a sequence of N frame image data (and potentially paths or other metadata like Tesseract output) and the formatted prompt.
  * Constructs and executes the API request.
  * Handles API responses, including retries and error management for a single call.
  * Parses the LLM's structured text output into a machine-readable format (e.g., Python dictionaries) according to the prompt's specifications.

### 3.7. LLM Analysis Pipeline Orchestration (`ocr_dataset_builder/llm_pipeline.py`)

* **Purpose**: To manage the LLM-based analysis for an entire dataset of frame sequences.
* **Responsibilities**:
  * Identifies and batches sequences of N frames from the output of `frame_pipeline.py`.
  * For each sequence, invokes `llm_processing.py`.
  * Collects and aggregates the structured LLM outputs.
  * Handles overall progress tracking, logging, and error management for the batch LLM analysis.
  * Provides a CLI for specifying input frame sequence directory (from `frame_pipeline.py`), output directory for LLM results, LLM model parameters, and batching configurations.
  * Saves the final aggregated and structured LLM analysis data (e.g., as JSON Lines).

## 4. Data Flow

1. **Input**: User provides `dataset_path` (containing video subdirectories) to `frame_pipeline.py`.
2. **`frame_pipeline.py`**: Processes videos in parallel.
    a.  For each video: uses `video_processing.py` to extract, resize, sample frames.
    b.  Saves frames (e.g., `output_path/video_id/frame_XXXXXX.png`) and copies metadata (e.g., `output_path/video_id/video.info.json`).
3. **(Optional) `tesseract_pipeline.py`**: User points this pipeline to the frame output directory from step 2.
    a.  Processes frames using `tesseract_processing.py`.
    b.  Saves raw text output (e.g., `tesseract_output_path/video_id/frame_XXXXXX.txt`).
4. **`llm_pipeline.py`**: User points this pipeline to the frame output directory (from step 2) and optionally to Tesseract output (from step 3).
    a.  Identifies sequences of N frames per video.
    b.  For each sequence: uses `llm_processing.py` to interact with the Gemini LLM using `ocr_image_multi_task_prompt.md`.
    c.  Receives structured text output from the LLM.
    d.  Parses and saves this rich analysis, often in a structured format like JSONL (e.g., `llm_output_path/video_id/analysis_batch_Y.jsonl`), linking back to frame numbers and video ID.

## 5. Key Design Decisions & Considerations

* **Modular Pipelines**: Separating frame extraction, optional Tesseract OCR, and LLM analysis into distinct, chained pipelines enhances modularity, testability, and allows for independent execution or re-runs of specific stages.
* **Parallelism**: Employed within each relevant pipeline (e.g., per-video in `frame_pipeline.py`, potentially per-frame or per-sequence in other pipelines) to maximize throughput.
* **Configuration**: CLIs for each pipeline provide flexibility.
* **Robustness**: Logging and error handling are critical at each stage.
* **Mirrored Output Structure**: `frame_pipeline.py` maintains a mirrored structure, simplifying data association for subsequent pipelines.
* **Prompt-Driven LLM Interaction**: The `ocr_image_multi_task_prompt.md` is central to the LLM's analytical capabilities.

## 6. Future Enhancements

* Integration of subtitle data (`.vtt` files) as additional context for the LLM within `llm_pipeline.py`.
* More sophisticated frame selection in `video_processing.py` (e.g., scene change detection).
* Support for various output formats for the LLM data.
* Containerization with Docker for all pipelines.
