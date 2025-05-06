# Data Formats: OCR Dataset Builder

This document specifies the expected data formats for inputs to and outputs from the various pipelines in the OCR Dataset Builder project.

## Table of Contents

- [Data Formats: OCR Dataset Builder](#data-formats-ocr-dataset-builder)
  - [Table of Contents](#table-of-contents)
  - [1. Input Video Dataset Structure](#1-input-video-dataset-structure)
  - [2. Frame Extraction Pipeline Output](#2-frame-extraction-pipeline-output)
  - [3. Tesseract OCR Pipeline Output](#3-tesseract-ocr-pipeline-output)
  - [4. LLM Analysis Pipeline Output](#4-llm-analysis-pipeline-output)

---

## 1. Input Video Dataset Structure

*   **Root Directory**: A main directory containing subdirectories for each video.
    *   **Example**: `/mnt/datasets/my_youtube_collection/`

*   **Video Subdirectory**: Each subdirectory within the root directory corresponds to a single video.
    *   **Naming**: Typically named with a video ID or descriptive title (e.g., `Video_ID_xyz/`, `My_Tutorial_Video/`).
    *   **Contents**:
        *   **Video File**: At least one video file (e.g., `.mp4`, `.mkv`, `.webm`). The `frame_pipeline.py` will attempt to find the most prominent video file if multiple exist.
        *   **Metadata File (Optional but Recommended)**: A `.info.json` file (as downloaded by tools like `yt-dlp`) containing metadata about the video (title, uploader, description, tags, etc.). The `frame_pipeline.py` copies this file to the output.
            *   **Example**: `My_Tutorial_Video.info.json`
        *   **Subtitles (Optional, Future Use)**: `.vtt` or `.srt` subtitle files. Currently not directly used by the pipelines but may be incorporated later.

**Example Structure:**

```
/path/to/video_dataset/
├── Video_A_ID/
│   ├── video_a_title.mp4
│   └── video_a_title.info.json
│   └── video_a_title.en.vtt (optional)
├── Another_Video_XYZ/
│   ├── another_vid.mkv
│   └── another_vid.info.json
└── ... (more video subdirectories)
```

---

## 2. Frame Extraction Pipeline Output

**Pipeline:** `ocr_dataset_builder/frame_pipeline.py`

This pipeline generates an output directory structure that mirrors the input video dataset.

*   **Output Root Directory**: Specified by the `--output_path` argument to `frame_pipeline.py`.
    *   **Example**: `./output/extracted_frames/`

*   **Mirrored Video Subdirectory**: For each processed input video subdirectory, a corresponding subdirectory is created here.
    *   **Example**: `./output/extracted_frames/Video_A_ID/`
    *   **Contents**:
        *   **Extracted Frames**: Image files, typically named sequentially.
            *   **Format**: `.png` (current default in `video_processing.py`)
            *   **Naming Convention**: `frame_{frame_number:06d}.png` (e.g., `frame_000000.png`, `frame_000001.png`). The frame number corresponds to the extracted frame sequence, influenced by `target_fps` and sampling.
        *   **Copied Metadata File**: The `.info.json` file from the input video subdirectory is copied here.
            *   **Example**: `video_a_title.info.json`
        *   **Checkpoint File (in `output_path` root)**: A log file (e.g., `.processed_video_dirs.log`) listing relative paths of successfully processed video directories.

**Example Output Structure (for `output_path = ./output/extracted_frames/`):**

```
./output/extracted_frames/
├── .processed_video_dirs.log  (Example checkpoint file name)
├── Video_A_ID/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   ├── ...
│   └── video_a_title.info.json
├── Another_Video_XYZ/
│   ├── frame_000000.png
│   ├── ...
│   └── another_vid.info.json
└── ...
```

---

## 3. Tesseract OCR Pipeline Output

**Pipeline:** `ocr_dataset_builder/tesseract_pipeline.py`

This pipeline processes frame directories and outputs OCR results.

*   **Output Root Directory**: Specified by the `--output_dir` argument to `tesseract_pipeline.py`.
    *   **Example**: `./output/tesseract_ocr_results/`

*   **Mirrored Frame Subdirectory**: For each processed input frame subdirectory, a corresponding subdirectory is created.
    *   **Example**: `./output/tesseract_ocr_results/Video_A_ID/`
    *   **Contents**:
        *   **OCR Results File**: A single JSON file containing OCR text for all frames in that subdirectory.
            *   **Naming Convention**: `tesseract_ocr.json` (current default).
            *   **Format**: A JSON object where keys are frame filenames (e.g., `frame_000000.png`) and values are the extracted text strings. Failed OCR attempts might have a placeholder value like `"<<< OCR_FAILED >>>"`.
            ```json
            {
              "frame_000000.png": "Text extracted from frame 0.",
              "frame_000001.png": "Text from frame 1...",
              "frame_000002.png": "<<< OCR_FAILED >>>"
              // ... more frames
            }
            ```
        *   **Checkpoint File (in `output_dir` root)**: A log file (e.g., `.processed_tesseract_dirs.log`) listing relative paths of successfully processed frame directories.

**Example Output Structure (for `output_dir = ./output/tesseract_ocr_results/`):**

```
./output/tesseract_ocr_results/
├── .processed_tesseract_dirs.log (Example checkpoint file name)
├── Video_A_ID/
│   └── tesseract_ocr.json
├── Another_Video_XYZ/
│   └── tesseract_ocr.json
└── ...
```

---

## 4. LLM Analysis Pipeline Output

**Pipeline:** `ocr_dataset_builder/llm_pipeline.py`

This pipeline generates structured analysis from the LLM for sequences of frames.

*   **Output Root Directory**: Specified by the `--output_dir` argument to `llm_pipeline.py`.
    *   **Example**: `./output/llm_analysis_results/`

*   **Mirrored Video Subdirectory**: For each processed input video subdirectory (which contains the frames), a corresponding subdirectory is created here.
    *   **Example**: `./output/llm_analysis_results/Video_A_ID/`
    *   **Contents**:
        *   **LLM Output Batch Files**: JSON files, each corresponding to one batch of N frames sent to the LLM.
            *   **Naming Convention**: `llm_output_batch_{batch_index+1:04d}.json` (e.g., `llm_output_batch_0001.json`).
            *   **Format**: The content of these files will be a JSON object representing the parsed output from the LLM, structured according to the `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`. This typically includes:
                *   `TASK 1: Raw OCR Output` (List of N strings/placeholders)
                *   `TASK 2: Augmented Imperfections` (List of N strings/placeholders)
                *   `TASK 3: Cleaned OCR Text` (List of N strings/placeholders)
                *   `TASK 4: Structured Markdown Output` (List of N Markdown strings/placeholders)
                *   `TASK 5: Narrative Summary` (Single block of text for the N-frame sequence)

                Refer to the `FINAL OUTPUT FORMAT` section in the prompt for the exact expected keys and structure.

            **Conceptual Example of `llm_output_batch_0001.json`:**
            ```json
            {
                "TASK 1: Raw OCR Output": [
                    "-- Frame 0 --\nRaw text for frame 0...",
                    "-- Frame 1 --\nF:0 + \"Appended text for frame 1...\""
                    // ... up to N frames
                ],
                "TASK 2: Augmented Imperfections": [
                    // ... similar structure ...
                ],
                "TASK 3: Cleaned OCR Text": [
                    // ... similar structure ...
                ],
                "TASK 4: Structured Markdown Output": [
                    // ... similar structure with Markdown blocks ...
                ],
                "TASK 5: Narrative Summary": "This N-frame sequence depicted..."
            }
            ```
        *   **Checkpoint File (in `output_dir` root)**: A log file (e.g., `.processed_llm_video_dirs.log`) listing relative paths of successfully processed video directories (meaning all their batches were processed).

**Example Output Structure (for `output_dir = ./output/llm_analysis_results/`):**

```
./output/llm_analysis_results/
├── .processed_llm_video_dirs.log (Example checkpoint file name)
├── Video_A_ID/
│   ├── llm_output_batch_0001.json
│   ├── llm_output_batch_0002.json
│   └── ...
├── Another_Video_XYZ/
│   ├── llm_output_batch_0001.json
│   └── ...
└── ...
```

---
*This document should be kept up-to-date with any changes to data structures or file naming conventions used by the pipelines.* 