# Data Formats: OCR Dataset Builder

This document specifies the expected data formats for inputs to and outputs from the various pipelines in the OCR Dataset Builder project.

## Table of Contents

- [Data Formats: OCR Dataset Builder](#data-formats-ocr-dataset-builder)
  - [Table of Contents](#table-of-contents)
  - [1. Input Video Dataset Structure](#1-input-video-dataset-structure)
  - [2. Frame Extraction Pipeline Output](#2-frame-extraction-pipeline-output)
  - [3. Tesseract OCR Pipeline Output](#3-tesseract-ocr-pipeline-output)
  - [4. LLM Analysis Pipeline Output](#4-llm-analysis-pipeline-output)
  - [5. Text LLM Refinement Pipeline Input (from Tesseract)](#5-text-llm-refinement-pipeline-input-from-tesseract)
  - [6. Text LLM Refinement Pipeline Output](#6-text-llm-refinement-pipeline-output)

---

## 1. Input Video Dataset Structure

- **Root Directory**: A main directory containing subdirectories for each video.
  - **Example**: `/mnt/datasets/my_youtube_collection/`

- **Video Subdirectory**: Each subdirectory within the root directory corresponds to a single video.
  - **Naming**: Typically named with a video ID or descriptive title (e.g., `Video_ID_xyz/`, `My_Tutorial_Video/`).
  - **Contents**:
    - **Video File**: At least one video file (e.g., `.mp4`, `.mkv`, `.webm`). The `frame_pipeline.py` will attempt to find the most prominent video file if multiple exist.
    - **Metadata File (Optional but Recommended)**: A `.info.json` file (as downloaded by tools like `yt-dlp`) containing metadata about the video (title, uploader, description, tags, etc.). The `frame_pipeline.py` copies this file to the output.
      - **Example**: `My_Tutorial_Video.info.json`
    - **Subtitles (Optional, Future Use)**: `.vtt` or `.srt` subtitle files. Currently not directly used by the pipelines but may be incorporated later.

**Example Structure:**

```bash
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

**Pipeline:** `ocr_dataset_builder/video/frame_pipeline.py`

This pipeline generates an output directory structure that mirrors the input video dataset.

- **Output Root Directory**: Specified by the `--output_path` argument to `frame_pipeline.py`.
  - **Example**: `./output/extracted_frames/`

- **Mirrored Video Subdirectory**: For each processed input video subdirectory, a corresponding subdirectory is created here.
  - **Example**: `./output/extracted_frames/Video_A_ID/`
  - **Contents**:
    - **Extracted Frames**: Image files, typically named sequentially.
      - **Format**: `.png` (current default in `video/processing.py`)
      - **Naming Convention**: `frame_{frame_number:06d}.png` (e.g., `frame_000000.png`, `frame_000001.png`). The frame number corresponds to the extracted frame sequence, influenced by `target_fps` and sampling.
    - **Copied Metadata File**: The `.info.json` file from the input video subdirectory is copied here.
      - **Example**: `video_a_title.info.json`
    - **Checkpoint File (in `output_path` root)**: A log file (e.g., `.processed_video_dirs.log`) listing relative paths of successfully processed video directories.

**Example Output Structure (for `output_path = ./output/extracted_frames/`):**

```bash
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

**Pipeline:** `ocr_dataset_builder/tesseract/pipeline.py`

This pipeline processes frame directories and outputs OCR results.

- **Output Root Directory**: Specified by the `--output_dir` argument to `tesseract_pipeline.py`.
  - **Example**: `./output/tesseract_ocr_results/`

- **Mirrored Frame Subdirectory**: For each processed input frame subdirectory, a corresponding subdirectory is created.
  - **Example**: `./output/tesseract_ocr_results/Video_A_ID/`
  - **Contents**:
    - **OCR Results File**: A single JSON file containing OCR text for all frames in that subdirectory.
      - **Naming Convention**: `tesseract_ocr.json` (current default).
      - **Format**: A JSON object where keys are frame filenames (e.g., `frame_000000.png`) and values are the extracted text strings. Failed OCR attempts might have a placeholder value like `"<<< OCR_FAILED >>>"`.

        ```json
            {
              "frame_000000.png": "Text extracted from frame 0.",
              "frame_000001.png": "Text from frame 1...",
              "frame_000002.png": "<<< OCR_FAILED >>>"
              // ... more frames
            }
        ```

    - **Checkpoint File (in `output_dir` root)**: A log file (e.g., `.processed_tesseract_dirs.log`) listing relative paths of successfully processed frame directories.

**Example Output Structure (for `output_dir = ./output/tesseract_ocr_results/`):**

```bash
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

**Pipeline:** `ocr_dataset_builder/llm/pipeline.py`

This pipeline generates structured analysis from the LLM for sequences of frames.

- **Output Root Directory**: Specified by the `--output_dir` argument to `llm_pipeline.py`.
  - **Example**: `./output/llm_analysis_results/`

- **Mirrored Video Subdirectory**: For each processed input video subdirectory (which contains the frames), a corresponding subdirectory is created here.
  - **Example**: `./output/llm_analysis_results/Video_A_ID/`
  - **Contents**:
    - **LLM Output Batch Files**: JSON files, each corresponding to one batch of N frames sent to the LLM.
      - **Naming Convention**: `llm_output_batch_{batch_index+1:04d}.json` (e.g., `llm_output_batch_0001.json`).
      - **Format**: The content of these files will be a JSON object representing the parsed output from the LLM, structured according to the `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`. This typically includes:
        - `TASK 1: Raw OCR Output` (List of N strings/placeholders)
        - `TASK 2: Augmented Imperfections` (List of N strings/placeholders)
        - `TASK 3: Cleaned OCR Text` (List of N strings/placeholders)
        - `TASK 4: Structured Markdown Output` (List of N Markdown strings/placeholders)
        - `TASK 5: Narrative Summary` (Single block of text for the N-frame sequence)

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

    - **Checkpoint File (in `output_dir` root)**: A log file (e.g., `.processed_llm_video_dirs.log`) listing relative paths of successfully processed video directories (meaning all their batches were processed).

**Example Output Structure (for `output_dir = ./output/llm_analysis_results/`):**

```bash
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

---

## 5. Text LLM Refinement Pipeline Input (from Tesseract)

**Source Pipeline:** `ocr_dataset_builder/tesseract/pipeline.py` (Expected Output Format)
**Consuming Pipeline:** `ocr_dataset_builder/llm/llm_text_pipeline.py`

This section describes the expected output format from the Tesseract OCR pipeline, which serves as the input for the `llm_text_pipeline.py`.

- **Input Root Directory**: A directory specified by `--input_dir` to `llm_text_pipeline.py`.
  - **Example**: `./output/tesseract_text_output/`

- **Mirrored Video Subdirectory**: For each processed video, a subdirectory is expected, mirroring the structure from frame extraction.
  - **Example**: `./output/tesseract_text_output/Video_A_ID/`
  - **Contents**:
    - **OCR Text Files**: Individual plain text files, one for each frame that underwent OCR.
      - **Naming Convention**: `frame_{frame_number:06d}.txt` (e.g., `frame_000000.txt`, `frame_000001.txt`). The name should ideally match the corresponding frame image file name (sans extension) and contain the raw Tesseract OCR text for that frame.
      - **Content**: Raw text output from Tesseract. If OCR failed or produced no text for a frame, the file might be empty or contain a specific placeholder if defined by the Tesseract pipeline.

**Example Input Structure (for `input_dir = ./output/tesseract_text_output/` to `llm_text_pipeline.py`):**

```bash
./output/tesseract_text_output/
├── Video_A_ID/
│   ├── frame_000000.txt
│   ├── frame_000001.txt
│   ├── ...
│   └── frame_XXXXXX.txt
├── Another_Video_XYZ/
│   ├── frame_000000.txt
│   ├── ...
└── ...
```

**Note on Tesseract Pipeline Output:**
To align with this input requirement for the Text LLM Refinement Pipeline, the `tesseract_pipeline.py` should be configured or modified to output individual `.txt` files per frame instead of a single consolidated JSON per video directory.

---

## 6. Text LLM Refinement Pipeline Output

**Pipeline:** `ocr_dataset_builder/llm/llm_text_pipeline.py`

This pipeline generates structured text refinements (Tasks 3, 4, 5) from the LLM for sequences of Tesseract OCR texts.

- **Output Root Directory**: Specified by the `--output_dir` argument to `llm_text_pipeline.py`.
  - **Example**: `./output/llm_text_refined_output/`

- **Mirrored Video Subdirectory**: For each processed input video subdirectory (which contains the Tesseract `.txt` files), a corresponding subdirectory is created here.
  - **Example**: `./output/llm_text_refined_output/Video_A_ID/`
  - **Contents**:
    - **LLM Output Batch Files**: JSON files, each corresponding to one batch of frame texts (e.g., 60 frames) sent to the LLM.
      - **Naming Convention**: `batch_{start_frame_index:06d}_{end_frame_index:06d}.json` (e.g., `batch_000000_000059.json`).
      - **Format**: The content of these files will be a JSON object representing the parsed output from the LLM, structured according to the `ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md`. This includes:
        - `video_id`: Identifier for the video.
        - `batch_info`: Details about the frame batch (start/end index, count).
        - `llm_output`:
          - `task3_cleaned_text`: List of cleaned text strings (one per frame in the batch).
          - `task4_markdown_text`: List of markdown strings (one per frame in the batch).
          - `task5_summary`: Single text block summarizing the content of the frames in the batch.
        - `token_counts`: Input and output token counts for the LLM call.
        - `processing_stats`: Timestamp and duration of processing.

            **Conceptual Example of `batch_000000_000059.json`:**

            ```json
            {
              "video_id": "Video_A_ID",
              "batch_info": {
                "start_frame_index": 0,
                "end_frame_index": 59,
                "num_frames_in_batch": 60
              },
              "llm_output": {
                "task3_cleaned_text": [
                  "Cleaned text for frame 0...",
                  "Cleaned text for frame 1..."
                  // ... up to 60 entries
                ],
                "task4_markdown_text": [
                  "### Frame 0 Content\n- Detail 1\n- Detail 2",
                  "### Frame 1 Content\n```python\nprint(\"Hello\")\n```"
                  // ... up to 60 entries
                ],
                "task5_summary": "This batch of 60 frames covers the initial setup and first examples of a Python tutorial on Flask..."
              },
              "token_counts": {
                "input_tokens": 12050,
                "output_tokens": 8750
              },
              "processing_stats": {
                "timestamp": "2023-10-27T10:30:00Z",
                "duration_seconds": 45.7
              }
            }
            ```

    - **Checkpoint File (in `output_dir` root)**: A JSON file (e.g., `llm_text_pipeline_checkpoint.json`) listing relative paths of successfully processed video directories (meaning all their batches were processed).

**Example Output Structure (for `output_dir = ./output/llm_text_refined_output/`):**

```bash
./output/llm_text_refined_output/
├── llm_text_pipeline_checkpoint.json (Example checkpoint file name)
├── Video_A_ID/
│   ├── batch_000000_000059.json
│   ├── batch_000060_000119.json
│   └── ...
├── Another_Video_XYZ/
│   ├── batch_000000_000059.json
│   └── ...
└── ...
```

---
*This document should be kept up-to-date with any changes to data structures or file naming conventions used by the pipelines.*
