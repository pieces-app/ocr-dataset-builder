# OCR Training Dataset Builder from YouTube Videos

This project provides tools to automatically extract frames from YouTube videos, preparing them for use in training Optical Character Recognition (OCR) models. The eventual goal is to use these frames with a multi-modal LLM to generate rich training data based on the visual content and (optionally) subtitles.

## 🌟 Features

*   🎞️ **Frame Extraction**: Extracts frames from video files at a configurable FPS.
*   ✂️ **Frame Sampling**: Optionally samples a maximum number of frames per video.
*   🖼️ **Image Resizing**: Optionally resizes frames to a maximum dimension while maintaining aspect ratio.
*   ⚙️ **Parallel Processing**: Efficiently processes multiple video **directories** in parallel using `concurrent.futures.ProcessPoolExecutor`.
*   📝 **Metadata Handling**: Copies associated metadata files (e.g., `.info.json`) to the output directory.
*   📊 **Progress Tracking**: Uses `tqdm` for progress bars during video processing.
*   🖼️ **LLM Multimodal Analysis**: Processes sequences of video frames with an LLM (e.g., Gemini) for detailed, multi-task visual and OCR analysis (Tasks 1-5 from images).
*   ✍️ **LLM Text Refinement**: Takes Tesseract OCR text output for frame sequences and uses an LLM (e.g., Gemini 2.5 Pro) to clean the text, convert it to markdown, and generate a contextual summary (Tasks 3-5 from text).

## 📊 Current Status (as of last update)

*   ✅ **Milestone 1: Setup & Configuration**: Environment setup (Conda, Poetry), API key handling (`.env`), and Vertex AI client verification are complete.
*   ✅ **Milestone 2: Frame Extraction Pipeline**:
    *   `ocr_dataset_builder/video/processing.py` (`extract_frames` function) is implemented and functional.
    *   `ocr_dataset_builder/video/frame_pipeline.py` is implemented and functional, enabling parallel processing, metadata handling, slicing, CLI, and checkpointing.
*   ✅ **Milestone 3: LLM Prompt Adaptation & Initial LLM Processing Code**:
    *   The core `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` has been adapted for video frame sequences.
    *   `ocr_dataset_builder/llm/processing.py` for image-based LLM interaction is implemented and functional.
*   ✅ **Milestone 4: LLM Multimodal Pipeline Development**:
    *   `ocr_dataset_builder/llm/pipeline.py` for image-based analysis is implemented and functional, with checkpointing, parallel processing, and standardized logging.
*   ✅ **Milestone 5: Tesseract OCR Integration (Optional/Baseline)**:
    *   `ocr_dataset_builder/tesseract/processing.py` and `ocr_dataset_builder/tesseract/pipeline.py` are implemented and functional, including checkpointing and logging.
*   🚧 **Milestone 6: Full Pipeline Integration & Output Generation**: Integrating `video/frame_pipeline.py` -> (optional `tesseract/pipeline.py`) -> `llm/pipeline.py` (image) and `llm/text_pipeline.py` (text) and generating final structured outputs. Development of the text LLM pipeline is in progress.
*   ✅ **Documentation Foundational Work**: `docs/PIPELINE_GUIDE.md`, `docs/DATA_FORMATS.md`, `docs/DESIGN.md`, and `docs/MILESTONES.md` have been updated to reflect current progress and the new text refinement pipeline. `docs/TEXT_LLM_PIPELINE_GUIDE.md` created.
*   ✅ **Codebase Refactoring**: Project structure updated with `video`, `tesseract`, and `llm` submodules. Imports and documentation reflect these changes.
*   ⏳ **New Text LLM Refinement Pipeline (Phase 2 - Implementation)**:
    *   `ocr_dataset_builder/llm/text_processing.py` (handles LLM interaction for text).
    *   `ocr_dataset_builder/llm/text_pipeline.py` (orchestrates the text refinement process).

## 🧩 Core Components

1.  **`ocr_dataset_builder/video/processing.py`**:
    *   Core logic for extracting, resizing, and sampling frames from individual video files.
    *   `extract_frames()`: Main function for per-video frame processing.
    *   `get_human_readable_size()`: Utility for file sizes.

2.  **`ocr_dataset_builder/video/frame_pipeline.py`**:
    *   Orchestrates the frame extraction process across an entire dataset of videos using `ocr_dataset_builder/video/processing.py`.
    *   Handles directory traversal, metadata copying, parallel execution for video directories.
    *   Provides a CLI (`fire`) for dataset processing.

3.  **`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`**:
    *   Detailed multi-task prompt for the Gemini LLM, adapted for video frame sequence analysis (per-frame Tasks 1-4, per-sequence Task 5).

4.  **`ocr_dataset_builder/llm/processing.py`**:
    *   Handles the direct interaction with the multi-modal LLM (e.g., Gemini via Vertex AI).
    *   Takes a sequence of frames and the prompt to get the structured textual analysis.
    *   Manages API calls, responses, and error handling for a single LLM request.

5.  **`ocr_dataset_builder/llm/pipeline.py`**:
    *   Orchestrates the process of sending frame sequences (from `video/frame_pipeline.py` output) to the LLM via `llm/processing.py`.
    *   Manages batching of frame sequences, collecting LLM results, and preparing them for final output generation.

6.  **`ocr_dataset_builder/tesseract/processing.py` (Optional/Baseline)**:
    *   Performs OCR on individual image frames using the Tesseract OCR engine.
    *   Can be used to generate a baseline OCR text for comparison or as an input to other processes.

7.  **`ocr_dataset_builder/tesseract/pipeline.py` (Optional/Baseline)**:
    *   Orchestrates the application of Tesseract OCR (via `tesseract/processing.py`) over a dataset of extracted frames.
    *   Expected to output individual `.txt` files per frame for consumption by the Text LLM Refinement Pipeline.

8.  **`ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md` (New)**:
    *   Specialized prompt for guiding an LLM to refine sequences of Tesseract OCR text, focusing on text cleaning (Task 3), markdown conversion (Task 4), and summarization (Task 5).

9.  **`ocr_dataset_builder/llm/llm_text_processing.py` (New - Planned)**:
    *   Handles direct interaction with an LLM for text-based refinement tasks using the `ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md`.

10. **`ocr_dataset_builder/llm/llm_text_pipeline.py` (New - Planned)**:
    *   Orchestrates the process of sending Tesseract OCR text sequences (e.g., from `tesseract/pipeline.py` output) to an LLM via `llm/llm_text_processing.py` for refinement.

## 🛠️ Setup

### Prerequisites

*   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
*   [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ocr-dataset-builder
```

### 2. Create Conda Environment

The `install-conda-env.sh` script automates the creation of the Conda environment using the `pyproject.toml` file for dependencies via Poetry.

```bash
bash install-conda-env.sh
```
This will create a Conda environment named `ocr-dataset-builder`. Activate it:
```bash
conda activate ocr-dataset-builder
```

### 3. Environment Variables (for LLM Interaction)

For the planned LLM interaction part of the project, you will need an API key.
Copy the `.env.example` to `.env` and fill in your `GOOGLE_API_KEY`:

```bash
cp .env.example .env
# Now edit .env and add your GOOGLE_API_KEY
# GOOGLE_API_KEY="your_api_key_here"
```
This key will be used by `scripts/example-vertex.py` and the future LLM processing module.

## 🚀 Usage

This project involves multiple pipelines. For detailed CLI arguments and usage, please refer to `docs/PIPELINE_GUIDE.md`, `docs/TEXT_LLM_PIPELINE_GUIDE.md`.

### 1. Frame Extraction

Use `ocr_dataset_builder/video/frame_pipeline.py` to process a dataset of videos and extract frames.

**Example:**

```bash
python -m ocr_dataset_builder.video.frame_pipeline process_videos \
    --dataset_path "/path/to/your/video_dataset" \
    --output_path "./output/extracted_frames" \
    --target_fps 1
```

### 2. Tesseract OCR (Optional Baseline)

If baseline OCR is needed, `ocr_dataset_builder/tesseract/pipeline.py` can be used on the output of the frame extraction.

**Example:**
```bash
python -m ocr_dataset_builder.tesseract.pipeline run \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/tesseract_ocr_results"
```

### 3. LLM-based Multimodal Analysis

Once frames are extracted, `ocr_dataset_builder/llm/pipeline.py` can be used to process sequences of these frames with the Gemini LLM.

**Example (Multimodal Image Analysis):**
```bash
python -m ocr_dataset_builder.llm.pipeline run \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/llm_analysis_results" \
    --batch_size 30
```

### 4. LLM-based Text Refinement (New - Planned)

After running the Tesseract OCR pipeline (which should output `.txt` files per frame), `ocr_dataset_builder/llm/llm_text_pipeline.py` can be used to refine these texts.

**Example (Text Refinement):**

```bash
python -m ocr_dataset_builder.llm.llm_text_pipeline run \
    --input_dir "./output/tesseract_text_output" \
    --output_dir "./output/llm_text_refined_output" \
    --frames_per_batch 60
```

## 🧑‍💻 Development

This project uses `ruff` for linting and `black`