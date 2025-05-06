# OCR Training Dataset Builder from YouTube Videos

This project provides tools to automatically extract frames from YouTube videos, preparing them for use in training Optical Character Recognition (OCR) models. The eventual goal is to use these frames with a multi-modal LLM to generate rich training data based on the visual content and (optionally) subtitles.

## 🌟 Features

*   🎞️ **Frame Extraction**: Extracts frames from video files at a configurable FPS.
*   ✂️ **Frame Sampling**: Optionally samples a maximum number of frames per video.
*   🖼️ **Image Resizing**: Optionally resizes frames to a maximum dimension while maintaining aspect ratio.
*   ⚙️ **Parallel Processing**: Efficiently processes multiple video **directories** in parallel using `concurrent.futures.ProcessPoolExecutor`.
*   📝 **Metadata Handling**: Copies associated metadata files (e.g., `.info.json`) to the output directory.
*   📊 **Progress Tracking**: Uses `tqdm` for progress bars during video processing.

## 📊 Current Status (as of last update)

*   ✅ **Milestone 1: Setup & Configuration**: Environment setup (Conda, Poetry), API key handling (`.env`), and Vertex AI client verification (`ocr_dataset_builder/examples/example-vertex.py`) are complete.
*   ✅ **Milestone 2: Frame Extraction Pipeline**:
    *   `ocr_dataset_builder/video_processing.py` (`extract_frames` function) is implemented.
    *   `ocr_dataset_builder/frame_pipeline.py` is implemented, enabling parallel processing of video directories, metadata copying, dataset slicing, CLI control, and robust checkpointing.
*   🚧 **Milestone 3: LLM Prompt Adaptation & Initial LLM Processing Code**:
    *   The core `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` has been significantly adapted for video frame sequences.
    *   Core setup for `ocr_dataset_builder/llm_processing.py` (logging, pathing) is done; Gemini interaction logic and output parsing refinement is ongoing.
*   🚧 **Milestone 4: LLM Pipeline Development**:
    *   `ocr_dataset_builder/llm_pipeline.py` structure established with checkpointing, parallel processing capabilities (for directories and batches), and standardized logging. Further development on result aggregation and output formatting is in progress.
*   🚧 **Milestone 5: Tesseract OCR Integration (Optional/Baseline)**:
    *   `ocr_dataset_builder/tesseract_processing.py` and `ocr_dataset_builder/tesseract_pipeline.py` have been updated with standardized logging. The pipeline includes checkpointing. Role and integration finalization pending.
*   ⏳ **Milestone 6: Full Pipeline Integration & Output Generation**: Integrating `frame_pipeline.py` -> (optional `tesseract_pipeline.py`) -> `llm_pipeline.py` and generating final structured output.
*   ✅ **Documentation Foundational Work**: `docs/PIPELINE_GUIDE.md` and `docs/DATA_FORMATS.md` are substantially complete. Core scripts have improved logging and structure. `docs/MILESTONES.md` and `docs/DESIGN.md` have been updated to reflect current progress.

## 🧩 Core Components

1.  **`ocr_dataset_builder/video_processing.py`**:
    *   Core logic for extracting, resizing, and sampling frames from individual video files.
    *   `extract_frames()`: Main function for per-video frame processing.
    *   `get_human_readable_size()`: Utility for file sizes.

2.  **`ocr_dataset_builder/frame_pipeline.py`**:
    *   Orchestrates the frame extraction process across an entire dataset of videos using `video_processing.py`.
    *   Handles directory traversal, metadata copying, parallel execution for video directories.
    *   Provides a CLI (`fire`) for dataset processing.

3.  **`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`**:
    *   Detailed multi-task prompt for the Gemini LLM, adapted for video frame sequence analysis (per-frame Tasks 1-4, per-sequence Task 5).

4.  **`ocr_dataset_builder/llm_processing.py`**:
    *   Handles the direct interaction with the multi-modal LLM (e.g., Gemini via Vertex AI).
    *   Takes a sequence of frames and the prompt to get the structured textual analysis.
    *   Manages API calls, responses, and error handling for a single LLM request.

5.  **`ocr_dataset_builder/llm_pipeline.py`**:
    *   Orchestrates the process of sending frame sequences (from `frame_pipeline.py` output) to the LLM via `llm_processing.py`.
    *   Manages batching of frame sequences, collecting LLM results, and preparing them for final output generation.

6.  **`ocr_dataset_builder/tesseract_processing.py` (Optional/Baseline)**:
    *   Performs OCR on individual image frames using the Tesseract OCR engine.
    *   Can be used to generate a baseline OCR text for comparison or as an input to other processes.

7.  **`ocr_dataset_builder/tesseract_pipeline.py` (Optional/Baseline)**:
    *   Orchestrates the application of Tesseract OCR (via `tesseract_processing.py`) over a dataset of extracted frames.

## 📂 Project Structure

```
ocr-dataset-builder/
├── .env.example             # Example for environment variables (e.g., API keys)
├── .gitignore
├── Dockerfile               # For containerized deployment (future)
├── MILESTONES.md            # Project milestones
├── README.md                # This file
├── docs/
│   └── DESIGN.md            # Design choices and architecture
├── install-conda-env.sh     # Script to create Conda environment
├── ocr_dataset_builder/
│   ├── __init__.py
│   ├── examples/            # Example scripts
│   │   └── example-vertex.py # Example for Vertex AI API usage
│   ├── prompts/
│   │   └── ocr_image_multi_task_prompt.md # LLM prompt
│   ├── frame_pipeline.py    # Orchestrates frame extraction
│   ├── video_processing.py   # Single video frame extraction
│   ├── llm_pipeline.py       # Orchestrates LLM analysis
│   ├── llm_processing.py     # Single LLM interaction
│   ├── tesseract_pipeline.py # Orchestrates Tesseract OCR (optional)
│   └── tesseract_processing.py # Single frame Tesseract OCR (optional)
├── poetry.lock
├── pyproject.toml           # Project metadata and dependencies (Poetry)
└── scripts/                 # Utility scripts (if any)
```

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

This project involves multiple pipelines. For detailed CLI arguments and usage, please refer to `docs/PIPELINE_GUIDE.md`.

### 1. Frame Extraction

Use `ocr_dataset_builder/frame_pipeline.py` to process a dataset of videos and extract frames.

**Example:**

```bash
python ocr_dataset_builder/frame_pipeline.py process_videos \
    --dataset_path "/path/to/your/video_dataset" \
    --output_path "./output/extracted_frames" \
    --target_fps 1
```

### 2. Tesseract OCR (Optional Baseline)

If baseline OCR is needed, `ocr_dataset_builder/tesseract_pipeline.py` can be used on the output of the frame extraction.

**Example:**
```bash
python ocr_dataset_builder/tesseract_pipeline.py \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/tesseract_ocr_results"
```

### 3. LLM-based Analysis

Once frames are extracted, `ocr_dataset_builder/llm_pipeline.py` will be used to process sequences of these frames with the Gemini LLM.

**Example:**
```bash
python ocr_dataset_builder/llm_pipeline.py run \
    --input_dir "./output/extracted_frames" \
    --output_dir "./output/llm_analysis_results" \
    --batch_size 30
```

## 🧑‍💻 Development

This project uses `ruff` for linting and `black` for formatting, managed via `poethepoet`.

*   **Format code:**
    ```bash
    poe format
    ```
*   **Lint code:**
    ```bash
    poe lint
    ```
    Or, to auto-fix linting issues where possible:
    ```bash
    poe lint --fix
    ```

## Next Steps

*   **Finalize `llm_processing.py`**: Ensure robust interaction with the Gemini API, including error handling and reliable parsing of the structured output based on `ocr_image_multi_task_prompt.md`.
*   **Complete `llm_pipeline.py` Functionality**: Implement final LLM result aggregation and saving in the defined structured output format (e.g., JSON files per batch).
*   **Integrate Pipelines**: Ensure smooth data flow: `frame_pipeline.py` output feeds into `llm_pipeline.py` (and optionally `tesseract_pipeline.py` if used as a distinct step).
*   **Refine Tesseract Pipeline (if pursued)**: Solidify its role and complete any remaining integration details.
*   **Testing & Iteration**: Thoroughly test each pipeline and the integrated flow with various scenarios and data.
*   **Finalize Core Documentation**: Review and polish `README.md`, `docs/DESIGN.md`, `docs/MILESTONES.md`, `docs/PIPELINE_GUIDE.md`, and `docs/DATA_FORMATS.md`.
*   **Expand Documentation**: Develop other planned documents (`docs/PROMPT_ENGINEERING_GUIDE.md`, `docs/TROUBLESHOOTING.md`, etc.) as the project matures.
