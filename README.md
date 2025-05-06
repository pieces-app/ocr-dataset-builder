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
    *   `ocr_dataset_builder/frame_pipeline.py` (formerly `pipeline.py`) is implemented, enabling parallel processing of video directories for frame extraction, metadata copying, dataset slicing, and CLI control via `fire`.
*   🚧 **Milestone 3: LLM Prompt Adaptation & Initial LLM Processing Code**:
    *   The core `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` has been significantly adapted for video frame sequences.
    *   Initial version of `ocr_dataset_builder/llm_processing.py` exists for interacting with the Gemini API.
*   ⏳ **Milestone 4: LLM Pipeline Development**:
    *   Development of `ocr_dataset_builder/llm_pipeline.py` to manage sequences of frames, interact with `llm_processing.py`, and handle LLM outputs.
*   ⏳ **Milestone 5: Tesseract OCR Integration (Optional/Baseline)**:
    *   Initial versions of `ocr_dataset_builder/tesseract_processing.py` and `ocr_dataset_builder/tesseract_pipeline.py` exist. Their role (e.g., baseline OCR, pre-processing) and integration need to be finalized.
*   ⏳ **Milestone 6: Full Pipeline Integration & Output Generation**: Integrating `frame_pipeline.py` -> (optional `tesseract_pipeline.py`) -> `llm_pipeline.py` and generating final structured output.

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

This project involves multiple pipelines.

### 1. Frame Extraction

Use `ocr_dataset_builder/frame_pipeline.py` to process a dataset of videos and extract frames.

**Example:**

```bash
python ocr_dataset_builder/frame_pipeline.py process_videos \
    --dataset_path "/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/" \
    --output_path "./extracted_frames" \
    --target_fps 1 \
    --max_workers 4 \
    --start_index 0 \
    --end_index 10
```

**Parameters for `frame_pipeline.py`'s `process_videos` command are detailed in its docstrings/help output.** (Self-correction: Link to `docs/PIPELINE_GUIDE.md` once it exists)

### 2. Tesseract OCR (Optional Baseline)

If baseline OCR is needed, `ocr_dataset_builder/tesseract_pipeline.py` can be used on the output of the frame extraction.
(Further details and example command to be added once CLI is finalized)

### 3. LLM-based Analysis

Once frames are extracted, `ocr_dataset_builder/llm_pipeline.py` will be used to process sequences of these frames with the Gemini LLM.
(Further details and example command to be added once CLI is finalized and module is more developed)

**Parameters for `extract_frames` (used internally by `frame_pipeline.py`):**

*   `video_path`: Path to the input video file.
*   `output_dir`: Directory to save extracted frames.
*   `target_fps` (int, optional): Target frames per second for extraction. Default: 1.
*   `max_dimension` (int | None, optional): Max dimension for resizing frames. Default: 1024.
*   `max_frames_per_video` (int | None, optional): Max frames to sample per video. Default: None (save all).

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

*   **Finalize `llm_processing.py`**: Ensure robust interaction with the Gemini API, including error handling and parsing of the structured output based on `ocr_image_multi_task_prompt.md`.
*   **Develop `llm_pipeline.py`**:
    *   Implement logic to read frame sequences from `frame_pipeline.py`'s output.
    *   Batch frame sequences appropriately for `llm_processing.py`.
    *   Manage the overall workflow of sending data to the LLM and collecting results.
    *   Define and implement a CLI using `fire`.
*   **Refine Tesseract Pipeline (if pursued)**: Clarify the role and finalize the implementation and CLI for `tesseract_pipeline.py` and `tesseract_processing.py`.
*   **Integrate Pipelines**: Create a master flow: `frame_pipeline.py` output feeds into `llm_pipeline.py` (and optionally `tesseract_pipeline.py`).
*   **Output Generation**: Define and implement the final structured output format (e.g., JSONL) for the LLM analysis.
*   **Testing & Iteration**: Thoroughly test each pipeline and the integrated flow with various scenarios and data.
*   **Documentation**:
    *   Update `docs/DESIGN.md` and `docs/MILESTONES.md` to reflect the detailed structure.
    *   Create `docs/PIPELINE_GUIDE.md` with detailed usage for each pipeline.
    *   Create `docs/DATA_FORMATS.md`.
*   Continue development based on `docs/MILESTONES.md`.
