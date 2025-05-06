# OCR Training Dataset Builder from YouTube Videos

This project provides tools to automatically extract frames from YouTube videos, preparing them for use in training Optical Character Recognition (OCR) models. The eventual goal is to use these frames with a multi-modal LLM to generate rich training data based on the visual content and (optionally) subtitles.

## 🌟 Features

*   🎞️ **Frame Extraction**: Extracts frames from video files at a configurable FPS.
*   ✂️ **Frame Sampling**: Optionally samples a maximum number of frames per video.
*   🖼️ **Image Resizing**: Optionally resizes frames to a maximum dimension while maintaining aspect ratio.
*   ⚙️ **Parallel Processing**: Designed to process multiple videos in parallel for efficiency (when implemented in a pipeline script).
*   📝 **Metadata Handling**: Copies associated metadata files (e.g., `.info.json`) to the output directory.
*   📊 **Progress Tracking**: Uses `tqdm` for progress bars during video processing.

## 🧩 Core Components

1.  **`ocr_dataset_builder/video_processing.py`**:
    *   Contains the core logic for extracting frames from individual video files.
    *   `extract_frames()`: The main function that handles video reading, frame selection, resizing, sampling, and saving.
    *   `get_human_readable_size()`: Utility function for displaying file sizes.

2.  **`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`**:
    *   A detailed multi-task prompt designed for a Gemini-based LLM.
    *   The prompt is structured to take a sequence of N frames and produce:
        *   Per-frame Raw OCR, Augmented OCR, Cleaned OCR, and Structured Markdown.
        *   Per-sequence Narrative Summary.
    *   This prompt is central to the project's goal of generating rich, structured OCR training data.

3.  **Pipeline Script (e.g., `run_pipeline.py` - to be developed/reinstated)**:
    *   Orchestrates the frame extraction process across an entire dataset of videos.
    *   Handles directory traversal, finding video and metadata files.
    *   Manages parallel execution of `extract_frames` for multiple videos.
    *   Provides a command-line interface for running the pipeline.

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
│   ├── prompts/
│   │   └── ocr_image_multi_task_prompt.md # LLM prompt
│   └── video_processing.py  # Frame extraction logic
├── poetry.lock
├── pyproject.toml           # Project metadata and dependencies (Poetry)
└── scripts/                 # Utility scripts (if any)
    └── example-vertex.py    # Example for Vertex AI API usage
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

## 🚀 Usage (Frame Extraction)

Once the pipeline script (e.g., `run_pipeline.py`) is implemented, you will typically run it from the command line.

**Example (conceptual):**

```bash
python -m ocr_dataset_builder.pipeline \
    --dataset_path "/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/" \
    --output_path "./extracted_frames" \
    --target_fps 1 \
    --max_workers 4 \
    --start_index 0 \
    --end_index 10
```

**Parameters for `extract_frames` (used by the pipeline):**

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

*   Implement/Reinstate the main pipeline script (`ocr_dataset_builder/pipeline.py` or similar) to manage the overall frame extraction workflow.
*   Develop the LLM interaction module (`llm_processing.py`) to process extracted frames using the Gemini prompt.
