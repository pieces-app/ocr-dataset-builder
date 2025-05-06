# OCR Dataset Builder

A tool designed to create datasets for training multimodal OCR and context understanding models by processing frames from YouTube videos using Gemini via Vertex AI.

## Overview

This project takes a directory of YouTube videos (including metadata and potentially video files) and processes them to generate structured training data. It extracts frames, sends them in batches to a Gemini model for analysis based on a multi-task prompt (`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`), and outputs the results in a structured format (e.g., JSON Lines).

For detailed information on the design and processing pipeline, please see [`docs/design.md`](docs/design.md).

## Milestones

Project milestones are tracked in [`milestones.md`](milestones.md).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ocr-dataset-builder
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    The project uses `setuptools` for packaging. Install the core dependencies and optionally the `dev` dependencies:
    ```bash
    pip install .
    # For development (linting, formatting, testing):
    pip install .[dev]
    ```
4.  **API Key Setup:**
    - You will need access to Google Cloud Vertex AI.
    - Configure authentication (e.g., via `gcloud auth application-default login`).
    - Set up your Google Cloud Project ID, potentially using a `.env` file:
      ```
      GOOGLE_CLOUD_PROJECT="your-project-id"
      GOOGLE_CLOUD_LOCATION="your-region" # e.g., us-central1
      ```

## Usage

*(Details to be added once the main CLI script using `fire` is implemented)*

The general workflow will involve running a Python script from the command line, pointing it to the directory containing the YouTube video data and specifying an output location.

Example (Conceptual):
```bash
python ocr_dataset_builder/main.py --video_dir /mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/ --output_file ./ocr_dataset.jsonl --batch_size 60
```

## Development

This project uses `ruff`, `black`, `isort`, and `autoflake` for linting and formatting. Use `poethepoet` to run these tools:

```bash
poe format
poe linter
```

Tests are run using `pytest`:
```bash
pytest
```
