# Project Milestones - OCR Dataset Builder

This document outlines the key milestones for the OCR Dataset Builder project.

- [ ] **Milestone 1: Setup & Configuration**
    - [ ] Initialize project structure (`pyproject.toml`, `README.md`, `docs/`, etc.).
    - [ ] Set up virtual environment and install dependencies.
    - [ ] Configure API key handling (e.g., using `.env` and `python-dotenv`).
    - [ ] Verify Vertex AI client setup (`example-vertex.py`).

- [ ] **Milestone 2: Frame Extraction Module**
    - [ ] Implement function/class using `opencv-python` to extract frames from a video file at 1 frame per second.
    - [ ] Define frame storage strategy (temporary files vs. persistent naming).
    - [ ] Test frame extraction on sample videos.

- [ ] **Milestone 3: LLM Interaction Module**
    - [ ] Adapt the prompt from `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` for batch frame processing.
        - [ ] Define specific instructions for Tasks 1-4 (per frame).
        - [ ] Define specific instructions for Task 5 (per batch).
        - [ ] Finalize the adapted Task 4 structure for raw video frames.
    - [ ] Implement function using `google-genai` to call the Gemini 2.5 Pro model via Vertex AI with multimodal input (60 frames + adapted prompt).
    - [ ] Handle API responses and extract outputs for each task.
    - [ ] Implement error handling and retries for API calls.

- [ ] **Milestone 4: Data Assembly & Output Generation**
    - [ ] Implement function to load video metadata from `.info.json`.
    - [ ] Implement logic to structure the final output per batch (JSON format) combining:
        - Frame paths
        - LLM task outputs (Tasks 1-5)
        - Video ID
        - Selected metadata
    - [ ] Define output storage strategy (e.g., JSON Lines file).

- [ ] **Milestone 5: Pipeline Orchestration**
    - [ ] Develop the main script (`ocr_dataset_builder/main.py` or similar).
    - [ ] Use `fire` to create a command-line interface for the script (e.g., specifying input video directory, output path, batch size).
    - [ ] Integrate modules: Frame Extraction, Batching, LLM Interaction, Data Assembly, Output Saving.
    - [ ] Add logging (`rich`) and progress indicators.

- [ ] **Milestone 6: Initial Dataset Generation & Testing**
    - [ ] Run the pipeline on a small subset of videos (e.g., 1-3 videos).
    - [ ] Verify the generated output structure and content.
    - [ ] Perform basic quality checks on the LLM outputs.

- [ ] **Milestone 7: Evaluation & Refinement**
    - [ ] Analyze the results from the initial run.
    - [ ] Identify areas for improvement (e.g., prompt tuning, Task 4 structure, error handling).
    - [ ] Refine the pipeline components based on findings.

- [ ] **Milestone 8: Full Dataset Generation**
    - [ ] Run the refined pipeline on the entire video dataset (`/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/`).
    - [ ] Monitor the process for errors or performance issues.
    - [ ] Finalize the complete dataset. 