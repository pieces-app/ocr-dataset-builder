# Project Milestones - OCR Dataset Builder

This document outlines the key milestones for the OCR Dataset Builder project.

- [x] **Milestone 1: Setup & Configuration**
    - [x] Initialize project structure (`pyproject.toml`, `README.md`, `docs/`, etc.).
    - [x] Set up virtual environment and install dependencies.
    - [x] Configure API key handling (e.g., using `.env`).
    - [x] Verify Vertex AI client setup (`ocr_dataset_builder/examples/example-vertex.py`).

- [x] **Milestone 2: Frame Extraction Pipeline**
    - [x] Implement `extract_frames` function in `ocr_dataset_builder/video_processing.py`.
    - [x] Develop `ocr_dataset_builder/frame_pipeline.py` for batch processing video directories (parallel processing, metadata copying, CLI with `fire`, progress bars).
    - [x] Test frame extraction pipeline on sample videos.

- [🚧] **Milestone 3: LLM Prompt Adaptation & Core LLM Interaction** (Partially In Progress)
    - [x] Adapt `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` for video frame sequences (per-frame Tasks 1-4, per-sequence Task 5, revised few-shot examples).
    - [🚧] Develop/finalize `ocr_dataset_builder/llm_processing.py` for robust interaction with the Gemini API for a single sequence of frames (API calls, response parsing, error handling for single call).

- [ ] **Milestone 4: LLM Analysis Pipeline Development**
    - [ ] Develop `ocr_dataset_builder/llm_pipeline.py` to:
        - [ ] Ingest frame sequences (output from `frame_pipeline.py`).
        - [ ] Manage batching of sequences for `llm_processing.py`.
        - [ ] Orchestrate calls to `llm_processing.py` for each sequence.
        - [ ] Aggregate LLM results.
        - [ ] Implement CLI, progress tracking, and overall error management.
    - [ ] Define and implement final structured output format (e.g., JSONL) for LLM analysis results.

- [ ] **Milestone 5: Tesseract OCR Pipeline (Optional Baseline)**
    - [ ] Develop/finalize `ocr_dataset_builder/tesseract_processing.py` for Tesseract OCR on single frames.
    - [ ] Develop/finalize `ocr_dataset_builder/tesseract_pipeline.py` to apply OCR across datasets of frames, including CLI and output generation (e.g., per-frame text files or structured JSON).
    - [ ] Determine its exact role (e.g., baseline, input to LLM) and integration point.

- [ ] **Milestone 6: Full System Integration & Initial Testing**
    - [ ] Ensure smooth data flow: `frame_pipeline.py` -> (optional `tesseract_pipeline.py`) -> `llm_pipeline.py`.
    - [ ] Run the fully integrated system on a small, diverse subset of videos.
    - [ ] Verify the end-to-end process, data integrity, and output formats.
    - [ ] Perform basic quality checks on the final LLM outputs.

- [ ] **Milestone 7: Evaluation & Refinement**
    - [ ] Analyze results from the initial integrated test run.
    - [ ] Identify areas for improvement (prompt tuning, pipeline efficiency, error handling, data reconstruction from LLM output, etc.).
    - [ ] Refine individual pipelines and the overall system based on findings.

- [ ] **Milestone 8: Full Dataset Generation**
    - [ ] Run the refined system on a larger portion or the entire video dataset.
    - [ ] Monitor for performance, cost, and potential issues at scale.
    - [ ] Finalize the generated dataset.

- [ ] **Milestone 9: Documentation & Cleanup** (Expanded)
    - [x] Update `README.md`, `docs/DESIGN.md`, `docs/MILESTONES.md` (initial pass complete, further updates as needed).
    - [ ] Create `docs/PIPELINE_GUIDE.md` with detailed usage for each pipeline.
    - [ ] Create `docs/DATA_FORMATS.md` detailing all intermediate and final data structures.
    - [ ] Create `docs/PROMPT_ENGINEERING_GUIDE.md` (if deeper explanation than in `DESIGN.md` is needed).
    - [ ] Create `docs/TROUBLESHOOTING.md`.
    - [ ] Create `docs/CONTRIBUTING.md` (if applicable).
    - [ ] Final code cleanup, comments, and ensure project is well-organized for handoff or future work. 