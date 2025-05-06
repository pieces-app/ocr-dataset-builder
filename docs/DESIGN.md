# Design Document: OCR Training Dataset Builder

## 1. Introduction

This document outlines the design and architecture of the OCR Training Dataset Builder. The primary goal is to process YouTube video content (frames and metadata) to generate a high-quality dataset for training multi-modal OCR and visual understanding models, leveraging a sophisticated LLM prompt for analysis.

## 2. High-Level Architecture

The system is envisioned to operate in several key stages:

1.  **Dataset Ingestion**: The system takes a path to a dataset typically containing subdirectories for each video. Each subdirectory is expected to hold video files and metadata (e.g., `.info.json`, `.vtt` subtitles).
2.  **Frame Extraction & Processing**: Videos are processed to extract relevant frames. This stage involves:
    *   Identifying video files.
    *   Extracting frames at a specified rate (e.g., 1 FPS).
    *   Optionally resizing frames to a maximum dimension.
    *   Optionally sampling a maximum number of frames per video.
    *   Saving processed frames to a structured output directory, mirroring the input structure.
    *   Copying associated metadata files to the output directory.
3.  **LLM-based Analysis (Future Milestone)**:
    *   Sequences of extracted frames will be sent to a multi-modal LLM (e.g., Gemini via Vertex AI).
    *   The LLM will use the `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md` to perform detailed analysis on each frame and summarize the sequence.
4.  **Output Generation**: The structured output from the LLM will be saved, likely in a JSON Lines format, associating the analysis with the source frames and video.

## 3. Core Modules and Components

### 3.1. Frame Extraction (`ocr_dataset_builder/video_processing.py`)

*   **Purpose**: To efficiently extract and prepare frames from individual video files.
*   **Key Function**: `extract_frames()`
    *   **Input**: Video file path, output directory, target FPS, max dimension for resizing, max frames to sample.
    *   **Process**:
        1.  Opens video using OpenCV (`cv2.VideoCapture`).
        2.  Calculates frame interval based on native FPS and target FPS.
        3.  Iterates through video, extracting frames at the calculated interval.
        4.  If `max_dimension` is set, resizes frames maintaining aspect ratio.
        5.  Collects all candidate frames (path and image data).
        6.  If `max_frames_per_video` is set and candidate count exceeds it, performs random sampling.
        7.  Saves selected frames to the output directory with names like `frame_XXXXXX.jpg` (XXXXXX = second mark).
    *   **Output**: List of paths to saved frames.
    *   **Error Handling**: Logs errors and aims to be robust (e.g., skips problematic frames/videos).
*   **Utilities**: `get_human_readable_size()` for logging.

### 3.2. Pipeline Orchestration (e.g., `run_pipeline.py` - to be developed/reinstated)

*   **Purpose**: To manage the processing of an entire dataset of videos.
*   **Responsibilities**:
    *   Traverse the input dataset directory structure.
    *   For each video subdirectory:
        *   Locate the video file (handling various extensions).
        *   Locate the metadata file (e.g., `.info.json`).
        *   Create a corresponding output subdirectory.
        *   Copy the metadata file to the output subdirectory.
        *   Invoke `extract_frames()` from `video_processing.py` for the video.
    *   **Parallel Processing**: Utilize `concurrent.futures.ProcessPoolExecutor` to process multiple video directories in parallel, improving throughput.
    *   **Dataset Slicing**: Allow processing of a subset of videos using `start_index` and `end_index` parameters.
    *   **CLI Interface**: Provide a command-line interface (e.g., using `fire`) for easy execution and parameterization.
    *   **Progress Reporting**: Outer `tqdm` progress bar for directories, inner for frames (handled by `extract_frames`).

### 3.3. LLM Prompt (`ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`)

*   **Purpose**: To guide the LLM in analyzing frame sequences and producing structured data.
*   **Design for Frame Sequences**:
    *   **Input**: A sequence of N frames from a single video, presented sequentially.
    *   **Task Adaptation**:
        *   Tasks 1-4 (Raw OCR, Augmented OCR, Cleaned OCR, Structured Markdown) are applied **Per Frame**.
        *   Task 5 (Narrative Summary) is applied **Per Sequence**, summarizing the activity across all N frames.
    *   **Speaker Attribution**: Retains detailed rules for speaker identification and dialogue attribution, to be applied based on content visible within frames.
    *   **Output Structure**: Defined with clear delimiters (`-- Frame X --`) for per-frame outputs, followed by a single block for the per-sequence summary. This facilitates parsing.
    *   **Few-Shot Examples**: Adapted to reflect the per-frame/per-sequence output format, using conceptual translations of desktop screenshots to video frame content.

### 3.4. LLM Interaction Module (e.g., `llm_processing.py` - Future Milestone)

*   **Purpose**: To handle communication with the LLM API (e.g., Google Vertex AI for Gemini).
*   **Responsibilities**:
    *   Take a sequence of frame image data (and potentially paths or other metadata).
    *   Load and format the multi-task prompt.
    *   Construct the API request with the prompt and image data.
    *   Handle API calls, including authentication (e.g., using `GOOGLE_API_KEY` from `.env`).
    *   Manage API responses, including retries or error handling.
    *   Parse the LLM's structured text output into a machine-readable format (e.g., Python dictionaries).

## 4. Data Flow

1.  User specifies `dataset_path`, `output_path`, and other processing parameters (FPS, workers, etc.) to the pipeline script.
2.  The pipeline script scans `dataset_path` for video subdirectories.
3.  For each video directory processed (in parallel):
    a.  Video file and `.info.json` are identified.
    b.  An output subdirectory is created under `output_path`.
    c.  `.info.json` is copied to the output subdirectory.
    d.  `extract_frames()` is called with the video path and its corresponding output subdirectory.
    e.  `extract_frames()` reads the video, extracts, resizes, samples, and saves frames as `frame_XXXXXX.jpg` in the designated output folder.
4.  (Future) The LLM processing module will:
    a.  Read sequences of frames from the `output_path`.
    b.  Send frame data and the prompt to the LLM.
    c.  Receive structured text output.
    d.  Parse and save this output, linking it back to the source video/frames.

## 5. Key Design Decisions & Considerations

*   **Modularity**: Separating frame extraction, pipeline orchestration, and LLM interaction into distinct modules enhances maintainability and testability.
*   **Parallelism**: Processing videos in parallel is crucial for handling large datasets efficiently.
*   **Configuration**: Using command-line arguments for key parameters allows flexibility.
*   **Robustness**: Logging and error handling at various stages help in diagnosing issues without catastrophic failure.
*   **Mirrored Output Structure**: Replicating the input dataset structure in the output directory for frames and metadata simplifies data management and association.
*   **Prompt Adaptability**: The core LLM prompt, originally for desktop screenshots, was carefully adapted to handle sequences of video frames, maintaining its analytical depth while accommodating the new input modality.
*   **Environment Management**: Using Conda and Poetry (`pyproject.toml`, `install-conda-env.sh`) ensures a reproducible and isolated environment.

## 6. Future Enhancements

*   Integration of subtitle data (`.vtt` files) as additional context for the LLM.
*   More sophisticated frame selection logic (e.g., scene change detection, content-based filtering) beyond simple FPS and random sampling.
*   Support for various output formats for the LLM data.
*   Containerization with Docker for easier deployment. 