# OCR Dataset Builder - Design Document

## 1. Goal

To create a dataset for training multimodal OCR and context understanding models by processing frames from YouTube videos using a Gemini large language model.

## 2. Data Source

- **Location:** `/mnt/nvme-fast0/datasets/pieces/pieces-ocr-v-0-1-0/`
- **Structure:** Contains subdirectories, each corresponding to a YouTube video.
- **Contents (per video):**
    - `*.info.json`: Metadata file (title, description, ID, duration, tags, categories, etc.).
    - `*.en.vtt` / `*.en-orig.vtt`: Subtitle files.
    - Associated video file (location assumed accessible, potentially referenced in `.info.json`).

## 3. Processing Pipeline (Per Video)

1.  **Frame Extraction:**
    - Extract frames from the video file at a rate of 1 frame per second.
    - Store extracted frames temporarily or with persistent, structured naming (e.g., `/path/to/frames/{video_id}/frame_{second}.jpg`).

2.  **Batching:**
    - Group extracted frames into batches of 60 consecutive frames (representing 1 minute of video content).

3.  **LLM Processing (Per Batch of 60 Frames):**
    - **Model:** Gemini 2.5 Pro (via Vertex AI).
    - **Input:**
        - 60 image frames.
        - An adapted version of the prompt defined in `ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md`.
    - **Prompt Adaptation:**
        - **Overall:** The prompt needs modification to accept a *batch* of 60 frames as input instead of a single desktop image.
        - **Task 1 (Raw OCR):** Run per frame. Extract all visible text from the frame.
        - **Task 2 (Augmented OCR):** Run per frame. Introduce realistic imperfections to the Task 1 output for that frame.
        - **Task 3 (Cleaned OCR):** Run per frame. Clean the Task 2 output for that frame.
        - **Task 4 (Structured Markdown):** Run per frame. **Adaptation Required:** Since the input is a raw video frame, not a desktop screenshot, redefine "Application". Instead, focus on identifying key content zones or elements *within the frame*.
            - Example Structure per Frame:
                ```markdown
                ### Frame Content Analysis: [Timestamp or Frame Index]
                #### Primary Subject: (e.g., Code Editor View, Presentation Slide, Talking Head, Gameplay Scene)
                #### Key Text Elements: (List significant text blocks identified)
                #### Visible UI Elements: (e.g., Buttons, Menus shown in a tutorial)
                #### Inferred Action/Topic: (Brief description of what's happening in this specific frame)
                ```
        - **Task 5 (Narrative Summary):** Run *once per batch*. Synthesize the information from the 60 processed frames (Tasks 1-4 outputs) and the frame sequence to generate a 1-3 paragraph narrative describing the overall activity or topic covered during that 1-minute segment.
    - **API Call:** Structure the Vertex AI API call to handle multimodal input (60 images + adapted text prompt).

4.  **Output Generation (Per Batch):**
    - Create a JSON object or similar data record containing:
        - `image_paths`: (list[str]) Paths to the 60 frame image files in the batch.
        - `task_1_outputs`: (list[str]) List of 60 raw OCR strings.
        - `task_2_outputs`: (list[str]) List of 60 augmented OCR strings.
        - `task_3_outputs`: (list[str]) List of 60 cleaned OCR strings.
        - `task_4_outputs`: (list[str]) List of 60 structured Markdown analysis strings (using the adapted Task 4 definition).
        - `task_5_output`: (str) Single narrative summary for the batch.
        - `youtube_video_id`: (str) From `.info.json`.
        - `metadata`: (dict) Dictionary containing selected small fields from `.info.json`: `title`, `duration`, `channel_id`, `channel_url`, `view_count`, `age_limit`, `tags`, `categories`, `webpage_url`.

5.  **Data Storage:**
    - Store the generated output records (e.g., as JSON lines in a file, entries in a database).

## 4. Implementation Details

- **Libraries:** `opencv-python` (frame extraction), `google-genai` / `google-generativeai` (Vertex AI client for Gemini), `yt-dlp` (video info/download), `fire` (CLI), `rich` (console output), `python-dotenv`.
- **Workflow Orchestration:** A Python script, potentially using `fire` for the CLI, to iterate through video directories, manage extraction, batching, LLM calls, and output saving.
- **Error Handling:** Implement retries for LLM calls, handle missing video files or corrupted data gracefully.

## 5. Future Considerations

- Explore different frame selection strategies (e.g., keyframes, subtitle timing).
- Experiment with different batch sizes.
- Refine Task 4 structure based on initial results.
- Add speaker diarization using subtitle timing if needed for Task 4 adaptation. 