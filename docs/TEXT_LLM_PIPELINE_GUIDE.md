# Text LLM Refinement Pipeline Guide

This document describes how to use the `llm_text_pipeline.py` script to process sequences of Tesseract OCR text outputs using a Large Language Model (LLM) like Gemini 2.5 Pro. The pipeline takes raw text extracted from video frames, sends it to an LLM for refinement (Task 3), markdown conversion (Task 4), and contextual summarization (Task 5).

## Overview

The `llm_text_pipeline.py` script is designed to:

1.  Scan an input directory for subdirectories. Each subdirectory is assumed to represent a video and contain individual `.txt` files of OCR text for its frames (e.g., `frame_0000.txt`, `frame_0001.txt`, ...).
2.  Process each video's frame texts in batches (e.g., 60 frames at a time).
3.  For each batch, concatenate the text from the frames.
4.  Use the `ocr_dataset_builder.llm_text_processing` module to send this concatenated text along with a specialized prompt (`ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md`) to the configured LLM.
5.  Parse the LLM's structured response to extract:
    *   Task 3: Cleaned and corrected text for each frame.
    *   Task 4: Markdown representation of the cleaned text for each frame.
    *   Task 5: A contextual summary based on all frames in the batch.
6.  Save the parsed output as a JSON file in the specified output directory, organized by video ID and batch.
7.  Maintain a checkpoint file to resume processing from where it left off.
8.  Calculate and log estimated costs based on token usage.

## Input Data Format

The pipeline expects the input data (`--input_dir`) to be structured as follows:

```
<input_dir>/
  <video_id_1>/
    frame_0000.txt
    frame_0001.txt
    frame_0002.txt
    ...
    frame_NNNN.txt
  <video_id_2>/
    frame_0000.txt
    frame_0001.txt
    ...
  ...
```

*   `<input_dir>`: The main directory containing processed video data.
*   `<video_id_1>`, `<video_id_2>`, etc.: Subdirectories, where each directory name is a unique identifier for a video. These are typically the names of the original video files (without extension) or some other unique ID.
*   `frame_XXXX.txt`: Plain text files containing the raw OCR output from Tesseract for each corresponding frame. The files should be named sequentially.

This structure is expected to be the output of a Tesseract OCR pipeline (e.g., `ocr_dataset_builder/tesseract/pipeline.py`).

## Output Data Format

For each processed batch of frames from a video, a JSON file will be created in the output directory (`--output_dir`). The structure will be:

```
<output_dir>/
  <video_id_1>/
    batch_000000_000059.json
    batch_000060_000119.json
    ...
  <video_id_2>/
    batch_000000_000059.json
    ...
```

Each JSON file (e.g., `batch_000000_000059.json`) will contain:

```json
{
  "video_id": "<video_id_1>",
  "batch_info": {
    "start_frame_index": 0,
    "end_frame_index": 59,
    "num_frames_in_batch": 60
  },
  "llm_output": {
    "task3_cleaned_text": [
      "Cleaned text for frame 0...",
      "Cleaned text for frame 1...",
      ...
    ],
    "task4_markdown_text": [
      "Markdown for frame 0...",
      "Markdown for frame 1...",
      ...
    ],
    "task5_summary": "Contextual summary for frames 0-59..."
  },
  "token_counts": {
    "input_tokens": <count>,
    "output_tokens": <count>
  },
  "processing_stats": {
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
    "duration_seconds": <seconds>
  }
}
```

## CLI Usage

The script is run using `python -m ocr_dataset_builder.llm.llm_text_pipeline run ...`.

### Arguments

*   `--input_dir` (str, required): Path to the directory containing subdirectories of Tesseract OCR `.txt` files (see Input Data Format).
*   `--output_dir` (str, required): Path to the directory where LLM output JSON files will be saved.
*   `--prompt_path` (str, optional): Path to the LLM prompt file. Defaults to `ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md`.
*   `--frames_per_batch` (int, optional): Number of frames to process in a single call to the LLM. Defaults to `60`.
*   `--model_name` (str, optional): Name of the Gemini model to use (e.g., `gemini-1.5-pro-latest`, `gemini-2.5-pro-preview-03-25`). Defaults to the value of `LLM_MODEL_NAME` environment variable or `gemini-2.5-pro-preview-03-25`.
*   `--max_input_dirs` (int, optional): Maximum number of input video directories to process. Useful for testing. Defaults to `None` (process all).
*   `--max_workers_dirs` (int, optional): Maximum number of worker processes for processing video directories in parallel. Defaults to `cpu_count() // 2`.
*   `--log_level` (str, optional): Logging level (e.g., `DEBUG`, `INFO`, `WARNING`). Defaults to `INFO`.
*   `--checkpoint_file` (str, optional): Path to the checkpoint file. Defaults to `llm_text_pipeline_checkpoint.json` in the output directory.
*   `--skip_cost_calculation` (bool, optional): If set, skips cost calculation. Defaults to `False`.
*   `--dry_run` (bool, optional): If set, simulates processing without making LLM calls or writing files. Useful for testing directory scanning and batching logic. Defaults to `False`.

### Example Command

```bash
python -m ocr_dataset_builder.llm.llm_text_pipeline run \
    --input_dir "/path/to/tesseract_output_texts/" \
    --output_dir "/path/to/llm_text_output/" \
    --frames_per_batch 60 \
    --model_name "gemini-2.5-pro-preview-03-25" \
    --max_workers_dirs 8
```

## Environment Variables

*   `GOOGLE_API_KEY`: Your Google API key for Gemini.
*   `VERTEX_AI_PROJECT_ID`: Your Google Cloud Project ID for Vertex AI (if using Vertex AI backend).
*   `VERTEX_AI_LOCATION`: Your Google Cloud Project Location (e.g., `us-central1`) for Vertex AI.
*   `LLM_MODEL_NAME`: Can be set to specify a default model if `--model_name` is not provided.

## Important Considerations

*   **API Costs and Quotas:** Processing large datasets can incur significant costs and hit API rate limits. Monitor your usage and consider using `--max_input_dirs` for initial runs.
*   **Tesseract Output Quality:** The quality of the LLM refinement depends heavily on the quality of the input Tesseract OCR text. Ensure your Tesseract pipeline is optimized.
*   **Prompt Engineering:** The default prompt (`ocr_text_refinement_prompt.md`) is a starting point. You may need to adapt it for specific types of content or desired output nuances.
*   **Error Handling:** The pipeline includes basic error handling, but for large-scale processing, robust monitoring and retries might be necessary. 