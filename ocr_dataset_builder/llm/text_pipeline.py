import concurrent.futures
import json
import logging
import math
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.progress import Progress

# Import necessary functions from the text processing module
from ocr_dataset_builder.llm.text_processing import (
    DEFAULT_MODEL_NAME as DEFAULT_TEXT_MODEL_NAME, # Rename to avoid clash
    initialize_gemini_client,
    load_prompt,
    parse_text_llm_response,
    process_text_input,
)

# Import cost calculation utility from the new location
from ocr_dataset_builder.llm.utils.costing import calculate_gemini_cost

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)

# Load environment variables
load_dotenv()

# Default prompt path for text refinement
DEFAULT_TEXT_PROMPT_PATH = Path(
    "ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md"
)

# --- Helper Functions ---

def extract_frame_number(filename: str) -> int | None:
    """Extracts the frame number from filenames like 'frame_000123.jpg'"""
    match = re.search(r"frame_(\d+)\.", filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None

def _process_single_batch(
    batch_texts: list[tuple[int, str]], # List of (frame_index, raw_text)
    batch_start_index: int,
    batch_end_index: int,
    video_id: str,
    prompt_text: str,
    model_name: str,
    output_dir: Path,
    skip_cost_calculation: bool,
) -> dict | None:
    """Processes a single batch of text frames, calls LLM, saves output."""
    batch_log_prefix = f"[{video_id} Batch {batch_start_index}-{batch_end_index}]"
    logging.info(f"{batch_log_prefix} Processing {len(batch_texts)} frames.")

    # Ensure batch_texts is sorted by frame index for consistent processing
    sorted_batch_texts = sorted(batch_texts, key=lambda x: x[0])

    # 0. Extract raw texts for final output (must match order of Task 3/4)
    raw_ocr_texts_for_batch = [text for _, text in sorted_batch_texts]

    # 1. Format concatenated input text for LLM
    concatenated_input = ""
    for frame_index, text in sorted_batch_texts:
        concatenated_input += f"--- Frame {frame_index} ---\n{text}\n\n"
    concatenated_input = concatenated_input.strip()

    if not concatenated_input:
        logging.warning(f"{batch_log_prefix} Batch resulted in empty concatenated text. Skipping.")
        # Create a placeholder output indicating skip?
        # For now, just return None, let caller handle.
        return None

    # 2. Initialize Gemini Client (do this in the worker process)
    client = initialize_gemini_client()
    if not client:
        logging.error(f"{batch_log_prefix} Failed to initialize Gemini client in worker.")
        return None

    # 3. Process with LLM
    llm_start_time = time.time()
    raw_response, input_tokens, output_tokens = process_text_input(
        client=client,
        concatenated_batch_text=concatenated_input,
        prompt_text=prompt_text,
        model_name=model_name,
        # print_output=False, # Control via log level
        # print_counts=False,
    )
    llm_duration = time.time() - llm_start_time

    if raw_response is None:
        logging.error(f"{batch_log_prefix} LLM processing failed.")
        return None # Indicate failure

    # 4. Parse LLM Response
    parsed_llm_data = parse_text_llm_response(raw_response)
    if not parsed_llm_data:
        logging.error(f"{batch_log_prefix} Failed to parse LLM response.")
        # Save raw response for debugging?
        # For now, treat as failure
        return None

    # 5. Calculate Cost
    cost = 0.0
    if not skip_cost_calculation and input_tokens is not None and output_tokens is not None:
        try:
            cost = calculate_gemini_cost(model_name, input_tokens, output_tokens)
            logging.info(f"{batch_log_prefix} Estimated cost: ${cost:.6f}")
        except Exception as cost_e:
            logging.warning(f"{batch_log_prefix} Cost calculation failed: {cost_e}")
    elif not skip_cost_calculation:
        logging.warning(f"{batch_log_prefix} Token counts missing, cannot calculate cost.")

    # 6. Prepare Final Output Structure
    output_data = {
        "video_id": video_id,
        "batch_info": {
            "start_frame_index": batch_start_index,
            "end_frame_index": batch_end_index,
            "num_frames_in_batch": len(batch_texts),
        },
        "task1_raw_ocr_text": raw_ocr_texts_for_batch, # Add the raw OCR text list
        "llm_output": parsed_llm_data, # Contains task3_cleaned_text, task4_markdown_text, task5_summary
        "token_counts": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "processing_stats": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "llm_duration_seconds": round(llm_duration, 2),
            "estimated_cost_usd": round(cost, 6)
        },
    }

    # 7. Save Output JSON
    output_filename = f"batch_{batch_start_index:06d}_{batch_end_index:06d}.json"
    output_json_path = output_dir / video_id / output_filename
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logging.info(f"{batch_log_prefix} Successfully saved output to {output_json_path}")
        return output_data # Return the data for aggregation if needed
    except Exception as save_e:
        logging.error(f"{batch_log_prefix} Failed to save output JSON {output_json_path}: {save_e}")
        return None


def _process_tesseract_json(
    tesseract_dir_info: tuple[Path, Path, str], # input_root, output_root, relative_path
    prompt_text: str,
    model_name: str,
    frames_per_batch: int,
    skip_cost_calculation: bool,
    dry_run: bool,
) -> tuple[str, int, int, float]: # relative_path, successful_batches, failed_batches, total_cost
    """Worker function to process one video directory (one tesseract_ocr.json)."""
    input_root, output_root, relative_path = tesseract_dir_info
    input_dir_abs = input_root / relative_path
    tesseract_json_path = input_dir_abs / "tesseract_ocr.json"
    log_prefix = f"[{relative_path}]"

    successful_batches = 0
    failed_batches = 0
    total_cost = 0.0

    if dry_run:
        logging.info(f"{log_prefix} [DRY RUN] Would process {tesseract_json_path}")
        # Simulate finding some frames and batches
        num_simulated_frames = 100
        num_simulated_batches = math.ceil(num_simulated_frames / frames_per_batch)
        logging.info(f"{log_prefix} [DRY RUN] Simulating {num_simulated_frames} frames -> {num_simulated_batches} batches.")
        return relative_path, num_simulated_batches, 0, 0.0 # Simulate all batches successful

    if not tesseract_json_path.is_file():
        logging.error(f"{log_prefix} Tesseract JSON not found: {tesseract_json_path}")
        return relative_path, 0, 1, 0.0 # Treat missing JSON as failure

    try:
        with open(tesseract_json_path, "r", encoding="utf-8") as f:
            tesseract_data = json.load(f)
    except Exception as e:
        logging.error(f"{log_prefix} Failed to load/parse {tesseract_json_path}: {e}")
        return relative_path, 0, 1, 0.0

    # Extract frame texts and sort them by frame number
    frame_texts_unsorted = []
    for filename, text in tesseract_data.items():
        frame_num = extract_frame_number(filename)
        if frame_num is not None:
            frame_texts_unsorted.append((frame_num, text if text else "")) # Use empty string if OCR text is null/None
        else:
            logging.warning(f"{log_prefix} Could not extract frame number from '{filename}'. Skipping this frame.")

    if not frame_texts_unsorted:
        logging.warning(f"{log_prefix} No valid frame texts extracted from {tesseract_json_path}. Skipping directory.")
        return relative_path, 0, 0, 0.0 # No batches to fail

    # Sort frames by frame number
    sorted_frame_texts = sorted(frame_texts_unsorted, key=lambda x: x[0])
    num_frames = len(sorted_frame_texts)
    num_batches = math.ceil(num_frames / frames_per_batch)
    logging.info(f"{log_prefix} Loaded {num_frames} frames from JSON. Processing in {num_batches} batches.")

    # Process batches sequentially for now to avoid overwhelming API / client init issues
    for i in range(num_batches):
        batch_start_frame_num_in_list = i * frames_per_batch
        batch_end_frame_num_in_list = batch_start_frame_num_in_list + frames_per_batch
        current_batch_data = sorted_frame_texts[batch_start_frame_num_in_list:batch_end_frame_num_in_list]

        if not current_batch_data:
            logging.warning(f"{log_prefix} Batch {i+1}/{num_batches} is empty. Skipping.")
            continue

        # Get actual start/end frame indices from the batch data for naming/metadata
        actual_start_index = current_batch_data[0][0]
        actual_end_index = current_batch_data[-1][0]

        batch_result = _process_single_batch(
            batch_texts=current_batch_data,
            batch_start_index=actual_start_index,
            batch_end_index=actual_end_index,
            video_id=relative_path,
            prompt_text=prompt_text,
            model_name=model_name,
            output_dir=output_root,
            skip_cost_calculation=skip_cost_calculation,
        )

        if batch_result:
            successful_batches += 1
            total_cost += batch_result["processing_stats"].get("estimated_cost_usd", 0.0)
        else:
            failed_batches += 1
            logging.error(f"{log_prefix} Batch {i+1}/{num_batches} (Frames {actual_start_index}-{actual_end_index}) failed processing.")
            # Optional: Implement retry logic here?

    logging.info(f"{log_prefix} Finished processing. Success: {successful_batches}/{num_batches}, Failed: {failed_batches}/{num_batches}, Est Cost: ${total_cost:.6f}")
    return relative_path, successful_batches, failed_batches, total_cost


def run_text_llm_pipeline(
    input_dir: str,
    output_dir: str,
    prompt_path: str = str(DEFAULT_TEXT_PROMPT_PATH),
    frames_per_batch: int = 60,
    model_name: str = DEFAULT_TEXT_MODEL_NAME,
    max_workers_dirs: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None, # Slicing indices for directories
    checkpoint_file: str = "llm_text_pipeline_checkpoint.log",
    skip_cost_calculation: bool = False,
    dry_run: bool = False,
    log_level: str = "INFO",
):
    """
    Runs the LLM text refinement pipeline.

    Args:
        input_dir: Root directory containing Tesseract JSON outputs per video.
        output_dir: Root directory to save LLM refinement JSON outputs.
        prompt_path: Path to the text refinement LLM prompt file.
        frames_per_batch: Number of frames' text to batch per LLM call.
        model_name: Name of the Gemini model to use.
        max_workers_dirs: Max parallel workers for processing video directories.
                          Defaults to os.cpu_count() // 2.
        start_index: 0-based index of the first video directory to process (after checkpointing).
        end_index: 0-based index *after* the last directory to process (after checkpointing).
        checkpoint_file: Name/path for the checkpoint file tracking completed directories.
        skip_cost_calculation: If True, skips cost estimation.
        dry_run: If True, simulates the process without LLM calls or file writes.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # --- Setup Logging ---
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.getLogger().setLevel(numeric_level)
    logging.info(f"Log level set to {log_level.upper()}")

    # --- Path and Parameter Setup ---
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / checkpoint_file

    if not input_root.is_dir():
        logging.error(f"Input directory not found: {input_root}")
        return

    try:
        prompt_text = load_prompt(Path(prompt_path))
        logging.info(f"Loaded prompt from: {prompt_path}")
    except Exception as e:
        logging.error(f"Failed to load prompt: {e}. Exiting.", exc_info=True)
        return

    if not max_workers_dirs:
        max_workers_dirs = os.cpu_count() // 2 if os.cpu_count() else 1
        max_workers_dirs = max(1, max_workers_dirs) # Ensure at least 1
    logging.info(f"Using model: {model_name}")
    logging.info(f"Frames per batch: {frames_per_batch}")
    logging.info(f"Max directory workers: {max_workers_dirs}")
    logging.info(f"Input directory: {input_root}")
    logging.info(f"Output directory: {output_root}")
    logging.info(f"Checkpoint file: {checkpoint_path}")
    if dry_run:
        logging.warning("*** DRY RUN ACTIVATED *** No LLM calls or file writes will occur.")

    # --- Load Checkpoint --- # TODO: Use JSON for checkpoint?
    processed_dirs_from_checkpoint = set()
    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line in f:
                    processed_dirs_from_checkpoint.add(line.strip())
            logging.info(f"Loaded {len(processed_dirs_from_checkpoint)} processed directory paths from checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Could not read checkpoint file {checkpoint_path}: {e}. Processing all (or as per slice).", exc_info=True)
            processed_dirs_from_checkpoint.clear()
    else:
        logging.info(f"No checkpoint file found at {checkpoint_path}. Will process all relevant directories.")

    # --- Discover Input Directories --- #
    all_potential_dirs_abs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    all_potential_dirs_rel = [d.relative_to(input_root).as_posix() for d in all_potential_dirs_abs]
    logging.info(f"Found {len(all_potential_dirs_rel)} potential video directories in input.")

    # --- Filter Processed/Stale Dirs ---
    valid_processed_paths = {p for p in processed_dirs_from_checkpoint if (input_root / p).is_dir()}
    if len(valid_processed_paths) < len(processed_dirs_from_checkpoint):
        stale_count = len(processed_dirs_from_checkpoint) - len(valid_processed_paths)
        logging.info(f"{stale_count} stale paths (present in checkpoint but not found in input) removed from checkpoint list.")

    skipped_dirs_due_to_checkpoint = sorted([
        p for p in all_potential_dirs_rel if p in valid_processed_paths
    ])
    if skipped_dirs_due_to_checkpoint:
        logging.info(f"{len(skipped_dirs_due_to_checkpoint)} directories will be SKIPPED as they are already in the checkpoint:")
        # To avoid excessively long logs, show first few and then a count if many
        display_limit = 10
        for i, dir_path in enumerate(skipped_dirs_due_to_checkpoint):
            if i < display_limit:
                logging.info(f"  SKIPPED (checkpointed): {dir_path}")
            elif i == display_limit:
                logging.info(f"  ... and {len(skipped_dirs_due_to_checkpoint) - display_limit} more.")
                break
    else:
        logging.info("No directories were skipped due to checkpointing (either checkpoint is empty or all dirs are new).")

    relative_dirs_to_process = sorted([p for p in all_potential_dirs_rel if p not in valid_processed_paths])
    logging.info(f"{len(relative_dirs_to_process)} directories remain to be considered after checkpoint filtering.")

    # --- Apply Slicing --- #
    total_remaining = len(relative_dirs_to_process)
    actual_start = start_index if start_index is not None else 0
    actual_end = end_index if end_index is not None else total_remaining
    actual_start = max(0, actual_start)
    actual_end = min(total_remaining, actual_end)

    if actual_start >= actual_end:
        logging.info("No directories selected for processing based on current slice parameters and checkpointing.")
        return

    final_dirs_for_run = relative_dirs_to_process[actual_start:actual_end]
    num_dirs_to_process = len(final_dirs_for_run)

    if num_dirs_to_process > 0:
        logging.info(f"Targeting {num_dirs_to_process} directories for processing in this run (slice [{actual_start}:{actual_end}] of remaining dirs):")
        display_limit = 10
        for i, dir_path in enumerate(final_dirs_for_run):
            if i < display_limit:
                logging.info(f"  TO PROCESS: {dir_path}")
            elif i == display_limit:
                logging.info(f"  ... and {num_dirs_to_process - display_limit} more.")
                break
    else:
        logging.info("No directories targeted for processing in this run after slicing.")
        return

    # --- Prepare Data for Workers --- #
    worker_data = [
        (input_root, output_root, rel_path)
        for rel_path in final_dirs_for_run
    ]

    # --- Execute Pipeline --- #
    overall_successful_dirs = 0
    overall_failed_dirs = 0
    overall_total_cost = 0.0
    start_time_pipeline = time.time()

    with Progress() as progress_bar:
        dir_task = progress_bar.add_task("[cyan]Processing Directories...", total=num_dirs_to_process)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_dirs) as executor:
            futures = {
                executor.submit(
                    _process_tesseract_json,
                    dir_info,
                    prompt_text,
                    model_name,
                    frames_per_batch,
                    skip_cost_calculation,
                    dry_run,
                ): dir_info[2] # Map future to relative_path
                for dir_info in worker_data
            }

            for future in concurrent.futures.as_completed(futures):
                relative_path = futures[future]
                try:
                    returned_path, successful_batches, failed_batches, total_dir_cost = future.result()
                    
                    # Sanity check
                    if returned_path != relative_path:
                        logging.error(f"Path mismatch! Expected {relative_path}, got {returned_path}. Worker logic error.")
                        overall_failed_dirs += 1
                        continue # Don't checkpoint

                    overall_total_cost += total_dir_cost

                    # Consider a directory successful if all batches succeeded (or if there were no batches)
                    if failed_batches == 0:
                        overall_successful_dirs += 1
                        # Checkpoint successful directory completion
                        if not dry_run:
                            try:
                                with open(checkpoint_path, "a") as cp_f:
                                    cp_f.write(f"{returned_path}\\n")
                                logging.info(f"CHECKPOINTED: Successfully processed and recorded '{returned_path}'")
                            except Exception as cp_e:
                                logging.error(f"Failed to write to checkpoint file {checkpoint_path} for {returned_path}: {cp_e}")
                        else:
                            logging.info(f"[DRY RUN] Would have checkpointed: Successfully processed and recorded '{returned_path}'")
                    else:
                        overall_failed_dirs += 1
                        logging.warning(f"Directory {returned_path} finished with {failed_batches} failed batches. Not checkpointed.")

                except Exception as exc:
                    overall_failed_dirs += 1
                    logging.error(f"Directory {relative_path} generated an exception: {exc}", exc_info=True)
                    # traceback.print_exc() # Ensure traceback is logged
                
                progress_bar.update(dir_task, advance=1)

    # --- Final Summary --- #
    pipeline_duration = time.time() - start_time_pipeline
    logging.info("--- Text LLM Pipeline Finished ---")
    logging.info(f"Total directories targeted: {num_dirs_to_process}")
    logging.info(f"Successfully processed directories: {overall_successful_dirs}")
    logging.info(f"Failed directories: {overall_failed_dirs}")
    logging.info(f"Total estimated cost: ${overall_total_cost:.6f}")
    logging.info(f"Total pipeline duration: {pipeline_duration:.2f} seconds")
    logging.info(f"See checkpoint log at: {checkpoint_path}")


class TextLLMPipelineCLI:
    """CLI for the Text LLM Refinement Pipeline."""

    def run(
        self,
        input_dir: str, # Required: Input from tesseract
        output_dir: str, # Required: Output dir for refined JSONs
        prompt_path: str = str(DEFAULT_TEXT_PROMPT_PATH),
        frames_per_batch: int = 60,
        model_name: str = str(DEFAULT_TEXT_MODEL_NAME),
        max_workers_dirs: int | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        checkpoint_file: str = "llm_text_pipeline_checkpoint.log",
        skip_cost_calculation: bool = False,
        dry_run: bool = False,
        log_level: str = "INFO",
    ):
        """Runs the LLM text refinement pipeline on Tesseract JSON outputs.

        Args:
            input_dir: Root directory containing subdirectories of Tesseract OCR JSON outputs.
            output_dir: Root directory to save the refined LLM JSON outputs.
            prompt_path: Path to the text refinement LLM prompt file.
            frames_per_batch: Number of frames' text to batch per LLM call.
            model_name: Name of the Gemini model to use.
            max_workers_dirs: Max parallel workers for processing video directories.
                              Defaults to os.cpu_count() // 2.
            start_index: 0-based index of the first video directory to process (after checkpointing).
            end_index: 0-based index *after* the last directory to process (after checkpointing).
            checkpoint_file: Name/path for the checkpoint file tracking completed directories.
            skip_cost_calculation: If True, skips cost estimation.
            dry_run: If True, simulates the process without LLM calls or file writes.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        """
        run_text_llm_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            prompt_path=prompt_path,
            frames_per_batch=frames_per_batch,
            model_name=model_name,
            max_workers_dirs=max_workers_dirs,
            start_index=start_index,
            end_index=end_index,
            checkpoint_file=checkpoint_file,
            skip_cost_calculation=skip_cost_calculation,
            dry_run=dry_run,
            log_level=log_level,
        )

if __name__ == "__main__":
    fire.Fire(TextLLMPipelineCLI) 