import concurrent.futures
import json
import logging
import math
import os
import sys
import time  # Added for overall pipeline timing
from pathlib import Path

import fire
from dotenv import load_dotenv
from google.genai.types import CountTokensResponse
from rich.logging import RichHandler
from tqdm import tqdm

# from rich import print # Removed unused import


# Import necessary functions from the other modules
from .processing import (
    initialize_gemini_client,
    load_prompt,
    parse_llm_response,
    process_image_sequence,
)

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # RichHandler handles its own formatting
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)
# logging.getLogger().setLevel(logging.DEBUG) # Uncomment for cost debug logs


# Load environment variables
load_dotenv()

# Default prompt path (can be overridden via args if needed later)
DEFAULT_PROMPT_PATH = Path(
    "ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md"
)  # Wrapped long line
# Read default model name from env
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-pro-preview-03-25") # Updated default

# --- Cost Calculation Function ---

# Rates for Gemini 1.5 Pro (USD per 1 million tokens)
# Verify these rates with official Google Cloud documentation.
GEMINI_1_5_PRO_RATES_CONFIG = {
    "<128k": {"input": 3.50, "output": 10.50},
    ">=128k": {"input": 7.00, "output": 21.00},
    "threshold_k": 128,
}

MODEL_PRICING = {
    "gemini-1.5-pro-latest": GEMINI_1_5_PRO_RATES_CONFIG,
    "gemini-1.5-pro-001": GEMINI_1_5_PRO_RATES_CONFIG, # Explicit alias
    "gemini-2.5-pro-preview-03-25": GEMINI_1_5_PRO_RATES_CONFIG, # User-mentioned alias, assuming same rates
    # Add other 1.5 Pro variants if they share these rates
}

def calculate_gemini_cost(
    model_name: str,
    input_tokens_data: int | CountTokensResponse,
    output_tokens_data: int | CountTokensResponse,
) -> float:
    """
    Calculates the estimated cost for a Gemini 1.5 Pro API call based on token counts.

    Args:
        model_name: The name of the Gemini 1.5 Pro model variant used.
        input_tokens_data: The number of input tokens or CountTokensResponse.
        output_tokens_data: The number of output tokens or CountTokensResponse.

    Returns:
        The estimated cost in USD, or 0.0 if pricing is not defined for the model.
    """
    pricing_config = MODEL_PRICING.get(model_name)

    if not pricing_config:
        # Try a base name if a specific version like -001 isn't found
        base_model_name = "gemini-1.5-pro-latest" if "1.5-pro" in model_name else None
        if base_model_name:
            pricing_config = MODEL_PRICING.get(base_model_name)
        
        if not pricing_config:
            logging.warning(
                f"Pricing config not found for model '{model_name}'. Cost calculation will be 0."
            )
            return 0.0

    input_tokens = 0
    if isinstance(input_tokens_data, CountTokensResponse):
        input_tokens = input_tokens_data.total_tokens
    elif isinstance(input_tokens_data, int):
        input_tokens = input_tokens_data

    output_tokens = 0
    if isinstance(output_tokens_data, CountTokensResponse):
        output_tokens = output_tokens_data.total_tokens
    elif isinstance(output_tokens_data, int):
        output_tokens = output_tokens_data
    
    threshold_k = pricing_config.get("threshold_k", 128)
    threshold_tokens = threshold_k * 1000
    
    rates_key = "<128k" if input_tokens < threshold_tokens else ">=128k"
    specific_rates = pricing_config.get(rates_key)
    
    if not specific_rates or not isinstance(specific_rates, dict):
        logging.warning(
            f"Tiered rates for '{rates_key}' not found for model '{model_name}'. Cost calculation will be 0."
        )
        return 0.0

    input_rate = specific_rates.get("input", 0.0)  # Per 1M tokens
    output_rate = specific_rates.get("output", 0.0) # Per 1M tokens

    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    logging.debug(
        f"Cost calc ({model_name}, Tier: {rates_key}): "
        f"In:{input_tokens}tk @ ${input_rate:.2f}/M = ${input_cost:.6f}, "
        f"Out:{output_tokens}tk @ ${output_rate:.2f}/M = ${output_cost:.6f}, "
        f"Total=${total_cost:.6f}"
    )
    return total_cost


# --- Worker Function ---


def _process_frame_batch(
    batch_frame_paths: list[Path],
    output_json_path: Path,
    prompt_text: str,
    batch_index: int,
    total_batches_for_dir: int,
    model_name: str,
    video_dir_relative_path: str,
) -> tuple[str, str, float | None, int | None]:
    """
    Processes a single batch of frames using the LLM and calculates cost.

    Args:
        batch_frame_paths: List of paths for the frames in this batch.
        output_json_path: Path to save the resulting JSON output.
        prompt_text: The loaded LLM prompt text.
        batch_index: The index of this batch for logging.
        total_batches_for_dir: Total number of batches for this video dir for logging.
        model_name: The name of the Gemini model to use.
        video_dir_relative_path: Relative path to the video directory for logging.

    Returns:
        Tuple: (status, message, cost | None, frame_count | None)
    """
    status = "Error"
    message = "Initialization failure"
    cost = None
    num_frames = len(batch_frame_paths)
    batch_repr = f"{video_dir_relative_path} Batch {batch_index + 1}/{total_batches_for_dir}"
    logging.info(f"[{batch_repr}] Starting processing {num_frames} frames...")
    logging.info(f"[{batch_repr}] Using model: {model_name}")

    gemini_client = initialize_gemini_client()
    if not gemini_client:
        message = f"[{batch_repr}] Failed to initialize Gemini client."
        logging.error(message)
        return status, message, cost, None

    try:
        result_tuple = process_image_sequence(
            client=gemini_client,
            image_paths=batch_frame_paths,
            prompt_text=prompt_text,
            model_name=model_name, # Pass model_name to process_image_sequence
        )

        raw_response, input_tokens_data, output_tokens_data = None, None, None # Renamed for clarity
        if result_tuple:
            raw_response, input_tokens_data, output_tokens_data = result_tuple
            if raw_response is None:
                # This implies process_image_sequence failed and returned (None, X, Y)
                # The actual API error with traceback should have been logged by process_image_sequence
                message = f"[{batch_repr}] LLM API call failed (raw_response is None). Check preceding logs for 'Error during API call via Client' which includes the full traceback."
                logging.error(message)
                # Token data will be unreliable or missing
                logging.warning(f"[{batch_repr}] Token data will be missing or unreliable due to API call failure.")
                return status, message, None, num_frames # status is already "Error"
        else:
            # This case should ideally not be hit if process_image_sequence always returns a 3-tuple.
            message = f"[{batch_repr}] Failed to get a result tuple from process_image_sequence. This is unexpected. Check process_image_sequence implementation."
            logging.error(message)
            return status, message, cost, num_frames # status is "Error"

        if input_tokens_data is not None and output_tokens_data is not None:
            cost = calculate_gemini_cost(
                model_name, input_tokens_data, output_tokens_data
            )
            logging.info(f"[{batch_repr}] Estimated cost: ${cost:.4f}")
        else:
            logging.warning(
                f"[{batch_repr}] Tokens/char count data missing, cannot calc cost."
            )

        parsed_data = parse_llm_response(raw_response)
        if not parsed_data:
            # If raw_response was None, parse_llm_response would log "Cannot parse empty response text."
            message = f"[{batch_repr}] Failed parsing LLM response. This often occurs if the API call failed (see previous logs) or returned an empty/malformed response."
            logging.error(message)
            return status, message, cost, num_frames

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=4, ensure_ascii=False)

        status = "Success"
        message = (
            f"[{batch_repr}] Processed and saved to {output_json_path.name}"
        )
        logging.info(message)
        return status, message, cost, num_frames

    except Exception as e:
        message = f"[{batch_repr}] Unexpected error: {e}"
        logging.exception(message)  # Use .exception to log traceback
        return status, message, cost, num_frames


def _process_video_directory_llm(
    frame_dir_absolute_path: Path,
    frame_dir_relative_path: str,
    output_root: Path,
    prompt_text: str,
    batch_size: int,
    model_name: str,
    max_workers_for_batches: int | None,
) -> tuple[str, int, int, float, int]: # dir_relative_path, total_batches, successful_batches, total_cost, total_frames_in_dir
    """
    Processes all frame batches for a single video directory.
    Returns stats for this directory.
    """
    logging.info(
        f"Starting LLM processing for directory: {frame_dir_relative_path} using model {model_name}"
    )
    all_frames = sorted(frame_dir_absolute_path.glob("frame_*.png"))
    if not all_frames:
        all_frames = sorted(frame_dir_absolute_path.glob("frame_*.jpg"))

    if not all_frames:
        logging.warning(
            f"No frames found in {frame_dir_relative_path}. Skipping."
        )
        return (
            frame_dir_relative_path,
            0, # total_batches
            0, # successful_batches
            0.0, # total_cost
            0, # total_frames_in_dir
        )

    total_frames_in_dir = len(all_frames)
    num_batches = math.ceil(total_frames_in_dir / batch_size)
    logging.info(
        f"Directory {frame_dir_relative_path}: {total_frames_in_dir} frames, {num_batches} batches of size {batch_size}."
    )

    video_output_dir = output_root / frame_dir_relative_path
    video_output_dir.mkdir(parents=True, exist_ok=True)

    batch_futures = []
    successful_batches_count = 0
    dir_total_cost = 0.0
    # Ensure max_workers_for_batches is at least 1 if not None
    current_max_batch_workers = max_workers_for_batches
    if current_max_batch_workers is not None and current_max_batch_workers < 1:
        current_max_batch_workers = 1
    elif current_max_batch_workers is None: # if None, use a sensible default or make it sequential
        current_max_batch_workers = 1 # Defaulting to sequential if not specified to avoid overwhelming API

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=current_max_batch_workers 
    ) as batch_executor:
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            current_batch_frames = all_frames[batch_start:batch_end]
            output_file_name = f"llm_output_batch_{i:04d}.json"
            output_json_path = video_output_dir / output_file_name

            future = batch_executor.submit(
                _process_frame_batch,
                current_batch_frames,
                output_json_path,
                prompt_text,
                i, # batch_index
                num_batches, # total_batches_for_dir
                model_name,
                frame_dir_relative_path,
            )
            batch_futures.append(future)
        
        batch_progress_desc = f"{frame_dir_relative_path[:20]} Batches"
        for future in tqdm(
            concurrent.futures.as_completed(batch_futures),
            total=num_batches,
            desc=batch_progress_desc,
            leave=False,
            ncols=100
        ):
            try:
                status, msg, cost, _ = future.result()
                if status == "Success":
                    successful_batches_count += 1
                    if cost is not None:
                        dir_total_cost += cost
                else:
                    logging.error(f"Batch failed for {frame_dir_relative_path}: {msg}")
            except Exception as e:
                logging.error(f"Error processing a batch future for {frame_dir_relative_path}: {e}", exc_info=True)

    logging.info(
        f"Finished directory {frame_dir_relative_path}: {successful_batches_count}/{num_batches} batches successful. Total cost: ${dir_total_cost:.4f}"
    )
    return (
        frame_dir_relative_path,
        num_batches,
        successful_batches_count,
        dir_total_cost,
        total_frames_in_dir,
    )


def _perform_llm_processing_pass(
    input_root: Path,
    output_root: Path,
    prompt_text: str,
    batch_size: int,
    model_name: str,
    max_workers_dirs: int | None,
    max_workers_batches_per_dir: int | None,
    effective_start_index: int | None,
    effective_end_index: int | None,
    checkpoint_file_name: str,
) -> tuple[int, int, int, float, int]: # dirs_attempted, dirs_successful, dirs_failed, total_cost_pass, total_frames_pass
    """
    Performs a single pass of discovering and processing video directories with LLM.
    """
    checkpoint_path = output_root / checkpoint_file_name
    processed_dirs_from_checkpoint = set()
    malformed_checkpoint_entries = 0
    max_path_len_checkpoint = 1024 # Arbitrary reasonable limit for a single path

    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line_num, line in enumerate(f):
                    path_str = line.strip()
                    # Add check for malformed/long lines
                    if not path_str: # Skip empty lines
                        continue
                    if '\\n' in path_str or '\r' in path_str: # Check for embedded newlines
                        logging.warning(
                            f"Skipping malformed entry in checkpoint {checkpoint_path} (line ~{line_num+1}): Contains newline characters."
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    if len(path_str) > max_path_len_checkpoint:
                        logging.warning(
                            f"Skipping malformed entry in checkpoint {checkpoint_path} (line ~{line_num+1}): Exceeds max length {max_path_len_checkpoint}. Entry: \"{path_str[:80]}...\""
                        )
                        malformed_checkpoint_entries += 1
                        continue
                    
                    processed_dirs_from_checkpoint.add(path_str)

            logging.info(
                f"Loaded {len(processed_dirs_from_checkpoint)} valid processed dirs from checkpoint: {checkpoint_path}"
            )
            if malformed_checkpoint_entries > 0:
                 logging.warning(f"Skipped {malformed_checkpoint_entries} potentially malformed entries during checkpoint load.")

        except Exception as e:
            logging.error(f"Could not read LLM checkpoint {checkpoint_path}: {e}")
            processed_dirs_from_checkpoint.clear()
            malformed_checkpoint_entries = 0 # Reset counter on read error

    all_potential_input_dirs_abs = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    all_potential_input_dirs_rel = [
        d.relative_to(input_root).as_posix()
        for d in all_potential_input_dirs_abs
    ]
    logging.info(
        f"Found {len(all_potential_input_dirs_rel)} potential input directories in {input_root}."
    )

    valid_processed_relative_paths = {
        path_str
        for path_str in processed_dirs_from_checkpoint
        if (input_root / path_str).is_dir()
    }
    if len(valid_processed_relative_paths) < len(processed_dirs_from_checkpoint):
        logging.info(
            f"{len(processed_dirs_from_checkpoint) - len(valid_processed_relative_paths)} stale LLM dir paths removed from checkpoint."
        )

    relative_dirs_to_consider = sorted([
        rel_path
        for rel_path in all_potential_input_dirs_rel
        if rel_path not in valid_processed_relative_paths
    ])
    logging.info(
        f"{len(relative_dirs_to_consider)} input directories to consider for LLM processing after checkpoint."
    )

    total_dirs_to_consider = len(relative_dirs_to_consider)
    actual_start_index = effective_start_index if effective_start_index is not None else 0
    actual_end_index = (
        effective_end_index if effective_end_index is not None else total_dirs_to_consider
    )
    actual_start_index = max(0, actual_start_index)
    actual_end_index = min(total_dirs_to_consider, actual_end_index)

    if actual_start_index >= actual_end_index:
        logging.info("No new input directories for LLM processing in this pass.")
        return 0, 0, 0, 0.0, 0

    sliced_relative_dirs_for_this_run = relative_dirs_to_consider[
        actual_start_index:actual_end_index
    ]
    absolute_dirs_for_this_run_map = {
        rel_path: (input_root / rel_path)
        for rel_path in sliced_relative_dirs_for_this_run
    }
    target_dir_count_this_pass = len(sliced_relative_dirs_for_this_run)

    if target_dir_count_this_pass == 0:
        logging.info("No input directories selected for LLM processing this pass.")
        return 0, 0, 0, 0.0, 0

    logging.info(
        f"Targeting {target_dir_count_this_pass} input directories for LLM processing this pass (slice {actual_start_index} to {actual_end_index-1})."
    )

    pass_dirs_attempted = target_dir_count_this_pass
    pass_dirs_successful = 0
    pass_dirs_failed = 0
    pass_total_cost = 0.0
    pass_total_frames_processed = 0
    
    # Ensure max_workers_dirs is at least 1 if not None
    current_max_dir_workers = max_workers_dirs
    if current_max_dir_workers is not None and current_max_dir_workers < 1:
        current_max_dir_workers = 1
    elif current_max_dir_workers is None:
        current_max_dir_workers = 1 # Defaulting to sequential for dirs if not specified

    dir_futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=current_max_dir_workers) as dir_executor:
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = absolute_dirs_for_this_run_map[rel_path]
            future = dir_executor.submit(
                _process_video_directory_llm,
                abs_path,
                rel_path,
                output_root,
                prompt_text,
                batch_size,
                model_name,
                max_workers_batches_per_dir,
            )
            dir_futures.append(future)

        dir_progress_desc = "Processing Dirs (LLM Pass)"
        for future in tqdm(
            concurrent.futures.as_completed(dir_futures),
            total=target_dir_count_this_pass,
            desc=dir_progress_desc,
            ncols=100
        ):
            try:
                (
                    returned_relative_path,
                    total_batches_in_dir,
                    successful_batches_in_dir,
                    dir_cost,
                    frames_in_dir,
                ) = future.result()
                
                pass_total_cost += dir_cost
                pass_total_frames_processed += frames_in_dir # All frames in dir were targeted for batching

                # A directory is considered successful if all its batches were successful.
                # Or, if it had no batches (e.g. no frames) but didn't error.
                if total_batches_in_dir == 0 and successful_batches_in_dir == 0: # No frames, not an error
                    pass_dirs_successful += 1 
                    # Checkpoint empty/skipped dirs to avoid reprocessing them
                    try:
                        with open(checkpoint_path, "a") as cp_f:
                            cp_f.write(f"{returned_relative_path}\\n")
                        logging.debug(f"LLM Checkpointed empty/skipped dir: {returned_relative_path}")
                    except Exception as cp_e:
                        logging.error(f"Failed to checkpoint empty LLM dir {returned_relative_path}: {cp_e}")
                elif successful_batches_in_dir == total_batches_in_dir and total_batches_in_dir > 0:
                    pass_dirs_successful += 1
                    try:
                        with open(checkpoint_path, "a") as cp_f:
                            cp_f.write(f"{returned_relative_path}\\n")
                        logging.debug(f"LLM Checkpointed successful dir: {returned_relative_path}")
                    except Exception as cp_e:
                        logging.error(f"Failed to checkpoint successful LLM dir {returned_relative_path}: {cp_e}")
                else: # Partial success or full failure of batches within the directory
                    pass_dirs_failed += 1
                    logging.warning(
                        f"LLM Dir {returned_relative_path} had {successful_batches_in_dir}/{total_batches_in_dir} successful batches. Not fully checkpointed."
                    )
            except concurrent.futures.process.BrokenProcessPool as bpp_exc:
                pass_dirs_failed +=1 # The directory processing failed
                logging.error(f"An LLM directory worker process died (BrokenProcessPool): {bpp_exc}", exc_info=True)
            except Exception as e:
                pass_dirs_failed += 1 # The directory processing failed
                logging.error(
                    f"Error processing an LLM directory future: {e}", exc_info=True
                )
    return pass_dirs_attempted, pass_dirs_successful, pass_dirs_failed, pass_total_cost, pass_total_frames_processed


def run_llm_pipeline(
    input_dir: str = "./extracted_frames",
    output_dir: str = "./llm_output",
    batch_size: int = 60,
    max_workers_dirs: int | None = 2,  # Max parallel video directories
    max_workers_batches_per_dir: int | None = 2,  # Max parallel batches within one dir
    start_index: int | None = None,
    end_index: int | None = None,
    prompt_path: str = str(DEFAULT_PROMPT_PATH),
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_file_name: str = ".processed_llm_video_dirs.log",
    daemon_mode: bool = False,
    watch_interval_seconds: int = 300,
):
    """
    Main function to run the LLM processing pipeline.
    Can run once or in daemon mode to continuously monitor for new directories.
    """
    from rich import print # Local import for this main orchestrator

    print("--- Starting LLM Processing Pipeline --- ")
    try:
        prompt_text = load_prompt(Path(prompt_path))
        if not prompt_text:
            print(f"[red]Error: Prompt file '{prompt_path}' is empty or could not be read.[/red]")
            return
        print(f"[green]Prompt loaded successfully from: {prompt_path}[/green]")
    except FileNotFoundError:
        print(f"[red]Error: Prompt file not found at '{prompt_path}'[/red]")
        return
    except Exception as e:
        print(f"[red]An unexpected error occurred while loading prompt: {e}[/red]")
        return

    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"Input Frames Directory (for LLM): {input_root}")
    logging.info(f"Output LLM JSON Directory: {output_root}")
    checkpoint_full_path = output_root / checkpoint_file_name
    logging.info(f"LLM Checkpoint file path: {str(checkpoint_full_path)}")
    logging.info(f"LLM Model: {model_name}")
    logging.info(f"Batch Size per LLM call: {batch_size} frames")
    logging.info(f"Max Workers (Dirs): {max_workers_dirs if max_workers_dirs else 'Sequential'}")
    logging.info(f"Max Workers (Batches/Dir): {max_workers_batches_per_dir if max_workers_batches_per_dir else 'Sequential'}")

    if daemon_mode:
        logging.info(
            f"[DAEMON MODE] LLM Pipeline activated. Watch interval: {watch_interval_seconds}s. "
            f"Initial slice: start={start_index}, end={end_index}. Subsequent passes process all new."
        )
    else:
        logging.info(f"Single run mode for LLM Pipeline. Slice: start={start_index}, end={end_index}")

    run_iteration = 0
    overall_dirs_attempted = 0
    overall_dirs_successful = 0
    overall_dirs_failed = 0
    overall_total_cost = 0.0
    overall_total_frames_processed = 0
    
    first_pass_daemon = True
    pipeline_start_time = time.time()

    try:
        while True:
            run_iteration += 1
            logging.info(f"--- Starting LLM Processing Pass #{run_iteration} ---")
            pass_start_time = time.time()

            current_start_idx_for_pass = None
            current_end_idx_for_pass = None

            if daemon_mode:
                if first_pass_daemon:
                    current_start_idx_for_pass = start_index
                    current_end_idx_for_pass = end_index
                    first_pass_daemon = False
                else:
                    current_start_idx_for_pass = 0 # Process all new
                    current_end_idx_for_pass = None
            else: # Single run
                current_start_idx_for_pass = start_index
                current_end_idx_for_pass = end_index
            
            pass_attempted, pass_successful, pass_failed, pass_cost, pass_frames = _perform_llm_processing_pass(
                input_root=input_root,
                output_root=output_root,
                prompt_text=prompt_text,
                batch_size=batch_size,
                model_name=model_name,
                max_workers_dirs=max_workers_dirs,
                max_workers_batches_per_dir=max_workers_batches_per_dir,
                effective_start_index=current_start_idx_for_pass,
                effective_end_index=current_end_idx_for_pass,
                checkpoint_file_name=checkpoint_file_name,
            )
            
            pass_duration = time.time() - pass_start_time

            logging.info(f"--- LLM Pass #{run_iteration} Summary ---")
            logging.info(f"Input Dirs targeted in this pass: {pass_attempted}")
            logging.info(f"Successfully processed (dirs) in this pass: {pass_successful}")
            logging.info(f"Failed (dirs) in this pass: {pass_failed}")
            logging.info(f"Estimated cost for this pass: ${pass_cost:.4f}")
            logging.info(f"Frames processed in this pass: {pass_frames}")
            logging.info(f"LLM Pass #{run_iteration} duration: {pass_duration:.2f} seconds")

            overall_dirs_attempted += pass_attempted
            overall_dirs_successful += pass_successful
            overall_dirs_failed += pass_failed
            overall_total_cost += pass_cost
            overall_total_frames_processed += pass_frames
            
            if not daemon_mode:
                break

            logging.info(f"[DAEMON MODE] Next LLM scan in {watch_interval_seconds} seconds. (Ctrl+C to stop)")
            time.sleep(watch_interval_seconds)
            
    except KeyboardInterrupt:
        print("[bold red]LLM Pipeline KeyboardInterrupt. Shutting down gracefully...[/bold red]")
    finally:
        pipeline_duration = time.time() - pipeline_start_time
        print("--- Overall LLM Processing Summary ---")
        if daemon_mode:
            print(f"Total LLM passes executed: {run_iteration}")
        print(f"Total Input Directories Targeted: {overall_dirs_attempted}")
        print(f"Total Successfully Processed Dirs: {overall_dirs_successful}")
        print(f"Total Failed Dirs: {overall_dirs_failed}")
        print(f"Total Estimated Cost: ${overall_total_cost:.4f}")
        print(f"Total Frames Processed (across all targeted batches): {overall_total_frames_processed}")
        print(f"Total LLM Pipeline Duration: {pipeline_duration:.2f} seconds")
        print(f"See {checkpoint_full_path} for completed LLM input directories.")


class LLMPipelineCLI:
    """CLI for the LLM Processing Pipeline."""

    def run(
        self,
        input_dir: str = "./extracted_frames",
        output_dir: str = "./llm_output",
        batch_size: int = 60,
        max_workers_dirs: int | None = 8, # Default based on common machine capabilities
        max_workers_batches_per_dir: int | None = 8, # Default based on common machine capabilities
        start_index: int | None = None,
        end_index: int | None = None,
        prompt_path: str = str(DEFAULT_PROMPT_PATH),
        model_name: str = str(DEFAULT_MODEL_NAME), # Ensure str for fire
        checkpoint_log: str = ".processed_llm_video_dirs.log",
        daemon_mode: bool = False,
        watch_interval_seconds: int = 300,
    ):
        """
        Runs the LLM processing pipeline on extracted frame directories.

        Args:
            input_dir: Root directory containing subdirectories of extracted frames.
            output_dir: Root directory to save the LLM JSON outputs.
            batch_size: Number of frames to process in a single LLM call.
            max_workers_dirs: Max number of video directories to process in parallel.
            max_workers_batches_per_dir: Max number of frame batches to process in parallel within a single video directory.
            start_index: 0-based index of the first video directory to process (after checkpointing).
                         Applies to single run or the first pass of daemon mode.
            end_index: 0-based index *after* the last video directory to process (after checkpointing).
                       Applies to single run or the first pass of daemon mode.
            prompt_path: Path to the LLM prompt file.
            model_name: Name of the Gemini model to use.
            checkpoint_log: Name for the checkpoint log file in output_dir for processed video directories.
            daemon_mode: If True, run continuously, scanning for new directories at intervals.
            watch_interval_seconds: Interval in seconds for daemon mode scans.
        """
        # Simple validation for worker counts to avoid issues with ProcessPoolExecutor
        if max_workers_dirs is not None and max_workers_dirs < 1:
            logging.warning("max_workers_dirs cannot be less than 1. Setting to 1.")
            max_workers_dirs = 1
        if max_workers_batches_per_dir is not None and max_workers_batches_per_dir < 1:
            logging.warning("max_workers_batches_per_dir cannot be less than 1. Setting to 1.")
            max_workers_batches_per_dir = 1

        run_llm_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            max_workers_dirs=max_workers_dirs,
            max_workers_batches_per_dir=max_workers_batches_per_dir,
            start_index=start_index,
            end_index=end_index,
            prompt_path=prompt_path,
            model_name=model_name,
            checkpoint_file_name=checkpoint_log, # Pass CLI's checkpoint_log as checkpoint_file_name
            daemon_mode=daemon_mode,
            watch_interval_seconds=watch_interval_seconds,
        )

if __name__ == "__main__":
    fire.Fire(LLMPipelineCLI) 