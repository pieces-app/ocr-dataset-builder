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

# Ensure the package modules can be found
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import necessary functions from the other modules
from ocr_dataset_builder.llm_processing import (
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
DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")

# --- Cost Calculation Function ---


def calculate_gemini_cost(
    model_name: str,
    input_tokens: int | CountTokensResponse,
    output_tokens: int | CountTokensResponse,
) -> float:
    """
    Calculates the estimated cost for a Gemini API call based on token counts.

    Args:
        model_name: The name of the model used (e.g., "gemini-1.5-pro-latest").
        input_tokens: The number of input tokens.
        output_tokens: The number of output tokens (including thinking).

    Returns:
        The estimated cost in USD, or 0.0 if pricing is not defined.
    """
    # Rates per 1 million tokens in USD.

    (
        input_rate_low,
        output_rate_low,
        input_rate_high,
        output_rate_high,
        threshold_k,
    ) = (
        1.25,
        10.00,
        2.50,
        15.00,
        200,
    )

    threshold = threshold_k * 1000  # Convert k to actual token count
    if isinstance(input_tokens, CountTokensResponse):
        input_tokens = input_tokens.total_tokens
    if isinstance(output_tokens, CountTokensResponse):
        output_tokens = output_tokens.total_tokens

    current_input_tokens = input_tokens if input_tokens is not None else 0
    current_output_tokens = output_tokens if output_tokens is not None else 0

    if current_input_tokens <= threshold:
        input_rate = input_rate_low
        output_rate = output_rate_low
    else:
        input_rate = input_rate_high
        output_rate = output_rate_high

    input_cost = (current_input_tokens / 1_000_000) * input_rate
    output_cost = (current_output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    # Debug log with formatted costs
    logging.debug(
        f"Cost calc ({model_name}, Thresh: {threshold_k}k): "  # Wrapped long line
        f"In:{current_input_tokens}tk@${input_rate}/M=${input_cost:.6f}, "
        f"Out:{current_output_tokens}tk@${output_rate}/M=${output_cost:.6f}, "
        f"Total=${total_cost:.6f}"
    )

    return total_cost


# --- Worker Function ---


def _process_frame_batch(
    batch_frame_paths: list[Path],
    output_json_path: Path,
    prompt_text: str,
    batch_index: int,
    total_batches_for_dir: int,  # Renamed for clarity
    model_name: str,
    video_dir_relative_path: str,  # Added for context in logging
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
    batch_repr = (
        f"{video_dir_relative_path} Batch {batch_index + 1}/{total_batches_for_dir}"
    )
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
            model_name=model_name,  # Pass model_name
        )

        raw_response, input_tokens, output_tokens = None, None, None
        if result_tuple:
            raw_response, input_tokens, output_tokens = result_tuple
        else:
            message = f"[{batch_repr}] Failed getting LLM response/tokens."
            logging.error(message)
            return status, message, cost, num_frames

        if input_tokens is not None and output_tokens is not None:
            cost = calculate_gemini_cost(model_name, input_tokens, output_tokens)
            logging.info(f"[{batch_repr}] Estimated cost: ${cost:.4f}")
        else:
            logging.warning(f"[{batch_repr}] Tokens missing, cannot calc cost.")

        parsed_data = parse_llm_response(raw_response)
        if not parsed_data:
            message = f"[{batch_repr}] Failed parsing LLM response."
            logging.error(message)
            return status, message, cost, num_frames

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=4, ensure_ascii=False)

        status = "Success"
        message = f"[{batch_repr}] Processed and saved to {output_json_path.name}"
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
    max_workers_for_batches: int | None,  # Max workers for batches within this dir
) -> tuple[
    str, int, int, float
]:  # dir_relative_path, total_batches, successful_batches, total_cost
    """Processes all frame batches for a single video directory and manages its checkpointing status."""
    logging.info(f"Starting LLM processing for directory: {frame_dir_relative_path}")
    all_frames = sorted(frame_dir_absolute_path.glob("frame_*.png"))
    if not all_frames:
        all_frames = sorted(frame_dir_absolute_path.glob("frame_*.jpg"))

    if not all_frames:
        logging.warning(f"No frames found in {frame_dir_relative_path}. Skipping.")
        return frame_dir_relative_path, 0, 0, 0.0  # No batches, 0 successful, 0 cost

    total_frames = len(all_frames)
    num_batches = math.ceil(total_frames / batch_size)
    logging.info(
        f"Directory {frame_dir_relative_path}: {total_frames} frames, {num_batches} batches of size {batch_size}."
    )

    video_output_dir = output_root / frame_dir_relative_path
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Simple batch checkpointing within a video directory (optional, for very large videos)
    # For now, we assume a video directory is the unit of checkpointing for the main pipeline.

    batch_futures = []
    processed_batches_count = 0
    successful_batches_count = 0
    dir_total_cost = 0.0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers_for_batches,
        thread_name_prefix=f"llm_batch_{frame_dir_relative_path[:10]}",
    ) as batch_executor:
        for i in range(num_batches):
            start_frame_idx = i * batch_size
            end_frame_idx = start_frame_idx + batch_size
            current_batch_frames = all_frames[start_frame_idx:end_frame_idx]
            output_json_file = video_output_dir / f"llm_output_batch_{i+1:04d}.json"

            # Potentially skip already processed batches if we add intra-directory checkpointing later

            future = batch_executor.submit(
                _process_frame_batch,
                current_batch_frames,
                output_json_file,
                prompt_text,
                i,  # batch_index
                num_batches,  # total_batches_for_dir
                model_name,
                frame_dir_relative_path,  # Pass relative path for logging
            )
            batch_futures.append(future)

        for future in concurrent.futures.as_completed(batch_futures):
            processed_batches_count += 1
            try:
                status, _, cost, _ = future.result()
                if status == "Success":
                    successful_batches_count += 1
                    if cost is not None:
                        dir_total_cost += cost
                # Detailed error logging happens in _process_frame_batch
            except Exception as exc:
                logging.error(
                    f"Error processing a batch future for {frame_dir_relative_path}: {exc}"
                )

    if successful_batches_count == num_batches and num_batches > 0:
        logging.info(
            f"Successfully processed all {num_batches} batches for {frame_dir_relative_path}. Total cost: ${dir_total_cost:.4f}"
        )
        return (
            frame_dir_relative_path,
            num_batches,
            successful_batches_count,
            dir_total_cost,
        )
    elif num_batches == 0:  # No frames found, but directory existed
        logging.info(
            f"No batches to process for {frame_dir_relative_path} (no frames). Considered complete for checkpointing."
        )
        return frame_dir_relative_path, 0, 0, 0.0
    else:
        logging.error(
            f"Directory {frame_dir_relative_path}: Only {successful_batches_count}/{num_batches} batches succeeded. Will not checkpoint as fully complete."
        )
        return (
            frame_dir_relative_path,
            num_batches,
            successful_batches_count,
            dir_total_cost,
        )  # Still return info


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
):
    """
    Runs the LLM processing pipeline on extracted frames with checkpointing per video directory.
    Args:
        input_dir: Root directory containing the extracted frames (subdirs per video).
        output_dir: Root directory to save the LLM processing results & checkpoint.
        batch_size: Number of frames to process in each LLM call (per batch).
        max_workers_dirs: Max parallel video directories to process. Defaults to 2.
        max_workers_batches_per_dir: Max parallel batches for frames within a single directory. Defaults to 2.
        start_index: 0-based index of the first video directory to process (after checkpoint).
        end_index: 0-based index *after* the last video directory (after checkpoint).
        prompt_path: Path to the LLM prompt file.
        model_name: Gemini model name.
        checkpoint_file_name: Log file for successfully processed video directories.
    """
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("--- Starting LLM Pipeline ---")
    logging.info(f"Input Frame Directory: {input_root}")
    logging.info(f"Output JSON Directory: {output_root}")
    logging.info(f"Checkpoint file: {output_root / checkpoint_file_name}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(
        f"Max Workers Dirs: {max_workers_dirs if max_workers_dirs else 'CPU Count'}"
    )
    logging.info(
        f"Max Workers Batches per Dir: {max_workers_batches_per_dir if max_workers_batches_per_dir else 'Default'}"
    )
    logging.info(f"Using Prompt: {prompt_path}")
    logging.info(f"Using Model: {model_name}")

    try:
        prompt_text = load_prompt(prompt_path)
        logging.info(f"Loaded prompt: {prompt_path}")
    except Exception as e:
        logging.error(
            f"Failed to load prompt {prompt_path}: {e}. Exiting.", exc_info=True
        )
        return

    # --- Checkpoint Loading ---
    checkpoint_path = output_root / checkpoint_file_name
    processed_video_dirs_from_checkpoint = set()
    if checkpoint_path.is_file():
        try:
            with open(checkpoint_path, "r") as f:
                for line in f:
                    processed_video_dirs_from_checkpoint.add(line.strip())
            logging.info(
                f"Loaded {len(processed_video_dirs_from_checkpoint)} processed video dirs from checkpoint."
            )
        except Exception as e:
            logging.error(
                f"Could not read checkpoint {checkpoint_path}: {e}. Processing all."
            )
            processed_video_dirs_from_checkpoint.clear()

    # --- Re-indexing and Filtering video directories ---
    all_potential_video_dirs_abs = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    all_potential_video_dirs_rel = [
        d.relative_to(input_root).as_posix() for d in all_potential_video_dirs_abs
    ]
    logging.info(
        f"Found {len(all_potential_video_dirs_rel)} potential video directories in input."
    )

    valid_processed_rel_paths = {
        path_str
        for path_str in processed_video_dirs_from_checkpoint
        if (input_root / path_str).is_dir()
    }
    # ... (log stale paths removed) ...

    relative_video_dirs_to_consider = sorted(
        [
            rel_path
            for rel_path in all_potential_video_dirs_rel
            if rel_path not in valid_processed_rel_paths
        ]
    )
    logging.info(
        f"{len(relative_video_dirs_to_consider)} video directories to consider after checkpoint."
    )

    # ... (Apply start_index and end_index to relative_video_dirs_to_consider) ...
    total_dirs_to_consider = len(relative_video_dirs_to_consider)
    actual_start_index = start_index if start_index is not None else 0
    actual_end_index = end_index if end_index is not None else total_dirs_to_consider
    # ... (validate indices) ...
    if not (0 <= actual_start_index <= total_dirs_to_consider):
        logging.error(f"Invalid start_index for LLM pipeline.")
        return
    if not (actual_start_index <= actual_end_index <= total_dirs_to_consider):
        logging.error(f"Invalid end_index for LLM pipeline.")
        return

    sliced_relative_dirs_for_this_run = relative_video_dirs_to_consider[
        actual_start_index:actual_end_index
    ]
    target_dir_count = len(sliced_relative_dirs_for_this_run)

    if target_dir_count == 0:
        logging.info("No video directories selected for LLM processing in this run.")
        return

    logging.info(
        f"Targeting {target_dir_count} video directories for LLM processing this run."
    )

    overall_start_time = time.time()
    total_processed_dirs_this_run = 0
    total_successful_dirs_this_run = 0
    grand_total_cost_this_run = 0.0

    # Executor for video directories
    if not max_workers_dirs:
        max_workers_dirs = os.cpu_count() if os.cpu_count() else 1
    if not max_workers_batches_per_dir:
        max_workers_batches_per_dir = 2  # Default if none

    dir_futures = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers_dirs, mp_context=concurrent.futures.get_context("spawn")
    ) as dir_executor:
        for rel_path in sliced_relative_dirs_for_this_run:
            abs_path = input_root / rel_path
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

        progress_bar = tqdm(
            concurrent.futures.as_completed(dir_futures),
            total=target_dir_count,
            unit="dir",
            desc="LLM Video Dirs",
        )
        for future in progress_bar:
            total_processed_dirs_this_run += 1
            try:
                returned_rel_path, total_b, successful_b, dir_cost = future.result()
                grand_total_cost_this_run += dir_cost
                if (
                    total_b == 0 or successful_b == total_b
                ):  # All batches (even 0) succeeded
                    total_successful_dirs_this_run += 1
                    try:
                        with open(checkpoint_path, "a") as cp_f:
                            cp_f.write(f"{returned_rel_path}\n")
                        logging.info(
                            f"Checkpointed LLM dir: {returned_rel_path} (Cost: ${dir_cost:.4f})"
                        )
                    except Exception as cp_e:
                        logging.error(
                            f"Failed to write LLM checkpoint for {returned_rel_path}: {cp_e}"
                        )
                else:
                    logging.error(
                        f"LLM processing for dir {returned_rel_path} was partial ({successful_b}/{total_b} batches). Not checkpointed."
                    )

            except Exception as exc:
                # How to get rel_path if future itself failed hard?
                logging.error(
                    f"A video directory LLM task failed: {exc}", exc_info=True
                )

    overall_duration = time.time() - overall_start_time
    logging.info(f"--- LLM Pipeline Finished --- For This Run ---")
    logging.info(f"Total Video Dirs Targeted: {target_dir_count}")
    logging.info(
        f"Total Video Dirs Processed (attempted): {total_processed_dirs_this_run}"
    )
    logging.info(f"Total Video Dirs Fully Successful: {total_successful_dirs_this_run}")
    logging.info(
        f"Grand Total Estimated Cost This Run: ${grand_total_cost_this_run:.4f}"
    )
    logging.info(f"Total LLM pipeline processing time: {overall_duration:.2f} seconds.")
    logging.info(
        f"See {checkpoint_path} for successfully LLM-processed video directories."
    )


class LLMPipelineCLI:
    def run(
        self,
        input_dir: str = "./extracted_frames",
        output_dir: str = "./llm_output",
        batch_size: int = 60,
        max_workers_dirs: int | None = 2,
        max_workers_batches_per_dir: int | None = 2,
        start_index: int | None = None,
        end_index: int | None = None,
        prompt_path: str = str(DEFAULT_PROMPT_PATH),
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_log: str = ".processed_llm_video_dirs.log",
    ):
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
            checkpoint_file_name=checkpoint_log,
        )


if __name__ == "__main__":
    fire.Fire(LLMPipelineCLI)
