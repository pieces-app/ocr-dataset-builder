import concurrent.futures
import json
import logging
import math
import os
import sys

# import time # Removed unused import
from pathlib import Path

from dotenv import load_dotenv
import fire
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

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
    model_name: str, input_tokens: int, output_tokens: int
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

    if model_name == "gemini-2.5-pro-exp-03-25":
        logging.debug(f"Model {model_name} is free tier. Cost: $0.00")
        return 0.0

    # Pricing map: model -> (input_low, output_low, input_high, output_high, threshold_k)
    # Low rates apply <= threshold_k input tokens.
    # Thresholds are in thousands of tokens (e.g., 128k, 200k).
    pricing_map = {
        # Gemini 1.5 Pro (Adjust rates/threshold if official pricing differs)
        "gemini-1.5-pro-latest": (3.50, 10.50, 7.00, 21.00, 128),  # Threshold 128k
        # Gemini 1.5 Flash (Adjust rates/threshold)
        "gemini-1.5-flash-latest": (0.35, 1.05, 0.70, 2.10, 128),  # Threshold 128k
        # Gemini 2.5 Pro Preview (Paid Tier - from user table)
        "gemini-2.5-pro-paid": (1.25, 10.00, 2.50, 15.00, 200),  # Threshold 200k
    }

    if model_name not in pricing_map:
        logging.warning(
            f"Pricing not defined for model '{model_name}'. Assuming $0.00 cost."
        )  # Wrapped long line
        return 0.0

    (
        input_rate_low,
        output_rate_low,
        input_rate_high,
        output_rate_high,
        threshold_k,
    ) = pricing_map[model_name]

    threshold = threshold_k * 1000  # Convert k to actual token count

    if input_tokens <= threshold:
        input_rate = input_rate_low
        output_rate = output_rate_low
    else:
        input_rate = input_rate_high
        output_rate = output_rate_high

    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    # Debug log with formatted costs
    logging.debug(
        f"Cost calc ({model_name}, Thresh: {threshold_k}k): "  # Wrapped long line
        f"In:{input_tokens}tk@${input_rate}/M=${input_cost:.6f}, "
        f"Out:{output_tokens}tk@${output_rate}/M=${output_cost:.6f}, "
        f"Total=${total_cost:.6f}"
    )

    return total_cost


# --- Worker Function ---


def _process_frame_batch(
    batch_frame_paths: list[Path],
    output_json_path: Path,
    prompt_text: str,
    batch_index: int,
    total_batches: int,
    model_name: str,
) -> tuple[str, str, float | None, int | None]:
    """
    Processes a single batch of frames using the LLM and calculates cost.

    Args:
        batch_frame_paths: List of paths for the frames in this batch.
        output_json_path: Path to save the resulting JSON output.
        prompt_text: The loaded LLM prompt text.
        batch_index: The index of this batch for logging.
        total_batches: Total number of batches for this video dir for logging.
        model_name: The name of the Gemini model to use.

    Returns:
        Tuple: (status, message, cost | None, frame_count | None)
    """
    status = "Error"
    message = "Initialization failure"
    cost = None
    num_frames = len(batch_frame_paths)
    batch_repr = f"{output_json_path.parent.name} Batch {batch_index+1}/{total_batches}"
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
        logging.exception(message)
        return status, message, cost, num_frames


# --- Main Pipeline Function ---


def run_llm_pipeline(
    input_dir: str = "./extracted_frames",
    output_dir: str = "./llm_output",
    batch_size: int = 60,
    max_workers: int | None = 4,
    start_index: int | None = None,
    end_index: int | None = None,
    prompt_path: str = str(DEFAULT_PROMPT_PATH),
    model_name: str = DEFAULT_MODEL_NAME,
):
    """
    Runs the LLM processing pipeline on extracted frames.

    Args:
        input_dir: Root directory containing the extracted frames.
        output_dir: Root directory to save the LLM processing results.
        batch_size: Number of frames to process in each LLM call.
        max_workers: Max parallel processes for LLM calls. Defaults to 4.
        start_index: 0-based index of the first video subdirectory to process.
        end_index: 0-based index of the video subdirectory *after* last one.
        prompt_path: Path to the LLM prompt file.
        model_name: Gemini model name (defaults to GEMINI_MODEL_NAME env var).
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info("Starting LLM Pipeline.")
    logging.info(f"Input Frame Directory: {input_root.resolve()}")
    logging.info(f"Output JSON Directory: {output_root.resolve()}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Max Workers: {max_workers if max_workers else 'CPU Count'}")
    logging.info(f"Using Prompt: {prompt_path}")
    logging.info(f"Using Model: {model_name}")

    try:
        prompt_text = load_prompt(prompt_path)
        logging.info(f"Loaded prompt: {prompt_path}")
    except Exception:
        logging.error("Failed to load prompt. Exiting.")
        return

    all_video_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if not all_video_dirs:
        logging.error(f"No video subdirs found in {input_root}. Exiting.")
        return

    logging.info(f"Found {len(all_video_dirs)} potential video directories.")

    if start_index is not None or end_index is not None:
        actual_start = start_index if start_index is not None else 0
        actual_end = end_index if end_index is not None else len(all_video_dirs)
        video_dirs_to_process = all_video_dirs[actual_start:actual_end]
        logging.info(
            f"Processing slice: Index {actual_start} to {actual_end-1} "
            f"({len(video_dirs_to_process)} dirs)"
        )
    else:
        video_dirs_to_process = all_video_dirs
        logging.info(f"Processing all {len(video_dirs_to_process)} directories.")

    if not video_dirs_to_process:
        logging.warning("No directories selected after slicing. Exiting.")
        return

    # --- Pre-calculate total frames and Prepare tasks ---
    tasks = []
    total_frames_overall = 0
    logging.info("Calculating total frames across selected directories...")
    for video_dir in video_dirs_to_process:
        frames = sorted(video_dir.glob("frame_*.jpg"))
        num_frames_in_dir = len(frames)
        if not frames:
            logging.warning(f"No frames found in {video_dir.name}, skipping.")
            continue

        total_frames_overall += num_frames_in_dir
        num_batches = math.ceil(num_frames_in_dir / batch_size)
        logging.info(
            f"Dir {video_dir.name}: {num_frames_in_dir} frames, {num_batches} batches."
        )

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch_frames = frames[batch_start:batch_end]

            output_subdir = output_root / video_dir.name
            output_json_filename = (
                f"batch_{i:04d}_frames_{batch_start:06d}-{batch_end-1:06d}.json"
            )
            output_json_path = output_subdir / output_json_filename

            tasks.append(
                {
                    "batch_frame_paths": batch_frames,
                    "output_json_path": output_json_path,
                    "prompt_text": prompt_text,
                    "batch_index": i,
                    "total_batches": num_batches,
                    "model_name": model_name,
                }
            )

    if not tasks:
        logging.warning("No tasks created. Check input dir/patterns.")
        return

    logging.info(
        f"Total frames to process across {len(video_dirs_to_process)} directories: {total_frames_overall}"
    )

    # --- Execute tasks in parallel and Aggregate Cost ---
    success_count = 0
    error_count = 0
    processed_frames_count = 0
    current_total_cost = 0.0

    logging.info(f"Submitting {len(tasks)} batch tasks to ProcessPoolExecutor...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_frame_batch, **task) for task in tasks]
        results = []
        with tqdm(total=len(tasks), desc="Processing Batches") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    status, message, cost, frame_count = future.result()
                    results.append((status, message, cost, frame_count))

                    if frame_count is not None:
                        processed_frames_count += frame_count

                    if status == "Success":
                        success_count += 1
                        if cost is not None:
                            current_total_cost += cost
                    else:
                        error_count += 1

                    if processed_frames_count > 0:
                        avg_cost_per_frame = current_total_cost / processed_frames_count
                        est_total_cost = avg_cost_per_frame * total_frames_overall
                        pbar.set_postfix(
                            {
                                "Est Cost": f"${est_total_cost:.2f}",
                                "Avg/Frame": f"${avg_cost_per_frame:.4f}",
                                "Frames": f"{processed_frames_count}/{total_frames_overall}",
                            }
                        )
                    else:
                        pbar.set_postfix(
                            {
                                "Est Cost": "$0.00",
                                "Avg/Frame": "$0.0000",
                                "Frames": f"0/{total_frames_overall}",
                            }
                        )

                except Exception as exc:
                    logging.error(f"Batch processing generated an exception: {exc}")
                    error_count += 1
                    results.append(("Error", f"Exception in future: {exc}", None, None))
                    pbar.set_postfix_str("Error in future")

                pbar.update(1)

    logging.info("--- Pipeline Execution Summary ---")
    logging.info(f"Total Batches Submitted: {len(tasks)}")
    logging.info(
        f"Total Frames Processed: {processed_frames_count}/{total_frames_overall}"
    )
    logging.info(f"Successful Batches: {success_count}")
    logging.info(f"Errored Batches: {error_count}")
    logging.info(f"Actual Total Cost for Completed Batches: ${current_total_cost:.4f}")
    if processed_frames_count > 0:
        final_avg_cost = current_total_cost / processed_frames_count
        logging.info(f"Final Average Cost per Processed Frame: ${final_avg_cost:.6f}")

    if error_count > 0:
        logging.warning("Some batches failed. Check logs for details.")


if __name__ == "__main__":
    fire.Fire(run_llm_pipeline)
