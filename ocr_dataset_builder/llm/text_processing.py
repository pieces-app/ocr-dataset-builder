import logging
import os
import re
import time
from pathlib import Path
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.logging import RichHandler

# Assuming these can be imported from the existing processing module
# If not, they would need to be redefined or copied here.
from ocr_dataset_builder.llm.image_processing import initialize_gemini_client, load_prompt

# Configure basic logging with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)

# Load environment variables
load_dotenv()

# Safety settings and generation config (can be adapted from image processing)
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

DEFAULT_TEMPERATURE = 0.7 # Adjusted for potentially more creative/deterministic text tasks
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 65000 # Gemini 1.5 Pro has large context, adjust as needed
DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-pro-preview-05-06")


def process_text_input(
    client: genai.Client,
    concatenated_batch_text: str,
    prompt_text: str,
    model_name: str = DEFAULT_MODEL_NAME,
    print_output: bool = False,
    print_counts: bool = False,
) -> tuple[str | None, int | None, int | None]:
    """
    Sends the concatenated batch of OCR texts and prompt to the LLM.

    Args:
        client: The initialized Gemini Client.
        concatenated_batch_text: A single string containing all OCR texts for the
                                 current batch, formatted with frame markers
                                 (e.g., "--- Frame 0 ---\nText\n--- Frame 1 ---\nText").
        prompt_text: The prompt text to guide the model (e.g., from
                     ocr_text_refinement_prompt.md).
        model_name: The name of the Gemini model to use.
        print_output: Whether to print the LLM response chunks to stdout.
        print_counts: Whether to print token counts.

    Returns:
        Tuple: (raw_text_response | None, input_tokens_count | None, output_tokens_count | None)
    """
    if not client:
        logging.error("Gemini client not initialized.")
        return None, None, None

    if not concatenated_batch_text.strip():
        logging.warning("Received empty or whitespace-only concatenated_batch_text. Skipping LLM call.")
        return "", 0, 0 # Return empty response and zero tokens

    logging.info(f"Processing text batch with model: {model_name}")

    # The prompt text is the first part, followed by the concatenated texts.
    parts_list = [prompt_text, concatenated_batch_text]

    generation_config = types.GenerateContentConfig(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_output_tokens=DEFAULT_MAX_TOKENS,
        # safety_settings=SAFETY_SETTINGS, # Safety settings included in client level or model level
        # response_modalities=["TEXT"], # For text-only models, this might not be needed or allowed
        # thinking_config=types.ThinkingConfig(include_thoughts=True), # Optional
    )
    
    # request_options = types.RequestOptions(timeout=600) # 10 minutes timeout

    try:
        # Calculate input tokens
        input_tokens_count = client.models.count_tokens(
            model=model_name,
            contents=parts_list,
        ).total_tokens
        logging.debug(f"Estimated input tokens: {input_tokens_count}")

        text_response = ""
        # Use generate_content for non-streaming if preferred for text,
        # or stream if responses can be long. Sticking to streaming for consistency.
        stream = client.models.generate_content_stream(
            model=model_name,
            contents=parts_list,
            config=generation_config,
            # request_options=request_options,
            # safety_settings=SAFETY_SETTINGS, # If not set at client/model level
        )

        for chunk in stream:
            if chunk.text:
                text_response += chunk.text
                if print_output:
                    print(chunk.text, end="", flush=True)
        
        if print_output: # Add a newline if we were printing chunks
            print()

        # Calculate output tokens
        output_tokens_count = client.models.count_tokens(
            model=model_name,
            contents=[text_response], # Model expects list of contents
        ).total_tokens
        logging.debug(f"Estimated output tokens: {output_tokens_count}")

        if print_counts:
            logging.info(f"--- Input Tokens: {input_tokens_count}")
            logging.info(f"--- Output Tokens: {output_tokens_count}")

        return text_response, input_tokens_count, output_tokens_count

    except Exception as e:
        logging.error(f"Error during LLM API call for text processing: {e}", exc_info=True)
        return None, None, None


def parse_text_llm_response(response_text: str) -> dict | None:
    """
    Parses the structured Markdown response from the LLM for text refinement tasks,
    splitting by predefined headers.
    Expected format is defined in 'ocr_text_refinement_prompt.md'.

    Args:
        response_text: The raw text output from the LLM, expected in structured Markdown.

    Returns:
        A dictionary containing parsed data for tasks 3, 4, and 5, or None if parsing fails.
        Example structure:
        {
            'task3_cleaned_text': ["text for frame 0", "text for frame 1", ...],
            'task4_markdown_text': ["md for frame 0", "md for frame 1", ...],
            'task5_summary': "Overall summary text..."
        }
    """
    if not response_text:
        logging.error("Cannot parse empty LLM response text (is None or empty string).")
        return None

    response_text = response_text.strip() # Remove leading/trailing overall whitespace
    if not response_text:
        logging.error("LLM response text is empty after stripping whitespace.")
        return None

    logging.info("Parsing text LLM response using header splitting...")
    parsed_data = {
        "task3_cleaned_text": [],
        "task4_markdown_text": [],
        "task5_summary": "",
    }

    # Define the exact headers to split by (ensure they match the prompt)
    task3_header = "==== TASK 3: CLEANED AND CORRECTED OCR TEXT ===="
    task4_header = "==== TASK 4: MARKDOWN REPRESENTATION ===="
    task5_header = "==== TASK 5: CONTEXTUAL SUMMARY AND KEY INFORMATION ===="

    # Split the response into parts based on task headers
    # We expect the response to start with Task 3, or at least have it.
    # Using re.split to keep the delimiters is complex; manual slicing is often clearer.

    try:
        task3_start_idx = response_text.find(task3_header)
        task4_start_idx = response_text.find(task4_header)
        task5_start_idx = response_text.find(task5_header)

        raw_task3_content = ""
        raw_task4_content = ""
        raw_task5_content = ""

        if task3_start_idx != -1:
            # Content for Task 3 is between its header and the next task header (or end of string)
            end_of_task3 = task4_start_idx if task4_start_idx != -1 else (task5_start_idx if task5_start_idx != -1 else len(response_text))
            raw_task3_content = response_text[task3_start_idx + len(task3_header):end_of_task3].strip()
        else:
            logging.warning(f"Header '{task3_header}' not found.")

        if task4_start_idx != -1:
            end_of_task4 = task5_start_idx if task5_start_idx != -1 else len(response_text)
            raw_task4_content = response_text[task4_start_idx + len(task4_header):end_of_task4].strip()
        else:
            logging.warning(f"Header '{task4_header}' not found.")

        if task5_start_idx != -1:
            raw_task5_content = response_text[task5_start_idx + len(task5_header):].strip()
            parsed_data["task5_summary"] = raw_task5_content
        else:
            logging.warning(f"Header '{task5_header}' not found.")

        # Helper function to parse frame data from a task block
        def parse_frames_from_block(block_content: str, task_name: str) -> list[str]:
            frames = []
            if not block_content:
                return frames
            
            # Split by frame headers. The first element will be empty or preamble, so skip it.
            # Using a regex to split but keep the frame numbers is tricky. 
            # Let's split by a generic frame header and then process each part.
            # A simple split by '-- Frame ' might be too naive if frame content contains this string.
            # Iterative find might be better.
            
            current_pos = 0
            frame_parts = []
            # Split by the generic part of the frame header "-- Frame "
            # This assumes frame headers are consistently formatted as "-- Frame X --"
            # and content for frame X follows immediately.
            split_parts = block_content.split("-- Frame ") 

            for part in split_parts:
                if not part.strip(): # Skip empty parts that might result from splitting
                    continue
                
                # The first line of 'part' should be like "0 --\nText..." or "12 --\nText..."
                match = re.match(r"(\d+)\s*--\n?(.*)", part, re.DOTALL)
                if match:
                    # frame_idx_str = match.group(1) # We don't strictly need the index for this list structure
                    frame_text = match.group(2).strip() # Strip whitespace around individual frame text
                    frames.append(frame_text)
                elif frame_parts: # if it's not the first part (which has no header)
                    # This case might happen if a split occurred mid-text, or format is unexpected
                    # For now, we assume the split primarily separates frame content correctly due to prompt.
                    logging.warning(f"Could not parse frame number from part in {task_name}: '{part[:100]}...'")
                    # frames.append(part.strip()) # As a fallback, add the whole part? Risky.
            return frames

        if raw_task3_content:
            parsed_data["task3_cleaned_text"] = parse_frames_from_block(raw_task3_content, "Task 3")
        
        if raw_task4_content:
            parsed_data["task4_markdown_text"] = parse_frames_from_block(raw_task4_content, "Task 4")

        # Check if any data was actually parsed for T3 or T4, or if summary was found
        if not parsed_data["task3_cleaned_text"] and \
           not parsed_data["task4_markdown_text"] and \
           not parsed_data["task5_summary"]:
            logging.error(
                "No task blocks (3, 4, or 5) could be meaningfully parsed using headers. "
                "LLM response might be malformed or headers are missing/incorrect."
            )
            logging.error(f"Full response for failed parse:\n{response_text}")
            return None # Indicate parsing failure

    except Exception as e:
        logging.error(f"An unexpected error occurred during header-based parsing: {e}", exc_info=True)
        logging.error(f"Full response that caused unexpected parsing error:\n{response_text}")
        return None

    logging.info(
        f"Parsed LLM response (header-split): "
        f"{len(parsed_data['task3_cleaned_text'])} Task3 items, "
        f"{len(parsed_data['task4_markdown_text'])} Task4 items, "
        f"{len(parsed_data['task5_summary'])} chars in summary."
    )
    return parsed_data


if __name__ == "__main__":
    print("[bold green]--- Running Text LLM Processing (text_processing.py) Direct Test ---[/bold green]")

    # 1. Setup: Load API Key, Initialize Client, Load Prompt
    if not os.getenv("GOOGLE_API_KEY"):
         # First, try to load from .env if not already in environment
        from dotenv import load_dotenv
        if load_dotenv():
            print("Loaded GOOGLE_API_KEY from .env file.")
        else: # If .env doesn't exist or key is not in it.
            print("[yellow]Warning: GOOGLE_API_KEY not found in environment or .env file.[/yellow]")
            # Attempt to get GOOGLE_API_KEY from user input
            api_key_input = input("Please enter your GOOGLE_API_KEY: ").strip()
            if api_key_input:
                os.environ["GOOGLE_API_KEY"] = api_key_input
                print("GOOGLE_API_KEY set from user input.")
            else:
                print("[red]Error: GOOGLE_API_KEY is required. Exiting test.[/red]")
                exit(1)
    
    # Initialize client (assuming Vertex AI setup is handled by initialize_gemini_client)
    # Ensure GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set if using Vertex
    if not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_CLOUD_LOCATION"):
        print("[yellow]Warning: GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set for Vertex AI.[/yellow]")
        # Add fallbacks or error out if Vertex is strictly needed by initialize_gemini_client
        # For now, assume initialize_gemini_client might handle non-Vertex or have its own checks.

    client = initialize_gemini_client()
    if not client:
        print("[red]Failed to initialize Gemini client. Exiting test.[/red]")
        exit(1)
    print("Gemini client initialized successfully.")

    # Define paths
    # Assumes CWD is the root of the ocr-dataset-builder project
    default_text_prompt_path = Path("ocr_dataset_builder/prompts/ocr_text_refinement_prompt.md")
    try:
        prompt_text_content = load_prompt(default_text_prompt_path)
        print(f"Text refinement prompt loaded from: {default_text_prompt_path}")
    except Exception as e:
        print(f"[red]Failed to load text prompt: {e}. Exiting test.[/red]")
        exit(1)

    # 2. Create Dummy Input Data (Concatenated Frame Texts)
    dummy_frame_texts = [
        "Frame 0: This is teh frist frame with a speling mistakee.",
        "Frame 1: Code snippet: ```python\nprint('Hello World')\n```",
        "Frame 2: Another line of text for frame two. It's qute simple.",
        "Frame 3: ", # Empty frame
        "Frame 4: Final frame with some more details and anuther typo."
    ]
    concatenated_input = ""
    for i, text in enumerate(dummy_frame_texts):
        concatenated_input += f"--- Frame {i} ---\n{text}\n\n"
    
    print(f"\n[bold blue]--- Sending Test Request to LLM ({DEFAULT_MODEL_NAME}) ---[/bold blue]")
    print("Input Text (Concatenated):")
    print(f"```\n{concatenated_input[:500]}...\n```")


    # 3. Process with LLM
    start_time = time.time()
    raw_response, input_tokens, output_tokens = process_text_input(
        client=client,
        concatenated_batch_text=concatenated_input.strip(),
        prompt_text=prompt_text_content,
        model_name=DEFAULT_MODEL_NAME, # Use the configured default
        print_output=True, # Stream output to console
        print_counts=True
    )
    duration = time.time() - start_time

    if raw_response is None:
        print(f"[red]LLM call failed after {duration:.2f}s. Check logs.[/red]")
        exit(1)

    print(f"--- LLM Raw Response Received (API call took {duration:.2f}s) ---")
    # print(f"Raw response snippet: {raw_response[:1000]}...") # For brevity

    # 4. Parse Response
    print("\n[bold blue]--- Parsing LLM Response ---[/bold blue]")
    parsed_start_time = time.time()
    parsed_output = parse_text_llm_response(raw_response)
    parsed_duration = time.time() - parsed_start_time

    if parsed_output:
        print(f"Parsing successful (took {parsed_duration:.3f}s).")
        print("Parsed Data:")
        # Adjusted to match the original expected structure for the test
        task3_texts = parsed_output.get("task3_cleaned_text", [])
        task4_mds = parsed_output.get("task4_markdown_text", [])
        task5_summary_text = parsed_output.get("task5_summary", "")

        print(f"  task3_cleaned_text: ({len(task3_texts)} items)")
        for i, item_data in enumerate(task3_texts):
            print(f"    Item {i}: {item_data[:100]}{'...' if len(item_data) > 100 else ''}")
        
        print(f"  task4_markdown_text: ({len(task4_mds)} items)")
        for i, item_data in enumerate(task4_mds):
            print(f"    Item {i}: {item_data[:100]}{'...' if len(item_data) > 100 else ''}")
        
        print(f"  task5_summary: {task5_summary_text[:200]}{'...' if len(task5_summary_text) > 200 else ''}")
        
        # Basic validation based on dummy input
        expected_frames = len(dummy_frame_texts)
        if len(task3_texts) == expected_frames:
            print("[green]Test for Task 3 frame count: PASS[/green]")
        else:
            print(f"[yellow]Test for Task 3 frame count: FAIL (Expected {expected_frames}, Got {len(task3_texts)})[/yellow]")

        if len(task4_mds) == expected_frames:
            print("[green]Test for Task 4 frame count: PASS[/green]")
        else:
            print(f"[yellow]Test for Task 4 frame count: FAIL (Expected {expected_frames}, Got {len(task4_mds)})[/yellow]")

        if task5_summary_text:
             print("[green]Test for Task 5 summary presence: PASS[/green]")
        else:
            print("[yellow]Test for Task 5 summary presence: FAIL (Summary is empty)[/yellow]")

    else:
        print("[red]Failed to parse LLM response.[/red]")
        print(f"Full response was:\n{raw_response}")

    print("\n[bold green]--- Text Processing Direct Test Finished ---[/bold green]") 