import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich import print

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()
# --- Configuration ---
# Model name - align with example or keep flexible?
# Using the specific one from example-vertex.py for now
# MODEL_NAME = "gemini-2.5-pro-preview-03-25"  # Explicitly set model
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-exp-03-25")
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]
# Generation Config Defaults (aligning with example-vertex.py)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 65_535  # Adjust based on expected output size

# --- Helper Functions ---


def load_prompt(prompt_path: str | Path) -> str:
    """Loads the prompt text from a file."""
    try:
        prompt_text = Path(prompt_path).read_text()
        logging.info(f"Successfully loaded prompt from {prompt_path}")
        return prompt_text
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {prompt_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading prompt file {prompt_path}: {e}")
        raise


def initialize_gemini_client() -> genai.Client | None:
    """Initializes and returns the Vertex AI Client."""
    try:
        load_dotenv()  # Load environment variables from .env file
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")

        if not project_id or not location:
            # Logged as error, raised as ValueError
            logging.error(
                "Env vars GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION missing."
            )
            raise ValueError("Missing GOOGLE_CLOUD_PROJECT/LOCATION env vars.")

        logging.info(
            f"Initializing Vertex AI Client for Project: {project_id}, Location: {location}"
        )

        # Initialize client targeting Vertex AI backend
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        logging.info("Vertex AI Client initialized successfully.")
        return client
    except ValueError as ve:
        logging.error(f"Configuration error: {ve}")
        return None
    except Exception as e:
        # Catch potential auth or other init errors
        logging.error(f"Error initializing Vertex AI Client: {e}", exc_info=True)
        return None


# --- Core Processing ---


def process_image_sequence(
    client: genai.Client,
    image_paths: list[str | Path],
    prompt_text: str,
    print_output: bool = False,
    print_counts: bool = False,
) -> tuple[str | None, int | None, int | None]:
    """
    Sends the image sequence and prompt using the Vertex AI Client.

    Args:
        client: The initialized Vertex AI Client.
        image_paths: A list of paths to the image frames in the sequence.
        prompt_text: The prompt text to guide the model.

    Returns:
        The raw text response from the model, or None on error.
    """
    if not client:
        logging.error("Vertex AI client not initialized.")
        return None

    logging.info(f"Processing sequence of {len(image_paths)} images.")

    # Prepare the input parts list following the user-provided example structure
    # prompt_text string followed by image Parts created from bytes
    parts_list = [prompt_text]
    for img_path_obj in map(Path, image_paths):  # Ensure paths are Path objects
        try:
            logging.debug(f"Reading image bytes: {img_path_obj}")

            # Determine MIME type based on file extension
            ext = img_path_obj.suffix.lower()
            if ext == ".jpg" or ext == ".jpeg":
                mime_type = "image/jpeg"
            elif ext == ".png":
                mime_type = "image/png"
            # Add other common types if needed (webp, heic, etc.)
            else:
                logging.warning(
                    f"Unsupported image extension {ext} for {img_path_obj}, skipping."
                )
                continue  # Skip this image

            # Read image bytes directly from file
            with open(img_path_obj, "rb") as f:
                image_bytes = f.read()

            # Create Part from bytes
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            parts_list.append(image_part)

        except FileNotFoundError:
            logging.error(f"Image file not found: {img_path_obj}. Skipping sequence.")
            return None
        except Exception as e:
            logging.error(f"Error processing image {img_path_obj}: {e}", exc_info=True)
            return None

    if len(parts_list) <= 1:  # Only prompt, no images added successfully
        logging.error("No valid images loaded to create parts.")
        return None

    # Define generation config
    generation_config = types.GenerateContentConfig(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_output_tokens=DEFAULT_MAX_TOKENS,
        safety_settings=SAFETY_SETTINGS,
        response_modalities=["TEXT"],
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
    )

    try:
        logging.info(f"Sending request to model {MODEL_NAME} via Vertex AI Client...")
        # Call generate_content using the client.models attribute
        # Pass the list of parts directly as contents, following user example
        num_input_tokens = client.models.count_tokens(
            model=MODEL_NAME,
            contents=parts_list,
        ).total_tokens
        text_response = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=parts_list,
            config=generation_config,
        ):
            if chunk.text:
                text_response += chunk.text
                if print_output:
                    print(chunk.text, end="")
        num_output_tokens = client.models.count_tokens(
            model=MODEL_NAME,
            contents=text_response,
        ).total_tokens
        if print_counts:
            print(f"--- Input Tokens: {num_input_tokens}")
            print(f"--- Output Tokens: {num_output_tokens}")
        return text_response, num_input_tokens, num_output_tokens
    except Exception as e:
        logging.error(f"Error during API call via Client: {e}", exc_info=True)
        return None


def parse_llm_response(response_text: str) -> dict | None:
    """
    Parses the structured text response from the LLM into a dictionary,
    handling the '<<< SAME_AS_PREVIOUS >>>' placeholder for Tasks 1-4.

    Args:
        response_text: The raw text output from the Gemini model.

    Returns:
        A dictionary containing parsed data for each task, or None if parsing fails.
        Example structure:
        {
            'task1_raw_ocr': [frame0_text, frame1_text, ...],
            'task2_augmented': [frame0_text, frame1_text, ...],
            'task3_cleaned': [frame0_text, frame1_text, ...],
            'task4_markdown': [frame0_md, frame1_md, ...],
            'task5_summary': summary_text
        }
    """
    if not response_text:
        logging.error("Cannot parse empty response text.")
        return None

    logging.info("Parsing LLM response (handling redundancy)...")
    parsed_data = {}
    redundancy_placeholder = "<<< SAME_AS_PREVIOUS >>>"

    # Regex patterns to find task blocks and frame separators
    task_pattern = re.compile(
        r"^====\s*(TASK \d+:[^=]+?)\s*====$", re.MULTILINE | re.IGNORECASE
    )
    # Match frame markers, capture index and the content until the next marker or end
    frame_pattern = re.compile(
        r"^--\s*Frame (\d+)\s*--$\n(.*?)(?=^--\s*Frame \d+\s*--$|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    task_matches = list(task_pattern.finditer(response_text))
    if len(task_matches) != 5:
        logging.warning(
            f"Expected 5 task blocks, found {len(task_matches)}. Parsing may be incomplete."
        )

    task_content = {}
    for i, match in enumerate(task_matches):
        start_pos = match.end()
        end_pos = (
            task_matches[i + 1].start()
            if i + 1 < len(task_matches)
            else len(response_text)
        )
        content = response_text[start_pos:end_pos].strip()
        task_content[i + 1] = content

    # Process Tasks 1-4 (Handle Redundancy)
    for task_num in range(1, 5):
        task_key_map = {
            1: "task1_raw_ocr",
            2: "task2_augmented",
            3: "task3_cleaned",
            4: "task4_markdown",
        }
        task_key = task_key_map[task_num]
        parsed_data[task_key] = []  # Initialize list for this task
        content = task_content.get(task_num)
        if not content:
            logging.warning(f"No content found for Task {task_num}.")
            continue

        frame_matches = list(frame_pattern.finditer(content))
        if not frame_matches:
            logging.warning(f"No frame markers found in Task {task_num}.")
            continue

        # Store frame content temporarily, indexed by frame number
        temp_frame_data = {}
        for f_match in frame_matches:
            try:
                frame_index = int(f_match.group(1))
                frame_content = f_match.group(2).strip()
                temp_frame_data[frame_index] = frame_content
            except (IndexError, ValueError) as e:
                logging.warning(
                    f"Error parsing frame marker/content in Task {task_num}: {e}"
                )
                continue

        if not temp_frame_data:
            logging.warning(f"No frames successfully parsed for Task {task_num}.")
            continue

        # Determine the max frame index found
        max_index = max(temp_frame_data.keys())

        # Populate the final list, handling redundancy
        for i in range(max_index + 1):
            current_content = temp_frame_data.get(i, "")  # Get content or empty string

            if current_content == redundancy_placeholder:
                if i > 0 and len(parsed_data[task_key]) > 0:
                    # Replace placeholder with content from the previous frame *of this task*
                    previous_content = parsed_data[task_key][i - 1]
                    parsed_data[task_key].append(previous_content)
                    logging.debug(
                        f"Task {task_num}, Frame {i}: Replaced redundancy marker."
                    )
                else:
                    # Invalid use of placeholder (Frame 0 or previous frame missing)
                    logging.warning(
                        f"Task {task_num}, Frame {i}: Invalid use of '{redundancy_placeholder}'. Appending empty."
                    )
                    parsed_data[task_key].append("")  # Append empty string as fallback
            else:
                # Regular content, just append it
                parsed_data[task_key].append(current_content)

        logging.debug(
            f"Task {task_num}: Processed {len(parsed_data[task_key])} frames (after redundancy handling)."
        )

    # Process Task 5 (No Redundancy Handling)
    content_task5 = task_content.get(5)
    parsed_data["task5_summary"] = content_task5 if content_task5 else ""
    if not content_task5:
        logging.warning("No content found for Task 5.")

    logging.info("Finished parsing LLM response.")
    return parsed_data


# --- Main Execution / Test Block ---

if __name__ == "__main__":
    logging.info("--- Running LLM Processing Test --- ")

    # 1. Define paths
    default_prompt_path = Path(
        "ocr_dataset_builder/prompts/ocr_image_multi_task_prompt.md"
    )
    frame_source_dir_name = "#26 Arrow function in JavaScript [tJOJPealurs]"
    frame_source_root = Path("extracted_frames")
    frame_source_dir = frame_source_root / frame_source_dir_name
    num_test_frames = 60

    if not frame_source_dir.is_dir():
        logging.error(
            f"Frame source directory not found: {frame_source_dir}\n"
            f"Please run frame extraction first or update the path."
        )
        exit()

    try:
        all_frames = sorted(frame_source_dir.glob("frame_*.jpg"))
        sample_image_paths = all_frames[:num_test_frames]
        if len(sample_image_paths) < num_test_frames:
            logging.warning(
                f"Found only {len(sample_image_paths)} frames in "
                f"{frame_source_dir}, expected {num_test_frames}. "
                f"Using available frames."
            )
            if not sample_image_paths:
                raise FileNotFoundError("No frames found in directory.")
    except Exception as e:
        logging.error(f"Error loading frames from {frame_source_dir}: {e}")
        exit()

    print(f"Using Prompt: {default_prompt_path}")
    print(f"Using {len(sample_image_paths)} Sample Images from: {frame_source_dir}")

    # 2. Load Prompt
    try:
        prompt = load_prompt(default_prompt_path)
    except Exception:
        print("Failed to load prompt. Exiting.")
        exit()

    # 3. Initialize Client (Changed variable name)
    gemini_vertex_client = initialize_gemini_client()
    if not gemini_vertex_client:
        print("Failed to initialize Vertex AI Client. Exiting.")
        exit()
    print("Vertex AI Client initialized successfully.")

    # 4. Process Sequence (using client)
    print(
        f"\n--- Calling Gemini API ({MODEL_NAME}) with "
        f"{len(sample_image_paths)} frames --- "
    )
    start_time = time.time()
    raw_response, num_input_tokens, num_output_tokens = process_image_sequence(
        gemini_vertex_client,
        sample_image_paths,
        prompt,
        print_output=True,
        print_counts=True,
    )
    end_time = time.time()
    duration = end_time - start_time

    if raw_response:
        print(f"--- Received Raw Response (API call took {duration:.2f}s) --- ")

        # 5. Parse Response
        print("\n--- Parsing Response --- ")
        parse_start_time = time.time()
        parsed_output = parse_llm_response(raw_response)
        parse_end_time = time.time()
        parse_duration = parse_end_time - parse_start_time

        if parsed_output:
            print(f"--- Input Tokens: {num_input_tokens}")
            print(f"--- Output Tokens: {num_output_tokens}")
            print(
                f"--- Parsed Output Summary (Parsing took {parse_duration:.3f}s) --- "
            )
            print(f" Task 1 Frames: {len(parsed_output.get('task1_raw_ocr', []))}")
            print(f" Task 2 Frames: {len(parsed_output.get('task2_augmented', []))}")
            print(f" Task 3 Frames: {len(parsed_output.get('task3_cleaned', []))}")
            print(f" Task 4 Frames: {len(parsed_output.get('task4_markdown', []))}")
            t5_summary = parsed_output.get("task5_summary", "")
            print(f" Task 5 Summary Length: {len(t5_summary)}")
            print(f"  Task 5 Summary Snippet: {t5_summary[:200]}...")
            print("\nTest finished successfully.")
        else:
            print("Failed to parse the response.")
    else:
        print(
            f"Failed to get a response from the Gemini API "
            f"(call duration: {duration:.2f}s). Check logs."
        )

    # Note: No dummy file cleanup needed anymore
