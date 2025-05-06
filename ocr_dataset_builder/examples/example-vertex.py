import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich import print


def generate():
    # Load environment variables from .env file
    load_dotenv()

    # Get project ID and location from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project_id or not location:
        raise ValueError(
            "Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION "
            "environment variables (e.g., in a .env file)"
        )

    print(f"Using Project: {project_id}, Location: {location}")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user", parts=[types.Part.from_text(text="""Hello there""")]
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=1,
        seed=0,
        max_output_tokens=20000,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
    )
    output_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            print(chunk.text, end="")
            output_text += chunk.text

    num_tokens = client.models.compute_tokens(
        model=model,
        contents=contents,
    )
    print(f"Number of compute tokens: {num_tokens}")

    num_count_tokens = client.models.count_tokens(
        model=model,
        contents=contents,
    )
    print(f"Number of count input tokens: {num_count_tokens.total_tokens}")

    num_output_tokens = client.models.count_tokens(
        model=model,
        contents=output_text,
    )
    print(f"Number of count output tokens: {num_output_tokens.total_tokens}")


generate()
