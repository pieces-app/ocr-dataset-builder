import logging
from pathlib import Path

try:
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    logging.error(
        "Required libraries (pytesseract, Pillow) not found. "
        f"Install with: pip install pytesseract Pillow. Error: {e}"
    )
    # Re-raise or exit if these are critical dependencies for module load
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_tesseract_install() -> bool:
    """Checks if Tesseract is accessible via pytesseract."""
    try:
        version_info = pytesseract.get_tesseract_version()
        if version_info:
            logging.info(f"Tesseract version check successful: {version_info}")
            return True
        else:
            # Should not happen if version is returned, but handle defensively
            logging.warning(
                "pytesseract.get_tesseract_version() returned an empty value."
            )
            return False
    except pytesseract.TesseractNotFoundError:
        logging.error(
            "Tesseract executable not found by pytesseract. "
            "Ensure Tesseract is installed and in your PATH, "
            "or set the TESSERACT_CMD environment variable."
        )
        return False
    except Exception as e:
        logging.error("Unexpected error during Tesseract version check: " f"{e}")
        return False


def process_image_with_tesseract(image_path: Path, language: str = "eng") -> str | None:
    """
    Processes a single image file using pytesseract to extract text.

    Args:
        image_path: The path to the image file.
        language: The language code for Tesseract (e.g., 'eng', 'fra').

    Returns:
        The extracted text as a string, or None if an error occurs.
    """
    if not image_path.is_file():
        logging.error(f"Tesseract input image not found: {image_path}")
        return None

    try:
        log_msg = f"Processing {image_path.name} (lang={language})..."
        logging.debug(log_msg)
        # Open image using Pillow (needed by pytesseract)
        img = Image.open(image_path)
        # Perform OCR
        ocr_text = pytesseract.image_to_string(img, lang=language)
        logging.debug(f"Pytesseract processed {image_path.name} successfully.")
        return ocr_text.strip()  # Strip leading/trailing whitespace
    except pytesseract.TesseractNotFoundError:
        # This error should ideally be caught by check_tesseract_install first
        logging.error(
            "Tesseract executable not found during processing. "
            "Ensure Tesseract is installed and in PATH."
        )
        return None
    except pytesseract.TesseractError as e:
        # Catch Tesseract-specific errors (e.g., invalid language)
        logging.error(f"Pytesseract error for {image_path.name}: {e}")
        return None
    except FileNotFoundError:
        # Pillow might raise this if the file disappears between check and open
        logging.error(f"Image file not found during open: {image_path}")
        return None
    except Exception as e:
        logging.error(
            "Unexpected error during pytesseract process for " f"{image_path.name}: {e}"
        )
        return None


# Example usage within this module (optional)
if __name__ == "__main__":
    # Use print directly for simple example output
    from rich import print

    logging.getLogger().setLevel(logging.DEBUG)  # Enable debug for testing
    print("--- Running Pytesseract Processing Checks ---")

    if check_tesseract_install():
        print("\n--- Testing Pytesseract on a dummy/test image --- ")
        test_image_path = Path(
            "extracted_frames/#1 Covid-19 Lockdown Bioinformatics-along [8yLd7PtIMmA]/frame_000001.jpg"
        )

        print(f"Attempting to process: {test_image_path}")
        extracted_text = process_image_with_tesseract(test_image_path)

        if extracted_text is not None:
            print("\n[green]--- Extracted Text ---[/green] ")
            print(f">>>\n{extracted_text}\n<<<")
            print("----------------------")
        else:
            print("\n[yellow]--- Pytesseract processing failed ---[/yellow]")
            print("(Check Tesseract installation, image validity, and logs)")

    else:
        print(
            "\n[red]--- Tesseract installation check failed via Pytesseract. ---[/red]"
        )
        print(
            "Please install Tesseract OCR and ensure it's in your "
            "system PATH or configure Pytesseract."
        )
