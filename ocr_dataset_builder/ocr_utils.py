import re

def clean_tesseract_ocr(text: str) -> str:
    """
    Performs basic cleaning of raw Tesseract OCR output.

    Cleaning steps:
    1. Normalize line endings to \\n.
    2. Split text into lines.
    3. Strip leading/trailing whitespace from each line.
    4. Filter out lines that become empty after stripping.
    5. Join the cleaned lines back with single newlines.
    6. Replace occurrences of 3 or more consecutive newlines with just two newlines.
    7. Replace multiple spaces within lines with a single space.
    """
    if not text: # Handle empty or None input
        return ""

    # 1. Normalize line endings and split into lines
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # 2. Strip leading/trailing whitespace from each line
    # 3. Filter out lines that become empty after stripping
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    # 4. Join the cleaned lines back with single newlines
    rejoined_text = "\n".join(cleaned_lines)

    # 5. Replace multiple spaces within lines with a single space
    # This is applied to the rejoined text to handle spaces correctly across original line breaks that might be removed
    # and then within the lines themselves.
    processed_text = re.sub(r'[ \t]+', ' ', rejoined_text) # Replace one or more spaces/tabs with a single space

    # 6. Replace occurrences of 3 or more consecutive newlines with just two newlines
    # This should be done after stripping individual lines and rejoining, then processing spaces
    # to correctly identify legitimate paragraph breaks vs. excessive spacing.
    # First, ensure all newlines are single after previous steps, then handle multiple newlines.
    # The previous join might create single newlines. If original text had N newlines, it became 1.
    # If it had M empty lines, they were removed. Now we look at sequences of actual newlines.
    # Let's refine step 4 and 5 for clarity.

    # Simpler approach after initial stripping and filtering:
    # Join lines, which gives single newlines between previously non-empty lines.
    if not cleaned_lines: # if all lines were empty or whitespace
        return ""
    
    single_newline_text = "\n".join(cleaned_lines)
    
    # Replace multiple spaces/tabs on each line (now that they are distinct lines)
    lines_spaced_normalized = [re.sub(r'[ \t]+', ' ', line) for line in single_newline_text.split('\n')]
    text_spaces_normalized = "\n".join(lines_spaced_normalized)
    
    # Reduce 3+ newlines to 2, and 2 newlines to 2 (effectively max 2 newlines)
    text_newlines_normalized = re.sub(r'\n{3,}', '\n\n', text_spaces_normalized)
    
    return text_newlines_normalized.strip() # Final strip for the whole block


if __name__ == '__main__':
    # Test cases
    raw_text_1 = "Line 1  with   extra spaces.\n\n\nLine 2 after triple newline.\r\nLine 3 with carriage return.\n\n\n\nLine 4 after even more newlines."
    print(f"Raw: Original Text 1\n---{raw_text_1}---")
    print(f"Cleaned: Text 1\n---{clean_tesseract_ocr(raw_text_1)}---")

    raw_text_2 = "   Leading and trailing spaces   \n\nJust one blank line here.\nAnd   another   line."
    print(f"\nRaw: Original Text 2\n---{raw_text_2}---")
    print(f"Cleaned: Text 2\n---{clean_tesseract_ocr(raw_text_2)}---")

    raw_text_3 = "LineA\nLineB\n\nLineC\n\n\nLineD"
    print(f"\nRaw: Original Text 3\n---{raw_text_3}---")
    print(f"Cleaned: Text 3\n---{clean_tesseract_ocr(raw_text_3)}---")
    
    raw_text_4 = ""
    print(f"\nRaw: Original Text 4 (Empty)\n---{raw_text_4}---")
    print(f"Cleaned: Text 4 (Empty)\n---{clean_tesseract_ocr(raw_text_4)}---")

    raw_text_5 = "   \n \n  \n   "
    print(f"\nRaw: Original Text 5 (Only Whitespace)\n---{raw_text_5}---")
    print(f"Cleaned: Text 5 (Only Whitespace)\n---{clean_tesseract_ocr(raw_text_5)}---")

    raw_text_6 = "Hello     World\nThis  is  a    test."
    print(f"\nRaw: Original Text 6 (Internal Spaces)\n---{raw_text_6}---")
    print(f"Cleaned: Text 6 (Internal Spaces)\n---{clean_tesseract_ocr(raw_text_6)}---")

    raw_text_7 = "Line with\ttabs\tand  spaces."
    print(f"\nRaw: Original Text 7 (Tabs and Spaces)\n---{raw_text_7}---")
    print(f"Cleaned: Text 7 (Tabs and Spaces)\n---{clean_tesseract_ocr(raw_text_7)}---") 