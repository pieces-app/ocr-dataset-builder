import random
import re

# --- Helper Functions ---

def _split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences.
    Aims to keep delimiters as part of the preceding sentence.
    """
    if not text:
        return []
    # Split by sentence-ending punctuation, keeping the delimiter
    parts = re.split(r'([.!?])', text)
    sentences = []
    current_sentence = ""
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i+1] # Sentence part + delimiter
        if sentence.strip():
            sentences.append(sentence.strip())
    if len(parts) % 2 == 1 and parts[-1].strip(): # Handle text after last delimiter
        sentences.append(parts[-1].strip())
    
    # If no delimiters were found, treat the whole text as one sentence
    if not sentences and text.strip():
        sentences = [text.strip()]
        
    return [s for s in sentences if s]

def _split_into_words(text: str) -> list[str]:
    """Splits text into words based on whitespace."""
    return text.split()

# --- Core Primitive Augmentation Functions ---

def augment_delete_sentences(text: str, probability: float) -> str:
    """Randomly deletes sentences from the text."""
    if not text or probability == 0:
        return text
    sentences = _split_into_sentences(text)
    if not sentences:
        return text
    
    augmented_sentences = [s for s in sentences if random.random() > probability]
    
    # Rejoin sentences. Add a space if the sentence doesn't end with punctuation.
    # This is a basic rejoin, might need refinement based on desired output.
    result = ""
    for i, s in enumerate(augmented_sentences):
        result += s
        if i < len(augmented_sentences) - 1:
            # Add space if current sentence doesn't end with common punctuation
            # or if next sentence doesn't start with space.
            if s and s[-1] not in ".!?":
                 result += " "
            elif augmented_sentences[i+1] and augmented_sentences[i+1][0].isalnum():
                 result += " " # Ensure space if next sentence doesn't naturally have one.
    return result

def augment_delete_words(text: str, probability: float) -> str:
    """Randomly deletes words from the text."""
    if not text or probability == 0:
        return text
    words = _split_into_words(text)
    if not words:
        return text
    
    augmented_words = [w for w in words if random.random() > probability]
    return " ".join(augmented_words)

def augment_duplicate_lines(text: str, probability: float, max_duplicates: int = 1) -> str:
    """Randomly duplicates entire lines in the text."""
    if not text or probability == 0:
        return text
    lines = text.splitlines()
    augmented_lines = []
    for line in lines:
        augmented_lines.append(line)
        if random.random() < probability:
            for _ in range(random.randint(1, max_duplicates)):
                augmented_lines.append(line)
    return "\\n".join(augmented_lines)

def augment_duplicate_partial_lines(
    text: str, 
    probability: float, 
    segment_mode: str, # 'words_start', 'words_end', 'random_ratio'
    segment_params: dict, # e.g., {'num_words': (1,2)} or {'ratio_range': (0.2,0.5)}
    max_duplicates: int = 1
) -> str:
    """
    Randomly duplicates parts of lines.
    The duplicated segment is inserted immediately after the original segment within the same line.
    """
    if not text or probability == 0:
        return text
    lines = text.splitlines()
    augmented_lines = []

    for line in lines:
        if not line.strip() or random.random() >= probability:
            augmented_lines.append(line)
            continue

        words = line.split()
        if not words:
            augmented_lines.append(line)
            continue
            
        num_duplicates = random.randint(1, max_duplicates)
        
        original_line_part = line # Fallback
        
        for _ in range(num_duplicates):
            idx_to_insert = -1
            segment_to_duplicate = ""

            if segment_mode == 'words_start' and words:
                n_words = random.randint(segment_params['num_words'][0], segment_params['num_words'][1])
                n_words = min(n_words, len(words))
                segment_words = words[:n_words]
                segment_to_duplicate = " ".join(segment_words)
                # Find where this segment ends to insert after it
                # This is approximate as it relies on splitting and rejoining.
                # A more robust way would involve char indices if precision is critical.
                temp_line = ""
                last_idx = 0
                for i, word in enumerate(words):
                    temp_line += word
                    if i < n_words -1 :
                        temp_line += " "
                    if i == n_words -1:
                        last_idx = len(temp_line)
                        break
                original_line_part = line[:last_idx] + " " + segment_to_duplicate + line[last_idx:]

            elif segment_mode == 'words_end' and words:
                n_words = random.randint(segment_params['num_words'][0], segment_params['num_words'][1])
                n_words = min(n_words, len(words))
                segment_words = words[-n_words:]
                segment_to_duplicate = " ".join(segment_words)
                
                # Find where this segment starts
                start_char_idx = line.rfind(segment_words[0], 0, line.rfind(segment_words[-1]) + len(segment_words[-1]))
                if start_char_idx != -1:
                     original_line_part = line[:start_char_idx] + segment_to_duplicate + " " + line[start_char_idx:]


            elif segment_mode == 'random_ratio' and words:
                if len(words) == 1: # Avoid issues with single word lines for ratio
                    segment_words = words
                else:
                    min_ratio, max_ratio = segment_params['ratio_range']
                    ratio = random.uniform(min_ratio, max_ratio)
                    seg_len = max(1, int(len(words) * ratio))
                    start_idx = random.randint(0, len(words) - seg_len)
                    segment_words = words[start_idx : start_idx + seg_len]
                
                segment_to_duplicate = " ".join(segment_words)
                
                # Insert segment next to itself. This can be tricky with spaces.
                # For simplicity, if "word1 word2 word3" and segment is "word2",
                # it becomes "word1 word2 word2 word3".
                # This implementation is a bit naive and might need refinement
                # for perfect whitespace handling around the duplicated segment.
                temp_words = []
                added = False
                for i, word in enumerate(words):
                    temp_words.append(word)
                    if word == segment_words[-1] and words[i-len(segment_words)+1:i+1] == segment_words and not added:
                        temp_words.append(segment_to_duplicate)
                        added = True
                original_line_part = " ".join(temp_words)
            
            line = original_line_part # Update line with the duplication for next potential duplication
        augmented_lines.append(line)
        
    return "\\n".join(augmented_lines)


def augment_merge_lines(text: str, probability: float) -> str:
    """Randomly merges a line with the next line."""
    if not text or probability == 0:
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    augmented_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]
        if i + 1 < len(lines) and random.random() < probability:
            # Merge with next line
            merged_line = current_line.strip() + " " + lines[i+1].strip()
            augmented_lines.append(merged_line)
            i += 2 # Skip next line as it's merged
        else:
            augmented_lines.append(current_line)
            i += 1
    return "\\n".join(augmented_lines)

def augment_split_lines(text: str, probability: float) -> str:
    """Randomly splits lines at a random word boundary."""
    if not text or probability == 0:
        return text
    lines = text.splitlines()
    augmented_lines = []
    for line in lines:
        if random.random() < probability:
            words = line.split()
            if len(words) > 1:
                split_point = random.randint(1, len(words) - 1)
                augmented_lines.append(" ".join(words[:split_point]))
                augmented_lines.append(" ".join(words[split_point:]))
            else:
                augmented_lines.append(line)
        else:
            augmented_lines.append(line)
    return "\\n".join(augmented_lines)

def augment_character_noise(text: str, probability: float, 
                            char_map: dict = None) -> str:
    """Introduces character-level noise (swaps, typos)."""
    if not text or probability == 0:
        return text
    
    default_char_map = {
        'l': ['1', '|'], '1': ['l', 'i'], 'o': ['0', '()'], '0': 'o', 'i': ['1', 'l', '!'], 
        's': ['5', '$'], '5': 's', 'a': ['@', '4'], 'e': ['3', '€'], 't': ['7', '+'],
        'S': ['$', '5'], 'G': ['6', '&'], 'B': ['8', 'ß'], 'g': ['9', 'q'], 'c': ['(', '['],
        'k': ['<'], 'z': ['2'], 'r': ['Я'] # Adding a few more
    }
    current_char_map = char_map if char_map else default_char_map

    augmented_chars = []
    for char in text:
        if random.random() < probability:
            replacement = current_char_map.get(char.lower()) # Check for lowercase version too
            if not replacement and char in current_char_map:
                 replacement = current_char_map.get(char)

            if replacement:
                if isinstance(replacement, list):
                    augmented_chars.append(random.choice(replacement))
                else:
                    augmented_chars.append(replacement)
            else:
                # If no specific mapping, maybe introduce a common typo like duplication or nearby key
                # For simplicity, we'll just keep it or swap with a random common char if desired
                # Or, just skip replacement if not in map
                augmented_chars.append(char)
        else:
            augmented_chars.append(char)
    return "".join(augmented_chars)

def augment_whitespace_noise(text: str, prob_missing_space: float, prob_extra_space: float) -> str:
    """Adds or removes spaces between words."""
    if not text or (prob_missing_space == 0 and prob_extra_space == 0) :
        return text
        
    # This is a simplified approach. A more robust one would use regex to find word boundaries.
    words = text.split(' ') # Split by single space
    if len(words) <= 1:
        return text

    new_text_parts = [words[0]]
    for i in range(1, len(words)):
        if not words[i-1] or not words[i]: # Avoid adding spaces around empty strings from multiple spaces
            if words[i-1]: new_text_parts.append(words[i-1])
            if words[i]: new_text_parts.append(words[i])
            continue

        r = random.random()
        if r < prob_missing_space:
            # Omit space (effectively join word with previous)
            if new_text_parts: # Ensure there's a previous part to append to
                 new_text_parts[-1] = new_text_parts[-1] + words[i]
            else: # Should not happen if words[0] was added
                 new_text_parts.append(words[i])

        elif r < prob_missing_space + prob_extra_space:
            new_text_parts.append("  ") # Add extra space
            new_text_parts.append(words[i])
        else:
            new_text_parts.append(" ") # Add normal space
            new_text_parts.append(words[i])
            
    # Filter out empty strings that might result from splitting "word  word" by " "
    return "".join(p for p in new_text_parts if p)


# --- Augmentation Setting Profiles ---

def setting_slight_stutter(text: str) -> str:
    """Setting 1: Minor repetitions, mainly at line starts/ends."""
    text = augment_delete_words(text, probability=0.01)
    text = augment_duplicate_lines(text, probability=0.05, max_duplicates=1)
    text = augment_duplicate_partial_lines(
        text, probability=0.10, 
        segment_mode='words_start', segment_params={'num_words': (2, 3)}, max_duplicates=1
    )
    text = augment_duplicate_partial_lines(
        text, probability=0.10, # Apply again for end words
        segment_mode='words_end', segment_params={'num_words': (2, 3)}, max_duplicates=1
    )
    text = augment_whitespace_noise(text, prob_missing_space=0.01, prob_extra_space=0.02) # Low missing, some extra
    text = augment_character_noise(text, probability=0.005)
    return text

def setting_gappy_and_fragmented(text: str) -> str:
    """Setting 2: Simulates diffs missing chunks of text."""
    text = augment_delete_sentences(text, probability=0.10)
    text = augment_delete_words(text, probability=0.15)
    text = augment_merge_lines(text, probability=0.02)
    text = augment_split_lines(text, probability=0.02)
    text = augment_character_noise(text, probability=0.01)
    return text

def setting_overly_eager_diff(text: str) -> str:
    """Setting 3: Diff algorithm is too sensitive, frequently duplicating lines and parts."""
    text = augment_delete_words(text, probability=0.02)
    text = augment_duplicate_lines(text, probability=0.20, max_duplicates=2) # Can duplicate more than once
    text = augment_duplicate_partial_lines(
        text, probability=0.25,
        segment_mode='random_ratio', segment_params={'ratio_range': (0.3, 0.5)}, max_duplicates=1
    )
    text = augment_character_noise(text, probability=0.005)
    return text

def setting_line_boundary_chaos(text: str) -> str:
    """Setting 4: Diff stitching severely messes up line breaks."""
    text = augment_delete_sentences(text, probability=0.02)
    text = augment_delete_words(text, probability=0.05)
    text = augment_merge_lines(text, probability=0.20)
    text = augment_split_lines(text, probability=0.15)
    text = augment_whitespace_noise(text, prob_missing_space=0.025, prob_extra_space=0.05)
    text = augment_character_noise(text, probability=0.01)
    return text

def setting_classic_bad_ocr(text: str) -> str:
    """Setting 5: Common Tesseract character-level misrecognition and word drops."""
    text = augment_delete_sentences(text, probability=0.03)
    text = augment_delete_words(text, probability=0.08)
    text = augment_duplicate_lines(text, probability=0.01)
    text = augment_duplicate_partial_lines(text, probability=0.01, segment_mode='random_ratio', segment_params={'ratio_range':(0.1,0.3)})
    text = augment_whitespace_noise(text, prob_missing_space=0.015, prob_extra_space=0.03)
    text = augment_character_noise(text, probability=0.05)
    return text

def setting_the_echo_chamber(text: str) -> str:
    """Setting 6: Extreme duplication from faulty diff logic."""
    text = augment_delete_words(text, probability=0.005) # Very low deletion
    text = augment_delete_sentences(text, probability=0.005)
    text = augment_duplicate_lines(text, probability=0.30, max_duplicates=3)
    text = augment_duplicate_partial_lines(
        text, probability=0.30,
        segment_mode='random_ratio', segment_params={'ratio_range': (0.2, 0.6)}, max_duplicates=2
    )
    text = augment_whitespace_noise(text, prob_missing_space=0.001, prob_extra_space=0.001)
    text = augment_character_noise(text, probability=0.001)
    return text

def setting_telegraphic_transmission(text: str) -> str:
    """Setting 7: Diff captures only keywords, dropping many connecting words."""
    # Higher chance of dropping common short words - implemented by higher overall word deletion.
    # A more complex version could specifically target short words.
    text = augment_delete_sentences(text, probability=0.15) # Higher for short sentences
    text = augment_delete_words(text, probability=0.25) 
    text = augment_merge_lines(text, probability=0.05)
    text = augment_split_lines(text, probability=0.05)
    text = augment_character_noise(text, probability=0.01)
    return text

def setting_jittery_frame_capture(text: str) -> str:
    """Setting 8: Minor instabilities cause slight text variations, leading to mixed omissions/duplications."""
    text = augment_delete_sentences(text, probability=0.02)
    text = augment_delete_words(text, probability=0.07)
    text = augment_duplicate_lines(text, probability=0.03)
    text = augment_duplicate_partial_lines(
        text, probability=0.15, 
        segment_mode='words_start', segment_params={'num_words': (1, 2)}, max_duplicates=1
    )
    text = augment_duplicate_partial_lines(
        text, probability=0.15, # Apply again for end words
        segment_mode='words_end', segment_params={'num_words': (1, 2)}, max_duplicates=1
    )
    text = augment_whitespace_noise(text, prob_missing_space=0.01, prob_extra_space=0.02)
    text = augment_character_noise(text, probability=0.02)
    return text

def setting_minimalist_diff_max_omission(text: str) -> str:
    """Setting 9: Overly conservative diff leads to significant omissions."""
    text = augment_delete_sentences(text, probability=0.25)
    text = augment_delete_words(text, probability=0.30)
    text = augment_duplicate_partial_lines(
        text, probability=0.02, 
        segment_mode='random_ratio', segment_params={'ratio_range': (0.05, 0.15)}, # tiny fragments
        max_duplicates=1
    ) 
    # No full line duplication
    text = augment_whitespace_noise(text, prob_missing_space=0.005, prob_extra_space=0.005)
    text = augment_character_noise(text, probability=0.005)
    return text

def setting_comprehensive_degradation(text: str) -> str:
    """Setting 10: A balanced mix of various error types at moderate levels."""
    text = augment_delete_sentences(text, probability=0.05)
    text = augment_delete_words(text, probability=0.10)
    text = augment_duplicate_lines(text, probability=0.10)
    text = augment_duplicate_partial_lines(
        text, probability=0.10,
        segment_mode='random_ratio', segment_params={'ratio_range': (0.20, 0.30)}, # 25% avg
        max_duplicates=1
    )
    text = augment_merge_lines(text, probability=0.05)
    text = augment_split_lines(text, probability=0.05)
    text = augment_whitespace_noise(text, prob_missing_space=0.015, prob_extra_space=0.03)
    text = augment_character_noise(text, probability=0.03)
    return text

# Example usage (for testing, not part of the library itself for PyTorch Dataset)
if __name__ == '__main__':
    sample_text = """This is the first sentence. It is a good sentence.
This is the second line, which also forms a sentence.
A third line here. And perhaps a fourth one? Yes!
Final line for testing purposes.
"""
    print("Original Text:\\n", sample_text)

    print("\\n--- Setting 1: Slight Stutter ---")
    print(setting_slight_stutter(sample_text))

    print("\\n--- Setting 2: Gappy and Fragmented ---")
    print(setting_gappy_and_fragmented(sample_text))
    
    print("\\n--- Setting 5: Classic Bad OCR ---")
    print(setting_classic_bad_ocr(sample_text))

    print("\\n--- Setting 10: Comprehensive Degradation ---")
    print(setting_comprehensive_degradation(sample_text))

    test_empty = ""
    test_short = "OneWord."
    print("\\n--- Setting 10 on empty ---")
    print(setting_comprehensive_degradation(test_empty))
    print("\\n--- Setting 10 on short ---")
    print(setting_comprehensive_degradation(test_short)) 