## Documentation: Frame Referencing Notation in LLM Outputs

This document explains the `F:i-1` frame referencing and content appending notation used in the multi-task LLM outputs for frame sequence analysis. This notation is employed within Tasks 1-4 (Raw OCR, Augmented Imperfections, Cleaned OCR, and Structured Markdown) to optimize the generation process.

### The Notation Explained

When processing a sequence of N frames, the LLM generates output for each frame (Frame 0 to N-1) for Tasks 1-4. To handle redundancy and continuity efficiently, the following notation is used for any frame `Frame i` (where `i > 0`), comparing its potential full content for a task with the fully reconstructed content of the previous frame (`Frame i-1`) for the *same task*:

1.  **Exact Match Placeholder: `F:i-1`**
    *   **Usage**: If the complete textual content that would be generated for `Frame i` (for a specific task) is *exactly identical* to the complete textual content of `Frame i-1` (for that same task), the output for `Frame i` will simply be the string `F:i-1`.
    *   **Example**: If Task 3 (Cleaned OCR) for Frame 5 is identical to Task 3 for Frame 4, the output for Frame 5 in Task 3 will be `F:4`.
    *   **Meaning**: The content for the current frame is a direct repetition of the referenced previous frame's content for that specific task.

2.  **Appended Content Placeholder: `F:i-1 + "\n" + AppendedText`**
    *   **Usage**: If the content for `Frame i` is not an exact match to `Frame i-1`, but it *starts with* the exact content of `Frame i-1` followed by a newline character (`\n`) and then additional, non-empty text (`AppendedText`), this notation is used.
    *   **Example**: If Task 1 (Raw OCR) for Frame 2 is the same as Frame 1 plus a new line of text, the output for Frame 2 in Task 1 might be `F:1 + "\nSpeaker: User2: New comment added."`. 
    *   **Meaning**: The content for the current frame consists of the full content of the referenced previous frame (for that task) with `AppendedText` added as new lines. Only the newly appended text is explicitly provided.

3.  **Full Content**
    *   **Usage**: If neither of the above conditions is met (i.e., the content for `Frame i` is different from `Frame i-1` in a way that is not a simple append), the full, complete textual content for `Frame i` (for that task) is outputted.
    *   **Meaning**: The frame introduces significant changes, or the changes are not purely additive to the end of the previous frame's content for that task.

**Important Considerations:**
*   **Frame 0**: The first frame (`Frame 0`) for any task always contains its full content and never uses this placeholder notation.
*   **Task Independence**: These referencing rules are applied independently for each of the Tasks 1 through 4.
*   **Content Reconstruction**: To interpret a frame that uses this notation, its content must be reconstructed by retrieving the content of the referenced frame (`F:i-1`), which itself might be a reference, requiring a chain back to an original full content. Appended text is then added as specified.

### Reasons for Using This Notation

Employing this frame referencing and appending notation offers several key benefits in the context of generating detailed, multi-task analyses for frame sequences with LLMs:

1.  **Reduced Token Count**:
    *   LLM processing costs and speed are often directly related to the number of tokens generated. By referencing previously generated identical or base content, this notation significantly reduces the number of tokens the LLM needs to output, especially for sequences with static or slowly changing visual elements.
    *   Instead of repeating large blocks of text, only a short reference string (e.g., `F:10`) or the delta (appended text) is generated.

2.  **Faster Inference Times**:
    *   Generating fewer tokens typically leads to faster inference times from the LLM. This is crucial for processing long sequences or when near real-time analysis is desired.
    *   The model can complete its generation task more quickly if it can identify and leverage these redundancy patterns.

3.  **Lower Operational Costs**:
    *   Most LLM APIs charge based on the number of input and output tokens. By minimizing the output tokens through this notation, the overall cost of processing large datasets of frame sequences can be substantially reduced.

4.  **More Concise Raw Output**:
    *   The raw output files from the LLM are smaller and easier to store and transmit, as they don't contain voluminous repetitions of identical text across frames for each task. While requiring a post-processing step to "rehydrate" the full content, the initial storage and handling are more efficient.

In summary, the `F:i-1` notation is a strategy to optimize the LLM's generation process for frame sequence analysis, making it more efficient in terms of token usage, speed, and cost, while still allowing for the full reconstruction of detailed per-frame information. 