# OcrMultimodalDataset: Integration and Usage Guide

## 1. Overview

The `OcrMultimodalDataset` is a PyTorch `Dataset` class designed to load and serve multimodal data consisting of:
- Frame images extracted from videos.
- Textual outputs from a Large Language Model (LLM) based on these frames (covering multiple analytical tasks).
- Textual outputs from the Tesseract OCR engine applied to these frames.

It intelligently aggregates data from potentially multiple LLM output batch files per video and handles a specialized `F:i-1` notation for efficient representation of redundant LLM outputs.

## 2. Expected Data Directory Structure

The dataset class expects a specific organization for your input data:

### 2.1. Frame Images

Frame images should be stored in a root directory, with subdirectories for each unique `VIDEO_ID`.

```
<frames_root_dir>/
├── VIDEO_ID_1/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ... (sorted frame image files, e.g., .png, .jpg, .jpeg)
├── VIDEO_ID_2/
│   ├── frame_0000.png
│   └── ...
└── ...
```

### 2.2. LLM Outputs

LLM outputs should also be in a root directory, with subdirectories for each `VIDEO_ID`. Within each `VIDEO_ID` directory, LLM outputs for the video's frames are expected to be split into one or more sequentially numbered JSON batch files.

```
<llm_outputs_root_dir>/
├── VIDEO_ID_1/
│   ├── llm_output_batch_0001.json
│   ├── llm_output_batch_0002.json
│   └── ...
├── VIDEO_ID_2/
│   ├── llm_output_batch_0001.json
│   └── ...
└── ...
```

**Structure of `llm_output_batch_xxxx.json` files:**

Each batch JSON file should be a dictionary containing:
- Keys for different LLM tasks (e.g., `task1_raw_ocr`, `task2_augmented`, `task3_cleaned`, `task4_markdown`). The values for these keys should be lists of strings, where each string is the LLM output for a frame processed in that batch. The order of items in these lists corresponds to the order of frames processed in that batch.
- A `task5_summary` key, whose value is a single string representing the narrative summary applicable to *all* frames processed within that specific batch file.
- (Optional but recommended) A `frame_ids` key, with a list of frame filenames processed in that batch, in the same order as the task lists. (Note: The current dataset loader does not strictly require `frame_ids` if the frames per batch can be correctly inferred, but it's good for data integrity).

Example `llm_output_batch_0001.json`:
```json
{
    "task1_raw_ocr": [
        "Raw OCR text for frame 0 of this batch...",
        "Raw OCR text for frame 1 of this batch..."
    ],
    "task2_augmented": [
        "Augmented OCR for frame 0...",
        "F:0 ...plus some changes for frame 1"
    ],
    "task3_cleaned": [
        "Cleaned OCR for frame 0...",
        "Cleaned OCR for frame 1..."
    ],
    "task4_markdown": [
        "```markdown\n### Frame 0 Analysis\n...\n```",
        "```markdown\n### Frame 1 Analysis\n...\n```"
    ],
    "task5_summary": "This is the narrative summary covering frames 0 and 1 of this batch for this video.",
    "frame_ids": ["frame_0000.png", "frame_0001.png"]
}
```

### 2.3. Tesseract OCR Outputs

Tesseract outputs are also organized by `VIDEO_ID` subdirectories. Each `VIDEO_ID` directory should contain a single `tesseract_ocr.json` file.

```
<tesseract_outputs_root_dir>/
├── VIDEO_ID_1/
│   └── tesseract_ocr.json
├── VIDEO_ID_2/
│   └── tesseract_ocr.json
└── ...
```

**Structure of `tesseract_ocr.json` files:**

This JSON file is a dictionary mapping frame filenames (e.g., `frame_0000.png`) directly to their corresponding Tesseract OCR text.

Example `tesseract_ocr.json`:
```json
{
    "frame_0000.png": "Tesseract output for frame_0000.png.",
    "frame_0001.png": "Tesseract output for frame_0001.png.",
    ...
}
```

## 3. Data Loading and Processing Logic

### 3.1. Initialization (`__init__`)

When `OcrMultimodalDataset` is initialized:
1.  It scans the `frames_root_dir` to identify all `VIDEO_ID`s.
2.  For each `VIDEO_ID`:
    a.  It lists all frame image files (e.g., `.png`, `.jpg`) and sorts them by name. This defines the canonical order of frames for that video.
    b.  It lists all `llm_output_batch_xxxx.json` files in the corresponding LLM output directory and sorts them by name (e.g., `llm_output_batch_0001.json`, `llm_output_batch_0002.json`, ...).
    c.  It iterates through these sorted batch files:
        i.  For each LLM task (Task 1-4, e.g., `task1_raw_ocr`), it appends the list of outputs from the current batch file to a video-specific concatenated list for that task.
        ii. The `task5_summary` string from the current batch file is stored and will be associated with all frames that were part of this batch.
    d.  It loads the `tesseract_ocr.json` file for the video.
3.  The dataset maintains an internal list of `samples`. Each entry in this list corresponds to a unique frame image and stores its path, `VIDEO_ID`, frame stem (filename without extension), and its 0-based index within its video.
4.  The concatenated LLM task data and Tesseract data are stored in dictionaries keyed by `VIDEO_ID`.

### 3.2. Item Retrieval (`__getitem__`)

When an item is requested from the dataset (e.g., `dataset[idx]`):
1.  It retrieves the frame's path, `VIDEO_ID`, stem, and its index within the video.
2.  The frame image is loaded using PIL and transformed if an `image_transform` was provided.
3.  For each LLM task (Task 1-4):
    a.  The corresponding raw LLM output string is fetched from the concatenated list for that video and task, using the frame's index within the video.
    b.  If this string uses the `F:index` notation (see below), the content is reconstructed.
4.  The `task5_summary` associated with the frame's original LLM batch is retrieved.
5.  Tesseract OCR text is retrieved using the frame's filename.
6.  All data is returned as a dictionary.

### 3.3. Handling `F:index` Notation

To reduce redundancy in LLM outputs, the dataset supports a special notation:
- `F:idx`: If an LLM task output for a frame is `F:idx` (e.g., `F:0`), it means the content for this frame is identical to the content of frame `idx` (0-indexed) *for the same task within the same video*.
- `F:idx appended text`: If there is text after `F:idx ` (e.g., `F:0 ... some additional notes`), this `appended text` is concatenated to the fully reconstructed content of frame `idx`.

This reconstruction is handled by the `_reconstruct_llm_output` method, which uses memoization (caching) within a single `__getitem__` call to efficiently resolve chains of references for each task.

## 4. Usage Example

```python
from torchvision import transforms
from torch.utils.data import DataLoader
from ocr_dataset_builder.pytorch_dataset import OcrMultimodalDataset # Assuming your file is here

# Define paths to your data
frames_root = "/path/to/your/frames_root_dir"
llm_root = "/path/to/your/llm_outputs_root_dir"
tesseract_root = "/path/to/your/tesseract_outputs_root_dir"

# Define an image transform (optional)
image_transformation = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the task keys your LLM outputs use
# Ensure 'task5_summary' is included if you have narrative summaries per batch.
# The dataset will also check for 'alternate_task2_key' if specified and the primary task2 key is missing.
llm_task_keys = [
    "task1_raw_ocr",
    "task2_augmented", 
    "task3_cleaned",
    "task4_markdown",
    "task5_summary"
]
alternate_task2 = "task2_augmented_imperfections"

# Instantiate the dataset
dataset = OcrMultimodalDataset(
    frames_root_dir=frames_root,
    llm_outputs_root_dir=llm_root,
    tesseract_outputs_root_dir=tesseract_root,
    image_transform=image_transformation,
    task_keys=llm_task_keys,
    alternate_task2_key=alternate_task2
    # video_ids_to_load=["VIDEO_ID_1"], # Optional: load only specific videos
)

# Check dataset length
print(f"Dataset loaded with {len(dataset)} samples.")

if len(dataset) > 0:
    # Get a sample
    sample = dataset[0]
    print(f"--- Sample 0 ---")
    print(f"Frame Path: {sample['frame_path']}")
    print(f"Image Tensor Shape: {sample['image'].shape}")
    print(f"Task 1 (Raw OCR): '{sample.get('task1_raw_ocr', '')[:70]}...'")
    print(f"Task 2 (Augmented): '{sample.get('task2_augmented', '')[:70]}...'")
    print(f"Task 3 (Cleaned): '{sample.get('task3_cleaned', '')[:70]}...'")
    print(f"Task 4 (Markdown): '{sample.get('task4_markdown', '')[:70]}...'")
    print(f"Task 5 (Narrative): '{sample.get('task5_summary', '')[:70]}...'")
    print(f"Tesseract OCR: '{sample.get('tesseract_ocr', '')[:70]}...'")

    # Use with a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(f"--- Batch ---")
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Task 1 (first in batch): {batch['task1_raw_ocr'][0][:70]}...")
        # ... process your batch
        break 
else:
    print("Dataset is empty. Please check paths and data structure.")

```

## 5. Design Journey & Considerations

The design of this dataset loader evolved through discussions:

-   **Initial LLM Output Structure**: We initially considered a single `llm_results.json` per video. However, to handle potentially large numbers of frames and outputs, the design shifted to using multiple `llm_output_batch_xxxx.json` files per video.
-   **`frame_ids` in LLM Batches**: While an explicit `frame_ids` list within each LLM batch JSON provides the most robust mapping from batch item index to global frame ID, the current loader was adapted to work even if this list is absent. It does so by assuming that the concatenated items from sorted batch files for a video map directly and sequentially to the sorted frame image files for that same video.
-   **`F:i-1` Notation**: This was introduced to efficiently represent LLM outputs where content is identical or largely similar to a preceding frame, saving storage and potentially LLM processing costs. The dataset handles the on-the-fly reconstruction of this content.
-   **Task Key Flexibility**: The dataset allows specifying the expected LLM task keys and an alternate key for "task2" to accommodate slight variations in output formats (e.g., `task2_augmented` vs. `task2_augmented_imperfections`).

This iterative process aimed to create a flexible yet robust dataset loader for complex multimodal OCR-related data. 