# Answer Bounding Box Extractor — POC

Extracts the spatial location of each question's answer from a PDF answer sheet.

## Output format

```json
{
  "3":   {"pages": [1],    "y_start": 240,  "y_end": 580},
  "1.1": {"pages": [1, 2], "y_start": 610,  "y_end": 190},
  "6.2": {"pages": [3],    "y_start": 80,   "y_end": 420}
}
```

- **keys** — normalized question numbers (`17(a)` → `17.1`, `33(a)(i)` → `33.1.1`)
- **pages** — ordered list of 1-based page numbers the answer spans
- **y_start** — pixel y-coord (200 DPI) of the top of the answer on the **first** page
- **y_end** — pixel y-coord (200 DPI) of the bottom of the answer on the **last** page

## Pipeline

```
PDF
 │
 ▼  pdf2image (200 DPI)
Page images (PIL)
 │
 ├─► GPT-4o (vision)
 │     Identifies question numbers + which pages each answer spans
 │
 └─► LayoutLMv3ImageProcessor (apply_ocr=True)
       Runs Tesseract OCR on each page → word texts + bounding boxes
       (falls back to pytesseract directly if transformers unavailable)
         │
         ▼
       Question label matching in word list
         y_start = y0 of question label word on first page
         y_end   = y0 of next question label (or page bottom) on last page
```

## Setup

### 1. Install Tesseract binary

| OS | Command |
|----|---------|
| Windows | Download installer from https://github.com/UB-Mannheim/tesseract/wiki |
| Ubuntu  | `sudo apt install tesseract-ocr` |
| macOS   | `brew install tesseract` |

If tesseract is not on `PATH` set the env var:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 2. Install Poppler (for pdf2image)

| OS | Command |
|----|---------|
| Windows | Download from https://github.com/oschwartz10612/poppler-windows/releases and set `POPPLER_PATH` |
| Ubuntu  | `sudo apt install poppler-utils` |
| macOS   | `brew install poppler` |

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
OPENAI_API_KEY=sk-...
POPPLER_PATH=C:\path\to\poppler\bin   # Windows only
TESSERACT_CMD=C:\path\to\tesseract.exe  # Windows, if not on PATH
OPENAI_MODEL=gpt-4o                   # optional, default gpt-4o
```

## Usage

```bash
# Basic — writes <pdf_name>_bbox.json next to the PDF
python poc_bbox_extractor.py input/1.pdf

# Custom output path
python poc_bbox_extractor.py input/1.pdf -o output/1_bbox.json

# With visualization (saves annotated page images)
python poc_bbox_extractor.py input/1.pdf --visualize
```

## Notes

- `y_start`/`y_end` are in **pixel coordinates at 200 DPI**, matching the coordinate
  space used by the YOLO diagram detector in the backend (`pdf_answer_processor.py`).
- For multi-page answers, `y_start` is on the first page and `y_end` on the last page.
  There are no y-coordinates recorded for the intermediate pages.
- If a question label cannot be located in the OCR output, `y_start` will be `null`
  and a warning is logged.
- The `--visualize` flag draws blue horizontal lines at `y_start` and red lines at
  `y_end` on each page image, useful for manual verification.
