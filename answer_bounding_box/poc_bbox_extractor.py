#!/usr/bin/env python3
"""
POC: Answer Bounding Box Extractor
===================================
Processes a PDF answer sheet and outputs spatial bounding box info per question.

Output format (target JSON):
{
    "3":   {"pages": [1],    "y_start": 240,  "y_end": 580},
    "1.1": {"pages": [1, 2], "y_start": 610,  "y_end": 190},
    "6.2": {"pages": [3],    "y_start": 80,   "y_end": 420},
    ...
}

- Keys: normalized question numbers (e.g. "17(a)" -> "17.1")
- pages: ordered list of page numbers (1-based) the answer spans
- y_start: pixel y-coord (200 DPI) of the top of the answer on the FIRST page
- y_end:   pixel y-coord (200 DPI) of the bottom of the answer on the LAST page

Pipeline:
  1. PDF → PIL images at 200 DPI (via pdf2image)
  2. LLM (GPT-4o) identifies question numbers + which pages each answer spans
  3. LayoutLMv3ImageProcessor (apply_ocr=True) extracts word-level bounding boxes
     per page (via Tesseract OCR internally)
  4. Question label matching in the word list gives y_start on first page
  5. y_end is derived from the next question label position (or page bottom for last Q)
"""

import os
import re
import json
import base64
import logging
import unicodedata
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import dotenv
dotenv.load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Dependency checks (graceful import failures with clear messages)
# ---------------------------------------------------------------------------
try:
    from pdf2image import convert_from_path
except ImportError:
    raise SystemExit("pdf2image not installed. Run: pip install pdf2image")

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow not installed. Run: pip install Pillow")

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai not installed. Run: pip install openai")

try:
    from transformers import LayoutLMv3ImageProcessor
    _LAYOUTLMV3_AVAILABLE = True
except ImportError:
    _LAYOUTLMV3_AVAILABLE = False
    logger.warning(
        "transformers not installed. LayoutLMv3 word-box extraction unavailable. "
        "Run: pip install transformers torch"
    )

try:
    import pytesseract
    _tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _PYTESSERACT_AVAILABLE = False
    logger.warning(
        "pytesseract not installed. Fallback word-box extraction unavailable. "
        "Run: pip install pytesseract  (also requires Tesseract binary)"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 200  # Must match the DPI used for PDF→image conversion

# Vertical padding (pixels at 200 DPI) subtracted from a detected y_end so it
# sits just above the first line of the *next* question rather than overlapping it.
BOTTOM_PADDING_PX = 4


# ---------------------------------------------------------------------------
# Helper: normalize question number (mirrors backend pdf_answer_processor.py)
# ---------------------------------------------------------------------------
def normalize_question_number(q_num: str) -> str:
    """
    Normalize question numbers from various formats to dot notation.

    Examples:
        "17(a)"   -> "17.1"
        "17(b)"   -> "17.2"
        "33(a)(i)"-> "33.1.1"
        "29.1"    -> "29.1"
        "1"       -> "1"
    """
    if not re.search(r"[()]", q_num):
        return q_num.strip()

    letter_map = {
        "(a)": ".1", "(b)": ".2", "(c)": ".3", "(d)": ".4",
        "(e)": ".5", "(f)": ".6", "(g)": ".7", "(h)": ".8",
    }
    roman_map = {
        "(viii)": ".8", "(vii)": ".7", "(vi)": ".6", "(iv)": ".4",
        "(iii)": ".3", "(ii)": ".2", "(v)": ".5", "(i)": ".1",
    }

    result = q_num.lower()
    for k, v in letter_map.items():
        result = result.replace(k, v)
    # roman must be longest-first to avoid "(i)" matching inside "(ii)"
    for k, v in roman_map.items():
        result = result.replace(k, v)
    result = result.replace("(", ".").replace(")", "")
    return result.strip()


# ---------------------------------------------------------------------------
# Helper: PIL image → base64 data URL
# ---------------------------------------------------------------------------
def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Word-box extraction via LayoutLMv3 (primary) or pytesseract (fallback)
# ---------------------------------------------------------------------------

def extract_word_boxes_layoutlmv3(
    image: Image.Image,
) -> List[Tuple[str, int, int, int, int]]:
    """
    Use LayoutLMv3ImageProcessor with apply_ocr=True to obtain word-level
    bounding boxes from a page image.

    Returns:
        List of (word, x0_px, y0_px, x1_px, y1_px) in 200 DPI pixel coords.
        Boxes are normalized [0..1000] internally; this function converts them
        back to pixel space.
    """
    processor = LayoutLMv3ImageProcessor(apply_ocr=True)
    encoding = processor(image, return_tensors="pt")

    words: List[str] = encoding.words[0]           # list of str
    raw_boxes = encoding.boxes[0]
    # boxes may be a torch.Tensor or a plain list depending on transformers version
    boxes: List[List[int]] = raw_boxes.tolist() if hasattr(raw_boxes, "tolist") else list(raw_boxes)

    img_w, img_h = image.size  # PIL: (width, height)

    result: List[Tuple[str, int, int, int, int]] = []
    for word, box in zip(words, boxes):
        x0 = int(box[0] * img_w / 1000)
        y0 = int(box[1] * img_h / 1000)
        x1 = int(box[2] * img_w / 1000)
        y1 = int(box[3] * img_h / 1000)
        result.append((word, x0, y0, x1, y1))

    return result


def extract_word_boxes_pytesseract(
    image: Image.Image,
) -> List[Tuple[str, int, int, int, int]]:
    """
    Fallback: use pytesseract image_to_data to get word-level bounding boxes.

    Returns:
        List of (word, x0_px, y0_px, x1_px, y1_px) in 200 DPI pixel coords.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    result: List[Tuple[str, int, int, int, int]] = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 10:  # skip very-low-confidence noise
            continue
        x0 = data["left"][i]
        y0 = data["top"][i]
        x1 = x0 + data["width"][i]
        y1 = y0 + data["height"][i]
        result.append((text, x0, y0, x1, y1))
    return result


def extract_word_boxes(image: Image.Image) -> List[Tuple[str, int, int, int, int]]:
    """Primary extraction via LayoutLMv3; falls back to pytesseract."""
    if _LAYOUTLMV3_AVAILABLE:
        try:
            return extract_word_boxes_layoutlmv3(image)
        except Exception as exc:
            logger.warning(f"LayoutLMv3 extraction failed ({exc}); falling back to pytesseract")

    if _PYTESSERACT_AVAILABLE:
        return extract_word_boxes_pytesseract(image)

    raise RuntimeError(
        "No OCR backend available. Install transformers+torch OR pytesseract+tesseract."
    )


# ---------------------------------------------------------------------------
# Question label matching in the word list
# ---------------------------------------------------------------------------

def _normalize_for_matching(text: str) -> str:
    """Lower-case, strip punctuation noise for fuzzy matching."""
    text = unicodedata.normalize("NFKC", text)
    return text.strip().lower()


def _tokens_to_label(words: List[Tuple[str, int, int, int, int]], start: int, length: int) -> str:
    """Concatenate `length` consecutive word texts starting at index `start`."""
    return "".join(w[0] for w in words[start: start + length])


def find_question_label_y(
    words: List[Tuple[str, int, int, int, int]],
    label: str,
    img_height: int,
    search_below_y: int = 0,
) -> Optional[int]:
    """
    Find the y-start (top pixel) of a question label in the OCR word list.

    Tries:
      1. Exact single-token match (e.g. word == "1.1")
      2. Two-token concatenation (e.g. "1" + ".1" = "1.1", or "1." + "1")
      3. Three-token concatenation

    Only looks at words with y0 >= search_below_y (to skip content above
    a previously found question on the same page).

    Returns the y0 pixel of the first matching token, or None if not found.
    """
    target = _normalize_for_matching(label)

    visible = [(i, w) for i, w in enumerate(words) if w[2] >= search_below_y]

    # 1-token
    for i, w in visible:
        if _normalize_for_matching(w[0]) == target:
            return w[2]  # y0

    # 2-token
    for i, w in visible:
        if i + 1 < len(words):
            concat = _normalize_for_matching(_tokens_to_label(words, i, 2))
            if concat == target:
                return w[2]

    # 3-token
    for i, w in visible:
        if i + 2 < len(words):
            concat = _normalize_for_matching(_tokens_to_label(words, i, 3))
            if concat == target:
                return w[2]

    # Fallback: if label has a leading numeric segment added by the LLM (e.g. "0.1.1"
    # on a sheet that writes "1.1"), strip the first segment and retry.
    # "0.1.1" -> "1.1",  "2.1.1" -> "1.1" (strip first dot-segment)
    if re.match(r"^\d+\.", target):
        stripped = re.sub(r"^\d+\.", "", target)
        if stripped and stripped != target:
            alt = find_question_label_y(words, stripped, img_height, search_below_y)
            if alt is not None:
                return alt

    return None


# ---------------------------------------------------------------------------
# LLM: per-page extraction
# ---------------------------------------------------------------------------

_PAGE_SYSTEM_PROMPT = dedent("""
    You are a precise document analysis assistant specialising in student exam answer sheets.
    Return only valid JSON, no extra text.
""")

_PAGE_USER_PROMPT = dedent("""
    This is page {page_num} of a student exam answer sheet.

    Report every answer block that STARTS on this page and every answer block that ENDS
    on this page (they can overlap — a single-page answer both starts and ends here).

    Use two lists in your JSON:
      "starts" : blocks whose question-number LABEL is printed on this page.
      "ends"   : blocks whose last written line is on this page.

    For each item include:
      question_number  – the exact label (normalized as below).
      y_frac           – a 0.0–1.0 decimal: how far DOWN the page the block starts (for
                         "starts") or ends (for "ends").
                         0.0 = very top of page, 1.0 = very bottom.

    NORMALIZATION (apply to question_number):
    - Remove a leading "Q." prefix: "Q.3" → "3".
    - Convert bracketed sub-parts: "1(a)" → "1.1", "6(i)" → "6.1", "33(a)(i)" → "33.1.1".
    - Keep verbatim otherwise: "1.7", "1.10", "6.2", "3".
    - Do NOT add extra prefix segments.

    WHAT COUNTS AS A QUESTION NUMBER:
    - Leaf-level printed labels like "1.7", "3", "6.2" that appear beside an answer block.
    - NEVER group sub-questions: "1.1" and "1.2" are separate, NOT merged under "1".

    WHAT IS NOT A QUESTION NUMBER:
    - Stand-alone letters A/B/C/D (MCQ answer choices).
    - Section headings like "SECTION A", "PART 1".

    Return ONLY (no markdown fences):
    {{"starts": [{{}}, ...], "ends": [{{}}, ...]}}

    Example:
    {{"starts": [{{"question_number": "1.7", "y_frac": 0.42}},
                 {{"question_number": "1.8", "y_frac": 0.68}}],
      "ends":   [{{"question_number": "1.6", "y_frac": 0.38}},
                 {{"question_number": "1.7", "y_frac": 0.64}}]}}
""")


def _process_single_page(
    client: OpenAI,
    page_img: Image.Image,
    page_num: int,
) -> Dict[str, Any]:
    """Send one page image to the LLM and return its starts/ends dict."""
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": _PAGE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": _PAGE_USER_PROMPT.format(page_num=page_num)},
                {"type": "image_url", "image_url": {"url": pil_to_data_url(page_img)}},
            ]},
        ],
        response_format={"type": "json_object"},
    )
    msg = completion.choices[0].message
    raw = json.loads(msg.content) if msg.content else {}
    logger.debug(f"  Page {page_num} LLM response: {json.dumps(raw)}")
    return raw


def get_question_page_spans(
    client: OpenAI,
    pages: List[Image.Image],
) -> Dict[str, Any]:
    """
    Process every page individually so the LLM sees each at full resolution.

    Returns the final output dict directly:
        {"1.7": {"pages": [2], "y_start": 980, "y_end": 1496}, ...}
    """
    # Accumulate starts and ends per normalized question number
    starts: Dict[str, Tuple[int, float]] = {}   # q -> (page_num, y_frac)
    ends:   Dict[str, Tuple[int, float]] = {}   # q -> (page_num, y_frac)

    for page_num, page_img in enumerate(pages, 1):
        logger.info(f"  Processing page {page_num}/{len(pages)} …")
        raw = _process_single_page(client, page_img, page_num)

        for item in raw.get("starts", []):
            q = _clean_question_number(str(item.get("question_number", "")).strip())
            if not q:
                continue
            y = float(item.get("y_frac", 0.0))
            if q not in starts:            # keep first occurrence (earliest page)
                starts[q] = (page_num, y)
                logger.debug(f"    start: '{q}' at page {page_num} y_frac={y:.2f}")

        for item in raw.get("ends", []):
            q = _clean_question_number(str(item.get("question_number", "")).strip())
            if not q:
                continue
            y = float(item.get("y_frac", 1.0))
            ends[q] = (page_num, y)        # keep last occurrence (latest page)
            logger.debug(f"    end:   '{q}' at page {page_num} y_frac={y:.2f}")

    # Build output — page range is [start_page .. end_page] inclusive
    all_questions = sorted(set(starts) | set(ends),
                           key=lambda q: (starts.get(q, ends.get(q))[0],
                                          starts.get(q, (0, 0))[1]))

    output: Dict[str, Any] = {}
    for q in all_questions:
        if q in starts:
            start_page, y_start_frac = starts[q]
        else:
            # end found without a matching start — assume it started at top of that page
            start_page, y_start_frac = ends[q][0], 0.0
            logger.warning(f"  No start found for '{q}'; assuming page {start_page} top.")

        if q in ends:
            end_page, y_end_frac = ends[q]
        else:
            # start found without a matching end — assume it ends at bottom of start page
            end_page, y_end_frac = start_page, 1.0
            logger.warning(f"  No end found for '{q}'; assuming page {start_page} bottom.")

        if end_page < start_page:
            logger.warning(f"  '{q}': end_page {end_page} < start_page {start_page}; swapping.")
            start_page, end_page = end_page, start_page

        page_list = list(range(start_page, end_page + 1))
        img_h_start = pages[start_page - 1].size[1]
        img_h_end   = pages[end_page   - 1].size[1]

        output[q] = {
            "pages":   page_list,
            "y_start": int(y_start_frac * img_h_start),
            "y_end":   int(y_end_frac   * img_h_end),
        }

    logger.info(f"  Total questions found: {len(output)}")
    return output


# ---------------------------------------------------------------------------
# Question number post-processing
# ---------------------------------------------------------------------------

# Regex for roman numeral suffixes like ".i", ".ii", ".iii" (not inside parens)
_DOTTED_ROMAN_RE = re.compile(
    r"\.(?P<r>viii|vii|vi|iv|iii|ii|v|i)(?=\.|$)",
    re.IGNORECASE,
)
_DOTTED_ROMAN_MAP = {
    "i": ".1", "ii": ".2", "iii": ".3", "iv": ".4",
    "v": ".5", "vi": ".6", "vii": ".7", "viii": ".8",
}


def _clean_question_number(raw: str) -> str:
    """
    Normalize a raw LLM question number:
      1. Strip leading "Q." or "q." prefix  (e.g. "Q.3" → "3", "Q.6.1" → "6.1").
      2. Apply normalize_question_number (handles (a)/(i) brackets).
      3. Strip a leading '0.' segment the LLM sometimes adds  (e.g. "0.1.1" → "1.1").
      4. Convert dotted roman suffixes  (.i → .1, .ii → .2 etc.).
    """
    result = raw.strip()

    # Step 1: strip leading "Q." or "q." prefix
    result = re.sub(r"^[Qq]\.", "", result)

    # Step 2: bracket normalization
    result = normalize_question_number(result)

    # Step 3: strip leading "0." segment
    if re.match(r"^0\.", result):
        result = result[2:]

    # Step 4: dotted roman suffixes (e.g. "6.i" → "6.1", "6.ii" → "6.2")
    def _replace_roman(m: re.Match) -> str:
        return _DOTTED_ROMAN_MAP.get(m.group("r").lower(), m.group(0))

    result = _DOTTED_ROMAN_RE.sub(_replace_roman, result)

    return result


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class AnswerBBoxExtractor:
    """
    Main class for the POC.  Call extract_from_pdf() to get the target JSON.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Full pipeline: PDF → target bounding box JSON.

        Returns:
            {
                "1.1": {"pages": [1], "y_start": 240, "y_end": 580},
                ...
            }
        """
        pdf_path = str(pdf_path)
        logger.info(f"Processing: {pdf_path}")

        # Step 1: PDF → images
        pages = self._convert_pdf_to_images(pdf_path)
        logger.info(f"  {len(pages)} page(s) converted at {DPI} DPI")

        # Step 2: per-page LLM calls → question spans + pixel coords
        output = get_question_page_spans(self.client, pages)
        return output

    # ------------------------------------------------------------------
    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        poppler_path = os.getenv("POPPLER_PATH", None)
        try:
            return convert_from_path(pdf_path, dpi=DPI, poppler_path=poppler_path)
        except Exception as exc:
            raise RuntimeError(
                f"PDF conversion failed. Ensure Poppler is installed "
                f"(set POPPLER_PATH if on Windows). Error: {exc}"
            )


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _debug_ocr(pdf_path: Path) -> None:
    """Print all OCR words + bounding boxes per page for diagnosis."""
    poppler_path = os.getenv("POPPLER_PATH", None)
    pages = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=poppler_path)
    for page_num, page_img in enumerate(pages, 1):
        print(f"\n=== Page {page_num} ===")
        words = extract_word_boxes(page_img)
        for word, x0, y0, x1, y1 in words:
            print(f"  y={y0:4d}-{y1:4d}  x={x0:4d}-{x1:4d}  '{word}'")
    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="POC: Extract per-question answer bounding boxes from a PDF answer sheet"
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path. Defaults to <pdf_basename>_bbox.json beside the input.",
        default=None,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save page images annotated with the detected bounding boxes.",
    )
    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Print OCR words + coords for every page (useful to diagnose label-matching failures).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    if args.debug_ocr:
        _debug_ocr(pdf_path)
        return

    out_path = Path(args.output) if args.output else pdf_path.with_name(pdf_path.stem + "_bbox.json")

    extractor = AnswerBBoxExtractor()
    result = extractor.extract_from_pdf(str(pdf_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Output written to: {out_path}")
    print(json.dumps(result, indent=2))

    # Always generate annotated visualizations alongside the JSON
    _visualize(pdf_path, result, out_path.parent)


# ---------------------------------------------------------------------------
# Visualization color palette
# ---------------------------------------------------------------------------

_BAND_COLORS = [
    (255, 100, 100, 60),   # red
    (100, 180, 255, 60),   # blue
    (100, 220, 120, 60),   # green
    (255, 200,  80, 60),   # yellow
    (200, 120, 255, 60),   # purple
    (255, 160,  60, 60),   # orange
    ( 80, 220, 220, 60),   # teal
    (255, 130, 190, 60),   # pink
]
_LINE_COLORS = [
    (200,  50,  50),
    ( 30, 130, 220),
    ( 30, 170,  70),
    (210, 160,   0),
    (150,  50, 200),
    (200, 110,  10),
    (  0, 170, 170),
    (200,  50, 130),
]


def _visualize(pdf_path: Path, result: Dict[str, Any], out_dir: Path) -> None:
    """
    Produce annotated JPEG images for every page, showing each answer block as:
      - a semi-transparent coloured band between y_start and y_end
      - thick horizontal lines at y_start (coloured) and y_end (red)
      - bold question-number label near each line
    Saved to <out_dir>/annotated/page_NNN.jpg
    """
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow ImageDraw not available; skipping visualization.")
        return

    vis_dir = out_dir / "annotated"
    vis_dir.mkdir(parents=True, exist_ok=True)

    poppler_path = os.getenv("POPPLER_PATH", None)
    pages = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=poppler_path)

    # Try to load a readable font; fall back to default
    try:
        font       = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        try:
            font       = ImageFont.truetype("DejaVuSans.ttf", 28)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            font = font_small = ImageFont.load_default()

    # Sort questions by first page + y_start so colors are assigned in reading order
    sorted_labels = sorted(
        result.keys(),
        key=lambda q: (result[q]["pages"][0], result[q].get("y_start") or 0),
    )

    # Build per-page annotation list:
    # page_num -> [(label, color_idx, band_top, band_bottom, is_start, is_end)]
    page_annots: Dict[int, list] = {}

    for color_idx, label in enumerate(sorted_labels):
        info  = result[label]
        pg_list = info["pages"]
        y_s   = info.get("y_start")
        y_e   = info.get("y_end")
        cidx  = color_idx % len(_BAND_COLORS)

        if len(pg_list) == 1:
            pg = pg_list[0]
            h  = pages[pg - 1].size[1]
            top = y_s if y_s is not None else 0
            bot = y_e if y_e is not None else h
            page_annots.setdefault(pg, []).append(
                (label, cidx, top, bot, True, True)
            )
        else:
            # First page: y_start → page bottom
            pg = pg_list[0]
            h  = pages[pg - 1].size[1]
            top = y_s if y_s is not None else 0
            page_annots.setdefault(pg, []).append(
                (label, cidx, top, h, True, False)
            )
            # Middle pages: full-page band (no start/end markers)
            for pg in pg_list[1:-1]:
                h = pages[pg - 1].size[1]
                page_annots.setdefault(pg, []).append(
                    (label, cidx, 0, h, False, False)
                )
            # Last page: page top → y_end
            pg = pg_list[-1]
            h  = pages[pg - 1].size[1]
            bot = y_e if y_e is not None else h
            page_annots.setdefault(pg, []).append(
                (label, cidx, 0, bot, False, True)
            )

    annotated_count = 0
    for page_num, page_img in enumerate(pages, 1):
        vis_path = vis_dir / f"page_{page_num:03d}.jpg"

        if page_num not in page_annots:
            page_img.save(str(vis_path), "JPEG", quality=90)
            continue

        # Composite semi-transparent bands
        overlay  = Image.new("RGBA", page_img.size, (0, 0, 0, 0))
        ov_draw  = ImageDraw.Draw(overlay)

        for label, cidx, top, bot, is_start, is_end in page_annots[page_num]:
            r, g, b, a = _BAND_COLORS[cidx]
            ov_draw.rectangle([(0, top), (page_img.width, bot)], fill=(r, g, b, a))

        base     = page_img.convert("RGBA")
        combined = Image.alpha_composite(base, overlay).convert("RGB")
        draw     = ImageDraw.Draw(combined)

        margin = 6
        for label, cidx, top, bot, is_start, is_end in page_annots[page_num]:
            lc = _LINE_COLORS[cidx]
            if is_start:
                draw.line([(0, top), (combined.width, top)], fill=lc, width=4)
                # White pill behind text for readability
                draw.rectangle(
                    [(margin, top + margin), (margin + 220, top + margin + 34)],
                    fill=(255, 255, 255),
                )
                draw.text((margin + 6, top + margin + 2), f"▼ {label}", fill=lc, font=font)
            if is_end:
                draw.line([(0, bot), (combined.width, bot)], fill=(190, 0, 0), width=3)
                draw.rectangle(
                    [(margin, bot - 38), (margin + 260, bot - 4)],
                    fill=(255, 255, 255),
                )
                draw.text((margin + 6, bot - 36), f"▲ {label} end", fill=(190, 0, 0),
                          font=font_small)

        combined.save(str(vis_path), "JPEG", quality=90)
        annotated_count += 1

    print(f"Annotated images ({annotated_count} pages with annotations): {vis_dir}/")


if __name__ == "__main__":
    main()
