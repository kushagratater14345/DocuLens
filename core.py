"""
core.py - DocuLens AI core engine

Contains:
- Tesseract auto-detection
- Image preprocessing
- OCR (image + PDF)
- Text cleaning
- Document classification (keyword based)
- Structured field extraction
- Currency normalization
- Confidence scoring
- Risk / anomaly detection
- Duplicate detection helpers
- Embedding generation (SentenceTransformer)
- ChromaDB document store wrapper
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_FOLDER / "screenshots"
DB_FOLDER = DATA_FOLDER / "db"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

# ----------------------------------------------------------------------------
# TESSERACT AUTO-DETECTION
# ----------------------------------------------------------------------------


class TesseractNotFoundError(Exception):
    pass


def find_tesseract() -> Optional[str]:
    """
    Try to locate the tesseract binary automatically.
    Checks (in order):
      1. Whatever is already configured in pytesseract
      2. shutil.which('tesseract')  (covers PATH-based installs)
      3. Common macOS Homebrew locations (Apple Silicon + Intel)
      4. Common Linux locations
    """
    candidates = []

    configured = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
    if configured:
        candidates.append(configured)

    which_result = shutil.which("tesseract")
    if which_result:
        candidates.append(which_result)

    candidates.extend(
        [
            "/opt/homebrew/bin/tesseract",  # macOS Apple Silicon
            "/usr/local/bin/tesseract",  # macOS Intel
            "/usr/bin/tesseract",  # Linux
            "/usr/local/opt/tesseract/bin/tesseract",
        ]
    )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return None


def ensure_tesseract_configured() -> str:
    path = find_tesseract()
    if not path:
        raise TesseractNotFoundError(
            "Tesseract OCR was not found on this system.\n\n"
            "Fix it with:\n"
            "  brew install tesseract\n\n"
            "If it's already installed, find its path with `which tesseract` "
            "and set it manually in core.py via:\n"
            "  pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'"
        )
    return path


# ----------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------------------------------


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Improves OCR accuracy via grayscale, resizing, denoising,
    contrast enhancement and adaptive thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small images - Tesseract performs better on larger text
    h, w = gray.shape[:2]
    if max(h, w) < 1600:
        scale = 1600 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive thresholding for uneven lighting/scans
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return thresh


def load_image_any(path: Path) -> np.ndarray:
    """Load an image file into an OpenCV BGR array, handling PIL fallback for exotic formats."""
    img = cv2.imread(str(path))
    if img is None:
        # Fallback via PIL (handles some formats cv2 struggles with)
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------


def ocr_image_array(img: np.ndarray) -> str:
    ensure_tesseract_configured()
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    if not text.strip():
        # Retry on raw grayscale without adaptive threshold - sometimes helps
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang="eng")
    return text


def ocr_file(path: Path) -> str:
    """
    OCR an image or PDF file. Raises clear exceptions on failure.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        return ocr_pdf(path)

    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported file extension: {ext}")

    img = load_image_any(path)
    if img is None:
        raise ValueError("Could not read the uploaded image file. It may be corrupted.")

    text = ocr_image_array(img)
    if not text.strip():
        raise ValueError(
            "OCR completed but no text was detected. Try a clearer, higher-resolution image."
        )
    return text


def ocr_pdf(path: Path) -> str:
    """
    Converts each page of a PDF into an image and OCRs it, combining the results.
    Requires poppler (via pdf2image). Gives a clear error if poppler is missing.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise RuntimeError(
            "pdf2image is not installed. Run: pip install pdf2image"
        ) from e

    try:
        pages = convert_from_path(str(path), dpi=300)
    except Exception as e:
        raise RuntimeError(
            "Failed to convert PDF to images. This usually means Poppler is not "
            "installed.\n\nFix it with:\n  brew install poppler\n\n"
            f"Original error: {e}"
        ) from e

    if not pages:
        raise ValueError("The PDF appears to have no pages.")

    all_text = []
    for i, page in enumerate(pages):
        arr = cv2.cvtColor(np.array(page.convert("RGB")), cv2.COLOR_RGB2BGR)
        page_text = ocr_image_array(arr)
        if page_text.strip():
            all_text.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

    combined = "\n\n".join(all_text)
    if not combined.strip():
        raise ValueError("OCR completed but no text was detected in the PDF.")
    return combined


# ----------------------------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------------------------


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common OCR currency mis-reads: %, S, or 3 mistaken for currency symbols
    # when immediately followed by digits (heuristic, applied cautiously)
    text = re.sub(r"(?<![A-Za-z0-9])%(?=\d)", "₹", text)
    # Strip trailing spaces per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def get_meaningful_lines(text: str) -> List[str]:
    lines = [l.strip() for l in text.split("\n")]
    return [l for l in lines if len(l) >= 2]


# ----------------------------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------------------------

CLASSIFICATION_KEYWORDS: Dict[str, List[str]] = {
    "invoice": [
        "invoice", "tax invoice", "invoice number", "invoice no", "subtotal",
        "total amount", "gstin", "bill to", "ship to", "due date",
    ],
    "receipt": [
        "receipt", "paid", "transaction", "cash", "change due", "change",
        "thank you for your purchase", "cashier",
    ],
    "bank_statement": [
        "account number", "balance", "statement", "debit", "credit",
        "opening balance", "closing balance", "ifsc", "account holder",
    ],
    "identity_document": [
        "date of birth", "identity", "passport", "national id", "aadhaar",
        "driving licence", "driving license", "permanent account number", "gender",
    ],
    "certificate": [
        "certificate", "awarded", "completion", "hereby certify", "certify that",
        "achievement", "participation",
    ],
    "bill": [
        "electricity bill", "utility bill", "water bill", "bill amount",
        "due amount", "meter reading", "billing period",
    ],
}


def classify_document(text: str) -> Tuple[str, float]:
    """
    Keyword/feature based classification.
    Returns (document_type, confidence 0-1).
    """
    lowered = text.lower()
    scores: Dict[str, int] = {}

    for doc_type, keywords in CLASSIFICATION_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in lowered:
                # Weight multi-word / more specific keywords higher
                score += 2 if " " in kw else 1
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return "unknown", 0.3

    total_possible = len(CLASSIFICATION_KEYWORDS[best_type]) * 2
    raw_confidence = best_score / total_possible

    # Penalize if another category scored nearly as high (ambiguous document)
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    if second_best > 0 and second_best >= best_score * 0.8:
        raw_confidence *= 0.8

    confidence = min(0.98, max(0.35, raw_confidence + 0.35))
    return best_type, round(confidence, 2)


# ----------------------------------------------------------------------------
# CURRENCY / NUMBER NORMALIZATION
# ----------------------------------------------------------------------------

CURRENCY_SYMBOLS = {
    "₹": "INR", "rs.": "INR", "rs": "INR", "inr": "INR",
    "$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
}


def normalize_amount(raw: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Takes a raw matched amount string like '₹4,599' or '$1,234.50' and
    returns (amount_as_float, currency_code).
    """
    if not raw:
        return None, None

    raw = raw.strip()
    currency = None

    for symbol, code in CURRENCY_SYMBOLS.items():
        if raw.lower().startswith(symbol) or symbol in raw.lower():
            currency = code
            break

    number_part = re.sub(r"[^\d.]", "", raw)
    if not number_part:
        return None, currency

    try:
        amount = float(number_part)
    except ValueError:
        return None, currency

    return amount, currency


AMOUNT_PATTERN = re.compile(
    r"(₹|\$|€|£|rs\.?|inr|usd|eur|gbp)\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE
)


def find_amount_near(text: str, keywords: List[str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Finds a currency amount on a line containing any of the given keywords,
    or on the following line (common OCR layout pattern).

    Keywords are matched with word boundaries (checked in priority order, i.e.
    the order given in `keywords`) so that e.g. "total" does not incorrectly
    match inside "subtotal".
    """
    lines = get_meaningful_lines(text)

    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for i, line in enumerate(lines):
            if pattern.search(line):
                match = AMOUNT_PATTERN.search(line)
                if not match and i + 1 < len(lines):
                    match = AMOUNT_PATTERN.search(lines[i + 1])
                if match:
                    return normalize_amount(match.group(0))
    return None, None


def find_any_amounts(text: str) -> List[Tuple[float, Optional[str]]]:
    results = []
    for match in AMOUNT_PATTERN.finditer(text):
        amt, cur = normalize_amount(match.group(0))
        if amt is not None:
            results.append((amt, cur))
    return results


DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"
)


def find_date_near(text: str, keywords: List[str], fallback_to_any: bool = True) -> Optional[str]:
    lines = get_meaningful_lines(text)

    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        for i, line in enumerate(lines):
            if pattern.search(line):
                match = DATE_PATTERN.search(lines[i])
                if not match and i + 1 < len(lines):
                    match = DATE_PATTERN.search(lines[i + 1])
                if match:
                    return match.group(0)

    if not fallback_to_any:
        return None

    # Fallback: first date found anywhere (only used for the primary "date" field)
    match = DATE_PATTERN.search(text)
    return match.group(0) if match else None


# ----------------------------------------------------------------------------
# VENDOR EXTRACTION
# ----------------------------------------------------------------------------

GENERIC_HEADER_WORDS = {
    "invoice", "invoice number", "invoice no", "date", "receipt",
    "bill", "total", "subtotal", "tax invoice", "amount", "gst",
    "payment", "statement", "certificate", "bill to", "ship to",
}


def extract_vendor(text: str) -> Optional[str]:
    lines = get_meaningful_lines(text)
    if not lines:
        return None

    # Look at the first 5 lines - vendor/store names are usually at the top
    for line in lines[:5]:
        candidate = line.strip(" :-|")
        lowered = candidate.lower()

        if not candidate:
            continue
        if lowered in GENERIC_HEADER_WORDS:
            continue
        if any(lowered.startswith(w) for w in ["invoice", "date", "receipt no", "gstin", "bill no"]):
            continue
        if DATE_PATTERN.search(candidate):
            continue
        if AMOUNT_PATTERN.search(candidate):
            continue
        if re.fullmatch(r"[\d\W]+", candidate):
            continue

        # Strip trailing generic words e.g. "Amazon Invoice" -> "Amazon"
        cleaned = re.sub(
            r"\b(invoice|receipt|bill|statement|tax invoice)\b",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip(" :-|")

        if cleaned and len(cleaned) >= 2:
            return cleaned

    return None


# ----------------------------------------------------------------------------
# FIELD EXTRACTION PER DOCUMENT TYPE
# ----------------------------------------------------------------------------


def extract_invoice_number(text: str) -> Optional[str]:
    """
    Searches line-by-line for an explicit "Invoice Number / No / #" label so
    that a bare mention of the word "Invoice" (e.g. in a title like
    "Amazon Invoice") is never mistaken for the field label.
    """
    lines = get_meaningful_lines(text)
    pattern = re.compile(
        r"invoice\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,20})",
        re.IGNORECASE,
    )
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in {"number", "no", "date", "invoice"}:
                return candidate
            # Label was on its own line with the value on the next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not DATE_PATTERN.search(next_line):
                    return next_line
    return None


def extract_transaction_id(text: str) -> Optional[str]:
    match = re.search(
        r"(transaction\s*(?:id|no|#)?\s*[:\-]?\s*)([A-Za-z0-9\-\/]{4,25})",
        text,
        re.IGNORECASE,
    )
    return match.group(2).strip() if match else None


def extract_account_number(text: str) -> Optional[str]:
    match = re.search(
        r"(account\s*(?:number|no|#)?\s*[:\-]?\s*)([A-Za-z0-9\-\*]{4,25})",
        text,
        re.IGNORECASE,
    )
    return match.group(2).strip() if match else None


def extract_payment_status(text: str) -> Optional[str]:
    lowered = text.lower()
    if "paid" in lowered:
        return "paid"
    if "unpaid" in lowered or "due" in lowered or "pending" in lowered:
        return "unpaid"
    return None


def extract_payment_method(text: str) -> Optional[str]:
    for method in ["cash", "credit card", "debit card", "upi", "card", "net banking", "wallet"]:
        if method in text.lower():
            return method.title()
    return None


def extract_fields(text: str, doc_type: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}

    if doc_type == "invoice":
        fields["vendor"] = extract_vendor(text)
        fields["invoice_number"] = extract_invoice_number(text)
        fields["date"] = find_date_near(text, ["date", "invoice date"])
        fields["due_date"] = find_date_near(
            text, ["due date", "payment due"], fallback_to_any=False
        )

        subtotal, sub_cur = find_amount_near(text, ["subtotal", "sub total"])
        gst, gst_cur = find_amount_near(text, ["gst", "tax", "vat"])
        total, total_cur = find_amount_near(text, ["total amount", "grand total", "total"])

        fields["subtotal"] = subtotal
        fields["gst"] = gst
        fields["tax"] = gst
        fields["total_amount"] = total
        fields["currency"] = total_cur or gst_cur or sub_cur or "INR"
        fields["payment_status"] = extract_payment_status(text)

    elif doc_type == "receipt":
        fields["vendor"] = extract_vendor(text)
        fields["date"] = find_date_near(text, ["date"])
        total, total_cur = find_amount_near(text, ["total", "amount paid", "grand total"])
        fields["total_amount"] = total
        fields["currency"] = total_cur or "INR"
        fields["payment_method"] = extract_payment_method(text)
        fields["transaction_id"] = extract_transaction_id(text)

    elif doc_type == "bank_statement":
        fields["bank_name"] = extract_vendor(text)
        fields["account_number"] = extract_account_number(text)
        start_date = find_date_near(
            text, ["from", "period start", "statement period"], fallback_to_any=False
        )
        end_date = find_date_near(text, ["to", "period end"], fallback_to_any=False)
        fields["statement_period"] = (
            f"{start_date} - {end_date}" if start_date and end_date else start_date
        )
        opening, open_cur = find_amount_near(text, ["opening balance"])
        closing, close_cur = find_amount_near(text, ["closing balance"])
        fields["opening_balance"] = opening
        fields["closing_balance"] = closing
        fields["currency"] = close_cur or open_cur or "INR"

    elif doc_type == "certificate":
        fields["organization"] = extract_vendor(text)
        fields["recipient"] = None
        recipient_match = re.search(
            r"(?:awarded to|presented to|this is to certify that)\s+([A-Za-z .]{3,50})",
            text,
            re.IGNORECASE,
        )
        if recipient_match:
            fields["recipient"] = recipient_match.group(1).strip()
        fields["certificate_type"] = None
        for ctype in ["completion", "participation", "achievement", "excellence", "merit"]:
            if ctype in text.lower():
                fields["certificate_type"] = ctype.title()
                break
        fields["date"] = find_date_near(text, ["date"])

    else:  # unknown / bill / identity_document fallback
        fields["vendor"] = extract_vendor(text)
        fields["date"] = find_date_near(text, ["date"])
        total, total_cur = find_amount_near(text, ["total", "amount", "bill amount"])
        fields["total_amount"] = total
        fields["currency"] = total_cur or "INR"

    # Never return the string "None"
    for k, v in list(fields.items()):
        if isinstance(v, str) and v.strip().lower() == "none":
            fields[k] = None

    return fields


# ----------------------------------------------------------------------------
# CONFIDENCE ENGINE
# ----------------------------------------------------------------------------

IMPORTANT_FIELDS_BY_TYPE = {
    "invoice": ["vendor", "invoice_number", "date", "total_amount", "gst"],
    "receipt": ["vendor", "date", "total_amount"],
    "bank_statement": ["bank_name", "account_number", "closing_balance"],
    "certificate": ["organization", "recipient", "certificate_type"],
}


def calculate_confidence(
    doc_type: str,
    classification_confidence: float,
    fields: Dict[str, Any],
    ocr_text_length: int,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Returns (confidence, status, breakdown) where breakdown explains exactly
    how the score was built, e.g. for an explainability panel in the UI.
    """
    important_fields = IMPORTANT_FIELDS_BY_TYPE.get(doc_type, list(fields.keys()))

    found_fields = [f for f in important_fields if fields.get(f) not in (None, "", [])]
    missing_fields = [f for f in important_fields if f not in found_fields]
    field_score = len(found_fields) / len(important_fields) if important_fields else 0.5

    # OCR quality proxy: more extracted text generally means cleaner OCR
    ocr_quality = min(1.0, ocr_text_length / 300)

    classification_component = round(classification_confidence * 0.35, 3)
    field_component = round(field_score * 0.5, 3)
    ocr_component = round(ocr_quality * 0.15, 3)

    confidence = classification_component + field_component + ocr_component
    confidence = round(min(0.99, max(0.05, confidence)), 2)

    if confidence >= 0.8:
        status = "HIGH"
    elif confidence >= 0.55:
        status = "MEDIUM"
    else:
        status = "LOW"

    breakdown = {
        "classification_contribution": classification_component,
        "field_coverage_contribution": field_component,
        "ocr_quality_contribution": ocr_component,
        "fields_found": found_fields,
        "fields_missing": missing_fields,
        "explanation": (
            f"Classification confidence contributed {classification_component}, "
            f"{len(found_fields)}/{len(important_fields)} key fields were found "
            f"(+{field_component}), and OCR text quality added +{ocr_component}."
        ),
    }

    return confidence, status, breakdown


# ----------------------------------------------------------------------------
# RISK / ANOMALY DETECTION
# ----------------------------------------------------------------------------


def detect_risk(
    doc_type: str,
    fields: Dict[str, Any],
    existing_documents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    flags: List[str] = []
    flag_points: List[Dict[str, Any]] = []
    risk_points = 0

    def flag_it(message: str, points: int):
        nonlocal risk_points
        flags.append(message)
        flag_points.append({"flag": message, "points": points})
        risk_points += points

    if doc_type == "invoice":
        subtotal = fields.get("subtotal")
        gst = fields.get("gst")
        total = fields.get("total_amount")

        if total is None:
            flag_it("Total amount missing", 2)
        if fields.get("invoice_number") is None:
            flag_it("Invoice number missing", 1)
        if fields.get("date") is None:
            flag_it("Date missing", 1)

        if gst is not None and total is not None and gst > total:
            flag_it("GST is greater than the total amount", 3)

        if subtotal is not None and gst is not None and total is not None:
            expected_total = round(subtotal + gst, 2)
            if abs(expected_total - total) > max(1.0, total * 0.02):
                flag_it("Amount mismatch detected (subtotal + GST ≠ total)", 3)

        if gst is not None and subtotal not in (None, 0) and gst > subtotal * 0.5:
            flag_it("GST unusually high compared with subtotal", 2)

        for amount_field in ("subtotal", "gst", "total_amount"):
            val = fields.get(amount_field)
            if val is not None and val < 0:
                flag_it(f"Negative amount detected in {amount_field}", 3)

    else:
        total = fields.get("total_amount")
        if total is not None and total < 0:
            flag_it("Negative amount detected", 3)

    # Duplicate detection against existing documents
    if existing_documents:
        for doc in existing_documents:
            same_vendor = doc.get("vendor") and doc.get("vendor") == fields.get("vendor")
            same_invoice_no = (
                doc.get("invoice_number")
                and doc.get("invoice_number") == fields.get("invoice_number")
            )
            same_amount = (
                doc.get("total_amount") is not None
                and fields.get("total_amount") is not None
                and abs(doc.get("total_amount") - fields.get("total_amount")) < 0.01
            )
            if same_vendor and same_invoice_no and same_amount:
                flag_it("Potential duplicate invoice detected", 4)
                break

    risk_score = round(min(1.0, risk_points / 10), 2)

    if risk_score >= 0.6:
        risk_level = "HIGH"
    elif risk_score >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "flags": flags,
        "flag_breakdown": flag_points,
    }


# ----------------------------------------------------------------------------
# FILE HASHING (duplicate file detection)
# ----------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ----------------------------------------------------------------------------
# EMBEDDING MODEL
# ----------------------------------------------------------------------------

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def build_searchable_text(doc_type: str, fields: Dict[str, Any], ocr_text: str) -> str:
    """
    Builds a rich text blob for embedding that captures both raw content
    and structured metadata, improving semantic search quality.
    """
    parts = [f"Document type: {doc_type}"]
    for key, value in fields.items():
        if value not in (None, "", []):
            parts.append(f"{key.replace('_', ' ')}: {value}")
    parts.append(ocr_text[:1500])
    return "\n".join(parts)


def get_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ----------------------------------------------------------------------------
# AUTO EXPENSE CATEGORIZATION (embedding similarity, no training required)
# ----------------------------------------------------------------------------

# Each category is defined by a short natural-language description. A new
# document is categorized by comparing its embedding against these
# descriptions with cosine similarity and picking the closest match.
EXPENSE_CATEGORIES: Dict[str, str] = {
    "Groceries & Food": "grocery store supermarket food restaurant cafe dining snacks",
    "Travel & Transport": "flight train taxi cab fuel petrol diesel travel booking transport ride",
    "Utilities & Bills": "electricity water gas internet phone utility bill recharge broadband",
    "Shopping & Electronics": "amazon flipkart online shopping electronics gadgets clothing retail purchase",
    "Health & Medical": "hospital pharmacy medicine doctor clinic medical health insurance",
    "Banking & Finance": "bank statement account balance loan emi credit debit transaction",
    "Education": "school college university course tuition fee certificate exam",
    "Housing & Rent": "rent lease apartment maintenance housing property",
    "Other": "miscellaneous general document other expense",
}

_category_embeddings: Optional[Dict[str, List[float]]] = None


def _get_category_embeddings() -> Dict[str, List[float]]:
    global _category_embeddings
    if _category_embeddings is None:
        model = get_embedding_model()
        descriptions = list(EXPENSE_CATEGORIES.values())
        vectors = model.encode(descriptions, normalize_embeddings=True)
        _category_embeddings = {
            name: vec.tolist() for name, vec in zip(EXPENSE_CATEGORIES.keys(), vectors)
        }
    return _category_embeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)) or 1e-8
    return float(np.dot(a_arr, b_arr) / denom)


def categorize_document(document_embedding: List[float]) -> Tuple[str, float]:
    """
    Assigns an expense category by comparing the document's own embedding
    against a fixed set of category description embeddings. No training,
    no extra model - reuses the same SentenceTransformer already loaded
    for semantic search.
    """
    category_vectors = _get_category_embeddings()
    best_category = "Other"
    best_score = -1.0

    for name, vector in category_vectors.items():
        score = _cosine_similarity(document_embedding, vector)
        if score > best_score:
            best_score = score
            best_category = name

    return best_category, round(max(0.0, best_score), 3)


# ----------------------------------------------------------------------------
# ASK DOCULENS - LOCAL RAG QUESTION ANSWERING
# ----------------------------------------------------------------------------

_qa_model = None
_qa_tokenizer = None
QA_MODEL_NAME = "google/flan-t5-small"


def get_qa_model():
    """
    Lazily loads a small local sequence-to-sequence model for answering
    questions grounded in retrieved document context. Runs entirely on CPU,
    no API key, no internet after the first download.
    """
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        _qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_NAME)
    return _qa_model, _qa_tokenizer


def build_qa_context(search_results: List[Dict[str, Any]], max_docs: int = 5) -> str:
    """
    Turns ChromaDB search results into a compact context block the QA model
    can reason over: one line per document with its key structured fields.
    """
    lines = []
    for i, r in enumerate(search_results[:max_docs]):
        meta = r.get("metadata", {})
        parts = [f"Document {i + 1}:"]
        for key in (
            "document_type", "vendor", "bank_name", "organization",
            "invoice_number", "date", "total_amount", "gst", "currency",
            "risk_level", "category",
        ):
            val = meta.get(key)
            if val not in (None, "", "None"):
                parts.append(f"{key}={val}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def answer_question(question: str, context: str) -> str:
    """
    Generates a natural-language answer to `question` grounded strictly in
    `context` (retrieved document metadata). If the model is unavailable,
    raises so the caller can fall back gracefully.
    """
    model, tokenizer = get_qa_model()

    prompt = (
        "Answer the question using only the information in the context. "
        "If the answer isn't in the context, say you don't have that information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=128)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


# ----------------------------------------------------------------------------
# CHROMADB STORE
# ----------------------------------------------------------------------------


class DocumentStore:
    """Wraps a persistent ChromaDB collection for DocuLens documents."""

    COLLECTION_NAME = "doculens_documents"

    def __init__(self, db_path: Path = DB_FOLDER):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB metadata values must be str, int, float, or bool - never None/list."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                clean[k] = ""
            elif isinstance(v, (list, dict)):
                clean[k] = str(v)
            else:
                clean[k] = v
        return clean

    def add_document(
        self,
        doc_id: str,
        embedding: List[float],
        document_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[document_text],
            metadatas=[self._sanitize_metadata(metadata)],
        )

    def search(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_embedding = get_embedding(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, n_results),
        )

        output = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            output.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": round(float(distances[i]), 4),
                }
            )
        return output

    def get_all(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        output = []
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        for i in range(len(ids)):
            output.append(
                {
                    "id": ids[i],
                    "metadata": metadatas[i],
                    "document": documents[i] if i < len(documents) else "",
                }
            )
        # Newest first (metadata stores an ISO timestamp under 'processed_at')
        output.sort(key=lambda d: d["metadata"].get("processed_at", ""), reverse=True)
        return output

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(ids=[doc_id])
        ids = results.get("ids", [])
        if not ids:
            return None
        return {
            "id": ids[0],
            "metadata": results["metadatas"][0],
            "document": results["documents"][0],
        }

    def delete(self, doc_id: str) -> bool:
        existing = self.get_by_id(doc_id)
        if not existing:
            return False
        self.collection.delete(ids=[doc_id])
        return True

    def find_related(
        self,
        doc_id: str,
        embedding: List[float],
        min_distance: float = 0.08,
        max_distance: float = 0.45,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Finds documents semantically close to `embedding` but excludes the
        document itself and near-identical matches (which are more likely
        duplicates than "related" documents). Used to surface e.g. a receipt
        and an invoice for the same purchase, or recurring monthly bills.
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit + 3,
            )
        except Exception:
            return []

        related = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            if ids[i] == doc_id:
                continue
            dist = float(distances[i])
            if min_distance <= dist <= max_distance:
                related.append(
                    {
                        "id": ids[i],
                        "metadata": metadatas[i],
                        "distance": round(dist, 4),
                    }
                )
            if len(related) >= limit:
                break

        return related

    def compute_stats(self) -> Dict[str, Any]:
        docs = self.get_all()

        total_docs = len(docs)
        total_spending = 0.0
        total_gst = 0.0
        high_risk_count = 0
        by_vendor: Dict[str, float] = {}
        by_type: Dict[str, int] = {}
        by_risk: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        by_category: Dict[str, float] = {}
        by_date: Dict[str, float] = {}

        for doc in docs:
            meta = doc["metadata"]

            amount = _to_float(meta.get("total_amount"))
            gst = _to_float(meta.get("gst"))

            if amount:
                total_spending += amount
            if gst:
                total_gst += gst

            risk_level = meta.get("risk_level", "LOW") or "LOW"
            by_risk[risk_level] = by_risk.get(risk_level, 0) + 1
            if risk_level == "HIGH":
                high_risk_count += 1

            doc_type = meta.get("document_type", "unknown") or "unknown"
            by_type[doc_type] = by_type.get(doc_type, 0) + 1

            vendor = meta.get("vendor")
            if vendor:
                by_vendor[vendor] = by_vendor.get(vendor, 0.0) + (amount or 0.0)

            category = meta.get("category") or "Other"
            by_category[category] = by_category.get(category, 0.0) + (amount or 0.0)

            processed_at = meta.get("processed_at", "")
            date_key = processed_at[:10] if processed_at else "unknown"
            by_date[date_key] = by_date.get(date_key, 0.0) + (amount or 0.0)

        spending_trend = [
            {"date": d, "amount": round(a, 2)}
            for d, a in sorted(by_date.items())
            if d != "unknown"
        ]

        return {
            "documents": total_docs,
            "total_spending": round(total_spending, 2),
            "total_gst": round(total_gst, 2),
            "high_risk": high_risk_count,
            "by_vendor": by_vendor,
            "by_type": by_type,
            "by_risk": by_risk,
            "by_category": {k: round(v, 2) for k, v in by_category.items()},
            "spending_trend": spending_trend,
        }


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
