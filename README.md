# DocuLens AI

**AI-powered Document Intelligence, Understanding & Semantic Search**

DocuLens AI is a locally-running document analysis platform. Upload an invoice,
receipt, bank statement, or certificate (PNG/JPG/PDF) and it will OCR the
document, classify it, extract structured fields, score a confidence level,
flag risky/anomalous documents, and make it semantically searchable — all with
free, open-source tools, no cloud APIs required.

---

## Features

- **OCR** — Tesseract + OpenCV preprocessing (denoise, CLAHE contrast, adaptive
  threshold) for accurate text extraction from images and PDFs.
- **Document classification** — keyword/feature-based classifier for invoice,
  receipt, bank statement, identity document, certificate, bill, unknown.
- **Structured extraction** — vendor, invoice number, date, totals, GST, and
  type-specific fields, with currency normalization (₹ / $ / € / £).
- **Confidence engine** — combines classification confidence, extracted-field
  coverage, and OCR quality into a HIGH / MEDIUM / LOW score.
- **Risk & anomaly detection** — rule-based checks: missing fields, GST > total,
  subtotal + GST ≠ total, negative amounts, duplicate invoices.
- **Semantic search** — SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
  stored in ChromaDB, so you can search in natural language
  ("show me Amazon invoices", "find expensive purchases").
- **Analytics dashboard** — total documents, spending, GST, high-risk count,
  document-type distribution.
- **Document history** — every processed document, newest first, with a detail
  view.

### New: differentiator features

- **Ask DocuLens (local RAG Q&A)** — ask plain-English questions ("How much
  did I spend on Amazon?") and get a generated answer grounded in your
  retrieved documents, using a small local model
  (`google/flan-t5-small`, ~300MB, CPU-only, no API key).
- **Auto expense categorization** — every document is automatically sorted
  into a spending category (Groceries, Travel, Utilities, Shopping,
  Health, Banking, Education, Housing, Other) using embedding similarity
  against category descriptions — no training required, reuses the same
  embedding model already loaded for search.
- **Explainability panel** — a "why this score?" breakdown showing exactly
  how the confidence score and risk score were computed, point by point.
- **Related document linking** — documents that are semantically similar
  (but not exact duplicates) are surfaced automatically, e.g. a receipt and
  an invoice for the same purchase.
- **Spending trend chart** — a lightweight SVG bar chart of spending by day,
  built with no external charting library.
- **Voice search** — a mic button next to search uses the browser's built-in
  Web Speech API to transcribe spoken queries (Chrome/Edge/Safari; no
  backend involved, works offline once the page is loaded).
- **Folder auto-ingest (`watch_folder.py`)** — watches a folder and
  automatically uploads any new PNG/JPG/JPEG/PDF dropped into it.
- **Export reports** — one-click download of everything as an `.xlsx`
  workbook or a formatted `.pdf` summary.

---

## Architecture

```
Doculens/
├── app.py              FastAPI application + routes
├── core.py              OCR, classification, extraction, risk, categorization, QA, embeddings, ChromaDB
├── ingest.py             Pipeline orchestration (upload -> stored result)
├── reports.py            Excel / PDF report generation
├── watch_folder.py       Optional folder-watcher for auto-ingest
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── screenshots/     Uploaded files are saved here
│   ├── db/              ChromaDB persistent storage
│   ├── reports/          Generated Excel/PDF exports
│   └── watch_inbox/      Default folder watched by watch_folder.py
│
└── static/
    ├── index.html        Dashboard UI
    ├── style.css
    └── app.js
```

**Pipeline:** Upload → Validate → Preprocess → OCR → Clean → Classify →
Extract → Confidence → Risk → Embed → Store in ChromaDB → JSON response →
Dashboard render.

All paths are built from `BASE_DIR = Path(__file__).resolve().parent`, so the
app works no matter what directory you launch it from.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| OCR | Tesseract, pytesseract, OpenCV, Pillow |
| AI / NLP | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| Frontend | HTML, CSS, vanilla JavaScript |
| PDF support | pdf2image (requires Poppler) |

No paid APIs. No OpenAI. No cloud OCR. Everything runs on your machine.

---

## ⚠️ A note on Python version (macOS)

Heavy ML packages (`sentence-transformers` → PyTorch, `chromadb`, `opencv-python`)
frequently lag behind the newest Python release with prebuilt wheels. **If you
are on Python 3.14 and `pip install` fails or tries to compile from source,**
create your virtual environment with **Python 3.11 or 3.12** instead — this is
the safest, most compatible choice today:

```bash
brew install python@3.12
python3.12 -m venv .venv
```

Everything else in this guide is identical either way.

---

## Installation

### 1. Install Tesseract (OCR engine)

You mentioned Tesseract is already installed — verify with:

```bash
which tesseract
```

If that returns nothing, install it:

```bash
brew install tesseract
```

DocuLens auto-detects Tesseract at `/opt/homebrew/bin/tesseract` (Apple
Silicon), `/usr/local/bin/tesseract` (Intel), or wherever `which tesseract`
points — no manual path configuration needed.

### 2. Install Poppler (required for PDF support)

```bash
brew install poppler
```

### 3. Set up the project

```bash
cd ~/Desktop/Doculens

# create the venv (use python3.12 if you hit install issues, see note above)
python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The first run will also download the `all-MiniLM-L6-v2` embedding model
(~90MB) from Hugging Face the first time `sentence-transformers` is used —
this requires an internet connection once, then works offline.

---

## Running the Application

```bash
source .venv/bin/activate
python -m uvicorn app:app --reload --port 8000
```

Open: **http://127.0.0.1:8000**

If port 8000 is already in use:

```bash
# find and kill whatever is using it
lsof -i :8000
kill -9 <PID>

# or just use a different port
python -m uvicorn app:app --reload --port 8001
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves the dashboard |
| POST | `/upload` | Upload + process a document (multipart `file`) |
| GET | `/search?q=...` | Semantic search over stored documents |
| GET | `/ask?q=...` | Ask a natural-language question (local RAG Q&A) |
| GET | `/documents` | List all documents, newest first |
| GET | `/documents/{id}` | Full detail for one document |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/stats` | Analytics (spending, GST, risk, type/category distribution, trend) |
| GET | `/export/excel` | Download an `.xlsx` report of all documents + stats |
| GET | `/export/pdf` | Download a `.pdf` report of all documents + stats |
| GET | `/health` | System health (OCR, database, embedding model, QA model) |

Example health response:

```json
{
  "status": "ok",
  "ocr": true,
  "ocr_path": "/opt/homebrew/bin/tesseract",
  "database": true,
  "embedding_model": true
}
```

---

## How OCR works

Images go through grayscale conversion, upscaling (if small), denoising,
CLAHE contrast enhancement, and adaptive thresholding before being passed to
Tesseract — this significantly improves accuracy on phone-camera photos and
low-quality scans. PDFs are rendered page-by-page into images (via
`pdf2image`/Poppler at 300 DPI) and OCR'd the same way, then combined.

## How semantic search works

Every processed document is converted into a rich text blob (document type +
extracted fields + OCR text) and embedded with `all-MiniLM-L6-v2` into a
384-dimensional vector, stored in a persistent ChromaDB collection
(`doculens_documents`) at `data/db/`. Searches embed your query the same way
and return the closest documents by cosine distance — so "expensive
purchases" can match a ₹4,599 invoice even without that exact wording.

## How risk detection works

Rule-based checks run on every extracted document: missing critical fields,
GST exceeding the total, subtotal + GST not matching the total (beyond a small
tolerance), negative amounts, and duplicate invoices (same vendor + invoice
number + amount as an existing document). Each triggered rule adds risk
points; the total maps to a LOW / MEDIUM / HIGH risk level and a 0–1 risk
score. Every triggered rule and its point value is returned in
`risk.flag_breakdown`, which powers the explainability panel in the UI.

## How Ask DocuLens works (local RAG)

`/ask` runs the same retrieval step as `/search` — embed the question, find
the closest stored documents in ChromaDB — then hands the retrieved
documents' structured fields to a small local sequence-to-sequence model
(`google/flan-t5-small`) as context, and asks it to answer using only that
context. This is the standard retrieve-then-generate ("RAG") pattern, just
sized for a laptop CPU instead of a cloud GPU. The model downloads once
(~300MB) on first use and then runs fully offline.

## How auto expense categorization works

Each document's own embedding (already computed for semantic search) is
compared via cosine similarity against a small fixed set of category
description embeddings (Groceries & Food, Travel & Transport, Utilities &
Bills, etc.) defined in `core.EXPENSE_CATEGORIES`. The closest match wins.
No separate model or training step — it reuses `all-MiniLM-L6-v2`.

## How related-document detection works

After a new document is embedded, `DocumentStore.find_related()` queries
ChromaDB for existing documents whose embedding distance falls in a
"semantically similar but not identical" band — close enough to be about the
same purchase or vendor, far enough not to be a straight duplicate. These
show up under **Related Documents** on the result panel.

## Using the folder watcher (auto-ingest)

Start your DocuLens server as usual, then in a second terminal:

```bash
source .venv/bin/activate
python watch_folder.py
```

By default it watches `data/watch_inbox/` and uploads to
`http://127.0.0.1:8000`. Drop a screenshot or PDF into that folder and it's
processed automatically — no need to touch the dashboard. Useful for a
"set it and forget it" demo moment. Override either setting:

```bash
python watch_folder.py --folder ~/Desktop/Inbox --url http://127.0.0.1:8001
```

---

## Testing Instructions

1. Start the server (see **Running the Application** above).
2. Open `http://127.0.0.1:8000` — the status pill should read **AI Engine
   Online**.
3. Upload a test invoice image (see example below).
4. Watch the processing steps (Uploading → Processing → OCR complete →
   Analyzing → Embedding → Stored successfully).
5. Confirm the extracted fields, confidence, and risk level look correct.
6. Check that **Analytics** and **Your Documents** update automatically.
7. Try a semantic search, e.g. `"Amazon invoices"` or `"expensive purchases"`.
8. Click a document in search results or history to open the detail modal.

### Test invoice example

Create a plain image (or screenshot) containing this text and upload it:

```
Amazon Invoice

Invoice Number: AMZ-1023
Date: 17/08/2026

Total Amount: ₹4,599
GST: ₹699
```

### Expected output

```
Document Type: Invoice
Confidence: ~90%+
Vendor: Amazon
Invoice Number: AMZ-1023
Date: 17/08/2026
Total: ₹4,599
GST: ₹699
Risk: LOW (or MEDIUM, since subtotal isn't provided in this example)
```

---

## Hackathon Demo Flow

1. Open DocuLens AI.
2. Upload an Amazon invoice → watch OCR, classification, and extraction run
   live.
3. Point out: vendor identified, invoice number, date, total, GST, confidence
   score, auto-assigned category, risk analysis — all in seconds, all local.
4. Open **"Why this score?"** to show the explainability breakdown — this is
   the moment that separates DocuLens from a black-box OCR demo.
5. Dashboard stats update immediately, including the spending trend chart and
   category breakdown.
6. Search: **"Show me Amazon invoices"** → semantic match, not just keyword
   search. Try the 🎤 mic button to search by voice.
7. Open the result → show AI insights, related documents, and extracted text.
8. Ask DocuLens: **"How much did I spend on Amazon?"** → show it generating a
   real sentence, not just a search result.
9. Upload a second invoice.
10. Search: **"Show expensive purchases"** → demonstrate semantic ranking.
11. Upload a duplicate of the first invoice → show the duplicate-detection
    risk flag firing, with the point breakdown visible in explainability.
12. Click **⬇ Excel** or **⬇ PDF** to show a one-click analytics report.
13. (Optional) Run `python watch_folder.py` in a second terminal, then drag a
    new screenshot into `data/watch_inbox/` and watch it auto-process without
    touching the dashboard.
14. Wrap with: *"Google Lens + OCR + document understanding + semantic search
    + a local RAG chatbot + financial risk analysis — running entirely
    locally with open-source tools, no API keys, no cloud costs."*

---

## Future Improvements

- LLM-based document Q&A / RAG chatbot over stored documents
- Multilingual OCR
- Handwriting recognition
- ML-based fraud detection (beyond rule-based risk scoring)
- Automatic expense categorization
- Email ingestion
- Cloud deployment
- Authentication & multi-user support
- Document sharing
- Export to Excel
- Deeper financial analytics
