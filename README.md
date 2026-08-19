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





