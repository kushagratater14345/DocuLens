"""
app.py - DocuLens AI FastAPI backend

Endpoints:
  GET    /                    -> serves the dashboard
  POST   /upload               -> process and store a document
  GET    /search?q=...         -> semantic search
  GET    /ask?q=...            -> natural-language Q&A over stored documents (RAG)
  GET    /documents            -> list all documents (newest first)
  GET    /documents/{id}       -> get one document's full detail
  DELETE /documents/{id}       -> delete a document
  GET    /stats                -> dashboard analytics
  GET    /export/excel         -> download an .xlsx report
  GET    /export/pdf           -> download a .pdf report
  GET    /health                -> system health check
"""

from __future__ import annotations

import shutil
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import core
import ingest
import reports

# ----------------------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_FOLDER = BASE_DIR / "static"
UPLOAD_FOLDER = core.UPLOAD_FOLDER

app = FastAPI(title="DocuLens AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazily-initialized global state so /health can report failures cleanly
_store = None
_store_error = None
_tesseract_path = None
_tesseract_error = None


def get_store():
    """Returns the ChromaDB-backed document store, initializing it on first use."""
    global _store, _store_error
    if _store is None and _store_error is None:
        try:
            _store = core.DocumentStore()
        except Exception as e:
            _store_error = str(e)
    if _store is None:
        raise HTTPException(
            status_code=500,
            detail=f"Database is not available: {_store_error}",
        )
    return _store


@app.on_event("startup")
def on_startup():
    global _tesseract_path, _tesseract_error
    try:
        _tesseract_path = core.find_tesseract()
        if not _tesseract_path:
            _tesseract_error = (
                "Tesseract OCR not found. Install it with: brew install tesseract"
            )
    except Exception as e:
        _tesseract_error = str(e)

    # Warm the document store (non-fatal if it fails; /health will report it)
    try:
        get_store()
    except HTTPException:
        pass


# ----------------------------------------------------------------------------
# STATIC FRONTEND
# ----------------------------------------------------------------------------

if STATIC_FOLDER.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")


@app.get("/")
def serve_index():
    index_path = STATIC_FOLDER / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found: static/index.html is missing.")
    return FileResponse(str(index_path))


# ----------------------------------------------------------------------------
# HEALTH
# ----------------------------------------------------------------------------


@app.get("/health")
def health():
    ocr_ok = _tesseract_path is not None
    db_ok = _store is not None

    embedding_ok = True
    embedding_error = None
    try:
        core.get_embedding_model()
    except Exception as e:
        embedding_ok = False
        embedding_error = str(e)

    qa_installed = True
    try:
        import transformers  # noqa: F401
    except ImportError:
        qa_installed = False

    return {
        "status": "ok" if (ocr_ok and db_ok and embedding_ok) else "degraded",
        "ocr": ocr_ok,
        "ocr_path": _tesseract_path,
        "ocr_error": _tesseract_error,
        "database": db_ok,
        "database_error": _store_error,
        "embedding_model": embedding_ok,
        "embedding_error": embedding_error,
        "qa_model_installed": qa_installed,
        "qa_model_note": "Loaded lazily on first /ask call (downloads ~300MB once).",
    }


# ----------------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------------


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was provided.")

    try:
        ext = ingest.validate_file(file.filename)
    except ingest.DoculensError as e:
        raise HTTPException(status_code=400, detail=str(e))

    safe_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOAD_FOLDER / safe_name

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        store = get_store()
        result = ingest.process_upload(saved_path, file.filename, store)
        return JSONResponse(content=result)
    except ingest.DoculensError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected processing error: {e}")


# ----------------------------------------------------------------------------
# SEARCH
# ----------------------------------------------------------------------------


@app.get("/search")
def search_documents(q: str = "", limit: int = 8):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    store = get_store()
    try:
        results = store.search(q.strip(), n_results=limit)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {"query": q, "results": results}


# ----------------------------------------------------------------------------
# ASK DOCULENS (RAG Q&A)
# ----------------------------------------------------------------------------

_qa_unavailable_reason = None


@app.get("/ask")
def ask_documents(q: str = "", limit: int = 5):
    """
    Retrieves the most relevant stored documents for the question, then asks
    a small local model to answer using only that retrieved context - the
    same retrieve-then-generate pattern as a full RAG pipeline, just with a
    lighter model so it runs comfortably on a laptop CPU.
    """
    global _qa_unavailable_reason

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    store = get_store()
    try:
        results = store.search(q.strip(), n_results=limit)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if not results:
        return {
            "question": q,
            "answer": "I don't have any documents to answer that from yet. Upload one first.",
            "sources": [],
        }

    context = core.build_qa_context(results)

    try:
        answer = core.answer_question(q.strip(), context)
    except Exception as e:
        _qa_unavailable_reason = str(e)
        raise HTTPException(
            status_code=503,
            detail=(
                "The local Q&A model isn't available yet. On first use it downloads "
                "google/flan-t5-small (~300MB) - make sure you're online, or install "
                f"'transformers' if it's missing. Details: {e}"
            ),
        )

    return {"question": q, "answer": answer, "sources": results}


# ----------------------------------------------------------------------------
# DOCUMENTS
# ----------------------------------------------------------------------------


@app.get("/documents")
def list_documents():
    store = get_store()
    try:
        docs = store.get_all()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load documents: {e}")
    return {"documents": docs, "count": len(docs)}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    store = get_store()
    doc = store.get_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    store = get_store()
    deleted = store.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "id": doc_id}


# ----------------------------------------------------------------------------
# STATS
# ----------------------------------------------------------------------------


@app.get("/stats")
def get_stats():
    store = get_store()
    try:
        return store.compute_stats()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")


# ----------------------------------------------------------------------------
# EXPORT / REPORTS
# ----------------------------------------------------------------------------


@app.get("/export/excel")
def export_excel():
    store = get_store()
    try:
        path = reports.generate_excel_report(store)
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"openpyxl is not installed. Run: pip install openpyxl ({e})",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate Excel report: {e}")

    return FileResponse(
        str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="doculens_report.xlsx",
    )


@app.get("/export/pdf")
def export_pdf():
    store = get_store()
    try:
        path = reports.generate_pdf_report(store)
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"reportlab is not installed. Run: pip install reportlab ({e})",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {e}")

    return FileResponse(str(path), media_type="application/pdf", filename="doculens_report.pdf")
