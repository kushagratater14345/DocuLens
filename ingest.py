"""
ingest.py - DocuLens AI ingestion pipeline

Orchestrates the full document processing pipeline:

UPLOAD -> VALIDATE -> PREPROCESS -> OCR -> CLEAN -> CLASSIFY ->
EXTRACT -> CONFIDENCE -> RISK -> EMBED -> STORE -> RESPOND
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import core


class DoculensError(Exception):
    """Raised for any pipeline error with a user-friendly message."""


def validate_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in core.ALLOWED_EXTENSIONS:
        raise DoculensError(
            f"Unsupported file type '{ext}'. Please upload a PNG, JPG, JPEG or PDF."
        )
    return ext


def process_upload(file_path: Path, original_filename: str, store: "core.DocumentStore") -> Dict[str, Any]:
    """
    Runs the complete pipeline on an already-saved file and stores the
    result in ChromaDB. Returns the full response payload.
    """
    # 1. OCR
    try:
        raw_text = core.ocr_file(file_path)
    except core.TesseractNotFoundError as e:
        raise DoculensError(str(e)) from e
    except (ValueError, RuntimeError) as e:
        raise DoculensError(str(e)) from e

    # 2. Clean text
    cleaned_text = core.clean_text(raw_text)
    if not cleaned_text:
        raise DoculensError("OCR produced empty text after cleaning. Try a clearer document.")

    # 3. Classification
    doc_type, class_confidence = core.classify_document(cleaned_text)

    # 4. Structured extraction
    fields = core.extract_fields(cleaned_text, doc_type)

    # 5. Confidence scoring (with explainability breakdown)
    confidence, confidence_status, confidence_breakdown = core.calculate_confidence(
        doc_type, class_confidence, fields, len(cleaned_text)
    )

    # 6. Risk / duplicate detection (compare against existing documents of same type)
    existing_docs = []
    try:
        for doc in store.get_all():
            meta = doc["metadata"]
            existing_docs.append(
                {
                    "vendor": meta.get("vendor") or None,
                    "invoice_number": meta.get("invoice_number") or None,
                    "total_amount": core._to_float(meta.get("total_amount")),
                }
            )
    except Exception:
        existing_docs = []

    risk = core.detect_risk(doc_type, fields, existing_docs)

    # 7. Build searchable text + embedding
    searchable_text = core.build_searchable_text(doc_type, fields, cleaned_text)
    try:
        embedding = core.get_embedding(searchable_text)
    except Exception as e:
        raise DoculensError(f"Failed to generate embedding: {e}") from e

    # 7b. Auto expense categorization (reuses the same embedding, no extra model)
    try:
        category, category_confidence = core.categorize_document(embedding)
    except Exception:
        category, category_confidence = "Other", 0.0

    # 7c. Related-document detection (semantically close but not a duplicate)
    # Run before storing so the new document doesn't match against itself.
    try:
        related = store.find_related(doc_id="__pending__", embedding=embedding)
    except Exception:
        related = []

    # 8. Persist to ChromaDB
    doc_id = str(uuid.uuid4())
    file_hash = core.compute_file_hash(file_path)
    processed_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "document_type": doc_type,
        "classification_confidence": class_confidence,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "confidence_breakdown": confidence_breakdown,
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_flags": risk["flags"],
        "risk_breakdown": risk.get("flag_breakdown", []),
        "category": category,
        "category_confidence": category_confidence,
        "source": original_filename,
        "file_hash": file_hash,
        "processed_at": processed_at,
        **fields,
    }

    try:
        store.add_document(
            doc_id=doc_id,
            embedding=embedding,
            document_text=searchable_text,
            metadata=metadata,
        )
    except Exception as e:
        raise DoculensError(f"Failed to store document in the database: {e}") from e

    ai_insights = build_ai_insights(doc_type, fields, risk, confidence_status, category)

    return {
        "id": doc_id,
        "filename": original_filename,
        "document_type": doc_type,
        "classification_confidence": class_confidence,
        "fields": fields,
        "confidence": confidence,
        "confidence_status": confidence_status,
        "confidence_breakdown": confidence_breakdown,
        "risk": risk,
        "category": category,
        "category_confidence": category_confidence,
        "related_documents": related,
        "extracted_text": cleaned_text,
        "ai_insights": ai_insights,
        "processed_at": processed_at,
    }


def build_ai_insights(doc_type, fields, risk, confidence_status, category=None) -> list:
    insights = []

    if doc_type != "unknown":
        insights.append(f"Valid {doc_type.replace('_', ' ')} structure detected")
    else:
        insights.append("Document type could not be confidently determined")

    if fields.get("vendor"):
        insights.append(f"Vendor identified as {fields['vendor']}")
    elif fields.get("bank_name"):
        insights.append(f"Bank identified as {fields['bank_name']}")
    elif fields.get("organization"):
        insights.append(f"Organization identified as {fields['organization']}")

    if fields.get("gst"):
        insights.append("GST amount detected")

    if fields.get("total_amount") or fields.get("closing_balance"):
        insights.append("Total amount extracted successfully")

    if category:
        insights.append(f"Auto-categorized as {category}")

    if risk["flags"]:
        insights.append(f"{len(risk['flags'])} risk flag(s) raised - review recommended")
    else:
        insights.append("No major anomalies detected")

    insights.append(f"Overall extraction confidence: {confidence_status}")

    return insights
