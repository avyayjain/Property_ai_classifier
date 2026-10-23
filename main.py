from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF
import pytesseract
import streamlit as st
from PIL import Image

import predict_document as classifier
import property_ai_review as review


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "review_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_STATUSES = ["needs_review", "confirmed", "corrected", "unresolved"]
SUPPORTED_TYPES = {"sale_deed", "property_tax_receipt", "encumbrance_certificate"}

# If Tesseract is installed in the standard Windows location, configure it.
# You can override this by setting the TESSERACT_CMD environment variable.
tesseract_override = os.environ.get("TESSERACT_CMD", "").strip()
if tesseract_override and Path(tesseract_override).exists():
    pytesseract.pytesseract.tesseract_cmd = tesseract_override
else:
    standard_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if standard_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(standard_tesseract)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def safe_filename(value: str, max_length: int = 80) -> str:
    """Make a user-supplied file name safe for local output files."""
    stem = Path(value or "uploaded_document").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (stem or "uploaded_document")[:max_length]


def unique_join(values: list[str], separator: str = " | ") -> str:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return separator.join(output)


def detect_document_template(
    page_texts: list[str],
) -> tuple[bool, list[str]]:
    """Detect explicit template indicators or strongly blank deed forms.

    A single word such as ``specimen`` is intentionally not enough: some
    populated synthetic test documents may contain that word in a disclaimer.
    """
    combined_text = "\n".join(
        str(text or "") for text in page_texts
    ).casefold()

    explicit_markers = (
        "model draft",
        "template",
        "sample document",
        "sample receipt",
        "illustrative only",
        "for illustration only",
        "may be customized according to",
        "blank form",
        "not for execution",
    )
    detected_markers = [
        marker for marker in explicit_markers
        if marker in combined_text
    ]

    # Repeated blank placeholders are supporting evidence only and are used
    # only with sale-deed context, not as a general-purpose rejection rule.
    placeholders = re.findall(r"\.{5,}|…{3,}|_{5,}", combined_text)
    sale_deed_context = (
        "sale deed" in combined_text
        and ("vendor" in combined_text or "vendee" in combined_text)
    )
    if (
        not detected_markers
        and sale_deed_context
        and len(placeholders) >= 8
    ):
        detected_markers.append("repeated_blank_placeholders")

    return bool(detected_markers), detected_markers


def detect_synthetic_document(
    page_texts: list[str],
) -> tuple[bool, list[str]]:
    """Detect explicit test-only or fictional-document wording."""
    combined_text = "\n".join(
        str(text or "") for text in page_texts
    ).casefold()

    markers = (
        "synthetic test",
        "synthetic document",
        "fictional data",
        "for software testing only",
        "not an official record",
        "no legal effect",
        "invented names",
    )
    detected_markers = [
        marker for marker in markers
        if marker in combined_text
    ]
    return bool(detected_markers), detected_markers


def detect_manual_or_guide(page_texts: list[str]) -> bool:
    """Conservative document-level safeguard for obvious instruction manuals."""
    combined = "\n".join(page_texts).casefold()
    phrases = (
        "user manual",
        "instruction manual",
        "operating manual",
        "technical manual",
        "service manual",
        "user guide",
        "how-to guide",
        "how to use this manual",
    )
    return any(phrase in combined for phrase in phrases)


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Read text from every PDF page; OCR pages with little/no embedded text."""
    result: list[dict[str, Any]] = []
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_index, pdf_page in enumerate(document, start=1):
            embedded_text = (pdf_page.get_text("text") or "").strip()
            if len(embedded_text.split()) >= 8:
                result.append(
                    {
                        "page_number": page_index,
                        "text": embedded_text,
                        "extraction_method": "pdf_text_layer",
                    }
                )
                continue

            try:
                pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                image_bytes = pixmap.tobytes("png")
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                ocr_text = (pytesseract.image_to_string(image, lang="eng", config="--psm 3") or "").strip()
                method = "tesseract_ocr"
                if len(ocr_text.split()) < len(embedded_text.split()):
                    # Retain a small existing text layer if OCR produced even less text.
                    text = embedded_text or ocr_text
                    method = "pdf_text_layer" if embedded_text else "tesseract_ocr"
                else:
                    text = ocr_text or embedded_text
                    if not ocr_text and embedded_text:
                        method = "pdf_text_layer"
                result.append(
                    {
                        "page_number": page_index,
                        "text": text,
                        "extraction_method": method,
                    }
                )
            except pytesseract.TesseractNotFoundError as exc:
                if embedded_text:
                    result.append(
                        {
                            "page_number": page_index,
                            "text": embedded_text,
                            "extraction_method": "pdf_text_layer",
                        }
                    )
                else:
                    raise RuntimeError(
                        "Tesseract was not found. Install Tesseract OCR or set "
                        "the TESSERACT_CMD environment variable to the full path "
                        "of tesseract.exe."
                    ) from exc
    finally:
        document.close()
    return result


def get_value(data: dict, key: str, default: Any = "") -> Any:
    value = data.get(key, default)
    return default if value is None else value


def apply_document_safeguards(
    report: dict,
    *,
    manual_detected: bool,
    template_detected: bool,
    template_markers: list[str],
    synthetic_detected: bool,
    synthetic_markers: list[str],
) -> dict:
    """Apply document-level safeguards to a completed review report.

    Manuals and blank templates have extracted candidates suppressed.
    Explicit synthetic documents retain candidates for software testing,
    but are not accepted as evidence of a real transaction.
    """
    def suppress_candidates(page_count_key: str) -> None:
        general = report.get("general_candidate_fields", []) or []
        deed = report.get("sale_deed_candidate_fields", []) or []
        report["suppressed_candidate_count"] = len(general) + len(deed)
        report["general_candidate_fields"] = []
        report["sale_deed_candidate_fields"] = []

        for page in report.get("pages", []) or []:
            page_general = page.get("candidate_fields", []) or []
            page_deed = page.get("sale_deed_candidate_fields", []) or []
            page[page_count_key] = len(page_general) + len(page_deed)
            page["candidate_fields"] = []
            page["sale_deed_candidate_fields"] = []

    # Priority 1: instructional manuals and guides.
    if manual_detected:
        report["document_status"] = "unknown_or_needs_review"
        report["document_level_safeguard"] = "instructional_manual_detected"
        report["reason"] = (
            "Instructional or manual wording was detected. "
            "Page-level predictions remain visible, but the overall "
            "document type is unresolved."
        )
        report["agreed_document_type"] = None
        report["agreed_type_candidates"] = []
        report["suggestions_passing_review_rules"] = []
        report["acceptance_rules_passed"] = False
        suppress_candidates("suppressed_manual_candidate_count")
        return report

    # Priority 2: blank templates.
    if template_detected:
        markers = ", ".join(template_markers) or "template indicators"
        report["document_status"] = "unknown_or_needs_review"
        report["document_level_safeguard"] = "document_template_detected"
        report["reason"] = (
            "Template or repeated blank-field indicators were detected: "
            f"{markers}. The overall document remains unresolved. "
            "Do not treat extracted values as completed transaction details."
        )
        report["agreed_document_type"] = None
        report["agreed_type_candidates"] = []
        report["suggestions_passing_review_rules"] = []
        report["acceptance_rules_passed"] = False
        suppress_candidates("suppressed_template_candidate_count")
        return report

    # Priority 3: populated synthetic test documents. Keep extracted values
    # for software testing, but never accept them as genuine property records.
    if synthetic_detected:
        markers = ", ".join(synthetic_markers) or "synthetic-document indicators"
        report["document_status"] = "unknown_or_needs_review"
        report["document_level_safeguard"] = "synthetic_test_document_detected"
        report["reason"] = (
            "Explicit synthetic or test-only wording was detected: "
            f"{markers}. Candidate fields are retained for software testing "
            "only. This document is not evidence of a real property transaction."
        )
        report["agreed_document_type"] = None
        report["agreed_type_candidates"] = []
        report["suggestions_passing_review_rules"] = []
        report["acceptance_rules_passed"] = False
        return report

    return report


def build_report(uploaded_name: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Classify pages and extract review-only candidate values."""
    pdf_pages = extract_pdf_pages(pdf_bytes)
    readable_pages = [p for p in pdf_pages if str(p.get("text", "")).strip()]
    if not readable_pages:
        raise ValueError(
            "No readable text was found in this PDF. Check that it is not blank, "
            "password-protected, or a damaged file, and verify Tesseract OCR."
        )

    # Load the two existing project models using the project's own API.
    baseline_model = classifier.load_baseline()
    tokenizer, transformer_model, device, label_mapping = classifier.load_transformer()

    page_results: list[dict[str, Any]] = []
    general_candidates: list[dict[str, Any]] = []
    sale_deed_candidates: list[dict[str, Any]] = []
    sale_deed_pages: list[int] = []

    for page_data in readable_pages:
        page_number = int(page_data.get("page_number", len(page_results) + 1))
        page_text = str(page_data.get("text", "") or "")

        prediction = classifier.classify_page(
            page_text,
            baseline_model,
            tokenizer,
            transformer_model,
            device,
            label_mapping,
        )

        baseline_label = review.get_baseline_label(prediction)
        baseline_score = review.get_baseline_score(prediction)
        bert_label = review.get_bert_label(prediction)
        bert_score = review.get_bert_score(prediction)
        classifier_status = review.get_classifier_status(prediction)
        explanation = review.explain_page_prediction(
            baseline_label,
            bert_label,
            classifier_status,
        )

        general_page_candidates = review.extract_candidates_from_page(page_text, page_number)
        general_candidates.extend(general_page_candidates)

        sale_deed_attempted = baseline_label == "sale_deed" or bert_label == "sale_deed"
        specialised_page_candidates: list[dict[str, Any]] = []
        if sale_deed_attempted:
            sale_deed_pages.append(page_number)
            specialised_page_candidates = review.extract_sale_deed_candidates_from_page(
                page_text,
                page_number,
            )
            sale_deed_candidates.extend(specialised_page_candidates)

        page_results.append(
            {
                "page_number": page_number,
                "extraction_method": page_data.get("extraction_method", "unknown"),
                "word_count": len(page_text.split()),
                "prediction": prediction,
                "baseline_label": baseline_label,
                "baseline_score": baseline_score,
                "bert_label": bert_label,
                "bert_score": bert_score,
                "models_agree": baseline_label == bert_label,
                "classifier_status": classifier_status,
                "page_explanation": explanation,
                "candidate_fields": general_page_candidates,
                "sale_deed_extraction_attempted": sale_deed_attempted,
                "sale_deed_candidate_fields": specialised_page_candidates,
                "text": page_text,
            }
        )

    decision = review.classify_document_type(page_results)
    full_texts = [str(page.get("text", "") or "") for page in pdf_pages]
    manual_detected = detect_manual_or_guide(full_texts)
    template_detected, template_markers = detect_document_template(full_texts)
    synthetic_detected, synthetic_markers = detect_synthetic_document(full_texts)

    report: dict[str, Any] = {
        "source_file": uploaded_name,
        "document_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "document_status": decision.get("document_status", "unknown_or_needs_review"),
        "document_level_safeguard": "none",
        "reason": decision.get("reason", "Review required."),
        "agreed_document_type": decision.get("agreed_document_type"),
        "agreed_type_candidates": decision.get("agreed_type_candidates", []),
        "suggestions_passing_review_rules": decision.get("suggestions_passing_review_rules", []),
        "acceptance_rules_passed": bool(decision.get("acceptance_rules_passed", False)),
        "general_candidate_fields": general_candidates,
        "sale_deed_extraction_attempted": bool(sale_deed_pages),
        "sale_deed_pages": sale_deed_pages,
        "sale_deed_candidate_fields": sale_deed_candidates,
        "pages": page_results,
        "limitations": list(getattr(review, "LIMITATIONS", [])),
    }

    return apply_document_safeguards(
        report,
        manual_detected=manual_detected,
        template_detected=template_detected,
        template_markers=template_markers,
        synthetic_detected=synthetic_detected,
        synthetic_markers=synthetic_markers,
    )


def consolidate_candidates(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Combine identical field/value/page candidates from both extractors."""
    grouped: dict[tuple[int, str, str], dict[str, Any]] = {}
    sources = [
        ("general_candidate_fields", "general_context_regex"),
        ("sale_deed_candidate_fields", "sale_deed_extractor.py"),
    ]

    for list_name, source_name in sources:
        candidates = report.get(list_name, []) or []
        for candidate in candidates:
            field = str(candidate.get("field", "unknown_field"))
            value = str(candidate.get("candidate_value", "") or "").strip()
            page_number = int(candidate.get("page_number", 0) or 0)
            if not value:
                continue

            key = (page_number, field.casefold(), value.casefold())
            row = grouped.get(key)
            if row is None:
                row = {
                    "candidate_id": hashlib.sha1("|".join(map(str, key)).encode("utf-8")).hexdigest()[:12],
                    "field": field,
                    "candidate_value": value,
                    "page_number": page_number,
                    "sources": [],
                    "methods": [],
                    "evidence": [],
                    "extractor_notes": [],
                }
                grouped[key] = row
            row["sources"].append(list_name)
            row["methods"].append(str(candidate.get("method", source_name) or source_name))
            evidence = str(candidate.get("evidence", "") or "").strip()
            note = str(candidate.get("review_note", "") or "").strip()
            if evidence:
                row["evidence"].append(evidence)
            if note:
                row["extractor_notes"].append(note)

    rows = list(grouped.values())
    rows.sort(key=lambda row: (row["page_number"], row["field"], row["candidate_value"].casefold()))
    for row in rows:
        row["sources"] = unique_join(row["sources"])
        row["methods"] = unique_join(row["methods"])
        row["evidence"] = unique_join(row["evidence"], separator="\n---\n")
        row["extractor_notes"] = unique_join(row["extractor_notes"])
    return rows


def create_review_csv(rows: list[dict[str, Any]]) -> str:
    """Convert human-review decisions to CSV text."""
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    fieldnames = [
        "source_file",
        "document_sha256",
        "candidate_id",
        "page_number",
        "field",
        "original_candidate_value",
        "review_status",
        "corrected_value",
        "reviewer_note",
        "sources",
        "methods",
        "evidence",
        "reviewed_at_utc",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()



def _normalise_comparison_value(value: str) -> str:
    """Normalize spacing and punctuation for cautious string comparisons."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def compare_document_bundle(document_reports: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flag possible inconsistencies across extracted candidate values.

    Each input is {"source_file": name, "report": report_dict}. This is a
    screening aid only: values are candidates and all flags require human review.
    """
    field_groups = {
        "property_address": {
            "property_address_candidate", "property_address", "premises_candidate",
        },
        "property_identifier": {
            "property_identifier_candidate", "property_identifier",
        },
        "assessment_identifier": {
            "assessment_identifier_candidate", "assessment_identifier",
        },
        "seller_name": {"seller_name_candidate", "seller_name"},
        "buyer_name": {"buyer_name_candidate", "buyer_name"},
        "execution_date": {"execution_date_candidate", "execution_date"},
        "sale_consideration": {"sale_consideration_candidate", "sale_consideration"},
        "tax_amount": {"tax_amount_candidate", "tax_amount"},
        "additional_charges": {"additional_charges_candidate", "additional_charges"},
        "total_paid": {"total_paid_candidate", "total_paid"},
        "owner_or_taxpayer_name": {
            "owner_or_taxpayer_name_candidate", "owner_or_taxpayer_name",
            "taxpayer_name_candidate", "taxpayer_name", "payer_name_candidate",
            "owner_taxpayer_name_candidate", "owner_taxpayer_name", "payer_name",
        },
    }
    field_to_group = {
        field.casefold(): group
        for group, names in field_groups.items()
        for field in names
    }
    grouped: dict[str, dict[str, list[str]]] = {}

    for item in document_reports:
        source_file = str(item.get("source_file", "unnamed_document.pdf"))
        report = item.get("report", {}) or {}
        candidates = []
        for key in ("general_candidate_fields", "sale_deed_candidate_fields"):
            candidates.extend(report.get(key, []) or [])

        # Support reports where candidates may only appear on page-level arrays.
        if not candidates:
            for page in report.get("pages", []) or []:
                candidates.extend(page.get("candidate_fields", []) or [])
                candidates.extend(page.get("sale_deed_candidate_fields", []) or [])

        for candidate in candidates:
            field = str(candidate.get("field", "")).casefold()
            group = field_to_group.get(field)
            value = str(candidate.get("candidate_value", "") or "").strip()
            if not group or not value:
                continue
            per_doc = grouped.setdefault(group, {}).setdefault(source_file, [])
            if value not in per_doc:
                per_doc.append(value)

    rows: list[dict[str, str]] = []
    for group, by_document in grouped.items():
        if len(by_document) < 2:
            continue
        all_normalised = {
            _normalise_comparison_value(value)
            for values in by_document.values()
            for value in values
            if _normalise_comparison_value(value)
        }
        if len(all_normalised) <= 1 and all(len(values) <= 1 for values in by_document.values()):
            status = "candidate_values_match"
            note = "The extracted candidate strings match after basic punctuation/spacing normalization. Verify source evidence."
        else:
            status = "potential_discrepancy"
            note = "Candidate values differ across documents or within a document. Check the original pages; this is not a legal conclusion."
        rows.append({
            "check": group,
            "status": status,
            "documents": " | ".join(sorted(by_document)),
            "candidate_values": " | ".join(
                f"{name}: {' / '.join(values)}"
                for name, values in sorted(by_document.items())
            ),
            "review_note": note,
        })

    rows.sort(key=lambda row: (row["status"] != "potential_discrepancy", row["check"]))
    return rows


def run_app():
    # ---------------------------------------------------------------------
    # Streamlit interface
    # ---------------------------------------------------------------------
    st.set_page_config(page_title="Property AI Document Review", page_icon="🏠", layout="wide")
    st.title("Property AI — Document Review")
    st.caption(
        "Experimental document classification and candidate extraction. "
        "Every extracted value must be checked against the original PDF."
    )

    with st.sidebar:
        st.subheader("Current capabilities")
        st.write("• PDF text-layer extraction with OCR fallback")
        st.write("• TF-IDF baseline + BERT-Tiny classification")
        st.write("• General and sale-deed candidate extraction")
        st.write("• Human review decisions saved to CSV")
        st.warning(
            "This app does not verify title, ownership, authenticity, registration validity, "
            "encumbrance status, or legal validity. Do not upload sensitive documents to a "
            "publicly exposed instance."
        )

    uploaded_files = st.file_uploader(
        "Upload one or more property documents (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload a document bundle to compare extracted candidate values. Each value remains unverified.",
    )

    if not uploaded_files:
        st.info("Upload one or more PDFs to start the review.")
        st.stop()

    bundle_reports: list[dict[str, Any]] = []
    tabs = st.tabs([f"{index + 1}. {safe_filename(file.name)}" for index, file in enumerate(uploaded_files)])

    for file_index, (uploaded_file, tab) in enumerate(zip(uploaded_files, tabs), start=1):
        with tab:
            pdf_bytes = uploaded_file.getvalue()
            document_hash = hashlib.sha256(pdf_bytes).hexdigest()
            # Include index so duplicate copies of the same PDF get unique widget keys.
            review_key = f"{document_hash[:12]}_{file_index}"

            with st.spinner(f"Processing {uploaded_file.name}: reading pages, classifying, and extracting candidates..."):
                try:
                    report = build_report(uploaded_file.name, pdf_bytes)
                    bundle_reports.append({"source_file": uploaded_file.name, "report": report})
                except Exception as exc:
                    st.error(f"Document processing failed for {uploaded_file.name}: {exc}")
                    st.exception(exc)
                    continue

            # Save the machine-generated report separately from human decisions.
            report_path = OUTPUT_DIR / f"{safe_filename(uploaded_file.name)}_{review_key}_review.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

            st.subheader("Document-level result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall status", str(report.get("document_status", "unknown_or_needs_review")))
            col2.metric("Agreed type", str(report.get("agreed_document_type") or "Unresolved"))
            col3.metric("Pages processed", len(report.get("pages", [])))
            st.write(report.get("reason", "No explanation returned."))
            if report.get("document_level_safeguard") != "none":
                st.warning(f"Document safeguard triggered: {report['document_level_safeguard']}")

            st.subheader("Page-level classification")
            page_summary = []
            for page_result in report.get("pages", []):
                page_summary.append(
                    {
                        "Page": page_result.get("page_number"),
                        "Text extraction": page_result.get("extraction_method"),
                        "Words": page_result.get("word_count"),
                        "TF-IDF label": page_result.get("baseline_label"),
                        "TF-IDF score": page_result.get("baseline_score"),
                        "BERT label": page_result.get("bert_label"),
                        "BERT score": page_result.get("bert_score"),
                        "Models agree": page_result.get("models_agree"),
                        "Page status": page_result.get("classifier_status"),
                    }
                )
            if page_summary:
                st.dataframe(page_summary, width="stretch", hide_index=True)
            else:
                st.warning("No page-level classification results are available.")

            st.download_button(
                "Download machine-generated review JSON",
                data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{safe_filename(uploaded_file.name)}_review.json",
                mime="application/json",
            )

            st.subheader("Consolidated candidate fields")
            candidates = consolidate_candidates(report)
            if not candidates:
                st.info("No candidate fields were extracted. This does not mean the document contains none.")
            else:
                st.write(
                    "Matching values from the general and specialised extractors are combined into one row. "
                    "The AI value remains unchanged; any correction is stored separately."
                )
                display_rows = [
                    {
                        "Page": row["page_number"],
                        "Field": row["field"],
                        "AI candidate": row["candidate_value"],
                        "Extractors": row["methods"],
                        "Consolidated from": row["sources"],
                    }
                    for row in candidates
                ]
                st.dataframe(display_rows, width="stretch", hide_index=True)

                st.subheader("Human-based decisions")
                st.write(
                    "Check the PDF itself before choosing `confirmed`. Select `corrected` only when you "
                    "can enter a value supported by the source document. Use `unresolved` when the evidence "
                    "is unclear and `needs_review` when you have not finished checking."
                )

                with st.form(f"human_review_form_{review_key}"):
                    submitted_decisions: list[dict[str, Any]] = []
                    for index, candidate in enumerate(candidates):
                        widget_prefix = f"{review_key}_{candidate['candidate_id']}"
                        st.markdown(f"#### {index + 1}. {candidate['field']} — page {candidate['page_number']}")
                        st.write(f"**AI candidate:** `{candidate['candidate_value']}`")
                        st.caption(f"Extractors: {candidate['methods']} | Sources: {candidate['sources']}")
                        with st.expander("View evidence and extractor notes", expanded=False):
                            st.text(candidate["evidence"] or "No evidence snippet was returned.")
                            if candidate["extractor_notes"]:
                                st.caption(candidate["extractor_notes"])

                        left, middle, right = st.columns([1, 1.4, 2])
                        with left:
                            status = st.selectbox(
                                "Review status",
                                options=REVIEW_STATUSES,
                                index=0,
                                key=f"status_{widget_prefix}",
                            )
                        with middle:
                            corrected_value = st.text_input(
                                "Corrected value (only if corrected)",
                                key=f"corrected_{widget_prefix}",
                            )
                        with right:
                            reviewer_note = st.text_input(
                                "Reviewer note",
                                key=f"note_{widget_prefix}",
                            )

                        submitted_decisions.append(
                            {
                                "source_file": uploaded_file.name,
                                "document_sha256": document_hash,
                                "candidate_id": candidate["candidate_id"],
                                "page_number": candidate["page_number"],
                                "field": candidate["field"],
                                "original_candidate_value": candidate["candidate_value"],
                                "review_status": status,
                                "corrected_value": corrected_value.strip(),
                                "reviewer_note": reviewer_note.strip(),
                                "sources": candidate["sources"],
                                "methods": candidate["methods"],
                                "evidence": candidate["evidence"],
                                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            }
                        )
                        st.divider()

                    submitted = st.form_submit_button("Save human-review decisions", type="primary")

                if submitted:
                    invalid_rows = [
                        row for row in submitted_decisions
                        if row["review_status"] == "corrected" and not row["corrected_value"]
                    ]
                    if invalid_rows:
                        st.error(
                            "For every row marked `corrected`, enter a corrected value. "
                            "No decisions were saved; amend those rows and submit again."
                        )
                    else:
                        csv_text = create_review_csv(submitted_decisions)
                        decisions_path = OUTPUT_DIR / f"{safe_filename(uploaded_file.name)}_{review_key}_human_review.csv"
                        decisions_path.write_text(csv_text, encoding="utf-8-sig", newline="")
                        st.session_state[f"saved_csv_{review_key}"] = csv_text
                        st.session_state[f"saved_path_{review_key}"] = str(decisions_path)
                        st.success(f"Human-review decisions saved to: {decisions_path}")

                saved_csv = st.session_state.get(f"saved_csv_{review_key}")
                if saved_csv:
                    st.download_button(
                        "Download human-review decisions (CSV)",
                        data=saved_csv.encode("utf-8-sig"),
                        file_name=f"{safe_filename(uploaded_file.name)}_human_review.csv",
                        mime="text/csv",
                        key=f"download_review_{review_key}",
                    )

            with st.expander("Extracted page text", expanded=False):
                for page_result in report.get("pages", []):
                    st.markdown(f"**Page {page_result.get('page_number')}** — {page_result.get('extraction_method')}")
                    st.text_area(
                        f"OCR/text page {page_result.get('page_number')}",
                        value=str(page_result.get("text", "") or ""),
                        height=180,
                        key=f"page_text_{review_key}_{page_result.get('page_number')}",
                    )

            with st.expander("Limitations", expanded=False):
                for limitation in report.get("limitations", []):
                    st.write(f"• {limitation}")

            st.caption(
                "Human-review decisions are saved separately from model output. The review CSV records "
                "decisions for the current submission; this version does not automatically reload prior "
                "CSV decisions into the form on a new app session."
            )

    if len(bundle_reports) > 1:
        st.subheader("Cross-document consistency review")
        st.caption(
            "These checks compare extracted candidate values only. A match does not establish truth, "
            "and a difference is a prompt for human review—not a legal conclusion."
        )
        comparison_rows = compare_document_bundle(bundle_reports)
        if comparison_rows:
            st.dataframe(comparison_rows, width="stretch", hide_index=True)
            if any(row["status"] == "potential_discrepancy" for row in comparison_rows):
                st.warning("Potential inconsistencies were found. Inspect the cited source pages before making any decision.")
            else:
                st.info("No differences were detected among the compared fields. Missing fields are not treated as matches.")
        else:
            st.info("No comparable candidate fields were found in at least two documents.")
    else:
        st.info("Upload at least two documents to run cross-document consistency checks.")



if __name__ == "__main__":
    run_app()
