"""Cross-document comparison helpers for Property AI."""

from __future__ import annotations

import re
from typing import Any


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
