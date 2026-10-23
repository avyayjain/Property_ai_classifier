"""
PROPERTY AI - CROSS-DOCUMENT CONSISTENCY CHECKER

Compares candidate fields from:
- Sale deed
- Property tax receipt
- Encumbrance certificate

The results identify textual matches, potential discrepancies,
missing values, and ambiguous candidates.

This tool does not verify title, legal ownership, authenticity,
registration validity, encumbrances, or legal safety.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "review_output"

DEFAULT_REPORTS = {
    "sale_deed": (
        OUTPUT_DIR / "synthetic_sale_deed_document_review.json"
    ),
    "property_tax_receipt": (
        OUTPUT_DIR
        / "synthetic_property_tax_receipt_document_review.json"
    ),
    "encumbrance_certificate": (
        OUTPUT_DIR
        / "synthetic_encumbrance_certificate_document_review.json"
    ),
}

EXPECTED_TYPES = {
    "sale_deed": "sale_deed",
    "property_tax_receipt": "property_tax_receipt",
    "encumbrance_certificate": "encumbrance_certificate",
}

DISCLAIMER = (
    "This report compares extracted text candidates only. A match does "
    "not prove ownership or legal validity, and a discrepancy does not "
    "by itself prove an error or legal problem. Verify the original "
    "documents and obtain qualified professional review."
)


# ================================================================
# NORMALISATION AND REPORT LOADING
# ================================================================

def normalize_value(value: str) -> str:
    """
    Normalise text for basic exact comparison.

    Case, whitespace, and punctuation are ignored. Spelling differences
    are intentionally NOT corrected or treated as matches.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def load_review_report(path: Path) -> dict:
    """Load and validate a JSON review report."""

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Review report not found:\n{path}\n"
            "Run property_ai_review.py on the source document first."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON review report: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    data["_loaded_report_path"] = str(path)
    return data


def get_candidates(report: dict, field: str) -> list[dict]:
    """Collect unique candidate records from a review report."""

    all_items = []

    for section_name in (
        "general_candidate_fields",
        "sale_deed_candidate_fields",
    ):
        items = report.get(section_name, [])
        if isinstance(items, list):
            all_items.extend(
                item for item in items if isinstance(item, dict)
            )

    output = []
    seen = set()

    for item in all_items:
        if item.get("field") != field:
            continue

        value = str(item.get("candidate_value", "")).strip()

        if not value:
            continue

        page = item.get("page_number")
        key = (value.casefold(), page)

        if key in seen:
            continue

        seen.add(key)

        output.append({
            "value": value,
            "page_number": page,
            "evidence": str(item.get("evidence", "")),
            "status": item.get("status", "needs_review"),
            "method": item.get("method", "unknown"),
        })

    return output


def prepare_documents(report_paths: dict[str, Path]) -> dict:
    """Load the review reports and check their provisional types."""

    documents = {}

    for document_key, path in report_paths.items():
        report = load_review_report(Path(path))
        expected_type = EXPECTED_TYPES[document_key]
        actual_type = report.get("agreed_document_type")

        documents[document_key] = {
            "document_key": document_key,
            "expected_type": expected_type,
            "actual_type": actual_type,
            "classification_status": report.get(
                "document_status",
                "unknown_or_needs_review",
            ),
            "type_matches_expectation": actual_type == expected_type,
            "report_path": report["_loaded_report_path"],
            "report": report,
        }

    return documents


# ================================================================
# CONSISTENCY STATUS
# ================================================================

def determine_consistency_status(
    sources: list[dict],
    required_source_count: int,
) -> str:
    """
    Determine the comparison status from available candidate values.

    sources items contain:
      values: list of candidate records
      type_matches_expectation: bool
    """

    # Do not make comparisons based on reports whose document type
    # was not accepted even as a provisional candidate.
    if any(
        not source["type_matches_expectation"]
        for source in sources
    ):
        return "document_type_needs_review"

    # More than one distinct candidate in a single report makes the
    # comparison ambiguous. A reviewer must resolve the ambiguity.
    for source in sources:
        values = [
            item["value"] for item in source["values"]
        ]
        unique_values = {
            normalize_value(value) for value in values
        }
        unique_values.discard("")

        if len(unique_values) > 1:
            return "multiple_candidates_needs_review"

    populated_values = []

    for source in sources:
        values = source["values"]
        if values:
            populated_values.append(values[0]["value"])

    if len(populated_values) < 2:
        return "insufficient_evidence"

    normalised = [
        normalize_value(value) for value in populated_values
    ]

    if len(set(normalised)) > 1:
        return "potential_discrepancy"

    if len(populated_values) < required_source_count:
        return "partial_match_missing_data"

    return "consistent_across_documents"


def describe_status(status: str) -> str:
    """Return a human-readable explanation of a comparison status."""

    descriptions = {
        "consistent_across_documents": (
            "The available candidate values match textually across "
            "all compared documents."
        ),
        "partial_match_missing_data": (
            "The available candidate values match, but one or more "
            "documents do not contain a usable candidate for this field."
        ),
        "potential_discrepancy": (
            "The candidate values differ after basic text normalisation. "
            "Check the original documents and OCR output."
        ),
        "multiple_candidates_needs_review": (
            "At least one document has multiple distinct candidate values "
            "for this field. Manual review is required."
        ),
        "insufficient_evidence": (
            "Fewer than two usable candidate values are available."
        ),
        "document_type_needs_review": (
            "At least one report's provisional document type does not "
            "match its expected category. Resolve document classification "
            "before relying on this comparison."
        ),
    }

    return descriptions.get(
        status,
        "Manual review is required.",
    )


# ================================================================
# COMPARISON ENGINE
# ================================================================

def compare_field(
    title: str,
    source_definitions: list[dict],
    documents: dict,
    required_source_count: int | None = None,
    limitation: str | None = None,
) -> dict:
    """Compare one semantic field across the requested source documents."""

    if required_source_count is None:
        required_source_count = len(source_definitions)

    sources = []

    for definition in source_definitions:
        document_key = definition["document_key"]
        field_name = definition["field"]
        document = documents[document_key]

        candidates = get_candidates(
            document["report"],
            field_name,
        )

        sources.append({
            "document_key": document_key,
            "expected_document_type": document["expected_type"],
            "actual_document_type": document["actual_type"],
            "report_path": document["report_path"],
            "field": field_name,
            "type_matches_expectation": (
                document["type_matches_expectation"]
            ),
            "values": candidates,
        })

    status = determine_consistency_status(
        sources,
        required_source_count,
    )

    result = {
        "comparison": title,
        "status": status,
        "explanation": describe_status(status),
        "sources": sources,
        # Textual agreement is not independent verification.
        # Every comparison must still be reviewed against the originals.
        "requires_manual_review": True,
    }

    if limitation:
        result["limitation"] = limitation

    return result


def build_comparisons(documents: dict) -> list[dict]:
    """Build the comparisons relevant to this property workflow."""

    property_id_sources = [
        {"document_key": "sale_deed",
         "field": "property_identifier_candidate"},
        {"document_key": "property_tax_receipt",
         "field": "property_identifier_candidate"},
        {"document_key": "encumbrance_certificate",
         "field": "property_identifier_candidate"},
    ]

    address_sources = [
        {"document_key": "sale_deed",
         "field": "property_address_candidate"},
        {"document_key": "property_tax_receipt",
         "field": "property_address_candidate"},
        {"document_key": "encumbrance_certificate",
         "field": "property_address_candidate"},
    ]

    buyer_owner_sources = [
        {"document_key": "sale_deed",
         "field": "buyer_name_candidate"},
        {"document_key": "property_tax_receipt",
         "field": "owner_name_candidate"},
    ]

    buyer_owner_limitation = (
        "The buyer named in a sale deed and the owner name printed on "
        "a tax receipt have different documentary contexts. Matching "
        "text is not proof of current ownership or title."
    )

    return [
        compare_field(
            title="Property identifier",
            source_definitions=property_id_sources,
            documents=documents,
        ),
        compare_field(
            title="Property address",
            source_definitions=address_sources,
            documents=documents,
        ),
        compare_field(
            title="Sale-deed buyer versus tax-receipt owner",
            source_definitions=buyer_owner_sources,
            documents=documents,
            limitation=buyer_owner_limitation,
        ),
    ]


# ================================================================
# REPORT GENERATION
# ================================================================

def create_consistency_report(
    report_paths: dict[str, Path],
    output_path: Path | None = None,
) -> dict:
    """Generate and save the cross-document consistency report."""

    documents = prepare_documents(report_paths)
    comparisons = build_comparisons(documents)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "cross_document_consistency",
        "document_sources": {
            key: {
                "report_path": document["report_path"],
                "expected_type": document["expected_type"],
                "actual_type": document["actual_type"],
                "classification_status": document[
                    "classification_status"
                ],
                "type_matches_expectation": document[
                    "type_matches_expectation"
                ],
            }
            for key, document in documents.items()
        },
        "comparisons": comparisons,
        "limitations": [
            DISCLAIMER,
            "Comparison ignores case, whitespace, and punctuation only.",
            "OCR spelling errors may produce false discrepancies.",
            "Missing values do not establish that information is absent "
            "from the original document.",
            "Applicability and legal requirements are not assessed here.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = (
            OUTPUT_DIR / "cross_document_consistency_report.json"
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("CROSS-DOCUMENT CONSISTENCY SUMMARY")
    print("=" * 65)

    for comparison in comparisons:
        print(f"\n{comparison['comparison']}")
        print(f"Status: {comparison['status']}")
        print(comparison["explanation"])

        for source in comparison["sources"]:
            values = [
                item["value"] for item in source["values"]
            ]
            print(
                f"  {source['document_key']} "
                f"({source['field']}): "
                f"{values if values else 'NOT FOUND'}"
            )

    print("\nReport saved to:")
    print(output_path)
    print("\n", DISCLAIMER)

    return report


# ================================================================
# COMMAND-LINE ENTRY POINT
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate fields across the three synthetic "
            "Property AI review reports."
        )
    )

    parser.add_argument(
        "--sale-deed",
        type=Path,
        default=DEFAULT_REPORTS["sale_deed"],
        help="Sale-deed JSON review report.",
    )
    parser.add_argument(
        "--tax-receipt",
        type=Path,
        default=DEFAULT_REPORTS["property_tax_receipt"],
        help="Property tax receipt JSON review report.",
    )
    parser.add_argument(
        "--encumbrance",
        type=Path,
        default=DEFAULT_REPORTS["encumbrance_certificate"],
        help="Encumbrance certificate JSON review report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "cross_document_consistency_report.json",
        help="Destination JSON report path.",
    )

    args = parser.parse_args()

    report_paths = {
        "sale_deed": args.sale_deed,
        "property_tax_receipt": args.tax_receipt,
        "encumbrance_certificate": args.encumbrance,
    }

    try:
        create_consistency_report(
            report_paths=report_paths,
            output_path=args.output,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()