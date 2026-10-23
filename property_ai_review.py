"""
PROPERTY AI - INTERACTIVE DOCUMENT REVIEW

Features:
1. Classify pages using TF-IDF and BERT-Tiny.
2. Extract general candidate fields using contextual boundaries.
3. Extract specialised sale-deed candidates.
4. Keep classification separate from extraction.
5. Save JSON review reports.
6. Evaluate OCR using CER and WER.

This software does not verify ownership, title, authenticity,
registration validity, encumbrances, or legal safety.
"""

import json
import re
import sys
from pathlib import Path

import predict_document as classifier
import sale_deed_extractor


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "review_output"

SUPPORTED_TYPES = {
    "sale_deed",
    "property_tax_receipt",
    "encumbrance_certificate",
}

UNKNOWN_LABEL = "other_or_unknown"

SALE_DEED_FIELDS = {
    "execution_date_candidate",
    "agreement_date_candidate",
    "seller_name_candidate",
    "buyer_name_candidate",
    "owner_name_candidate",
    "property_address_candidate",
    "property_identifier_candidate",
    "sale_consideration_candidate",
    "registration_reference_candidate",
    "registration_date_candidate",
    "amount_candidate",
    "other_amount_candidate",
    "date_candidate",
}

LIMITATIONS = [
    "Classification is experimental and trained on synthetic examples.",
    "Model scores are not calibrated probabilities.",
    "OCR can introduce errors, merge fields, or omit information.",
    "Extracted fields are candidates, not verified facts.",
    "This software does not establish ownership, title, authenticity, "
    "registration validity, encumbrance status, or legal validity.",
    "Verify candidate values against the original document and obtain "
    "qualified professional review.",
]


# ================================================================
# FIELD LABELS AND BOUNDARIES
# ================================================================

FIELD_LABELS = [
    r"seller\s+(?:full\s+)?name",
    r"seller\s+address",
    r"seller\s+(?:id|identification)",
    r"buyer\s+(?:full\s+)?name",
    r"buyer\s+address",
    r"buyer\s+(?:id|identification)",
    r"vendor\s+(?:full\s+)?name",
    r"purchaser\s+(?:full\s+)?name",
    r"transferor\s+(?:full\s+)?name",
    r"transferee\s+(?:full\s+)?name",
    r"(?:present\s+)?(?:property\s+)?owner\s+name",
    r"owner\s+address",
    r"taxpayer\s+name",
    r"owner\s*/\s*taxpayer",
    r"name\s+of\s+payer",
    r"payer\s+name",
    r"assessee\s+name",
    r"name\s+on\s+account",
    r"property\s+address",
    r"address\s+of\s+property",
    r"location\s+of\s+property",
    r"property\s+situated\s+at",
    r"description\s+of\s+property",
    r"premises\s+searched",
    r"premises",
    r"reference\s+property\s+(?:number|no\.?)",
    r"property\s+(?:id|identifier|identification\s+number|number|no\.?|type)",
    r"plot\s+(?:id|number|no\.?)",
    r"flat\s+(?:id|number|no\.?)",
    r"khasra\s+(?:number|no\.?)",
    r"khata\s+(?:number|no\.?)",
    r"survey\s+(?:number|no\.?)",
    r"municipal\s+property\s+(?:id|number|no\.?)",
    r"floor\s+area",
    r"property\s+category",
    r"property\s+type",
    r"sale\s+consideration(?:\s+amount)?",
    r"consideration\s+amount",
    r"amount\s+in\s+words",
    r"sale\s+price",
    r"purchase\s+price",
    r"registration\s+(?:number|no\.?|date|office)",
    r"date\s+of\s+registration",
    r"date\s+of\s+execution",
    r"execution\s+date",
    r"agreement\s+date",
    r"penalty\s+amount",
    r"additional\s+charges",
    r"property\s+tax\s+amount",
    r"tax\s+amount",
    r"tax\s+remitted",
    r"late\s+payment\s+charge",
    r"total\s+amount\s+paid",
    r"total\s+tax\s+paid",
    r"amount\s+paid",
    r"payment\s+status",
    r"status",
    r"payment\s+reference",
    r"payment\s+date",
    r"receipt\s+(?:number|no\.?|reference)",
    r"assessment\s+year",
    r"period\s+of\s+assessment",
    r"certificate\s+(?:reference|number|no\.?)",
    r"application\s+(?:number|no\.?)",
    r"issue\s+date",
    r"certificate\s+date",
    r"search\s+period",
    r"search\s+dates",
    r"period\s+examined",
    r"description\s+of\s+property",
    r"assessment\s+(?:id|identifier|number|no\.?|reference)",
    r"municipal\s+assessment\s+(?:number|no\.?|reference)",
    r"property\s+reference",
    r"total\s+amount\s+received",
    r"total\s+received",
    r"net\s+amount\s+received",
    r"amount\s+received",
    r"search\s+results?",
    r"name\s+of\s+applicant",
    r"application\s+by",
    r"applicant\s+name",
    r"purpose",
    r"notice\s+date",
    r"subject",
    r"date",
    r"from",
    r"to",
]

FIELD_LABELS_PATTERN = "|".join(
    sorted(FIELD_LABELS, key=len, reverse=True)
)

# Values stop at the next recognised field label, section heading,
# disclaimer, or end of text.
FIELD_STOP = (
    rf"(?=\s+(?:{FIELD_LABELS_PATTERN})\s*[:#\-]"
    r"|\s+\d{1,2}\.\s+[A-Z]"
    r"|\s+(?:this is a fictional|this fictional|"
    r"this specimen|this statement|all names|the names|"
    r"synthetic test document|synthetic document series|"
    r"no legal effect|invented names)\b"
    r"|\s+end of specimen\b|$)"
)

# ================================================================
# PATH AND TEXT HELPERS
# ================================================================

def resolve_path(raw_path: str) -> Path:
    """Resolve an absolute or project-relative input path."""
    cleaned = raw_path.strip().strip('"').strip("'")
    path = Path(cleaned).expanduser()

    if path.is_absolute():
        return path

    if path.exists():
        return path.resolve()

    return (PROJECT_DIR / path).resolve()


def normalise_text(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def evidence_around(
    text: str,
    start: int,
    end: int,
    radius: int = 180,
) -> str:
    """Build a compact evidence snippet around a matched value."""
    start_context = max(0, start - radius)
    end_context = min(len(text), end + radius)

    snippet = text[start_context:end_context].strip()

    if start_context > 0:
        snippet = "..." + snippet

    if end_context < len(text):
        snippet += "..."

    return snippet[:700]


def get_value(data: dict, *keys, default=None):
    """Read the first available key from a dictionary."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def safe_float(value, default=0.0) -> float:
    """Safely convert a value to a float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_baseline_label(prediction: dict) -> str:
    return str(
        get_value(
            prediction,
            "baseline_label",
            "tfidf_label",
            default=UNKNOWN_LABEL,
        )
    )


def get_baseline_score(prediction: dict) -> float:
    return safe_float(
        get_value(
            prediction,
            "baseline_score",
            "tfidf_score",
            default=0.0,
        )
    )


def get_bert_label(prediction: dict) -> str:
    return str(
        get_value(
            prediction,
            "bert_label",
            default=UNKNOWN_LABEL,
        )
    )


def get_bert_score(prediction: dict) -> float:
    return safe_float(
        get_value(
            prediction,
            "bert_score",
            default=0.0,
        )
    )


def get_classifier_status(prediction: dict) -> str:
    return str(
        get_value(
            prediction,
            "status",
            default="needs_review",
        )
    )


# ================================================================
# GENERAL FIELD EXTRACTION
# ================================================================

def extract_candidates_from_page(
    text: str,
    page_number: int,
) -> list[dict]:
    """
    Extract general candidate fields from page text.

    Extracted values are unverified candidates. The function does not
    establish that a field is legally valid or absent from a document.
    """

    flat_text = normalise_text(text)

    if not flat_text:
        return []

    candidates = []
    seen = set()

    def add(
        field: str,
        value: str,
        start: int,
        end: int,
        note: str,
    ) -> None:
        """Add a deduplicated candidate with evidence and review metadata."""

        value = re.sub(r"\s+", " ", str(value or ""))
        value = value.strip(" \t\r\n:;,.")

        value = re.sub(
            r"\s+(?:this fictional|this specimen|this statement|"
            r"all names|the names)\b.*$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n:;,.")

        if not value:
            return

        key = (field, value.casefold(), page_number)

        if key in seen:
            return

        seen.add(key)

        candidates.append({
            "field": field,
            "candidate_value": value,
            "page_number": page_number,
            "evidence": evidence_around(
                flat_text,
                start,
                end,
            ),
            "status": "needs_review",
            "method": "general_context_regex",
            "review_note": note,
        })

    def add_labelled(
        field: str,
        labels: str,
        note: str,
    ) -> None:
        """Extract a value after a labelled field with stop conditions."""

        pattern = re.compile(
            rf"(?P<label>\b(?:{labels})\s*[:#\-]\s*)"
            rf"(?P<value>.+?){FIELD_STOP}",
            re.IGNORECASE,
        )

        for match in pattern.finditer(flat_text):
            value = match.group("value").strip()
            start, end = match.span("value")

            add(
                field,
                value,
                start,
                end,
                note,
            )

    # ------------------------------------------------------------
    # DATE EXTRACTION WITH CONTEXTUAL FIELD NAMES
    # ------------------------------------------------------------

    date_pattern = re.compile(
        r"\b(?:"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        r"|"
        r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
        r"|"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r")\b",
        re.IGNORECASE,
    )

    weekday_pattern = (
        r"(?:monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday)"
    )


    # EXPLICIT_SEARCH_PERIOD_RANGE_EXTRACTION
    search_period_range_pattern = re.compile(
        r"\b(?:search\s+period|search\s+dates|period\s+examined)\b\s*[:#\-–]?\s*"
        r"(?:from\s*[:#\-–]?\s*)?"
        r"(?P<start>(?:"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"))"
        r"\s*(?:\bto\b|\bthrough\b|\buntil\b|[-–—])"
        r"\s*[:#\-–]?\s*"
        r"(?P<end>(?:"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}"
        r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"))\b",
        re.IGNORECASE,
    )

    # Record date spans handled by the explicit Search Period parser so
    # the generic date loop below does not emit duplicate date_candidate
    # entries for the same dates.
    handled_search_period_date_spans = set()

    for period_match in search_period_range_pattern.finditer(flat_text):
        start_value = period_match.group("start")
        start_position = period_match.span("start")
        handled_search_period_date_spans.add(start_position)

        add(
            "search_period_start_candidate",
            start_value,
            start_position[0],
            start_position[1],
            (
                "Start date parsed from the labelled Search Period range. "
                "Verify against the original document."
            ),
        )

        end_value = period_match.group("end")
        end_position = period_match.span("end")
        handled_search_period_date_spans.add(end_position)

        add(
            "search_period_end_candidate",
            end_value,
            end_position[0],
            end_position[1],
            (
                "End date parsed from the labelled Search Period range. "
                "Verify against the original document."
            ),
        )

    for match in date_pattern.finditer(flat_text):
        # Skip a date already recorded as a Search Period boundary.
        # This prevents duplicate generic date_candidate entries.
        if any(
            match.start() < span_end and match.end() > span_start
            for span_start, span_end in handled_search_period_date_spans
        ):
            continue

        before = flat_text[
            max(0, match.start() - 180):match.start()
        ]

        after = flat_text[
            match.end():min(len(flat_text), match.end() + 180)
        ]

        context = f"{before} {match.group(0)} {after}"

        # Detect dates listed under blocks, for example:
        # Block A - Monday, 12 October 2026
        # Block B - Tuesday, 13 October 2026
        # Block C - Wednesday, 14 October 2026
        block_match = re.search(
            rf"\bblock\s+([abc])\s*[-–—:]\s*"
            rf"(?:(?:{weekday_pattern})\s*,?\s*)?$",
            before,
            re.IGNORECASE,
        )

        if block_match:
            block_letter = block_match.group(1).lower()
            field = (
                f"block_{block_letter}_maintenance_date_candidate"
            )
            note = (
                "Date appears to be the scheduled maintenance date "
                f"for Block {block_letter.upper()}. "
                "Verify against the original document."
            )

        elif re.search(
            r"(?:date\s+of\s+execution|execution\s+date|executed\s+on)"
            r"\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "execution_date_candidate"
            note = (
                "Date appears near execution-related wording. "
                "Verify against the original document."
            )

        elif re.search(
            r"(?:registration\s+date|date\s+of\s+registration)"
            r"\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "registration_date_candidate"
            note = (
                "Date appears near registration-related wording. "
                "Verify against the original document."
            )

        elif re.search(
            r"agreement\s+date\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "agreement_date_candidate"
            note = (
                "Date appears near agreement-date wording. "
                "Verify against the original document."
            )

        elif re.search(
            r"payment\s+date\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "payment_date_candidate"
            note = (
                "Date appears near payment-date wording. "
                "Verify against the original document."
            )

        elif re.search(
            r"(?:issue\s+date|certificate\s+date)\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            if re.search(
                r"encumbrance\s+certificate|certificate\s+reference",
                context,
                re.IGNORECASE,
            ):
                field = "certificate_issue_date_candidate"
            else:
                field = "issue_date_candidate"

            note = (
                "Date appears near issue-date wording. "
                "Verify against the original document."
            )

        elif (
            re.search(r"\bfrom\s*[:\-]?\s*$", before, re.IGNORECASE)
            and re.search(r"search\s+period", before[-180:], re.IGNORECASE)
        ):
            field = "search_period_start_candidate"
            note = (
                "Date appears to be the start of a search period. "
                "Verify the period against the original document."
            )

        elif (
            re.search(r"\bto\s*[:\-]?\s*$", before, re.IGNORECASE)
            and re.search(r"search\s+period", before[-180:], re.IGNORECASE)
        ):
            field = "search_period_end_candidate"
            note = (
                "Date appears to be the end of a search period. "
                "Verify the period against the original document."
            )

        elif re.search(
            r"\b(?:notice\s+date|date\s+of\s+notice)"
            r"\s*[:#\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "notice_date_candidate"
            note = (
                "Date appears to be a notice date. "
                "Verify against the original document."
            )

        elif (
            re.search(r"\bdate\s*:\s*$", before, re.IGNORECASE)
            and re.search(
                r"\b(?:maintenance notice|notice of planned maintenance)\b",
                context,
                re.IGNORECASE,
            )
        ):
            field = "notice_date_candidate"
            note = (
                "Date appears to be the date of the maintenance notice. "
                "Verify against the original document."
            )

        elif (
            re.search(
                r"\b(?:week of|during|from|until|starting on|"
                r"commencing on)\s*$",
                before,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:maintenan\w*|servic\w*|scheduled|"
                r"water tanks|lighting)\b",
                context,
                re.IGNORECASE,
            )
        ):
            field = "scheduled_event_date_candidate"
            note = (
                "Date appears related to a scheduled event or maintenance. "
                "Verify the intended date and schedule against the original."
            )

        else:
            field = "date_candidate"
            note = (
                "Date detected, but its meaning is unclear. "
                "Check the surrounding text and original page."
            )

        add(
            field,
            match.group(0),
            match.start(),
            match.end(),
            note,
        )

    # ------------------------------------------------------------
    # MONETARY AMOUNTS
    # ------------------------------------------------------------

    money_pattern = re.compile(
        r"(?:"
        r"₹\s*\d[\d,]*(?:\.\d{1,2})?"
        r"|"
        r"\bINR\s*\d[\d,]*(?:\.\d{1,2})?"
        r"|"
        r"\bRs\.?\s*\d[\d,]*(?:\.\d{1,2})?"
        r")",
        re.IGNORECASE,
    )

    for match in money_pattern.finditer(flat_text):
        before = flat_text[
            max(0, match.start() - 150):match.start()
        ]

        if re.search(
                r"penalty\s+amount\s*[:\-]?\s*$",
                before,
                re.IGNORECASE,
        ):
            field = "penalty_amount_candidate"
            note = (
                "Amount appears near penalty-amount wording. "
                "Verify the amount and currency against the original document."
            )

        elif re.search(
                r"(?:late\s+payment\s+charge|late\s+fee)\s*[:\-]?\s*$",
                before,
                re.IGNORECASE,
        ):
            field = "late_payment_charge_candidate"
            note = (
                "Amount appears near late-payment-charge wording. "
                "Verify amount and currency against the original."
            )

        elif re.search(
            r"additional\s+charges\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "additional_charges_candidate"
            note = (
                "Amount appears near additional-charges wording. This label "
                "does not necessarily mean a penalty; verify against the source."
            )

        # EXPLICIT_AMOUNT_RECEIVED_PAID_MAPPING
        elif re.search(
            r"(?:total\s+amount\s+received|total\s+received|"
            r"net\s+amount\s+received|amount\s+received)\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "total_amount_paid_candidate"
            note = (
                "Amount appears near amount-received wording. "
                "Verify the meaning and amount against the original document."
            )

        elif re.search(
            r"(?:total\s+amount\s+paid|total\s+tax\s+paid|"
            r"amount\s+paid|total\s+paid)\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "total_amount_paid_candidate"
            note = (
                "Amount appears near total-paid wording. "
                "Verify amount and currency against the original."
            )

        elif re.search(
            r"(?:property\s+tax\s+amount|tax\s+amount|"
            r"property\s+tax\s+collected|tax\s+collected|"
            r"tax\s+deposited|tax\s+remitted)\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "tax_amount_candidate"
            note = (
                "Amount appears near tax-amount wording. "
                "Verify amount and currency against the original."
            )

        elif re.search(
            r"(?:sale\s+consideration(?:\s+amount)?|"
            r"consideration\s+amount|total\s+consideration|"
            r"sale\s+price|purchase\s+price)\s*[:\-]?\s*$",
            before,
            re.IGNORECASE,
        ):
            field = "sale_consideration_candidate"
            note = (
                "Amount appears near sale-price wording. "
                "Verify amount and currency against the original."
            )

        else:
            field = "amount_candidate"
            note = (
                "Amount detected. Confirm what it represents "
                "and verify it against the original document."
            )

        add(
            field,
            match.group(0),
            match.start(),
            match.end(),
            note,
        )

    # ------------------------------------------------------------
    # PARTY NAMES
    # ------------------------------------------------------------

    party_note = (
        "Name-like value detected. Confirm the name and legal role "
        "against the original document."
    )

    add_labelled(
        "seller_name_candidate",
        r"(?:seller|vendor|transferor|executant)(?:'s)?\s+(?:full\s+)?name",
        party_note,
    )

    add_labelled(
        "buyer_name_candidate",
        r"(?:buyer|purchaser|transferee)(?:'s)?\s+(?:full\s+)?name",
        party_note,
    )

    add_labelled(
        "owner_or_taxpayer_name_candidate",
        r"owner\s*/\s*taxpayer",
        (
            "The source labels this person as owner/taxpayer. The role is "
            "ambiguous and must not be treated as verified property ownership."
        ),
    )

    add_labelled(
        "owner_name_candidate",
        r"(?:present\s+)?(?:property\s+)?owner(?:'s)?\s+name",
        (
            "Owner-like label detected. Verify that the document "
            "explicitly identifies this person as an owner."
        ),
    )

    add_labelled(
        "payer_name_candidate",
        r"(?:name\s+of\s+payer|payer\s+name)",
        (
            "Payer name detected. This does not establish "
            "property ownership. Verify against the source document."
        ),
    )

    add_labelled(
        "taxpayer_name_candidate",
        r"(?:taxpayer\s+name|assessee\s+name|name\s+on\s+account)",
        (
            "Taxpayer or account-name label detected. This does not "
            "establish property ownership. Verify against the source."
        ),
    )

    # ------------------------------------------------------------
    # PROPERTY ADDRESS
    # ------------------------------------------------------------

    add_labelled(
        "property_address_candidate",
        r"(?:property\s+address|address\s+of\s+property|"
        r"location\s+of\s+property|premises\s+searched|"
        r"property\s+situated\s+at|description\s+of\s+property|premises)",
        (
            "Address-like text detected. Verify the complete property "
            "address against the original document."
        ),
    )

    # ------------------------------------------------------------
    # PROPERTY IDENTIFIER
    # ------------------------------------------------------------

    property_id_pattern = re.compile(
        r"\b(?:"
        r"property\s+(?:id|identifier|identification\s+number|number|no\.?)"
        r"|plot\s+(?:id|number|no\.?)"
        r"|flat\s+(?:id|number|no\.?)"
        r"|khasra\s+(?:number|no\.?)"
        r"|khata\s+(?:number|no\.?)"
        r"|survey\s+(?:number|no\.?)"
        r"|municipal\s+property\s+(?:id|number|no\.?)"
        r"|municipal\s+assessment\s+(?:id|identifier|number|no\.?)"
        r"|assessment\s+(?:id|identifier|number|no\.?)"
        r"|property\s+reference"
        r")\s*[:#\-]\s*"
        r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
        re.IGNORECASE,
    )

    for match in property_id_pattern.finditer(flat_text):
        # ASSESSMENT_ID_NOT_GENERIC_PROPERTY_ID
        matched_label = flat_text[match.start():match.start(1)]

        if re.search(
            r"\bassessment\s+(?:id|identifier|number|no\.?)\b",
            matched_label,
            re.IGNORECASE,
        ):
            continue

        add(
            "property_identifier_candidate",
            match.group(1),
            match.start(1),
            match.end(1),
            (
                "Property-identifier pattern detected. Verify the "
                "identifier against the original document."
            ),
        )

    # ------------------------------------------------------------
    # MUNICIPAL ASSESSMENT IDENTIFIER
    # ------------------------------------------------------------

    assessment_id_pattern = re.compile(
        r"\b(?:"
        r"municipal\s+assessment\s+"
        r"(?:id|identifier|number|no\.?|reference)"
        r"|assessment\s+"
        r"(?:id|identifier|number|no\.?|reference)"
        r")\s*[:#\-]\s*"
        r"([A-Z0-9][A-Z0-9/_\-]{1,60})"
        r"(?=\s|$|[,;])",
        re.IGNORECASE,
    )

    for match in assessment_id_pattern.finditer(flat_text):
        add(
            "assessment_identifier_candidate",
            match.group(1),
            match.start(1),
            match.end(1),
            (
                "Assessment identifier detected. Verify the identifier "
                "and its meaning against the original document."
            ),
        )

    # ------------------------------------------------------------
    # DOCUMENT REFERENCES
    # ------------------------------------------------------------

    reference_patterns = [
        (
            "registration_reference_candidate",
            r"\bregistration\s*(?:no\.?|number)\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
            "Registration reference detected. Verify against the original document.",
        ),
        (
            "receipt_reference_candidate",
            r"\breceipt\s*(?:no\.?|number|reference)\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
            "Receipt reference detected. Verify against the original document.",
        ),
        (
            "certificate_reference_candidate",
            r"\bcertificate\s*(?:reference|no\.?|number)\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
            "Certificate reference detected. Verify against the original document.",
        ),
        (
            "payment_reference_candidate",
            r"\bpayment\s+reference\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
            "Payment reference detected. Verify against the original document.",
        ),
        (
            "application_reference_candidate",
            r"\bapplication\s*(?:no\.?|number)\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/_\-]{1,60})",
            "Application reference detected. Verify against the original document.",
        ),
    ]

    for field, expression, note in reference_patterns:
        for match in re.finditer(
            expression,
            flat_text,
            re.IGNORECASE,
        ):
            add(
                field,
                match.group(1),
                match.start(1),
                match.end(1),
                note,
            )

    # ------------------------------------------------------------
    # ASSESSMENT YEAR
    # ------------------------------------------------------------

    assessment_year_pattern = re.compile(
        r"\b(?:assessment\s+year|financial\s+year|tax\s+year|"
        r"assessment\s+period|period\s+of\s+assessment|tax\s+period)\s*[:#\-]?\s*"
        r"((?:19|20)\d{2}\s*[-/–]\s*(?:(?:19|20)?\d{2}))\b",
        re.IGNORECASE,
    )

    for match in assessment_year_pattern.finditer(flat_text):
        value = re.sub(r"\s+", "", match.group(1))

        add(
            "assessment_year_candidate",
            value,
            match.start(1),
            match.end(1),
            (
                "Assessment year detected. Verify it against the "
                "original receipt."
            ),
        )

    # ------------------------------------------------------------
    # PAYMENT STATUS AND APPLICANT NAME
    # ------------------------------------------------------------

    payment_status_labels = r"payment\s+status"

    if re.search(
        r"(?:property|municipal|tax|dues).{0,100}\breceipt\b",
        flat_text,
        re.IGNORECASE,
    ):
        payment_status_labels = r"(?:payment\s+status|status)"

    add_labelled(
        "payment_status_candidate",
        payment_status_labels,
        "Payment status detected. Verify it against the original receipt.",
    )

    add_labelled(
        "applicant_name_candidate",
        r"(?:applicant\s+name|name\s+of\s+applicant|application\s+by|applicant)",
        "Applicant name detected. Confirm the name against the original document.",
    )

    return candidates


# ================================================================
# SPECIALISED SALE-DEED EXTRACTION
# ================================================================

def clean_specialized_candidate(
    field: str,
    value: str,
) -> str:
    """Clean obvious adjacent-field text without inventing values."""

    value = normalise_text(value)

    if field in {
        "seller_name_candidate",
        "buyer_name_candidate",
        "owner_name_candidate",
    }:
        value = re.split(
            r"\s+(?:(?:seller|buyer|vendor|purchaser|owner)\s+)?"
            r"(?:address|identification|id|premises)\s*:",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        value = re.sub(
            r"\s+(?:seller|buyer|vendor|purchaser|transferor|transferee)$",
            "",
            value,
            flags=re.IGNORECASE,
        )

    elif field == "property_address_candidate":
        value = re.split(
            r"\s+(?:"
            r"reference\s+property\s+number"
            r"|property\s+(?:id|type|number)"
            r"|floor\s+area"
            r"|sale\s+consideration"
            r"|registration\s+(?:number|date)"
            r")\s*[:\-]",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

    return value.strip(" \t\r\n:;,.")


def extract_sale_deed_candidates_from_page(
    text: str,
    page_number: int,
) -> list[dict]:
    """Call sale_deed_extractor.py and normalize its output."""

    raw_candidates = sale_deed_extractor.extract_page_candidates(
        text,
        page_number,
    )

    candidates = []
    seen = set()

    for item in raw_candidates:
        field = str(item.get("field", "unknown_field"))

        if field not in SALE_DEED_FIELDS:
            continue

        value = clean_specialized_candidate(
            field,
            item.get("candidate_value", ""),
        )

        if not value:
            continue

        key = (field, value.casefold(), page_number)

        if key in seen:
            continue

        seen.add(key)

        candidates.append({
            "field": field,
            "candidate_value": value,
            "page_number": item.get("page_number", page_number),
            "evidence": str(item.get("evidence", ""))[:700],
            "status": "needs_review",
            "method": "sale_deed_extractor.py",
            "review_note": (
                "Candidate extracted by sale_deed_extractor.py. "
                "Verify the value against the original document."
            ),
        })

    return candidates


# ================================================================
# PAGE-LEVEL CLASSIFICATION EXPLANATIONS
# ================================================================

def explain_page_prediction(
    baseline_label: str,
    bert_label: str,
    classifier_status: str,
) -> str:

    if baseline_label != bert_label:
        return (
            "The TF-IDF and BERT-Tiny models disagree. "
            "The document type cannot be resolved automatically."
        )

    if baseline_label == UNKNOWN_LABEL:
        return (
            "Both models agree on an unsupported or unknown document type. "
            "The document may be outside the current classifier's scope."
        )

    if baseline_label in SUPPORTED_TYPES:
        if classifier_status == "candidate_for_manual_review":
            return (
                f"Both models agree on {baseline_label}, and classifier "
                "acceptance rules passed. Original-document review is "
                "still required."
            )

        return (
            f"Both models agree on {baseline_label}, but the classifier "
            "acceptance rules did not pass. Manual review is required."
        )

    return (
        "The models returned a label outside the currently supported "
        "document types. Manual review is required."
    )


# ================================================================
# DOCUMENT-LEVEL CLASSIFICATION
# ================================================================

def classify_document_type(page_results: list[dict]) -> dict:
    """Determine document review status without overriding safeguards."""

    if not page_results:
        return {
            "document_status": "unknown_or_needs_review",
            "reason": "No readable pages were found.",
            "agreed_document_type": None,
            "agreed_type_candidates": [],
            "suggestions_passing_review_rules": [],
            "acceptance_rules_passed": False,
        }

    baseline_labels = [
        page["baseline_label"] for page in page_results
    ]
    bert_labels = [
        page["bert_label"] for page in page_results
    ]

    models_disagree = any(
        baseline != bert
        for baseline, bert in zip(baseline_labels, bert_labels)
    )

    all_unknown = all(
        baseline == UNKNOWN_LABEL and bert == UNKNOWN_LABEL
        for baseline, bert in zip(baseline_labels, bert_labels)
    )

    agreed_supported_types = {
        baseline
        for baseline, bert in zip(baseline_labels, bert_labels)
        if baseline == bert and baseline in SUPPORTED_TYPES
    }

    all_pages_agree_on_one_supported_type = (
        not models_disagree
        and len(set(baseline_labels)) == 1
        and len(set(bert_labels)) == 1
        and baseline_labels[0] == bert_labels[0]
        and baseline_labels[0] in SUPPORTED_TYPES
    )

    all_acceptance_rules_passed = all(
        page["classifier_status"] == "candidate_for_manual_review"
        for page in page_results
    )

    if all_unknown:
        return {
            "document_status": "unknown_or_needs_review",
            "reason": (
                "Both models predict an unsupported or unknown document "
                "type. The document may be outside the classifier's "
                "current scope."
            ),
            "agreed_document_type": None,
            "agreed_type_candidates": [],
            "suggestions_passing_review_rules": [],
            "acceptance_rules_passed": False,
        }

    if models_disagree:
        return {
            "document_status": "unknown_or_needs_review",
            "reason": (
                "The models disagree on the document type on one or more "
                "pages. Review the predictions and original document "
                "before relying on extracted information."
            ),
            "agreed_document_type": None,
            "agreed_type_candidates": sorted(agreed_supported_types),
            "suggestions_passing_review_rules": [],
            "acceptance_rules_passed": False,
        }

    if all_pages_agree_on_one_supported_type:
        candidate_type = baseline_labels[0]

        if all_acceptance_rules_passed:
            return {
                "document_status": "candidate_for_manual_review",
                "reason": (
                    "Both models agree on one supported document type and "
                    "the classifier acceptance rules passed. This remains "
                    "a candidate for human review, not legal verification."
                ),
                "agreed_document_type": candidate_type,
                "agreed_type_candidates": [candidate_type],
                "suggestions_passing_review_rules": [candidate_type],
                "acceptance_rules_passed": True,
            }

        return {
            "document_status": "provisional_type_candidate_needs_review",
            "reason": (
                "Both models agree on a supported document type, but one "
                "or more classifier acceptance rules did not pass. Review "
                "the predictions and original document."
            ),
            "agreed_document_type": None,
            "agreed_type_candidates": [candidate_type],
            "suggestions_passing_review_rules": [],
            "acceptance_rules_passed": False,
        }

    return {
        "document_status": "unknown_or_needs_review",
        "reason": (
            "The processed pages do not establish one consistent "
            "supported document type. The document may contain mixed "
            "types or pages outside the supported classifier scope."
        ),
        "agreed_document_type": None,
        "agreed_type_candidates": sorted(agreed_supported_types),
        "suggestions_passing_review_rules": [],
        "acceptance_rules_passed": False,
    }


# ================================================================
# PRINT HELPERS
# ================================================================

def print_candidate_section(
    title: str,
    candidates: list[dict],
) -> None:
    print(f"\n{title}:")

    if not candidates:
        print("  None detected.")
        return

    for item in candidates:
        print(
            f"  - {item['field']}: {item['candidate_value']} "
            f"(page {item['page_number']})"
        )
        print("      Evidence:", item["evidence"])
        print("      Review note:", item["review_note"])


# ================================================================
# MAIN DOCUMENT REVIEW
# ================================================================

def load_review_pages(path: Path) -> list[dict]:
    return classifier.load_document(path)


def classify_document(input_path: Path) -> dict:
    """Classify a document, extract candidates, and save a JSON report."""

    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Document file not found:\n{input_path}"
        )

    pages = load_review_pages(input_path)

    if not pages:
        raise ValueError("No readable pages were returned.")

    print("\nLoading TF-IDF baseline...")
    baseline = classifier.load_baseline()

    print("\nLoading BERT-Tiny...")
    tokenizer, transformer, device, label_mapping = (
        classifier.load_transformer()
    )

    print("Device:", device)

    page_results = []
    general_candidates = []
    sale_deed_candidates = []
    sale_deed_pages = []

    for page_index, page in enumerate(pages, start=1):
        page_number = page.get("page_number", page_index)
        page_text = str(page.get("text", "") or "")

        if not page_text.strip():
            print(f"\nSkipping empty page {page_number}.")
            continue

        print(f"\nProcessing page {page_number}...")

        prediction = classifier.classify_page(
            page_text,
            baseline,
            tokenizer,
            transformer,
            device,
            label_mapping,
        )

        baseline_label = get_baseline_label(prediction)
        baseline_score = get_baseline_score(prediction)
        bert_label = get_bert_label(prediction)
        bert_score = get_bert_score(prediction)
        classifier_status = get_classifier_status(prediction)

        page_explanation = explain_page_prediction(
            baseline_label,
            bert_label,
            classifier_status,
        )

        print(f"  TF-IDF prediction: {baseline_label}")
        print(f"  BERT-Tiny prediction: {bert_label}")
        print(f"  Explanation: {page_explanation}")

        # General extraction runs independently of classification.
        general_page_candidates = extract_candidates_from_page(
            page_text,
            page_number,
        )
        general_candidates.extend(general_page_candidates)

        # Attempt specialised extraction if either model predicts sale_deed.
        sale_deed_attempted = (
            baseline_label == "sale_deed"
            or bert_label == "sale_deed"
        )

        specialized_page_candidates = []

        if sale_deed_attempted:
            sale_deed_pages.append(page_number)

            specialized_page_candidates = (
                extract_sale_deed_candidates_from_page(
                    page_text,
                    page_number,
                )
            )
            sale_deed_candidates.extend(specialized_page_candidates)
            print("  Sale-deed extraction: attempted")
        else:
            print("  Sale-deed extraction: skipped")

        page_results.append({
            "page_number": page_number,
            "extraction_method": page.get(
                "extraction_method",
                page.get("method", "unknown"),
            ),
            "word_count": len(page_text.split()),
            "prediction": prediction,
            "baseline_label": baseline_label,
            "baseline_score": baseline_score,
            "bert_label": bert_label,
            "bert_score": bert_score,
            "models_agree": baseline_label == bert_label,
            "classifier_status": classifier_status,
            "page_explanation": page_explanation,
            "candidate_fields": general_page_candidates,
            "sale_deed_extraction_attempted": sale_deed_attempted,
            "sale_deed_candidate_fields": specialized_page_candidates,
        })

    decision = classify_document_type(page_results)

    report = {
        "source_file": str(input_path),
        "document_status": decision["document_status"],
        "reason": decision["reason"],
        "agreed_document_type": decision["agreed_document_type"],
        "agreed_type_candidates": decision["agreed_type_candidates"],
        "suggestions_passing_review_rules": (
            decision["suggestions_passing_review_rules"]
        ),
        "acceptance_rules_passed": decision["acceptance_rules_passed"],
        "general_candidate_fields": general_candidates,
        "sale_deed_extraction_attempted": bool(sale_deed_pages),
        "sale_deed_pages": sale_deed_pages,
        "sale_deed_candidate_fields": sale_deed_candidates,
        "pages": page_results,
        "limitations": LIMITATIONS,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}_document_review.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("DOCUMENT REVIEW SUMMARY")
    print("=" * 65)

    print("Document status:", report["document_status"])
    print("Reason:", report["reason"])
    print("Agreed document type:", report["agreed_document_type"])
    print(
        "Agreed type candidates (provisional):",
        report["agreed_type_candidates"],
    )
    print(
        "Suggestions passing review rules:",
        report["suggestions_passing_review_rules"],
    )
    print(
        "Acceptance rules passed:",
        report["acceptance_rules_passed"],
    )
    print("General candidate fields:", len(general_candidates))
    print(
        "Sale-deed extraction attempted:",
        report["sale_deed_extraction_attempted"],
    )
    print("Sale-deed pages:", sale_deed_pages)
    print("Sale-deed candidate fields:", len(sale_deed_candidates))

    print("\nClassification details:")

    for item in page_results:
        print(f"\n  Page {item['page_number']}")
        print(
            f"    TF-IDF: {item['baseline_label']} "
            f"(score: {item['baseline_score']:.4f})"
        )
        print(
            f"    BERT-Tiny: {item['bert_label']} "
            f"(score: {item['bert_score']:.4f})"
        )
        print("    Models agree:", item["models_agree"])
        print("    Classifier status:", item["classifier_status"])
        print("    Explanation:", item["page_explanation"])

    print_candidate_section(
        "General candidate fields",
        general_candidates,
    )
    print_candidate_section(
        "Specialized sale-deed candidate fields",
        sale_deed_candidates,
    )

    print("\nAll extracted candidates require manual verification.")
    print("Report saved to:")
    print(output_path)

    return report


# ================================================================
# OCR EVALUATION
# ================================================================

def edit_distance(reference: list, hypothesis: list) -> int:
    """Compute Levenshtein edit distance with linear auxiliary memory."""

    previous = list(range(len(hypothesis) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1]


def read_ocr_text(path: Path) -> str:
    """Read TXT output or join page text from supported OCR JSON."""

    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    if path.suffix.lower() == ".json":
        pages = classifier.load_document(path)

        return "\n".join(
            str(page.get("text", "") or "")
            for page in pages
        )

    return path.read_text(encoding="utf-8-sig")


def evaluate_ocr(
    ocr_path: Path,
    ground_truth_path: Path,
) -> dict:
    """Calculate CER and WER against a manually corrected transcription."""

    ocr_path = ocr_path.expanduser().resolve()
    ground_truth_path = ground_truth_path.expanduser().resolve()

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR output not found:\n{ocr_path}")

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth text not found:\n{ground_truth_path}"
        )

    ocr_text = read_ocr_text(ocr_path)
    truth_text = ground_truth_path.read_text(
        encoding="utf-8-sig"
    )

    reference_chars = list(truth_text)
    hypothesis_chars = list(ocr_text)
    reference_words = truth_text.split()
    hypothesis_words = ocr_text.split()

    char_errors = edit_distance(
        reference_chars,
        hypothesis_chars,
    )
    word_errors = edit_distance(
        reference_words,
        hypothesis_words,
    )

    result = {
        "ocr_file": str(ocr_path),
        "ground_truth_file": str(ground_truth_path),
        "character_error_rate": (
            char_errors / max(1, len(reference_chars))
        ),
        "word_error_rate": (
            word_errors / max(1, len(reference_words))
        ),
        "character_edit_distance": char_errors,
        "word_edit_distance": word_errors,
        "reference_characters": len(reference_chars),
        "reference_words": len(reference_words),
        "ocr_characters": len(hypothesis_chars),
        "ocr_words": len(hypothesis_words),
        "note": (
            "Ground truth must be a manually corrected transcription "
            "of the same source document. Lower CER and WER indicate "
            "a closer OCR transcription."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"{ocr_path.stem}_ocr_evaluation.json"
    )

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nOCR EVALUATION")
    print("=" * 65)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nReport saved to:")
    print(output_path)

    return result


# ================================================================
# INTERACTIVE MENU
# ================================================================

def interactive_menu() -> None:
    """Show the interactive menu."""

    while True:
        print("\n" + "=" * 65)
        print("PROPERTY AI - INTERACTIVE DOCUMENT REVIEW")
        print("=" * 65)
        print("Enter options and file paths at the prompts below.")
        print("You do not need to provide command-line arguments.")
        print("Type Q at the main menu to quit.")

        print("\n" + "-" * 65)
        print("MAIN MENU")
        print("-" * 65)
        print("1. Classify document and extract candidate fields")
        print("2. Evaluate OCR against ground-truth text")
        print("Q. Quit")

        choice = input("\nChoose an option: ").strip().lower()

        if choice == "q":
            print("Exiting Property AI review.")
            return

        if choice == "1":
            print("\nEnter an OCR JSON or TXT file path.")
            print(r"Example: .\synthetic_sale_deed.txt")

            raw_path = input("Document path: ").strip()

            if raw_path.lower() in {"q", "quit", "exit"}:
                continue

            if not raw_path:
                print("No path entered.")
                continue

            try:
                classify_document(resolve_path(raw_path))
            except Exception as exc:
                print("\nERROR: Document review failed.")
                print(f"{type(exc).__name__}: {exc}")

            continue

        if choice == "2":
            raw_ocr = input(
                "\nOCR output path (TXT or supported JSON): "
            ).strip()

            if raw_ocr.lower() in {"q", "quit", "exit"}:
                continue

            raw_truth = input(
                "Ground-truth TXT path: "
            ).strip()

            if raw_truth.lower() in {"q", "quit", "exit"}:
                continue

            if not raw_ocr or not raw_truth:
                print("Both paths are required.")
                continue

            try:
                evaluate_ocr(
                    resolve_path(raw_ocr),
                    resolve_path(raw_truth),
                )
            except Exception as exc:
                print("\nERROR: OCR evaluation failed.")
                print(f"{type(exc).__name__}: {exc}")

            continue

        print("Invalid option. Choose 1, 2, or Q.")


# ================================================================
# OPTIONAL COMMAND-LINE MODE
# ================================================================

def main() -> None:
    if len(sys.argv) == 1:
        interactive_menu()
        return

    command = sys.argv[1].strip().lower()

    try:
        if command == "classify":
            if len(sys.argv) < 3:
                raise ValueError(
                    "Provide an OCR JSON or TXT path."
                )

            classify_document(resolve_path(sys.argv[2]))

        elif command == "evaluate-ocr":
            if len(sys.argv) < 4:
                raise ValueError(
                    "Provide both OCR output and ground-truth TXT paths."
                )

            evaluate_ocr(
                resolve_path(sys.argv[2]),
                resolve_path(sys.argv[3]),
            )

        else:
            raise ValueError(
                "Unknown command. Use no arguments for the menu, "
                "'classify', or 'evaluate-ocr'."
            )

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
