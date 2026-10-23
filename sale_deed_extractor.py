
# sale_deed_extractor.py
"""Interactive candidate extractor for OCR text from property documents."""

import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "review_output"


DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r")\b",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"(?:(?:INR|Rs\.?|₹)\s*)\d[\d,]*(?:\.\d{1,2})?"
    r"|\d[\d,]*(?:\.\d{1,2})?\s*(?:rupees|only)",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_pages(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.suffix.lower() == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8-sig"))

        if isinstance(data, dict):
            raw_pages = data.get("pages", [])
        elif isinstance(data, list):
            raw_pages = data
        else:
            raise ValueError("Unsupported JSON structure.")

        pages = []
        for index, item in enumerate(raw_pages, start=1):
            if not isinstance(item, dict):
                continue

            text = str(item.get("text", "")).strip()
            if text:
                pages.append({
                    "page_number": item.get("page_number", item.get("page", index)),
                    "method": item.get(
                        "extraction_method", item.get("method", "unknown")
                    ),
                    "text": text,
                })

        if not pages:
            raise ValueError("No non-empty pages were found in the JSON.")

        return pages

    text = input_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("The text file is empty.")

    return [{"page_number": 1, "method": "text_file", "text": text}]


def extract_page_candidates(text: str, page_number: int) -> list[dict]:
    lines = [
        normalise(line)
        for line in text.splitlines()
        if normalise(line)
    ]

    candidates = []
    seen = set()

    def add(field: str, value: str, index: int):
        value = normalise(value).strip(" \t:;,.")
        key = (field, value.casefold(), str(page_number))

        if not value or key in seen:
            return
        seen.add(key)

        context = " ".join(
            lines[max(0, index - 1):min(len(lines), index + 2)]
        )

        candidates.append({
            "field": field,
            "candidate_value": value,
            "page_number": page_number,
            "evidence": context[:700],
            "status": "needs_review",
            "method": "context_regex",
        })

    label_patterns = [
        ("seller_name_candidate", re.compile(
            r"\b(?:seller|vendor|transferor|executant)(?:'s)?\s*"
            r"(?:full\s+)?name\s*[:\-]\s*(.{2,150})", re.I)),
        ("buyer_name_candidate", re.compile(
            r"\b(?:buyer|purchaser|transferee)(?:'s)?\s*"
            r"(?:full\s+)?name\s*[:\-]\s*(.{2,150})", re.I)),
        ("owner_name_candidate", re.compile(
            r"\b(?:present\s+)?(?:property\s+)?owner(?:'s)?\s*"
            r"name\s*[:\-]\s*(.{2,150})", re.I)),
        ("property_address_candidate", re.compile(
            r"\b(?:property\s+address|address\s+of\s+property|"
            r"property\s+situated\s+at|situated\s+at)\s*[:\-]?\s*"
            r"(.{5,250})", re.I)),
        ("registration_reference_candidate", re.compile(
            r"\b(?:registration\s*(?:no\.?|number)|"
            r"document\s*(?:no\.?|number)|deed\s*(?:no\.?|number)|"
            r"serial\s*(?:no\.?|number)|book\s*(?:no\.?|number))"
            r"\s*[:#\-]?\s*([A-Z0-9][A-Z0-9/\-]{1,60})", re.I)),
        ("property_identifier_candidate", re.compile(
            r"\b(?:"
            r"property\s+(?:id|identification\s+number|number|no\.?)"
            r"|plot\s+(?:id|number|no\.?)"
            r"|flat\s+(?:id|number|no\.?)"
            r"|khasra\s+(?:number|no\.?)"
            r"|khata\s+(?:number|no\.?)"
            r"|survey\s+(?:number|no\.?)"
            r"|municipal\s+property\s+(?:id|number|no\.?)"
            r")\s*[:#\-]\s*"
            r"([A-Z0-9][A-Z0-9/\-]{1,60})"
            r"(?=\s|$|[,;])",
            re.I
        )),
    ]

    for index, line in enumerate(lines):
        for field, pattern in label_patterns:
            match = pattern.search(line)
            if match:
                value = re.split(
                    r"\s+\b(?:date|address|buyer|seller|vendor|purchaser)"
                    r"\s*:",
                    match.group(1),
                    maxsplit=1,
                    flags=re.I,
                )[0]
                add(field, value, index)

        for match in DATE_PATTERN.finditer(line):
            context = " ".join(
                lines[max(0, index - 1):min(len(lines), index + 2)]
            ).lower()

            if re.search(r"\b(?:execution|executed|execution date)\b", context):
                field = "execution_date_candidate"
            elif re.search(
                r"\b(?:registration|registered|date of registration)\b",
                context,
            ):
                field = "registration_date_candidate"
            elif re.search(r"\bagreement\b", context):
                field = "agreement_date_candidate"
            else:
                field = "date_candidate"

            add(field, match.group(0), index)

        for match in MONEY_PATTERN.finditer(line):
            context = " ".join(
                lines[max(0, index - 1):min(len(lines), index + 2)]
            )

            if re.search(
                r"\b(?:sale consideration|consideration amount|"
                r"total consideration|sale price|purchase price)\b",
                context,
                re.I,
            ):
                field = "sale_consideration_candidate"
            elif re.search(
                r"\b(?:stamp duty|registration fee|circle rate|guidance value)\b",
                context,
                re.I,
            ):
                field = "other_amount_candidate"
            else:
                field = "amount_candidate"

            add(field, match.group(0), index)

    return candidates


def extract_document(input_path: Path) -> dict:
    pages = load_pages(input_path)
    page_reports = []
    all_candidates = []

    for page in pages:
        candidates = extract_page_candidates(
            page["text"], page["page_number"]
        )
        all_candidates.extend(candidates)

        page_reports.append({
            "page_number": page["page_number"],
            "extraction_method": page["method"],
            "character_count": len(page["text"]),
            "candidate_count": len(candidates),
        })

    return {
        "source_file": str(input_path.resolve()),
        "document_type_assumption": "sale_deed",
        "document_type_verified": False,
        "page_count": len(pages),
        "pages": page_reports,
        "candidate_fields": all_candidates,
        "limitations": [
            "Fields are pattern-matched candidates, not verified facts.",
            "OCR can omit, merge, or misread information.",
            "No candidate does not mean information is absent.",
            "This tool does not verify ownership, title, authenticity, "
            "registration, or encumbrance status.",
            "Verify candidates against the original document and obtain "
            "qualified legal review.",
        ],
    }


def save_report(report: dict, input_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_sale_deed_fields.json"

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def show_report(report: dict):
    print("\n" + "=" * 65)
    print("PROPERTY DOCUMENT FIELD REVIEW")
    print("=" * 65)
    print("Pages processed:", report["page_count"])
    print("Candidate fields:", len(report["candidate_fields"]))

    if not report["candidate_fields"]:
        print("\nNo candidates found. This does not prove fields are absent.")
        return

    for index, item in enumerate(report["candidate_fields"], start=1):
        print(f"\n[{index}] {item['field']}")
        print("    Value:    ", item["candidate_value"])
        print("    Page:     ", item["page_number"])
        print("    Status:   ", item["status"])
        print("    Evidence: ", item["evidence"])

    print("\nAll candidates require manual verification.")


def main():
    print("=" * 65)
    print("INTERACTIVE SALE-DEED FIELD EXTRACTOR")
    print("=" * 65)
    print("Enter an OCR JSON or TXT file path.")
    print("Type Q to quit.\n")

    while True:
        raw_path = input("OCR file path: ").strip().strip('"')

        if raw_path.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if not raw_path:
            print("Please enter a file path.\n")
            continue

        input_path = Path(raw_path).expanduser()

        if not input_path.is_absolute():
            input_path = PROJECT_DIR / input_path

        try:
            report = extract_document(input_path)
            show_report(report)

            output_path = save_report(report, input_path)
            print("\nReport saved to:")
            print(output_path.resolve())

        except Exception as exc:
            print(f"\nCould not process this file: {exc}")

        print("\nYou can enter another file or type Q to quit.\n")


if __name__ == "__main__":
    main()
