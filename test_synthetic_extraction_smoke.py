
from pathlib import Path
from collections import Counter

import property_ai_review as review


PROJECT_DIR = Path(__file__).resolve().parent

OCR_DIR = (
    PROJECT_DIR
    / "ocr_evaluation"
    / "synthetic_pdfs_layout_fixed"
    / "recognized_text"
)

PATTERNS = {
    "sale_deed": "SALE_DEED_*__dpi150_psm3.txt",
    "property_tax_receipt": "PROPERTY_TAX_RECEIPT_*__dpi150_psm3.txt",
    "encumbrance_certificate": "ENCUMBRANCE_CERTIFICATE_*__dpi150_psm3.txt",
    "other_or_unknown": "OTHER_OR_UNKNOWN_*__dpi150_psm3.txt",
}


def main() -> None:
    if not OCR_DIR.exists():
        raise FileNotFoundError(
            f"Synthetic OCR directory not found: {OCR_DIR}"
        )

    print("=" * 70)
    print("PROPERTY AI — SYNTHETIC EXTRACTION SMOKE TEST")
    print("=" * 70)
    print("Input:", OCR_DIR)

    total_files = 0
    total_candidates = 0
    empty_files = 0
    field_counts = Counter()

    for document_type, pattern in PATTERNS.items():
        files = sorted(OCR_DIR.glob(pattern))

        print(f"\n--- {document_type} ({len(files)} files) ---")

        if not files:
            print("No matching OCR files found.")
            continue

        for path in files:
            text = path.read_text(encoding="utf-8-sig")

            if not text.strip():
                empty_files += 1
                print(f"{path.name}: EMPTY OCR TEXT")
                continue

            candidates = review.extract_candidates_from_page(
                text,
                page_number=1,
            )

            total_files += 1
            total_candidates += len(candidates)

            current_fields = Counter(
                item.get("field", "unknown")
                for item in candidates
            )
            field_counts.update(current_fields)

            print(
                f"{path.name}: "
                f"{len(text.split())} words, "
                f"{len(candidates)} candidates"
            )

            if not candidates:
                print(
                    "  No candidates; inspect the source text "
                    "before changing extraction rules."
                )
            else:
                for item in candidates:
                    print(
                        f"  {item['field']}: "
                        f"{item['candidate_value']}"
                    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Files with non-empty OCR text:", total_files)
    print("Empty OCR files:", empty_files)
    print("Total candidate values:", total_candidates)

    print("\nCandidate counts by field:")
    for field, count in field_counts.most_common():
        print(f"  {field}: {count}")

    print(
        "\nLimitations: this tests extraction on synthetic OCR text. "
        "It does not measure real-world accuracy or verify legal facts."
    )


if __name__ == "__main__":
    main()
