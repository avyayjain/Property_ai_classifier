"""
Download official public reference documents for Property AI.

Downloads:
1. Delhi Government model sale deed PDF.
2. MCD property-tax payment user manual PDF.
3. Optionally extracts the receipt specimen page from that manual.

Also creates:
- Source manifest CSV
- Official portal links TXT
- README explaining the contents and limitations

Important:
- These are REFERENCE/SAMPLE documents, not a real-document
  validation dataset.
- Does not log in to portals, bypass CAPTCHAs, or download
  private property records.
- Does not invent or fabricate an encumbrance certificate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

VALIDATION_DIR = PROJECT_DIR / "real_document_validation"

# Keep official reference samples separate from real validation PDFs.
SAMPLE_DIR = VALIDATION_DIR / "sample_documents"

MANIFEST_PATH = SAMPLE_DIR / "sample_document_manifest.csv"
PORTALS_PATH = SAMPLE_DIR / "OFFICIAL_PORTAL_LINKS.txt"
README_PATH = SAMPLE_DIR / "README_SAMPLE_SET.md"


# ------------------------------------------------------------
# OFFICIAL PUBLIC SOURCES
# ------------------------------------------------------------

SOURCES = [
    {
        "sample_id": "SAMPLE_SALE_DEED_MODEL",
        "filename": "SAMPLE_SALE_DEED_MODEL_DELHI.pdf",
        "url": (
            "https://cdn.s3waas.gov.in/"
            "s3f79921bbae40a577928b76d2fc3edc2a/"
            "uploads/2018/11/2018112260.pdf"
        ),
        "document_type": "sale_deed",
        "role": "Official blank model draft",
        "notes": (
            "A model sale deed draft. It is not an executed or "
            "registered sale deed. Use for layout and extraction "
            "smoke tests only."
        ),
    },
    {
        "sample_id": "SAMPLE_MCD_PROPERTY_TAX_MANUAL",
        "filename": "SAMPLE_MCD_PROPERTY_TAX_USER_MANUAL.pdf",
        "url": "https://mcdonline.nic.in/PTR_UserManual.pdf",
        "document_type": "property_tax_receipt_reference",
        "role": "Official user manual with receipt specimen",
        "notes": (
            "The PDF is a property-tax payment user manual. It "
            "contains an illustrative receipt screenshot; it is "
            "not an independently issued tax receipt."
        ),
    },
]


# Delhi registration portals.
# These URLs are references only. This script does not scrape
# records or attempt to obtain a real encumbrance certificate.
PORTALS = [
    {
        "name": "Delhi e-Search",
        "url": "https://esearch.delhigovt.nic.in/",
        "purpose": (
            "Official Delhi registration-record search portal. "
            "Use the portal manually and only with authorised access."
        ),
    },
    {
        "name": "Delhi Online Registration Information System (DORIS)",
        "url": "https://doris.delhigovt.nic.in/",
        "purpose": (
            "The homepage currently displays a notice that Sub-Registrar "
            "offices have migrated to NGDRS and are no longer active in "
            "DORIS. Check the current official service before use."
        ),
    },
    {
        "name": "Delhi NGDRS search",
        "url": "https://ngdrs.delhi.gov.in/NGDRSDL/DLSearch/citizenloginsearch",
        "purpose": (
            "Delhi registration-search entry point. Availability and "
            "access requirements may vary."
        ),
    },
]


# Maximum download size: 40 MB per PDF.
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024

USER_AGENT = (
    "PropertyAI-ReferenceSampleDownloader/1.0 "
    "(public reference documents only)"
)


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def pdf_is_valid(path: Path) -> bool:
    """Basic check that a file begins with a PDF signature."""
    try:
        with path.open("rb") as file:
            return file.read(5) == b"%PDF-"
    except OSError:
        return False


def timestamp_utc() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def create_directories() -> None:
    """Create the reference-sample directory."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# PDF DOWNLOAD
# ------------------------------------------------------------

def download_pdf(
    source: dict,
    overwrite: bool = False,
) -> dict:
    """Download and verify one public PDF."""

    filename = source["filename"]
    output_path = SAMPLE_DIR / filename
    temporary_path = output_path.with_suffix(".pdf.part")

    base_result = {
        "sample_id": source["sample_id"],
        "filename": filename,
        "document_type": source["document_type"],
        "role": source["role"],
        "source_url": source["url"],
        "sha256": "",
        "status": "",
        "notes": source["notes"],
        "timestamp_utc": timestamp_utc(),
    }

    if output_path.exists() and not overwrite:
        if pdf_is_valid(output_path):
            base_result["status"] = "ALREADY_PRESENT"
            base_result["sha256"] = sha256_file(output_path)
            print(f"[EXISTS] {filename} — keeping existing file.")
        else:
            base_result["status"] = "EXISTING_FILE_NOT_VALID_PDF"
            base_result["notes"] += (
                " Existing file did not pass the PDF signature check. "
                "Run with --overwrite to replace it."
            )
            print(f"[WARNING] {filename} is not a valid PDF.")
        return base_result

    request = Request(
        source["url"],
        headers={"User-Agent": USER_AGENT},
    )

    try:
        print(f"[DOWNLOAD] {filename}")

        total_bytes = 0

        with urlopen(request, timeout=45) as response:
            content_type = response.headers.get(
                "Content-Type",
                "unknown",
            )

            first_chunk = response.read(8192)

            if not first_chunk.startswith(b"%PDF-"):
                raise ValueError(
                    "The source did not return a PDF. "
                    f"Content-Type: {content_type}"
                )

            with temporary_path.open("wb") as output:
                output.write(first_chunk)
                total_bytes += len(first_chunk)

                while True:
                    chunk = response.read(64 * 1024)

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "Download exceeded the 40 MB safety limit."
                        )

                    output.write(chunk)

        if not pdf_is_valid(temporary_path):
            raise ValueError("Downloaded file failed PDF validation.")

        temporary_path.replace(output_path)

        base_result["status"] = "DOWNLOADED"
        base_result["sha256"] = sha256_file(output_path)

        print(
            f"[OK] {filename} "
            f"({total_bytes:,} bytes)"
        )

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        base_result["status"] = "DOWNLOAD_FAILED"
        base_result["notes"] += (
            f" Download error: {type(exc).__name__}: {exc}"
        )

        print(
            f"[ERROR] {filename}: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return base_result


# ------------------------------------------------------------
# EXTRACT SPECIMEN RECEIPT PAGE
# ------------------------------------------------------------

def extract_receipt_specimen_page(
    manual_path: Path,
    overwrite: bool = False,
) -> dict:
    """
    Extract page 9 (1-based) from the 10-page MCD manual.

    The page contains a specimen receipt screenshot and other
    illustrative material. It is not a real issued receipt.
    """

    output_path = (
        SAMPLE_DIR / "SAMPLE_MCD_RECEIPT_SPECIMEN_PAGE.pdf"
    )

    result = {
        "sample_id": "SAMPLE_MCD_RECEIPT_SPECIMEN_PAGE",
        "filename": output_path.name,
        "document_type": "property_tax_receipt_reference",
        "role": "Extracted specimen page from MCD user manual",
        "source_url": next(
            source["url"]
            for source in SOURCES
            if source["sample_id"] == "SAMPLE_MCD_PROPERTY_TAX_MANUAL"
        ),
        "sha256": "",
        "status": "",
        "notes": (
            "Extracted from page 9 of the MCD user manual. "
            "Contains illustrative receipt imagery, not a real "
            "issued receipt. Reference-only."
        ),
        "timestamp_utc": timestamp_utc(),
    }

    if not manual_path.exists() or not pdf_is_valid(manual_path):
        result["status"] = "SKIPPED_MANUAL_UNAVAILABLE"
        print("[INFO] Receipt page extraction skipped: manual unavailable.")
        return result

    if output_path.exists() and not overwrite:
        if pdf_is_valid(output_path):
            result["status"] = "ALREADY_PRESENT"
            result["sha256"] = sha256_file(output_path)
            print(f"[EXISTS] {output_path.name}")
        else:
            result["status"] = "EXISTING_FILE_NOT_VALID_PDF"
        return result

    try:
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            from PyPDF2 import PdfReader, PdfWriter

        reader = PdfReader(str(manual_path))

        # Page index 8 = page 9, when counting pages from 1.
        page_index = 8

        if len(reader.pages) <= page_index:
            raise ValueError(
                "The downloaded MCD manual has fewer than 9 pages."
            )

        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])

        temporary_path = output_path.with_suffix(".pdf.part")

        with temporary_path.open("wb") as output:
            writer.write(output)

        if not pdf_is_valid(temporary_path):
            temporary_path.unlink(missing_ok=True)
            raise ValueError("Extracted page failed PDF validation.")

        temporary_path.replace(output_path)

        result["status"] = "EXTRACTED_FROM_REFERENCE_PDF"
        result["sha256"] = sha256_file(output_path)

        print(f"[OK] Extracted {output_path.name}")

    except ImportError:
        result["status"] = "SKIPPED_PDF_LIBRARY_MISSING"
        result["notes"] += (
            " Install a PDF library with "
            "'python -m pip install pypdf' and rerun."
        )
        print(
            "[INFO] Page extraction skipped. "
            "Install pypdf with: "
            ".\\.venv\\Scripts\\python.exe -m pip install pypdf"
        )

    except Exception as exc:
        result["status"] = "PAGE_EXTRACTION_FAILED"
        result["notes"] += (
            f" Extraction error: {type(exc).__name__}: {exc}"
        )
        print(
            f"[WARNING] Receipt page extraction failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return result


# ------------------------------------------------------------
# CREATE SOURCE REGISTER AND README
# ------------------------------------------------------------

def write_portal_links() -> None:
    """Write official portal links and explain their limitations."""

    lines = [
        "PROPERTY AI — OFFICIAL PORTAL REFERENCES",
        "=" * 55,
        "",
        "These links are references, not downloaded property records.",
        "The downloader does not bypass authentication or CAPTCHAs.",
        "",
    ]

    for portal in PORTALS:
        lines.extend([
            portal["name"],
            portal["url"],
            portal["purpose"],
            "",
        ])

    lines.extend([
        "IMPORTANT",
        "--------",
        "No real encumbrance certificate has been downloaded by this script.",
        "Use an authorised official service or obtain an authorised copy.",
        "Do not label a portal page or a blank form as an encumbrance certificate.",
        "",
    ])

    PORTALS_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_readme() -> None:
    """Describe which files were downloaded and how to interpret them."""

    content = """# Property AI — Official Reference Samples

This folder contains official public reference materials.

## Expected files

- `SAMPLE_SALE_DEED_MODEL_DELHI.pdf`
  - Blank model deed draft.
  - Not an executed or registered sale deed.

- `SAMPLE_MCD_PROPERTY_TAX_USER_MANUAL.pdf`
  - MCD property-tax payment user manual.
  - Contains an illustrative receipt screenshot.

- `SAMPLE_MCD_RECEIPT_SPECIMEN_PAGE.pdf`
  - Extracted page 9 from the manual, when PDF page extraction
    is available.
  - Contains illustrative specimen imagery, not an issued receipt.

- `sample_document_manifest.csv`
  - Source URLs, file status, hashes, and limitations.

- `OFFICIAL_PORTAL_LINKS.txt`
  - Official registration-search references.

## Important limitations

The source portals do not provide a freely downloadable,
standalone encumbrance certificate through this script.
No authenticated records are downloaded.

Do not count these materials as an independently verified
real-document validation set. Keep them separate from
`real_document_validation/documents`.

Use these PDFs for initial OCR, layout, and application smoke tests.
For genuine accuracy evaluation, obtain documents you are authorised
to use and independently record reference values from their originals.

Extracted candidates remain unverified. Property AI does not certify
ownership, title, authenticity, registration validity, or the absence
of encumbrances.
"""

    README_PATH.write_text(content, encoding="utf-8")


def write_manifest(rows: list[dict]) -> None:
    """Write all download, extraction and portal-reference outcomes."""

    columns = [
        "sample_id",
        "filename",
        "document_type",
        "role",
        "source_url",
        "sha256",
        "status",
        "notes",
        "timestamp_utc",
    ]

    temporary_path = MANIFEST_PATH.with_suffix(".csv.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(MANIFEST_PATH)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download official sample/reference PDFs for Property AI."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing PDFs and regenerate extracted page.",
    )

    args = parser.parse_args()

    create_directories()

    print("=" * 65)
    print("PROPERTY AI — OFFICIAL SAMPLE DOCUMENT DOWNLOADER")
    print("=" * 65)
    print("Output folder:", SAMPLE_DIR)
    print()

    manifest_rows = []

    # Download public source PDFs.
    for source in SOURCES:
        row = download_pdf(
            source,
            overwrite=args.overwrite,
        )
        manifest_rows.append(row)

    # Attempt to extract the specimen receipt page from the manual.
    manual_path = (
        SAMPLE_DIR / "SAMPLE_MCD_PROPERTY_TAX_USER_MANUAL.pdf"
    )

    receipt_page_row = extract_receipt_specimen_page(
        manual_path,
        overwrite=args.overwrite,
    )
    manifest_rows.append(receipt_page_row)

    # Portal reference only: no real encumbrance certificate downloaded.
    manifest_rows.append({
        "sample_id": "ENCUMBRANCE_SEARCH_PORTAL_REFERENCE",
        "filename": "",
        "document_type": "encumbrance_related_reference",
        "role": "Official portal link only",
        "source_url": "https://esearch.delhigovt.nic.in/",
        "sha256": "",
        "status": "MANUAL_RETRIEVAL_REQUIRED",
        "notes": (
            "The portal is not a standalone downloadable EC sample. "
            "Access it manually and use only authorised records."
        ),
        "timestamp_utc": timestamp_utc(),
    })

    write_portal_links()
    write_readme()
    write_manifest(manifest_rows)

    print()
    print("=" * 65)
    print("FINISHED")
    print("=" * 65)
    print("Folder:", SAMPLE_DIR)
    print("Manifest:", MANIFEST_PATH)
    print("Portal links:", PORTALS_PATH)
    print("README:", README_PATH)
    print()
    print(
        "IMPORTANT: These are reference samples, not verified real "
        "property documents."
    )
    print(
        "No real encumbrance certificate was automatically downloaded."
    )


if __name__ == "__main__":
    main()