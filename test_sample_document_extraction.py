
from pathlib import Path

import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image

import property_ai_review as review


PROJECT_DIR = Path(__file__).resolve().parent

SAMPLE_DIR = (
    PROJECT_DIR
    / "real_document_validation"
    / "sample_documents"
)

# Keep sample OCR output separate from real-document validation.
OCR_OUTPUT_DIR = SAMPLE_DIR / "ocr_text"

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)


def extract_candidates(text: str, page_number: int) -> list[dict]:
    """Run general candidate extraction without classifying the document."""
    if not text.strip():
        return []

    return review.extract_candidates_from_page(
        text,
        page_number=page_number,
    )


def save_text(path: Path, text: str) -> None:
    """Save extracted text for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_candidates(
    method: str,
    text: str,
    page_number: int,
) -> int:
    """Print a limited candidate summary and return its count."""
    if not text.strip():
        print(f"    {method}: no readable text")
        return 0

    candidates = extract_candidates(text, page_number)

    print(
        f"    {method}: {len(text.split())} words, "
        f"{len(candidates)} candidates"
    )

    for item in candidates[:10]:
        print(
            f"      - {item['field']}: "
            f"{item['candidate_value']}"
        )

    if len(candidates) > 10:
        print(f"      ... {len(candidates) - 10} more candidates")

    return len(candidates)


def main() -> None:
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(SAMPLE_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No sample PDFs found in:")
        print(SAMPLE_DIR)
        return

    print("=" * 65)
    print("PROPERTY AI — PDF AND OCR SMOKE TEST")
    print("=" * 65)

    total_pages = 0
    total_candidates = 0
    failures = 0
    ocr_pages = 0

    for pdf_path in pdf_files:
        print(f"\nDocument: {pdf_path.name}")

        try:
            # Open as a PDF document, not as a UTF-8 text file.
            with pymupdf.open(pdf_path) as pdf:
                print(f"  Pages: {len(pdf)}")

                for index, page in enumerate(pdf, start=1):
                    page_number = index
                    total_pages += 1

                    page_dir = OCR_OUTPUT_DIR / pdf_path.stem
                    native_path = (
                        page_dir / f"page_{page_number:03d}_native.txt"
                    )
                    ocr_path = (
                        page_dir / f"page_{page_number:03d}_ocr.txt"
                    )

                    # First, extract any existing PDF text layer.
                    native_text = page.get_text("text").strip()

                    if native_text:
                        save_text(native_path, native_text)
                        total_candidates += print_candidates(
                            "PDF text layer",
                            native_text,
                            page_number,
                        )
                    else:
                        print(
                            f"  Page {page_number}: "
                            "PDF text layer is empty"
                        )

                    # OCR image-containing pages, and any page without
                    # embedded text. This helps with scanned documents.
                    has_images = bool(page.get_images(full=True))
                    should_ocr = has_images or not native_text

                    if not should_ocr:
                        continue

                    try:
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2),
                            colorspace=pymupdf.csRGB,
                            alpha=False,
                        )

                        image = Image.frombytes(
                            "RGB",
                            (pix.width, pix.height),
                            pix.samples,
                        )

                        ocr_text = pytesseract.image_to_string(
                            image,
                            lang="eng",
                            config="--psm 3",
                        ).strip()

                        save_text(ocr_path, ocr_text)
                        ocr_pages += 1

                        total_candidates += print_candidates(
                            "Tesseract OCR",
                            ocr_text,
                            page_number,
                        )

                        if not ocr_text:
                            print(
                                f"    OCR returned no readable text "
                                f"on page {page_number}"
                            )

                    except Exception as exc:
                        failures += 1
                        # Do not print extracted document contents.
                        print(
                            f"    OCR ERROR on page {page_number}: "
                            f"{type(exc).__name__}: {exc}"
                        )

        except Exception as exc:
            failures += 1
            print(f"  PDF ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 65)
    print("SMOKE TEST SUMMARY")
    print("=" * 65)
    print("PDF files found:", len(pdf_files))
    print("Pages opened:", total_pages)
    print("Pages sent to OCR:", ocr_pages)
    print("Candidates detected:", total_candidates)
    print("Errors:", failures)
    print("Saved text folder:", OCR_OUTPUT_DIR)
    print()
    print("This is a smoke test, not an accuracy evaluation.")
    print("Extracted fields remain unverified candidates.")
    print("Check the original PDF before treating values as correct.")


if __name__ == "__main__":
    main()
