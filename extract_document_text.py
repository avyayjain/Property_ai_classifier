
import json
import os
import sys
from pathlib import Path

import pymupdf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MIN_EMBEDDED_TEXT_LENGTH = 40
MIN_REVIEW_TEXT_LENGTH = 20

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

# Your installed Tesseract location
TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")

# Make Tesseract available to this Python process
if (TESSERACT_DIR / "tesseract.exe").exists():
    os.environ["PATH"] = (
        str(TESSERACT_DIR)
        + os.pathsep
        + os.environ.get("PATH", "")
    )


# --------------------------------------------------
# Extract text from PDF or image
# --------------------------------------------------

def extract_document(input_path: Path, language: str = "eng"):
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {input_path.suffix}"
        )

    pages = []

    with pymupdf.open(input_path) as document:
        for page_index, page in enumerate(document):
            # First, try extracting existing PDF text
            text = page.get_text("text").strip()
            method = "embedded_text"

            # If the page contains little text, try OCR
            if len(text) < MIN_EMBEDDED_TEXT_LENGTH:
                try:
                    text_page = page.get_textpage_ocr(
                        language=language,
                        dpi=300,
                        full=True,
                    )

                    ocr_text = page.get_text(
                        "text",
                        textpage=text_page,
                    ).strip()

                    if ocr_text:
                        text = ocr_text
                        method = "ocr"

                except Exception as exc:
                    raise RuntimeError(
                        f"OCR failed on page {page_index + 1}. "
                        "Check Tesseract installation and language data. "
                        f"Details: {exc}"
                    ) from exc

            word_count = len(text.split())

            pages.append({
                "page_number": page_index + 1,
                "extraction_method": method,
                "word_count": word_count,
                "text": text,
                "needs_review": len(text) < MIN_REVIEW_TEXT_LENGTH,
            })

    return pages


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1].strip('"'))
    else:
        raw_path = input(
            "Enter the full path to your PDF or image: "
        ).strip().strip('"')

        input_path = Path(raw_path)

    try:
        pages = extract_document(input_path)

        json_path = input_path.with_name(
            input_path.stem + "_extracted.json"
        )

        text_path = input_path.with_name(
            input_path.stem + "_extracted.txt"
        )

        # Save structured output
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                pages,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Save readable output
        with text_path.open("w", encoding="utf-8") as file:
            for page in pages:
                file.write(
                    f"\n--- PAGE {page['page_number']} "
                    f"[{page['extraction_method']}] ---\n"
                )
                file.write(page["text"])
                file.write("\n")

        print("\nExtraction completed.")
        print(f"Pages processed: {len(pages)}")
        print(f"Text output: {text_path}")
        print(f"JSON output: {json_path}")

        for page in pages:
            print(
                f"Page {page['page_number']}: "
                f"{page['extraction_method']}, "
                f"{page['word_count']} words"
            )

            if page["needs_review"]:
                print("  REVIEW: Little or no text extracted.")

        print(
            "\nNext: inspect the extracted text "
            "before running document classification."
        )

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

