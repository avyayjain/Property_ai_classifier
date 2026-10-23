
from pathlib import Path
import json
import sys

import pymupdf
import pytesseract
from PIL import Image, ImageOps, ImageEnhance


SUPPORTED_IMAGES = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}


def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


def ocr_image(image, lang="eng", psm=3):
    processed = preprocess_image(image)
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(
        processed, lang=lang, config=config
    ).strip()


def extract_pdf(path, lang="eng", psm=3,
                min_text_chars=40, dpi=250):
    pages = []

    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            embedded = page.get_text("text").strip()

            if len(embedded) >= min_text_chars:
                text = embedded
                method = "embedded_text"
            else:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )

                text = ocr_image(image, lang, psm)
                method = "tesseract_ocr"

                if embedded and len(embedded) > len(text):
                    text = embedded
                    method = "embedded_text_fallback"

            pages.append({
                "page": page_index + 1,
                "method": method,
                "text": text,
                "character_count": len(text)
            })

    return pages


def extract_image(path, lang="eng", psm=3):
    with Image.open(path) as image:
        text = ocr_image(image, lang, psm)

    return [{
        "page": 1,
        "method": "tesseract_ocr",
        "text": text,
        "character_count": len(text)
    }]


def main():
    print("=" * 60)
    print("PROPERTY AI - OCR DOCUMENT EXTRACTOR")
    print("=" * 60)

    # Ask for the input path in the console
    input_path = input(
        "\nEnter the full path of your PDF or image: "
    ).strip().strip('"').strip("'")

    if not input_path:
        print("ERROR: No file path entered.")
        return

    source = Path(input_path).expanduser()

    if not source.is_file():
        print(f"ERROR: File not found: {source}")
        return

    # Optional settings
    lang = input(
        "OCR language [eng]: "
    ).strip() or "eng"

    psm_input = input(
        "Page segmentation mode [3]: "
    ).strip()

    psm = int(psm_input) if psm_input else 3

    dpi_input = input(
        "PDF OCR resolution DPI [250]: "
    ).strip()

    dpi = int(dpi_input) if dpi_input else 250

    # Set Tesseract path if using the standard Windows location
    default_tesseract = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if Path(default_tesseract).is_file():
        pytesseract.pytesseract.tesseract_cmd = default_tesseract
    else:
        tesseract_path = input(
            "Enter full path to tesseract.exe: "
        ).strip().strip('"').strip("'")

        if not tesseract_path or not Path(tesseract_path).is_file():
            print("ERROR: tesseract.exe was not found.")
            return

        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    suffix = source.suffix.lower()

    try:
        if suffix == ".pdf":
            pages = extract_pdf(
                source, lang=lang, psm=psm, dpi=dpi
            )
        elif suffix in SUPPORTED_IMAGES:
            pages = extract_image(source, lang=lang, psm=psm)
        else:
            print(f"ERROR: Unsupported file type: {suffix}")
            return

    except Exception as error:
        print(f"\nERROR while processing document: {error}")
        return

    output_dir = Path("ocr_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{source.stem}.ocr.json"
    text_path = output_dir / f"{source.stem}.ocr.txt"

    payload = {
        "source_file_name": source.name,
        "note": (
            "OCR can contain errors. Verify extracted information "
            "against the original document."
        ),
        "pages": pages
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    combined_text = "\n\n".join(
        f"===== PAGE {page['page']} | "
        f"METHOD: {page['method']} =====\n{page['text']}"
        for page in pages
    )

    text_path.write_text(combined_text, encoding="utf-8")

    print("\n" + "=" * 60)
    print("OCR COMPLETED")
    print("=" * 60)
    print(f"Pages processed: {len(pages)}")

    for page in pages:
        print(
            f"Page {page['page']}: {page['method']} | "
            f"{page['character_count']} characters"
        )

    print(f"\nText output: {text_path.resolve()}")
    print(f"JSON output: {json_path.resolve()}")


if __name__ == "__main__":
    main()
