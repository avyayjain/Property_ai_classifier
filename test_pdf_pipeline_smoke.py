"""PDF-to-report integration smoke tests for Property AI.

Uses synthetic documents only.

The full-report test invokes the project's actual models, so those models
must be available in the local environment.
"""

from __future__ import annotations

import io
import unittest

import pymupdf

import main


SYNTHETIC_TEXT = """
SYNTHETIC TEST - NOT OFFICIAL
SYNTHETIC DOCUMENT SERIES
RESIDENTIAL SALE DEED - TEST COPY
FICTIONAL DATA - FOR SOFTWARE TESTING ONLY - NOT AN OFFICIAL RECORD

Seller name: Sana Nair
Buyer name: Aarav Arora
Premises: Flat 1436, Example Residency, Sector 18, New Delhi
Property identifier: TEST-PROP-00003
Sale consideration: INR 12,363,584
Deed reference: SYN-DOC-000003
Execution date: 18 May 2024

This fictional document is for software testing only.
It does not record an actual property transaction.
SYNTHETIC TEST DOCUMENT - NOT OFFICIAL - NO LEGAL EFFECT.
Invented names, references, addresses, dates, and amounts.
"""


def make_text_pdf() -> bytes:
    """Create a synthetic PDF with an embedded searchable text layer."""

    document = pymupdf.open()

    try:
        page = document.new_page(width=595, height=842)

        remaining_space = page.insert_textbox(
            pymupdf.Rect(45, 45, 550, 800),
            SYNTHETIC_TEXT,
            fontsize=11,
            lineheight=1.4,
        )

        if remaining_space < 0:
            raise RuntimeError(
                "Synthetic fixture text did not fit on the PDF page."
            )

        return document.tobytes()

    finally:
        document.close()


def make_image_only_pdf() -> bytes:
    """Create an image-only PDF from the synthetic text fixture."""

    text_pdf_bytes = make_text_pdf()

    source = pymupdf.open(
        stream=text_pdf_bytes,
        filetype="pdf",
    )

    scanned = pymupdf.open()

    try:
        source_page = source[0]

        # Render the source page to an image.
        pixmap = source_page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False,
        )

        image_bytes = pixmap.tobytes("png")

        # Copy only the page image into a new PDF.
        # This deliberately removes the searchable text layer.
        target_page = scanned.new_page(
            width=source_page.rect.width,
            height=source_page.rect.height,
        )

        target_page.insert_image(
            target_page.rect,
            stream=image_bytes,
        )

        return scanned.tobytes()

    finally:
        source.close()
        scanned.close()


class PDFPipelineSmokeTests(unittest.TestCase):

    def test_embedded_text_pdf_is_readable(self):
        pdf_bytes = make_text_pdf()

        pages = main.extract_pdf_pages(pdf_bytes)

        self.assertEqual(len(pages), 1)
        self.assertEqual(
            pages[0]["extraction_method"],
            "pdf_text_layer",
        )

        text = pages[0]["text"]

        self.assertIn("SYNTHETIC", text.upper())
        self.assertIn("Sana Nair", text)
        self.assertIn("Aarav Arora", text)

    def test_image_only_pdf_uses_ocr(self):
        pdf_bytes = make_image_only_pdf()

        pages = main.extract_pdf_pages(pdf_bytes)

        self.assertEqual(len(pages), 1)
        self.assertEqual(
            pages[0]["extraction_method"],
            "tesseract_ocr",
        )

        self.assertTrue(
            pages[0]["text"].strip(),
            "Tesseract returned no readable text.",
        )

    def test_full_report_keeps_synthetic_deed_unresolved(self):
        pdf_bytes = make_text_pdf()

        report = main.build_report(
            "SYNTHETIC_PIPELINE_TEST.pdf",
            pdf_bytes,
        )

        # The safeguard must override any agreement between classifiers.
        self.assertEqual(
            report["document_level_safeguard"],
            "synthetic_test_document_detected",
        )

        self.assertEqual(
            report["document_status"],
            "unknown_or_needs_review",
        )

        self.assertIsNone(
            report["agreed_document_type"],
        )

        self.assertFalse(
            report["acceptance_rules_passed"],
        )

        self.assertEqual(
            report["agreed_type_candidates"],
            [],
        )

        self.assertEqual(
            report["suggestions_passing_review_rules"],
            [],
        )

        # Candidate extraction is useful for testing, but not verification.
        all_candidates = (
            report.get("general_candidate_fields", [])
            + report.get("sale_deed_candidate_fields", [])
        )

        self.assertTrue(
            all_candidates,
            "Expected candidate fields from the synthetic deed.",
        )

        self.assertTrue(
            all(
                candidate.get("status") == "needs_review"
                for candidate in all_candidates
            ),
            "Every extracted candidate must remain marked for review.",
        )

        self.assertEqual(
            report["pages"][0]["extraction_method"],
            "pdf_text_layer",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)