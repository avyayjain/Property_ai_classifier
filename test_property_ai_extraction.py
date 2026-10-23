"""Regression tests for Property AI's general field extractor."""

import unittest
from pathlib import Path

import property_ai_review


PROJECT_DIR = Path(__file__).resolve().parent


def read_txt(filename: str) -> str:
    """Read a project test document."""
    path = PROJECT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Test document not found: {path}"
        )

    return path.read_text(encoding="utf-8-sig")


def read_ocr_json(filename: str) -> str:
    """Read page text from a supported OCR JSON file."""
    path = PROJECT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Test document not found: {path}"
        )

    pages = property_ai_review.classifier.load_document(path)

    return "\n".join(
        str(page.get("text", "") or "")
        for page in pages
    )


def extract(text: str) -> list[dict]:
    """Call the production extractor without loading the ML models."""
    return property_ai_review.extract_candidates_from_page(
        text=text,
        page_number=1,
    )


def values_for(
    candidates: list[dict],
    field: str,
) -> list[str]:
    """Return every extracted value for a field."""
    return [
        item["candidate_value"]
        for item in candidates
        if item["field"] == field
    ]


class PropertyAIExtractionTests(unittest.TestCase):
    """Exact-value and robustness tests using fictional documents."""

    @classmethod
    def setUpClass(cls):
        cls.tax_text = read_txt(
            "synthetic_property_tax_receipt.txt"
        )
        cls.encumbrance_text = read_txt(
            "synthetic_encumbrance_certificate.txt"
        )
        cls.sale_deed_text = read_txt(
            "synthetic_sale_deed.txt"
        )
        cls.notice_text = read_ocr_json(
            "ocr_output/sample-scanned-page.ocr.json"
        )

        cls.tax = extract(cls.tax_text)
        cls.encumbrance = extract(cls.encumbrance_text)
        cls.sale_deed = extract(cls.sale_deed_text)
        cls.notice = extract(cls.notice_text)

    def assert_candidate(
        self,
        candidates: list[dict],
        field: str,
        expected: str,
    ):
        """Assert that the exact expected candidate value is present."""
        actual_values = values_for(candidates, field)

        self.assertIn(
            expected,
            actual_values,
            msg=(
                f"\nField: {field}"
                f"\nExpected: {expected!r}"
                f"\nActual: {actual_values!r}"
            ),
        )

    # ============================================================
    # BASELINE TESTS
    # ============================================================

    def test_tax_receipt_owner_address_and_identifier(self):
        self.assert_candidate(
            self.tax,
            "owner_name_candidate",
            "Meera Verma",
        )
        self.assert_candidate(
            self.tax,
            "property_address_candidate",
            (
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ),
        )
        self.assert_candidate(
            self.tax,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )

    def test_payer_name_is_not_mislabelled_as_owner(self):
        text = (
            "Property Tax Receipt "
            "Name of payer - Ananya Das "
            "Municipal assessment number - ASM-00019 "
            "Property reference - TEST-PROP-00019"
        )
        candidates = extract(text)

        self.assert_candidate(
            candidates,
            "payer_name_candidate",
            "Ananya Das",
        )
        self.assertEqual(
            values_for(candidates, "owner_name_candidate"),
            [],
        )

    def test_penalty_amount_is_not_mislabelled_as_generic_amount(self):
        text = (
            "Property Tax Receipt "
            "Tax Amount: INR 13,533 "
            "Penalty Amount: INR 750 "
            "Amount Received: Rs. 14,283"
        )
        candidates = extract(text)

        self.assert_candidate(
            candidates,
            "tax_amount_candidate",
            "INR 13,533",
        )
        self.assert_candidate(
            candidates,
            "penalty_amount_candidate",
            "INR 750",
        )
        self.assert_candidate(
            candidates,
            "total_amount_paid_candidate",
            "Rs. 14,283",
        )
        self.assertEqual(
            values_for(candidates, "amount_candidate"),
            [],
        )

    def test_alternate_tax_receipt_labels(self):
        text = """PROPERTY TAX REMITTANCE RECORD
Owner / taxpayer: Aarav Das
Assessment reference: ASM-00014
Location of property: Flat 231, Fictional Heights, Sector 18, Gurugram, Delhi
Property ID: TEST-PROP-00014
Tax remitted: Rs. 13,496
Additional charges: Rs. 500
Total received: INR 13,996
Period of assessment: 2023-24
"""
        candidates = extract(text)

        self.assert_candidate(
            candidates, "owner_or_taxpayer_name_candidate", "Aarav Das"
        )
        self.assertEqual(values_for(candidates, "owner_name_candidate"), [])
        self.assert_candidate(candidates, "assessment_identifier_candidate", "ASM-00014")
        self.assert_candidate(candidates, "property_identifier_candidate", "TEST-PROP-00014")
        self.assert_candidate(
            candidates, "property_address_candidate",
            "Flat 231, Fictional Heights, Sector 18, Gurugram, Delhi",
        )
        self.assert_candidate(candidates, "tax_amount_candidate", "Rs. 13,496")
        self.assert_candidate(candidates, "additional_charges_candidate", "Rs. 500")
        self.assert_candidate(candidates, "total_amount_paid_candidate", "INR 13,996")
        self.assert_candidate(candidates, "assessment_year_candidate", "2023-24")
        self.assertEqual(values_for(candidates, "amount_candidate"), [])

    def test_alternate_encumbrance_search_date_labels(self):
        text = """PROPERTY REGISTRATION SEARCH REPORT
Application by - Reyansh Bansal
Premises searched - Flat 301, Sample Enclave, Demo Park, New Delhi, Uttar Pradesh
Property ID - TEST-PROP-00025
Search number - SYN-EC-000025
Period examined - 01/01/2017 to 31/12/2022
"""
        candidates = extract(text)

        self.assert_candidate(candidates, "applicant_name_candidate", "Reyansh Bansal")
        self.assert_candidate(
            candidates, "property_address_candidate",
            "Flat 301, Sample Enclave, Demo Park, New Delhi, Uttar Pradesh",
        )
        self.assert_candidate(candidates, "property_identifier_candidate", "TEST-PROP-00025")
        self.assert_candidate(candidates, "search_period_start_candidate", "01/01/2017")
        self.assert_candidate(candidates, "search_period_end_candidate", "31/12/2022")
        self.assertEqual(values_for(candidates, "date_candidate"), [])

    def test_alternate_search_dates_label(self):
        text = "Search dates: 1 January 2020 - 31 December 2026"
        candidates = extract(text)
        self.assert_candidate(candidates, "search_period_start_candidate", "1 January 2020")
        self.assert_candidate(candidates, "search_period_end_candidate", "31 December 2026")
        self.assertEqual(values_for(candidates, "date_candidate"), [])

    def test_tax_receipt_amounts(self):
        self.assert_candidate(
            self.tax,
            "tax_amount_candidate",
            "INR 12,500",
        )
        self.assert_candidate(
            self.tax,
            "late_payment_charge_candidate",
            "INR 0",
        )
        self.assert_candidate(
            self.tax,
            "total_amount_paid_candidate",
            "INR 12,500",
        )

    def test_tax_receipt_dates_and_references(self):
        self.assert_candidate(
            self.tax,
            "payment_date_candidate",
            "10 October 2026",
        )
        self.assert_candidate(
            self.tax,
            "assessment_year_candidate",
            "2026-2027",
        )
        self.assert_candidate(
            self.tax,
            "receipt_reference_candidate",
            "TEST-TAX-2026-001",
        )
        self.assert_candidate(
            self.tax,
            "payment_reference_candidate",
            "TEST-PAY-2026-001",
        )
        self.assert_candidate(
            self.tax,
            "payment_status_candidate",
            "PAID",
        )

    def test_encumbrance_certificate_dates(self):
        self.assert_candidate(
            self.encumbrance,
            "certificate_issue_date_candidate",
            "18 October 2026",
        )
        self.assert_candidate(
            self.encumbrance,
            "search_period_start_candidate",
            "1 January 2016",
        )
        self.assert_candidate(
            self.encumbrance,
            "search_period_end_candidate",
            "31 December 2025",
        )

    def test_encumbrance_certificate_identity_fields(self):
        self.assert_candidate(
            self.encumbrance,
            "property_address_candidate",
            (
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ),
        )
        self.assert_candidate(
            self.encumbrance,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )
        self.assert_candidate(
            self.encumbrance,
            "certificate_reference_candidate",
            "TEST-EC-2026-001",
        )
        self.assert_candidate(
            self.encumbrance,
            "application_reference_candidate",
            "TEST-APP-2026-001",
        )
        self.assert_candidate(
            self.encumbrance,
            "applicant_name_candidate",
            "Meera Verma",
        )

    def test_sale_deed_party_and_property_fields(self):
        self.assert_candidate(
            self.sale_deed,
            "seller_name_candidate",
            "Aarav Sharma",
        )
        self.assert_candidate(
            self.sale_deed,
            "buyer_name_candidate",
            "Meera Verma",
        )
        self.assert_candidate(
            self.sale_deed,
            "property_address_candidate",
            (
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ),
        )
        self.assert_candidate(
            self.sale_deed,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )

    def test_sale_deed_transaction_fields(self):
        self.assert_candidate(
            self.sale_deed,
            "execution_date_candidate",
            "15 September 2026",
        )
        self.assert_candidate(
            self.sale_deed,
            "registration_date_candidate",
            "16 September 2026",
        )
        self.assert_candidate(
            self.sale_deed,
            "sale_consideration_candidate",
            "INR 75,00,000",
        )
        self.assert_candidate(
            self.sale_deed,
            "registration_reference_candidate",
            "TEST-REG-2026-001",
        )

    def test_maintenance_notice_dates(self):
        self.assert_candidate(
            self.notice,
            "notice_date_candidate",
            "6 October 2026",
        )
        self.assert_candidate(
            self.notice,
            "scheduled_event_date_candidate",
            "12 October 2026",
        )

    def test_candidate_metadata_is_preserved(self):
        """Every candidate should preserve its review safeguards."""
        all_candidates = (
            self.tax
            + self.encumbrance
            + self.sale_deed
            + self.notice
        )

        self.assertTrue(
            all_candidates,
            "No candidate fields were extracted from the fixtures.",
        )

        for item in all_candidates:
            with self.subTest(
                field=item.get("field"),
                value=item.get("candidate_value"),
            ):
                self.assertEqual(item["page_number"], 1)
                self.assertEqual(item["status"], "needs_review")
                self.assertTrue(item.get("evidence"))
                self.assertTrue(item.get("review_note"))

    # ============================================================
    # FIELD-BOUNDARY TESTS
    # ============================================================

    def test_adjacent_fields_do_not_leak_into_values(self):
        text = (
            "Owner Name: Meera Verma "
            "Property Category: Residential Flat "
            "Tax Amount: INR 12,500 "
            "Late Payment Charge: INR 0 "
            "Total Amount Paid: INR 12,500 "
            "Payment Status: PAID"
        )

        candidates = extract(text)

        self.assertEqual(
            values_for(candidates, "owner_name_candidate"),
            ["Meera Verma"],
        )
        self.assertEqual(
            values_for(candidates, "tax_amount_candidate"),
            ["INR 12,500"],
        )
        self.assertEqual(
            values_for(candidates, "late_payment_charge_candidate"),
            ["INR 0"],
        )
        self.assertEqual(
            values_for(candidates, "total_amount_paid_candidate"),
            ["INR 12,500"],
        )
        self.assertEqual(
            values_for(candidates, "payment_status_candidate"),
            ["PAID"],
        )

    def test_property_address_stops_at_next_field(self):
        text = (
            "Property Address: Flat 204, Fictional Residency, "
            "Sector 10, New Delhi, Delhi "
            "Owner Name: Meera Verma "
            "Property Category: Residential Flat"
        )

        candidates = extract(text)

        self.assertEqual(
            values_for(candidates, "property_address_candidate"),
            [
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ],
        )

    def test_identical_amounts_keep_distinct_meanings(self):
        text = (
            "Tax Amount: INR 12,500 "
            "Total Amount Paid: INR 12,500"
        )

        candidates = extract(text)

        self.assertEqual(
            values_for(candidates, "tax_amount_candidate"),
            ["INR 12,500"],
        )
        self.assertEqual(
            values_for(candidates, "total_amount_paid_candidate"),
            ["INR 12,500"],
        )

    def test_empty_text_returns_no_candidates(self):
        self.assertEqual(extract(""), [])
        self.assertEqual(extract("   \n\t   "), [])

    # ============================================================
    # FORMATTING AND OCR-ROBUSTNESS TESTS
    # ============================================================

    def test_ocr_line_breaks_and_extra_whitespace(self):
        text = """
        PROPERTY TAX RECEIPT

        OWNER NAME:
        Meera Verma

        PROPERTY ADDRESS:
        Flat 204, Fictional Residency, Sector 10,
        New Delhi, Delhi

        TAX AMOUNT:
        INR 12,500

        PAYMENT STATUS:
        PAID
        """

        candidates = extract(text)

        self.assert_candidate(
            candidates,
            "owner_name_candidate",
            "Meera Verma",
        )
        self.assert_candidate(
            candidates,
            "property_address_candidate",
            (
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ),
        )
        self.assert_candidate(
            candidates,
            "tax_amount_candidate",
            "INR 12,500",
        )
        self.assert_candidate(
            candidates,
            "payment_status_candidate",
            "PAID",
        )

    def test_ocr_case_and_separator_variations(self):
        text = (
            "pAyMeNt DaTe: 10/10/2026 "
            "ASSESSMENT YEAR - 2026/2027 "
            "PROPERTY ID: TEST-PROP-204 "
            "OWNER NAME: Meera Verma "
            "TAX AMOUNT - INR12,500"
        )

        candidates = extract(text)

        self.assert_candidate(
            candidates,
            "payment_date_candidate",
            "10/10/2026",
        )
        self.assert_candidate(
            candidates,
            "assessment_year_candidate",
            "2026/2027",
        )
        self.assert_candidate(
            candidates,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )
        self.assert_candidate(
            candidates,
            "owner_name_candidate",
            "Meera Verma",
        )
        self.assert_candidate(
            candidates,
            "tax_amount_candidate",
            "INR12,500",
        )

    def test_rupee_symbol_amount(self):
        text = (
            "Property Tax Receipt "
            "Tax Amount: ₹12,500 "
            "Late Payment Charge: ₹0"
        )

        candidates = extract(text)

        self.assert_candidate(
            candidates,
            "tax_amount_candidate",
            "₹12,500",
        )
        self.assert_candidate(
            candidates,
            "late_payment_charge_candidate",
            "₹0",
        )

    def test_corrupted_field_label_is_not_guessed(self):
        # "Nane" is a simulated OCR corruption of "Name".
        # The extractor must not silently reinterpret it.
        text = (
            "Owner Nane: Meera Verma "
            "Tax Amount: INR 12,500"
        )

        candidates = extract(text)

        self.assertEqual(
            values_for(candidates, "owner_name_candidate"),
            [],
        )
        self.assert_candidate(
            candidates,
            "tax_amount_candidate",
            "INR 12,500",
        )

    # ============================================================
    # MEDIUM AND SEVERE OCR REGRESSION TESTS
    # ============================================================

    def test_maintenance_notice_address_boundary_on_ocr_outputs(self):
        medium_text = read_ocr_json(
            "ocr_output/synthetic_maintenance_notice_medium.ocr.json"
        )
        severe_text = read_ocr_json(
            "ocr_output/synthetic_maintenance_notice_severe.ocr.json"
        )

        medium_candidates = extract(medium_text)
        severe_candidates = extract(severe_text)

        medium_addresses = values_for(
            medium_candidates,
            "property_address_candidate",
        )
        severe_addresses = values_for(
            severe_candidates,
            "property_address_candidate",
        )

        self.assertEqual(
            medium_addresses,
            [
                "Flat 204, Fictional Residency, Sector 10, "
                "New Delhi, Delhi"
            ],
        )

        self.assertEqual(len(severe_addresses), 1)

        severe_address = severe_addresses[0]

        self.assertTrue(
            severe_address.startswith(
                "Flat 204, Fictional Residency, Sector 10,"
            ),
            msg=f"Unexpected severe-scan address: {severe_address!r}",
        )

        self.assertNotRegex(
            severe_address,
            r"(?i)subject\s*:|dear residents|water tanks|issued by",
            msg=(
                "Address candidate incorrectly includes later document text."
            ),
        )

        self.assert_candidate(
            medium_candidates,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )
        self.assert_candidate(
            severe_candidates,
            "property_identifier_candidate",
            "TEST-PROP-204",
        )

    def test_block_specific_dates_on_medium_and_severe_ocr(self):
        filenames = [
            "ocr_output/synthetic_maintenance_notice_medium.ocr.json",
            "ocr_output/synthetic_maintenance_notice_severe.ocr.json",
        ]

        expected = {
            "block_a_maintenance_date_candidate": "12 October 2026",
            "block_b_maintenance_date_candidate": "13 October 2026",
            "block_c_maintenance_date_candidate": "14 October 2026",
        }

        for filename in filenames:
            with self.subTest(filename=filename):
                text = read_ocr_json(filename)
                candidates = extract(text)

                for field, expected_date in expected.items():
                    self.assert_candidate(
                        candidates,
                        field,
                        expected_date,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)