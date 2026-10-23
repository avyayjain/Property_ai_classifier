"""Detector and document-report safeguard regression tests."""

import ast
import re
import unittest
from pathlib import Path


MANUAL_TEXT = """
Municipal Corporation Delhi
Property Tax Payment
Mobile App User Manual
Instructions for logging in and downloading a receipt.
"""

BLANK_DEED_TEXT = """
Note: This is a model draft and may be customized according
to requirements of individual cases.
SALE DEED
Vendor: ___________________
Vendee: ___________________
Property number: ....................
Sale consideration: ....................
Date of execution: ....................
"""

SYNTHETIC_TEXT = """
SYNTHETIC TEST - NOT OFFICIAL
SYNTHETIC DOCUMENT SERIES
RESIDENTIAL SALE DEED - TEST COPY
FICTIONAL DATA - FOR SOFTWARE TESTING ONLY - NOT AN OFFICIAL RECORD
Seller name: Sana Nair
Buyer name: Aarav Arora
Property identifier: TEST-PROP-00003
Sale consideration: INR 12,363,584
Execution date: 18 May 2024
Specimen statement: this fictional document is for software testing only.
SYNTHETIC TEST DOCUMENT - NOT OFFICIAL - NO LEGAL EFFECT
Invented names, references, addresses, dates, and amounts.
"""


def load_functions():
    """Load functions from main.py without executing the Streamlit app."""

    main_path = Path(__file__).with_name("main.py")
    tree = ast.parse(
        main_path.read_text(encoding="utf-8"),
        filename=str(main_path),
    )

    required = {
        "detect_manual_or_guide",
        "detect_document_template",
        "detect_synthetic_document",
        "apply_document_safeguards",
    }

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in required
    }

    missing = required - functions.keys()
    if missing:
        raise AssertionError(
            f"Missing functions in main.py: {sorted(missing)}"
        )

    module = ast.Module(
        body=[functions[name] for name in sorted(required)],
        type_ignores=[],
    )

    namespace = {"re": re}
    exec(compile(module, str(main_path), "exec"), namespace)
    return namespace


def make_report():
    """A representative report with populated candidate lists."""

    general = [{"field": "execution_date_candidate"}]
    deed = [{"field": "seller_name_candidate"}]

    return {
        "document_status": "candidate_for_manual_review",
        "document_level_safeguard": "none",
        "reason": "Original classifier decision.",
        "agreed_document_type": "sale_deed",
        "agreed_type_candidates": ["sale_deed"],
        "suggestions_passing_review_rules": ["sale_deed"],
        "acceptance_rules_passed": True,
        "general_candidate_fields": general.copy(),
        "sale_deed_candidate_fields": deed.copy(),
        "pages": [
            {
                "page_number": 1,
                "candidate_fields": general.copy(),
                "sale_deed_candidate_fields": deed.copy(),
                "classifier_status": "candidate_for_manual_review",
            }
        ],
    }


class DocumentSafeguardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_functions()

    def test_mcd_manual_is_detected(self):
        self.assertTrue(
            self.functions["detect_manual_or_guide"]([MANUAL_TEXT])
        )

    def test_blank_deed_template_is_detected(self):
        detected, markers = self.functions[
            "detect_document_template"
        ]([BLANK_DEED_TEXT])

        self.assertTrue(detected)
        self.assertIn("model draft", markers)

    def test_synthetic_document_is_detected(self):
        detected, markers = self.functions[
            "detect_synthetic_document"
        ]([SYNTHETIC_TEXT])

        self.assertTrue(detected)
        self.assertIn("synthetic test", markers)

    def test_synthetic_document_is_not_mistaken_for_blank_template(self):
        detected, _ = self.functions[
            "detect_document_template"
        ]([SYNTHETIC_TEXT])

        self.assertFalse(detected)

    def test_synthetic_document_is_not_mistaken_for_manual(self):
        self.assertFalse(
            self.functions["detect_manual_or_guide"]([SYNTHETIC_TEXT])
        )

    def test_manual_report_suppresses_candidates(self):
        apply = self.functions["apply_document_safeguards"]
        report = make_report()

        apply(
            report,
            manual_detected=True,
            template_detected=False,
            template_markers=[],
            synthetic_detected=False,
            synthetic_markers=[],
        )

        self.assertEqual(
            report["document_level_safeguard"],
            "instructional_manual_detected",
        )
        self.assertEqual(
            report["document_status"], "unknown_or_needs_review"
        )
        self.assertIsNone(report["agreed_document_type"])
        self.assertFalse(report["acceptance_rules_passed"])
        self.assertEqual(report["agreed_type_candidates"], [])
        self.assertEqual(report["suggestions_passing_review_rules"], [])
        self.assertEqual(report["general_candidate_fields"], [])
        self.assertEqual(report["sale_deed_candidate_fields"], [])
        self.assertEqual(report["suppressed_candidate_count"], 2)
        self.assertEqual(
            report["pages"][0]["suppressed_manual_candidate_count"], 2
        )
        self.assertEqual(report["pages"][0]["candidate_fields"], [])
        self.assertEqual(
            report["pages"][0]["sale_deed_candidate_fields"], []
        )

    def test_template_report_suppresses_candidates(self):
        apply = self.functions["apply_document_safeguards"]
        report = make_report()

        apply(
            report,
            manual_detected=False,
            template_detected=True,
            template_markers=["model draft"],
            synthetic_detected=False,
            synthetic_markers=[],
        )

        self.assertEqual(
            report["document_level_safeguard"],
            "document_template_detected",
        )
        self.assertEqual(
            report["document_status"], "unknown_or_needs_review"
        )
        self.assertIsNone(report["agreed_document_type"])
        self.assertFalse(report["acceptance_rules_passed"])
        self.assertEqual(report["general_candidate_fields"], [])
        self.assertEqual(report["sale_deed_candidate_fields"], [])
        self.assertEqual(report["suppressed_candidate_count"], 2)
        self.assertEqual(
            report["pages"][0]["suppressed_template_candidate_count"], 2
        )
        self.assertEqual(report["pages"][0]["candidate_fields"], [])
        self.assertEqual(
            report["pages"][0]["sale_deed_candidate_fields"], []
        )

    def test_synthetic_report_retains_candidates_but_is_unresolved(self):
        apply = self.functions["apply_document_safeguards"]
        report = make_report()

        apply(
            report,
            manual_detected=False,
            template_detected=False,
            template_markers=[],
            synthetic_detected=True,
            synthetic_markers=["synthetic test"],
        )

        self.assertEqual(
            report["document_level_safeguard"],
            "synthetic_test_document_detected",
        )
        self.assertEqual(
            report["document_status"], "unknown_or_needs_review"
        )
        self.assertIsNone(report["agreed_document_type"])
        self.assertFalse(report["acceptance_rules_passed"])
        self.assertEqual(report["agreed_type_candidates"], [])
        self.assertEqual(report["suggestions_passing_review_rules"], [])
        self.assertTrue(report["general_candidate_fields"])
        self.assertTrue(report["sale_deed_candidate_fields"])
        self.assertTrue(report["pages"][0]["candidate_fields"])
        self.assertTrue(
            report["pages"][0]["sale_deed_candidate_fields"]
        )

    def test_manual_safeguard_has_priority(self):
        apply = self.functions["apply_document_safeguards"]
        report = make_report()

        apply(
            report,
            manual_detected=True,
            template_detected=True,
            template_markers=["model draft"],
            synthetic_detected=True,
            synthetic_markers=["synthetic test"],
        )

        self.assertEqual(
            report["document_level_safeguard"],
            "instructional_manual_detected",
        )
        self.assertEqual(report["general_candidate_fields"], [])

    def test_no_safeguard_preserves_original_decision(self):
        apply = self.functions["apply_document_safeguards"]
        report = make_report()

        apply(
            report,
            manual_detected=False,
            template_detected=False,
            template_markers=[],
            synthetic_detected=False,
            synthetic_markers=[],
        )

        self.assertEqual(
            report["document_status"], "candidate_for_manual_review"
        )
        self.assertEqual(report["agreed_document_type"], "sale_deed")
        self.assertTrue(report["acceptance_rules_passed"])
        self.assertEqual(report["document_level_safeguard"], "none")
        self.assertTrue(report["general_candidate_fields"])


if __name__ == "__main__":
    unittest.main(verbosity=2)