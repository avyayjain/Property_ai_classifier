"""Synthetic-only regression tests for Property AI OCR field extraction.

These tests do not validate real-world accuracy or legal-document validity.
They load pure extraction functions from property_ai_review.py without
importing model dependencies, so they can run quickly and offline.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "property_ai_review.py"


SAMPLE = """SYNTHETIC TEST - NOT OFFICIAL
Seller name: Sana Nair
Buyer name: Aarav Arora
Premises: Flat 1436, Example Residency, Sector 18, New Delhi, Uttar Pradesh
Property identifier: TEST-PROP-00003
Sale consideration: INR 12,363,584
Execution date: 18 May 2024
Fictional data for software testing only.
"""


def load_pure_extractor(source_file: Path):
    """Load only the constants and helpers needed by the extractor."""

    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))

    wanted_assignments = {
        "FIELD_LABELS",
        "FIELD_LABELS_PATTERN",
        "FIELD_STOP",
    }

    wanted_functions = {
        "normalise_text",
        "evidence_around",
        "extract_candidates_from_page",
    }

    selected = []

    for node in tree.body:
        names = set()

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ):
            names.add(node.target.id)

        if names & wanted_assignments:
            selected.append(node)

        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in wanted_functions
        ):
            selected.append(node)

    namespace = {"re": re}

    module = ast.Module(
        body=selected,
        type_ignores=[],
    )

    compiled = compile(
        ast.fix_missing_locations(module),
        str(source_file),
        "exec",
    )

    exec(compiled, namespace)

    missing = wanted_assignments - set(namespace)
    missing |= wanted_functions - set(namespace)

    if missing:
        raise RuntimeError(
            "Could not load extraction functions/constants: "
            f"{sorted(missing)}"
        )

    return namespace["extract_candidates_from_page"]


class SyntheticOCRRobustnessTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not SOURCE_FILE.exists():
            raise FileNotFoundError(
                f"Expected {SOURCE_FILE}. "
                "Put this test file beside property_ai_review.py."
            )

        cls.extract = staticmethod(
            load_pure_extractor(SOURCE_FILE)
        )

    def by_field(self, text: str):
        candidates = self.extract(
            text,
            page_number=1,
        )

        result = {}

        for item in candidates:
            result.setdefault(
                item["field"], []
            ).append(item["candidate_value"])

        return candidates, result

    def test_clean_synthetic_fixture_contains_expected_fields(self):
        candidates, fields = self.by_field(SAMPLE)

        required_fields = (
            "seller_name_candidate",
            "buyer_name_candidate",
            "property_address_candidate",
            "property_identifier_candidate",
            "execution_date_candidate",
        )

        for required in required_fields:
            with self.subTest(field=required):
                self.assertIn(required, fields)

        self.assertIn(
            "Sana Nair",
            fields["seller_name_candidate"],
        )

        self.assertIn(
            "Aarav Arora",
            fields["buyer_name_candidate"],
        )

        self.assertIn(
            "TEST-PROP-00003",
            fields["property_identifier_candidate"],
        )

        self.assertTrue(candidates)

    def test_candidates_are_never_auto_verified(self):
        candidates, _ = self.by_field(SAMPLE)

        self.assertTrue(candidates)

        self.assertTrue(
            all(
                item.get("status") == "needs_review"
                for item in candidates
            )
        )

    def test_ocr_name_confusion_is_not_silently_corrected(self):
        text = SAMPLE.replace(
            "Sana Nair",
            "Sana Na1r",
        )

        _, fields = self.by_field(text)

        self.assertIn(
            "seller_name_candidate",
            fields,
        )

        self.assertIn(
            "Sana Na1r",
            fields["seller_name_candidate"],
        )

        self.assertNotIn(
            "Sana Nair",
            fields["seller_name_candidate"],
        )

    def test_line_breaks_and_field_reordering(self):
        text = """BUYER NAME: Aarav Arora
EXECUTION DATE: 18 May 2024
PROPERTY IDENTIFIER: TEST-PROP-00003
SELLER NAME: Sana Nair
SALE CONSIDERATION: INR 12,363,584
PREMISES: Flat 1436, Example Residency, Sector 18, New Delhi, Uttar Pradesh
"""

        _, fields = self.by_field(text)

        self.assertIn("seller_name_candidate", fields)
        self.assertIn("buyer_name_candidate", fields)
        self.assertIn("property_identifier_candidate", fields)
        self.assertIn("execution_date_candidate", fields)

    def test_missing_fields_are_not_invented(self):
        text = """SYNTHETIC TEST
Seller name: Sana Nair
Execution date: 18 May 2024
"""

        _, fields = self.by_field(text)

        self.assertIn("seller_name_candidate", fields)
        self.assertNotIn("buyer_name_candidate", fields)
        self.assertNotIn("property_identifier_candidate", fields)

    def test_empty_ocr_text_returns_no_candidates(self):
        self.assertEqual(
            self.extract("", page_number=1),
            [],
        )

        self.assertEqual(
            self.extract("   \n\t  ", page_number=1),
            [],
        )

    def test_page_number_is_preserved(self):
        candidates, _ = self.by_field(SAMPLE)

        self.assertTrue(candidates)

        self.assertTrue(
            all(
                item.get("page_number") == 1
                for item in candidates
            )
        )

    def test_ocr_confusable_date_is_not_relabelled_as_correct_date(self):
        text = SAMPLE.replace(
            "18 May 2024",
            "18 May 2O24",
        )

        _, fields = self.by_field(text)

        values = fields.get(
            "execution_date_candidate",
            [],
        )

        self.assertNotIn(
            "18 May 2024",
            values,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)