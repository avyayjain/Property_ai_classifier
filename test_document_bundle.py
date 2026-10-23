"""Unit tests for synthetic document-bundle consistency screening."""
from __future__ import annotations

import unittest

from document_bundle import compare_document_bundle


class DocumentBundleComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = staticmethod(compare_document_bundle)

    def test_conflicting_property_identifiers_are_flagged(self):
        rows = self.compare([
            {"source_file": "sale_deed.pdf", "report": {
                "sale_deed_candidate_fields": [{"field": "property_identifier_candidate", "candidate_value": "PROP-100"}]
            }},
            {"source_file": "tax_receipt.pdf", "report": {
                "general_candidate_fields": [{"field": "property_identifier_candidate", "candidate_value": "PROP-200"}]
            }},
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "potential_discrepancy")
        self.assertEqual(rows[0]["check"], "property_identifier")

    def test_matching_addresses_are_reported_as_matching_candidates(self):
        rows = self.compare([
            {"source_file": "a.pdf", "report": {"general_candidate_fields": [
                {"field": "property_address_candidate", "candidate_value": "Flat 2, Example Road"}
            ]}},
            {"source_file": "b.pdf", "report": {"general_candidate_fields": [
                {"field": "property_address_candidate", "candidate_value": "Flat 2 Example Road"}
            ]}},
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "candidate_values_match")

    def test_absent_field_in_one_document_is_not_assumed_to_match(self):
        rows = self.compare([
            {"source_file": "a.pdf", "report": {"general_candidate_fields": [
                {"field": "property_identifier_candidate", "candidate_value": "PROP-100"}
            ]}},
            {"source_file": "b.pdf", "report": {"general_candidate_fields": []}},
        ])
        self.assertEqual(rows, [])

    def test_multiple_different_values_within_one_document_are_flagged(self):
        rows = self.compare([
            {"source_file": "a.pdf", "report": {"general_candidate_fields": [
                {"field": "property_identifier_candidate", "candidate_value": "PROP-100"},
                {"field": "property_identifier_candidate", "candidate_value": "PROP-200"},
            ]}},
            {"source_file": "b.pdf", "report": {"general_candidate_fields": [
                {"field": "property_identifier_candidate", "candidate_value": "PROP-100"}
            ]}},
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "potential_discrepancy")

    def test_tax_identifier_is_not_conflated_with_property_identifier(self):
        rows = self.compare([
            {"source_file": "a.pdf", "report": {"general_candidate_fields": [
                {"field": "property_identifier_candidate", "candidate_value": "123"}
            ]}},
            {"source_file": "b.pdf", "report": {"general_candidate_fields": [
                {"field": "assessment_identifier_candidate", "candidate_value": "456"}
            ]}},
        ])
        self.assertEqual(rows, [])

    def test_no_candidates_returns_empty_list(self):
        self.assertEqual(self.compare([
            {"source_file": "a.pdf", "report": {}},
            {"source_file": "b.pdf", "report": {}},
        ]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
