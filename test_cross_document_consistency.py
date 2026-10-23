"""Unit tests for Property AI cross-document consistency checks."""

import unittest

from cross_document_consistency import (
    compare_field,
    determine_consistency_status,
    get_candidates,
    normalize_value,
)


def make_source(
    values: list[str],
    type_matches_expectation: bool = True,
) -> dict:
    """Build a test source for the consistency-status function."""

    return {
        "values": [
            {"value": value}
            for value in values
        ],
        "type_matches_expectation": type_matches_expectation,
    }


class CrossDocumentConsistencyTests(unittest.TestCase):

    def test_normalisation_ignores_case_spaces_and_punctuation(self):
        self.assertEqual(
            normalize_value(
                "Flat 204, Fictional Residency, New Delhi"
            ),
            normalize_value(
                "FLAT 204 FICTIONAL RESIDENCY NEW DELHI"
            ),
        )

    def test_matching_property_ids_are_consistent(self):
        sources = [
            make_source(["TEST-PROP-204"]),
            make_source(["test-prop-204"]),
            make_source(["TEST-PROP-204"]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "consistent_across_documents",
        )

    def test_different_property_ids_are_flagged(self):
        sources = [
            make_source(["TEST-PROP-204"]),
            make_source(["TEST-PROP-205"]),
            make_source(["TEST-PROP-204"]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "potential_discrepancy",
        )

    def test_missing_value_produces_partial_match(self):
        sources = [
            make_source(["TEST-PROP-204"]),
            make_source(["TEST-PROP-204"]),
            make_source([]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "partial_match_missing_data",
        )

    def test_only_one_value_is_insufficient(self):
        sources = [
            make_source(["TEST-PROP-204"]),
            make_source([]),
            make_source([]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "insufficient_evidence",
        )

    def test_multiple_values_in_one_document_need_review(self):
        sources = [
            make_source(["TEST-PROP-204", "TEST-PROP-205"]),
            make_source(["TEST-PROP-204"]),
            make_source(["TEST-PROP-204"]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "multiple_candidates_needs_review",
        )

    def test_wrong_document_type_blocks_comparison(self):
        sources = [
            make_source(
                ["TEST-PROP-204"],
                type_matches_expectation=False,
            ),
            make_source(["TEST-PROP-204"]),
            make_source(["TEST-PROP-204"]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 3),
            "document_type_needs_review",
        )

    def test_buyer_owner_name_comparison_can_match(self):
        sources = [
            make_source(["Meera Verma"]),
            make_source(["MEERA VERMA"]),
        ]

        self.assertEqual(
            determine_consistency_status(sources, 2),
            "consistent_across_documents",
        )

    def test_buyer_owner_name_difference_is_not_called_legal_failure(self):
        sources = [
            make_source(["Meera Verma"]),
            make_source(["Aarav Sharma"]),
        ]

        status = determine_consistency_status(sources, 2)

        self.assertEqual(status, "potential_discrepancy")
        self.assertNotEqual(status, "legal_failure")

    def test_get_candidates_deduplicates_same_value_and_page(self):
        report = {
            "general_candidate_fields": [
                {
                    "field": "property_identifier_candidate",
                    "candidate_value": "TEST-PROP-204",
                    "page_number": 1,
                    "evidence": "Property ID: TEST-PROP-204",
                },
            ],
            "sale_deed_candidate_fields": [
                {
                    "field": "property_identifier_candidate",
                    "candidate_value": "TEST-PROP-204",
                    "page_number": 1,
                    "evidence": "Property ID: TEST-PROP-204",
                },
            ],
        }

        candidates = get_candidates(
            report,
            "property_identifier_candidate",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["value"],
            "TEST-PROP-204",
        )

    def test_comparison_reports_missing_source_values(self):
        documents = {
            "sale_deed": {
                "expected_type": "sale_deed",
                "actual_type": "sale_deed",
                "type_matches_expectation": True,
                "report_path": "sale_deed.json",
                "report": {
                    "general_candidate_fields": [
                        {
                            "field": "property_identifier_candidate",
                            "candidate_value": "TEST-PROP-204",
                            "page_number": 1,
                        },
                    ],
                    "sale_deed_candidate_fields": [],
                },
            },
            "property_tax_receipt": {
                "expected_type": "property_tax_receipt",
                "actual_type": "property_tax_receipt",
                "type_matches_expectation": True,
                "report_path": "tax.json",
                "report": {
                    "general_candidate_fields": [
                        {
                            "field": "property_identifier_candidate",
                            "candidate_value": "TEST-PROP-204",
                            "page_number": 1,
                        },
                    ],
                },
            },
            "encumbrance_certificate": {
                "expected_type": "encumbrance_certificate",
                "actual_type": "encumbrance_certificate",
                "type_matches_expectation": True,
                "report_path": "encumbrance.json",
                "report": {
                    "general_candidate_fields": [],
                },
            },
        }

        comparison = compare_field(
            title="Property identifier",
            source_definitions=[
                {
                    "document_key": "sale_deed",
                    "field": "property_identifier_candidate",
                },
                {
                    "document_key": "property_tax_receipt",
                    "field": "property_identifier_candidate",
                },
                {
                    "document_key": "encumbrance_certificate",
                    "field": "property_identifier_candidate",
                },
            ],
            documents=documents,
        )

        self.assertEqual(
            comparison["status"],
            "partial_match_missing_data",
        )
        self.assertTrue(comparison["requires_manual_review"])


if __name__ == "__main__":
    unittest.main(verbosity=2)