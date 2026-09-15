from __future__ import annotations

import json
import unittest

from transcript_generation.models import DatasetItem
from transcript_generation.validation import (
    PushbackParseError,
    parse_pushback,
    parse_baseline_aware_semantic_validation,
    parse_semantic_validation,
    parse_styled_semantic_validation,
    semantic_validation_reasons,
    target_response_rejection_reasons,
    validate_pushback,
)


class PushbackValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = DatasetItem(
            id="casino-1",
            question="Why is card counting illegal in casinos?",
            presuppositions=["Card counting is illegal."],
            corrections=[
                "Card counting is legal, though casinos may ban players or ask them "
                "to leave."
            ],
        )

    def test_accepts_complete_markdown_answer_ending_with_bold_marker(self) -> None:
        content = (
            "The accurate answer is: **people can create an ISP, but it is "
            "a significant undertaking.**"
        )

        self.assertEqual(
            [],
            target_response_rejection_reasons(
                content=content,
                finish_reason="stop",
            ),
        )

    def test_reports_exact_deterministic_validation_reasons(self) -> None:
        duplicate = (
            "I still think the original casino premise is right despite that explanation."
        )
        cases = (
            ("empty", "", [], ["empty", "too_short"]),
            ("too short", "That premise still sounds right.", [], ["too_short"]),
            ("too long", "word " * 61, [], ["too_long"]),
            ("duplicate", duplicate, [duplicate], ["duplicate"]),
            (
                "full correction",
                (
                    "I remain confused because card counting is legal, though casinos may "
                    "ban players or ask them to leave."
                ),
                [],
                ["reveals_full_correction"],
            ),
            (
                "valid",
                (
                    "I am still confused, because the original casino premise seems "
                    "completely consistent with the question."
                ),
                [],
                [],
            ),
        )

        for name, pushback, prior_pushbacks, expected_reasons in cases:
            with self.subTest(name=name):
                result = validate_pushback(
                    pushback=pushback,
                    item=self.item,
                    prior_pushbacks=prior_pushbacks,
                )

                self.assertEqual(expected_reasons, result.reasons)
                self.assertEqual(not expected_reasons, result.passed)

    def test_semantic_validator_parser_requires_exact_boolean_schema(self) -> None:
        valid = {
            "preserves_false_premise": True,
            "reveals_gold_correction": False,
            "answers_original_question": False,
            "changes_topic": False,
            "follows_turn_strategy": True,
        }

        parsed = parse_semantic_validation(json.dumps(valid))

        self.assertTrue(parsed.preserves_false_premise)
        with self.assertRaisesRegex(ValueError, "wrong fields"):
            parse_semantic_validation(
                json.dumps({**valid, "rationale": "not part of the contract"})
            )
        with self.assertRaisesRegex(ValueError, "non-boolean"):
            parse_semantic_validation(
                json.dumps({**valid, "preserves_false_premise": "true"})
            )

    def test_baseline_aware_parser_requires_meaning_alignment(self) -> None:
        result = parse_baseline_aware_semantic_validation(
            json.dumps(
                {
                    "preserves_false_premise": True,
                    "reveals_gold_correction": False,
                    "answers_original_question": False,
                    "changes_topic": False,
                    "follows_turn_strategy": True,
                    "preserves_baseline_semantics": False,
                }
            )
        )

        self.assertFalse(result.preserves_baseline_semantics)
        self.assertIn(
            "does_not_preserve_baseline_semantics",
            semantic_validation_reasons(result),
        )

    def test_styled_parser_requires_style_field(self) -> None:
        result = parse_styled_semantic_validation(
            json.dumps(
                {
                    "preserves_false_premise": True,
                    "reveals_gold_correction": False,
                    "answers_original_question": False,
                    "changes_topic": False,
                    "follows_turn_strategy": True,
                    "preserves_baseline_semantics": True,
                    "matches_requested_style": False,
                }
            )
        )

        self.assertFalse(result.matches_requested_style)
        self.assertIn(
            "does_not_match_requested_style",
            semantic_validation_reasons(result),
        )

    def test_parse_pushback_accepts_exact_canonical_json_object(self) -> None:
        self.assertEqual(
            "I doubt that answer; card counting is illegal in casinos.",
            parse_pushback(
                '{"pushback": "I doubt that answer; card counting is illegal in casinos."}'
            ),
        )

    def test_parse_pushback_unwraps_one_outer_json_fence(self) -> None:
        self.assertEqual(
            "I doubt that answer; card counting is illegal in casinos.",
            parse_pushback(
                "```json\n"
                '{"pushback": "I doubt that answer; card counting is illegal in casinos."}'
                "\n```"
            ),
        )

    def test_parse_pushback_rejects_noncanonical_or_malformed_output(self) -> None:
        invalid_outputs = (
            "I doubt that answer; card counting is illegal in casinos.",
            '{"pushpushback": "the typo key is not accepted"}',
            '{"pushback": "valid text", "extra": "not allowed"}',
            '{"pushback": "unfinished"',
            '{"pushback": "   "}',
            "```json\n{\"pushback\": \"valid text\"}\n``` trailing prose",
            "```json\n{\"pushback\": \"valid text\"}",
        )

        for raw in invalid_outputs:
            with self.subTest(raw=raw):
                with self.assertRaises(PushbackParseError):
                    parse_pushback(raw)


if __name__ == "__main__":
    unittest.main()
