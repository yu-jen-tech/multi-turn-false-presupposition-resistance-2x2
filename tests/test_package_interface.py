from __future__ import annotations

import unittest

import transcript_generation


class PackageInterfaceTests(unittest.TestCase):
    def test_package_exposes_primary_generation_interface_only(self) -> None:
        expected_names = {
            "ApiCallMetadata",
            "BatchGenerationPlan",
            "CallPurpose",
            "ChatClient",
            "ChatMessage",
            "ChatResult",
            "DatasetItem",
            "DatasetProvenance",
            "DialogueTurn",
            "GenerationSettings",
            "OpenAICompatibleChatClient",
            "PushbackRecord",
            "PushbackSource",
            "SCENARIOS",
            "Scenario",
            "SemanticValidationResult",
            "SurfaceStyleValidationResult",
            "TranscriptGenerator",
            "TranscriptRecord",
            "ValidationResult",
            "generate_transcripts",
            "load_transcript",
        }

        self.assertEqual(expected_names, set(transcript_generation.__all__))
        self.assertFalse(
            {
                "model_path_component",
                "utc_now",
                "validate_pushback",
                "write_transcript",
            }
            & set(vars(transcript_generation))
        )


if __name__ == "__main__":
    unittest.main()
