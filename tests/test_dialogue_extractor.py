from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dialogue_extractor import DIALOGUE_COLUMNS, extract_dialogues
from transcript_generation import (
    ApiCallMetadata,
    DatasetItem,
    DialogueTurn,
    SCENARIOS,
    TranscriptRecord,
)
from transcript_generation.models import utc_now
from transcript_generation.storage import write_transcript


def metadata() -> ApiCallMetadata:
    now = utc_now()
    return ApiCallMetadata(
        purpose="target",
        model="model/a",
        temperature=0.0,
        started_at=now,
        finished_at=now,
        latency_seconds=0,
    )


def transcript_for(
    item_id: str,
    question: str,
    *,
    turn_count: int = 5,
) -> TranscriptRecord:
    item = DatasetItem(
        id=item_id,
        question=question,
        presuppositions="False premise.",
        corrections="Gold correction.",
    )
    turns = [
        DialogueTurn(
            turn=1,
            user=question,
            user_source="dataset",
            assistant=f"{item_id} response 1",
            target_messages=[],
            api_metadata=metadata(),
        )
    ]
    for turn in range(2, turn_count + 1):
        assistant = f"{item_id} response {turn}"
        if turn == 5:
            assistant = f"{assistant}  \n"
        turns.append(
            DialogueTurn(
                turn=turn,
                user=f"{item_id} prompt {turn}",
                user_source="simulator",
                strategy="confusion",
                assistant=assistant,
                target_messages=[],
                api_metadata=metadata(),
            )
        )

    return TranscriptRecord(
        generated_at=utc_now(),
        item=item,
        scenario=SCENARIOS[0],
        simulator_model="simulator",
        target_model="model/a",
        target_system_prompt="system",
        turns=turns,
        pushbacks=[],
    )


class DialogueExtractorTests(unittest.TestCase):
    def test_package_provides_module_cli_entrypoint(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dialogue_extractor", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Extract pure 5-turn dialogues", result.stdout)

    def test_extract_dialogues_writes_one_row_per_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript_dir = Path(tmp) / "transcript"
            output_dir = Path(tmp) / "dialogue"
            write_transcript(
                transcript_for("item-1", "Question one?"),
                transcript_dir,
                overwrite=False,
            )
            write_transcript(
                transcript_for("item-2", "Question two?"),
                transcript_dir,
                overwrite=False,
            )

            written = extract_dialogues(
                transcript_dir=transcript_dir,
                output_dir=output_dir,
            )

            expected_path = (
                output_dir
                / "model__a"
                / "formal_without_simulator_history"
                / "dialogue.csv"
            )
            self.assertEqual([expected_path], written)

            with expected_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(list(DIALOGUE_COLUMNS), list(rows[0].keys()))
            self.assertEqual(2, len(rows))
            self.assertEqual("Question one?", rows[0]["question"])
            self.assertEqual("item-1 response 1", rows[0]["response_1"])
            self.assertEqual("item-1 prompt 2", rows[0]["prompt_2"])
            self.assertEqual("item-1 response 5", rows[0]["response_5"])
            self.assertEqual("Question two?", rows[1]["question"])

    def test_extract_dialogues_requires_complete_five_turn_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript_dir = Path(tmp) / "transcript"
            write_transcript(
                transcript_for("item-1", "Question one?", turn_count=4),
                transcript_dir,
                overwrite=False,
            )

            with self.assertRaisesRegex(ValueError, "exactly 5 turns"):
                extract_dialogues(
                    transcript_dir=transcript_dir,
                    output_dir=Path(tmp) / "dialogue",
                )


if __name__ == "__main__":
    unittest.main()
