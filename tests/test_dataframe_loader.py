import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.to_dataframe import INPUT_PATH, load_dataframe


class LoadDataframeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.directory = Path(self.temp_directory.name)

    def test_loads_csv_by_extension(self) -> None:
        input_path = self.directory / "records.csv"
        input_path.write_text("name,score\nAda,10\nGrace,9\n", encoding="utf-8")

        dataframe = load_dataframe(input_path)

        pd.testing.assert_frame_equal(
            dataframe,
            pd.DataFrame({"name": ["Ada", "Grace"], "score": [10, 9]}),
        )

    def test_uses_editable_input_path_when_no_path_is_given(self) -> None:
        dataframe = load_dataframe()

        self.assertTrue(INPUT_PATH.exists())
        self.assertEqual(len(dataframe), 3004)

    def test_loads_json_array_by_extension(self) -> None:
        input_path = self.directory / "records.json"
        input_path.write_text(
            json.dumps([{"name": "Ada", "score": 10}]), encoding="utf-8"
        )

        dataframe = load_dataframe(input_path)

        pd.testing.assert_frame_equal(
            dataframe,
            pd.DataFrame({"name": ["Ada"], "score": [10]}),
        )

    def test_loads_jsonl_and_skips_blank_lines(self) -> None:
        input_path = self.directory / "records.jsonl"
        input_path.write_text(
            '{"name": "Ada", "score": 10}\n\n{"name": "Grace", "score": 9}\n',
            encoding="utf-8",
        )

        dataframe = load_dataframe(input_path)

        pd.testing.assert_frame_equal(
            dataframe,
            pd.DataFrame(
                {"name": ["Ada", "Grace"], "score": [10, 9]}
            ),
        )

    def test_explicit_format_supports_files_without_extension(self) -> None:
        input_path = self.directory / "records"
        input_path.write_text("name,score\nAda,10\n", encoding="utf-8")

        dataframe = load_dataframe(input_path, file_format="csv")

        self.assertEqual(dataframe.to_dict(orient="records"), [{"name": "Ada", "score": 10}])

    def test_rejects_unsupported_extension(self) -> None:
        input_path = self.directory / "records.txt"
        input_path.write_text("name\nAda\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Could not determine"):
            load_dataframe(input_path)

    def test_rejects_non_line_jsonl_options(self) -> None:
        input_path = self.directory / "records.jsonl"
        input_path.write_text('{"name": "Ada"}\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "lines=True"):
            load_dataframe(input_path, lines=False)


if __name__ == "__main__":
    unittest.main()
