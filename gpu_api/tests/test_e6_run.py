"""Regression tests for E6's DeepSeek JSON output protocol."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import existing_ids  # noqa: E402
from e6_run import parse_label  # noqa: E402


class ParseLabelTest(unittest.TestCase):
    def test_accepts_exact_json_labels(self) -> None:
        self.assertEqual(parse_label('{"label":"A"}'), "A")
        self.assertEqual(parse_label('{"label":"B"}'), "B")
        self.assertEqual(parse_label('{"label":"No preference"}'), "No preference")

    def test_rejects_non_contract_outputs(self) -> None:
        self.assertIsNone(parse_label('{"label":"A", "reason":"..."}'))
        self.assertIsNone(parse_label('{"choice":"A"}'))
        self.assertIsNone(
            parse_label("Response A adheres to the principle of being helpful.")
        )

    def test_saved_failure_is_terminal_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgements.jsonl"
            path.write_text(
                '{"request_id": "ok", "status": "ok"}\n'
                '{"request_id": "failed", "status": "failed"}\n',
                encoding="utf-8",
            )
            self.assertEqual(existing_ids(path), {"ok", "failed"})


if __name__ == "__main__":
    unittest.main()
