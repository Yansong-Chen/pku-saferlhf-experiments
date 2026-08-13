"""Regression tests for E5's DeepSeek JSON safety contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from e5_run import parse_deepseek_safety_label  # noqa: E402


class DeepSeekSafetyParserTest(unittest.TestCase):
    def test_accepts_exact_safety_labels(self) -> None:
        self.assertEqual(parse_deepseek_safety_label('{"label":"safe"}'), "safe")
        self.assertEqual(parse_deepseek_safety_label('{"label":"unsafe"}'), "unsafe")

    def test_rejects_non_contract_outputs(self) -> None:
        self.assertIsNone(parse_deepseek_safety_label('{"label":"safe", "reason":"..."}'))
        self.assertIsNone(parse_deepseek_safety_label('{"label":"uncertain"}'))
        self.assertIsNone(parse_deepseek_safety_label("The response is safe."))


if __name__ == "__main__":
    unittest.main()
