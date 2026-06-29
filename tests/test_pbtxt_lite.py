"""Tests for stdlib pbtxt parser fallback."""

import unittest
from pathlib import Path

from triton_validator.config_parser import load_model_config_from_pbtxt
from triton_validator.pbtxt_lite import parse_model_config_pbtxt

_EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "model_repository"
    / "example_model"
    / "config.pbtxt"
)


class TestPbtxtLite(unittest.TestCase):
    def test_parse_example_model(self):
        text = _EXAMPLE_CONFIG.read_text(encoding="utf-8")
        parsed = parse_model_config_pbtxt(text)
        self.assertEqual(parsed["name"], "example_model")
        self.assertEqual(parsed["max_batch_size"], 8)
        self.assertEqual(parsed["input"][0]["name"], "INPUT0")
        self.assertEqual(parsed["input"][0]["dims"], (3, 224, 224))

    def test_load_via_lite_fallback_path(self):
        spec = load_model_config_from_pbtxt(_EXAMPLE_CONFIG)
        self.assertEqual(spec.inputs[0].data_type, "TYPE_FP32")


if __name__ == "__main__":
    unittest.main()
