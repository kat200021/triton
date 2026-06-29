"""Unit tests for shape validation (no Triton server required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from triton_validator.config_parser import load_model_config_from_pbtxt
from triton_validator.shape_validator import ShapeValidationError, validate_request_inputs

_REPO = Path(__file__).resolve().parents[1]
_EXAMPLE_CONFIG = _REPO / "model_repository" / "example_model" / "config.pbtxt"


class TestConfigParser(unittest.TestCase):
    def test_load_example_model_config(self):
        spec = load_model_config_from_pbtxt(_EXAMPLE_CONFIG)
        self.assertEqual(spec.name, "example_model")
        self.assertEqual(spec.max_batch_size, 8)
        self.assertEqual(len(spec.inputs), 1)
        self.assertEqual(spec.inputs[0].name, "INPUT0")
        self.assertEqual(spec.inputs[0].dims, (3, 224, 224))

    def test_round_trip_pbtxt_write_read(self):
        spec = load_model_config_from_pbtxt(_EXAMPLE_CONFIG)
        from tritonclient.grpc import model_config_pb2
        from google.protobuf import text_format

        proto = model_config_pb2.ModelConfig()
        proto.name = spec.name
        proto.max_batch_size = spec.max_batch_size
        for inp in spec.inputs:
            model_input = proto.input.add()
            model_input.name = inp.name
            model_input.data_type = model_config_pb2.DataType.Value(inp.data_type)
            model_input.dims.extend(inp.dims)

        with tempfile.NamedTemporaryFile("w", suffix=".pbtxt", delete=False) as fh:
            fh.write(text_format.MessageToString(proto))
            path = fh.name

        reloaded = load_model_config_from_pbtxt(path)
        self.assertEqual(reloaded.name, spec.name)
        self.assertEqual(reloaded.inputs[0].dims, spec.inputs[0].dims)


class TestShapeValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_model_config_from_pbtxt(_EXAMPLE_CONFIG)

    def test_valid_batched_input(self):
        validate_request_inputs(
            self.spec,
            {
                "INPUT0": {
                    "shape": (2, 3, 224, 224),
                    "data_type": "TYPE_FP32",
                }
            },
        )

    def test_invalid_batch_rank(self):
        with self.assertRaises(ShapeValidationError) as ctx:
            validate_request_inputs(
                self.spec,
                {
                    "INPUT0": {
                        "shape": (3, 224, 224),
                        "data_type": "TYPE_FP32",
                    }
                },
            )
        self.assertIn("expected 4 dimensions", str(ctx.exception))

    def test_invalid_channel_dim(self):
        with self.assertRaises(ShapeValidationError) as ctx:
            validate_request_inputs(
                self.spec,
                {
                    "INPUT0": {
                        "shape": (1, 1, 224, 224),
                        "data_type": "TYPE_FP32",
                    }
                },
            )
        self.assertIn("dimension 1 expected 3", str(ctx.exception))

    def test_missing_required_input(self):
        with self.assertRaises(ShapeValidationError) as ctx:
            validate_request_inputs(self.spec, {})
        self.assertIn("missing", str(ctx.exception))

    def test_wrong_dtype(self):
        with self.assertRaises(ShapeValidationError) as ctx:
            validate_request_inputs(
                self.spec,
                {
                    "INPUT0": {
                        "shape": (1, 3, 224, 224),
                        "data_type": "TYPE_INT32",
                    }
                },
            )
        self.assertIn("data_type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
