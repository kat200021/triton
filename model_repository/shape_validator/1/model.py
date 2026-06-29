import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

# Prefer bundled validator beside model.py; fall back to repo src/ for dev.
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, "..", "..", ".."))
for _path in (_MODEL_DIR, os.path.join(_REPO_ROOT, "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from triton_validator.config_parser import (  # noqa: E402
    load_model_config_from_pbtxt,
    resolve_target_config_path,
)
from triton_validator.shape_validator import (  # noqa: E402
    ShapeValidationError,
    validate_request_inputs,
)


def _get_parameter(model_config: dict, key: str, default: str | None = None) -> str | None:
    params = model_config.get("parameters", {})
    entry = params.get(key)
    if entry is None:
        return default
    return entry.get("string_value", default)


def _numpy_dtype_to_triton(dtype: np.dtype) -> str:
    """Map numpy dtype to Triton TYPE_* string (pb_utils covers the common cases)."""
    return pb_utils.numpy_to_triton_dtype(np.dtype(dtype))


def _extract_request_tensors(request) -> dict[str, dict]:
    tensors: dict[str, dict] = {}
    for tensor in request.inputs():
        arr = tensor.as_numpy()
        tensors[tensor.name()] = {
            "shape": tuple(int(d) for d in arr.shape),
            "data_type": _numpy_dtype_to_triton(arr.dtype),
        }
    return tensors


class TritonPythonModel:
    """Validation middleware: check inputs, optionally forward to a target model."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model_repository = args["model_repository"]

        self.target_model = _get_parameter(self.model_config, "TARGET_MODEL")
        if not self.target_model:
            raise pb_utils.TritonModelException(
                "Parameter TARGET_MODEL is required (name of model to validate against)."
            )

        self.target_version = _get_parameter(self.model_config, "TARGET_MODEL_VERSION")
        forward = (_get_parameter(self.model_config, "FORWARD_ON_SUCCESS", "false") or "false").lower()
        self.forward_on_success = forward in {"1", "true", "yes"}

        strict = (_get_parameter(self.model_config, "STRICT_INPUTS", "true") or "true").lower()
        self.strict_inputs = strict not in {"0", "false", "no"}

        config_path = resolve_target_config_path(
            self.model_repository,
            self.target_model,
            self.target_version,
        )
        self.target_config = load_model_config_from_pbtxt(config_path)

        if self.forward_on_success:
            self.forward_output_names = [
                output["name"] for output in self.model_config.get("output", [])
            ]
            if not self.forward_output_names:
                raise pb_utils.TritonModelException(
                    "FORWARD_ON_SUCCESS is enabled but this model defines no outputs."
                )

        pb_utils.Logger.log_info(
            f"shape_validator: loaded config for target '{self.target_model}' "
            f"from {config_path}"
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                tensors = _extract_request_tensors(request)
                validate_request_inputs(
                    self.target_config,
                    tensors,
                    strict=self.strict_inputs,
                )
            except ShapeValidationError as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Input validation failed: {exc}")
                    )
                )
                continue

            if self.forward_on_success:
                forward_request = pb_utils.InferenceRequest(
                    model_name=self.target_model,
                    requested_output_names=self.forward_output_names,
                    inputs=request.inputs(),
                    model_version=self.target_version if self.target_version else -1,
                )
                forward_response = forward_request.exec()
                if forward_response.has_error():
                    responses.append(forward_response)
                else:
                    responses.append(forward_response)
                continue

            # Pass-through mode: echo validated inputs on outputs with matching names.
            output_tensors = []
            for output_cfg in self.model_config.get("output", []):
                out_name = output_cfg["name"]
                in_tensor = pb_utils.get_input_tensor_by_name(request, out_name)
                if in_tensor is None:
                    responses.append(
                        pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(
                                f"Pass-through output '{out_name}' has no matching input."
                            )
                        )
                    )
                    break
                output_tensors.append(
                    pb_utils.Tensor(out_name, in_tensor.as_numpy())
                )
            else:
                responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def finalize(self):
        pb_utils.Logger.log_info("shape_validator: finalized")
