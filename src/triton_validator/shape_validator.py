"""Validate inference request tensor shapes against a ModelConfigSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from triton_validator.config_parser import InputSpec, ModelConfigSpec

# Triton uses -1 for variable-size dimensions in config.pbtxt.
VARIABLE_DIM = -1


@dataclass(frozen=True)
class ShapeValidationError(Exception):
    """Raised when request tensors do not match the model configuration."""

    message: str
    input_name: str | None = None

    def __str__(self) -> str:
        if self.input_name:
            return f"[{self.input_name}] {self.message}"
        return self.message


def _compare_dim(expected: int, actual: int, dim_index: int, input_name: str) -> None:
    if expected == VARIABLE_DIM:
        if actual < 0:
            raise ShapeValidationError(
                f"dimension {dim_index} must be non-negative, got {actual}",
                input_name=input_name,
            )
        return
    if expected != actual:
        raise ShapeValidationError(
            f"dimension {dim_index} expected {expected}, got {actual}",
            input_name=input_name,
        )


def _validate_shape_against_dims(
    actual_shape: Sequence[int],
    expected_dims: Sequence[int],
    *,
    input_name: str,
    supports_batching: bool,
) -> None:
    actual = tuple(int(d) for d in actual_shape)
    expected = tuple(int(d) for d in expected_dims)

    if supports_batching:
        if len(actual) != len(expected) + 1:
            raise ShapeValidationError(
                f"expected {len(expected) + 1} dimensions (batch + {len(expected)}), "
                f"got {len(actual)}: {actual}",
                input_name=input_name,
            )
        batch_size = actual[0]
        if batch_size < 0:
            raise ShapeValidationError(
                f"batch dimension must be non-negative, got {batch_size}",
                input_name=input_name,
            )
        for idx, (exp, act) in enumerate(zip(expected, actual[1:], strict=False)):
            _compare_dim(exp, act, idx + 1, input_name)
        return

    if len(actual) != len(expected):
        raise ShapeValidationError(
            f"expected {len(expected)} dimensions {expected}, got {len(actual)}: {actual}",
            input_name=input_name,
        )
    for idx, (exp, act) in enumerate(zip(expected, actual, strict=False)):
        _compare_dim(exp, act, idx, input_name)


def validate_tensor(
    *,
    input_name: str,
    actual_shape: Sequence[int],
    actual_dtype: str,
    spec: InputSpec,
    supports_batching: bool,
) -> None:
    """Validate a single tensor name, dtype, and shape."""
    if actual_dtype != spec.data_type:
        raise ShapeValidationError(
            f"expected data_type {spec.data_type}, got {actual_dtype}",
            input_name=input_name,
        )

    if spec.is_shape_tensor:
        # Shape tensors carry int64 metadata; only check element count loosely.
        expected_elements = 1
        for d in spec.dims:
            expected_elements *= max(d, 1) if d != VARIABLE_DIM else 1
        actual_elements = 1
        for d in actual_shape:
            actual_elements *= int(d)
        if spec.dims and expected_elements > 1 and actual_elements != expected_elements:
            raise ShapeValidationError(
                f"shape tensor expected {expected_elements} elements, "
                f"got shape {tuple(actual_shape)} ({actual_elements} elements)",
                input_name=input_name,
            )
        return

    _validate_shape_against_dims(
        actual_shape,
        spec.dims,
        input_name=input_name,
        supports_batching=supports_batching,
    )


def validate_request_inputs(
    model_config: ModelConfigSpec,
    request_tensors: Mapping[str, Mapping[str, Any]],
    *,
    strict: bool = True,
) -> None:
    """
    Validate all tensors in an inference request against model config inputs.

    Parameters
    ----------
    model_config:
        Parsed target model configuration.
    request_tensors:
        Mapping of input name -> {"shape": tuple[int, ...], "data_type": str}.
    strict:
        When True, reject unknown input names present in the request.
    """
    provided = set(request_tensors)
    expected_by_name = model_config.input_by_name

    for spec in model_config.inputs:
        if spec.optional and spec.name not in provided:
            continue
        if spec.name not in provided:
            raise ShapeValidationError(
                f"required input '{spec.name}' is missing from request",
                input_name=spec.name,
            )

    if strict:
        unknown = provided - set(expected_by_name)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ShapeValidationError(f"unexpected input(s) in request: {names}")

    for name, tensor in request_tensors.items():
        spec = expected_by_name.get(name)
        if spec is None:
            continue
        validate_tensor(
            input_name=name,
            actual_shape=tensor["shape"],
            actual_dtype=tensor["data_type"],
            spec=spec,
            supports_batching=model_config.supports_batching,
        )
