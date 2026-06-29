"""Parse Triton ModelConfig from config.pbtxt or runtime JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

ModelConfigPath = Union[str, Path]


def _import_model_config_pb2():
    """Load Triton's ModelConfig protobuf module (shipped with tritonclient)."""
    try:
        from tritonclient.grpc import model_config_pb2 as mc

        return mc
    except ImportError as exc:
        raise ImportError(
            "tritonclient is required to parse config.pbtxt. "
            "Install with: pip install 'tritonclient[grpc]'"
        ) from exc


@dataclass(frozen=True)
class InputSpec:
    name: str
    data_type: str
    dims: tuple[int, ...]
    optional: bool = False
    allow_ragged_batch: bool = False
    is_shape_tensor: bool = False


@dataclass(frozen=True)
class ModelConfigSpec:
    """Normalized view of a Triton ModelConfig for validation."""

    name: str
    max_batch_size: int
    inputs: tuple[InputSpec, ...] = field(default_factory=tuple)

    @property
    def input_by_name(self) -> dict[str, InputSpec]:
        return {spec.name: spec for spec in self.inputs}

    @property
    def supports_batching(self) -> bool:
        return self.max_batch_size > 0


def _normalize_data_type(data_type: Any) -> str:
    if isinstance(data_type, str):
        if data_type.isdigit():
            mc = _import_model_config_pb2()
            return mc.DataType.Name(int(data_type))
        return data_type
    if isinstance(data_type, int):
        mc = _import_model_config_pb2()
        return mc.DataType.Name(data_type)
    if hasattr(data_type, "name"):
        return str(data_type.name)
    return str(data_type)


def _dims_from_proto(dims: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(d) for d in dims)


def _input_spec_from_proto(model_input) -> InputSpec:
    return InputSpec(
        name=model_input.name,
        data_type=_normalize_data_type(model_input.data_type),
        dims=_dims_from_proto(model_input.dims),
        optional=bool(model_input.optional),
        allow_ragged_batch=bool(model_input.allow_ragged_batch),
        is_shape_tensor=bool(model_input.is_shape_tensor),
    )


def model_config_from_proto(model_config) -> ModelConfigSpec:
    return ModelConfigSpec(
        name=model_config.name,
        max_batch_size=int(model_config.max_batch_size),
        inputs=tuple(_input_spec_from_proto(inp) for inp in model_config.input),
    )


def _load_model_config_from_pbtxt_protobuf(pbtxt: str) -> ModelConfigSpec:
    from google.protobuf import text_format

    mc = _import_model_config_pb2()
    config = mc.ModelConfig()
    text_format.Merge(pbtxt, config)
    return model_config_from_proto(config)


def _load_model_config_from_pbtxt_lite(pbtxt: str) -> ModelConfigSpec:
    from triton_validator.pbtxt_lite import parse_model_config_pbtxt

    return load_model_config_from_json(parse_model_config_pbtxt(pbtxt))


def load_model_config_from_pbtxt(path: ModelConfigPath) -> ModelConfigSpec:
    """Parse a config.pbtxt file into a ModelConfigSpec."""
    pbtxt = Path(path).read_text(encoding="utf-8")
    try:
        return _load_model_config_from_pbtxt_protobuf(pbtxt)
    except ImportError:
        return _load_model_config_from_pbtxt_lite(pbtxt)


def load_model_config_from_json(config: Union[str, Mapping[str, Any]]) -> ModelConfigSpec:
    """Parse Triton runtime model_config JSON (from Python backend initialize args)."""
    if isinstance(config, str):
        config = json.loads(config)

    inputs: list[InputSpec] = []
    for inp in config.get("input", []):
        inputs.append(
            InputSpec(
                name=inp["name"],
                data_type=inp["data_type"],
                dims=tuple(int(d) for d in inp.get("dims", [])),
                optional=bool(inp.get("optional", False)),
                allow_ragged_batch=bool(inp.get("allow_ragged_batch", False)),
                is_shape_tensor=bool(inp.get("is_shape_tensor", False)),
            )
        )

    return ModelConfigSpec(
        name=config.get("name", ""),
        max_batch_size=int(config.get("max_batch_size", 0)),
        inputs=tuple(inputs),
    )


def resolve_target_config_path(
    model_repository: str,
    target_model: str,
    target_version: str | None = None,
) -> Path:
    """
    Resolve config.pbtxt for a model in the Triton model repository.

    Triton stores config at <repo>/<model>/config.pbtxt (not under version dirs).
    The Python backend may pass either the repository root or the loading model's
    directory as ``model_repository``, so walk upward until the target is found.
    """
    version_hint = f" (version {target_version})" if target_version else ""
    current = Path(model_repository).resolve()
    searched: list[Path] = []

    while True:
        config_path = current / target_model / "config.pbtxt"
        searched.append(config_path)
        if config_path.is_file():
            return config_path
        if current.parent == current:
            break
        current = current.parent

    paths = "\n".join(f"  - {path}" for path in searched)
    raise FileNotFoundError(
        f"Target model config not found for '{target_model}'{version_hint}. "
        f"Searched:\n{paths}"
    )
