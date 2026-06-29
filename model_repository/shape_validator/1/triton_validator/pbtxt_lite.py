"""Stdlib-only parser for the subset of config.pbtxt needed for input validation."""

from __future__ import annotations

import re
from typing import Any

_BLOCK_RE = re.compile(r"\{([^{}]*)\}", re.DOTALL)
_STRING_FIELD_RE = re.compile(r'(\w+):\s*"([^"]*)"')
_SCALAR_FIELD_RE = re.compile(r"(\w+):\s*(\S+)")
_DIMS_RE = re.compile(r"dims:\s*\[\s*([^\]]*)\]")


def _parse_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def _parse_dims(raw: str) -> tuple[int, ...]:
    dims: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        dims.append(int(token))
    return tuple(dims)


def _parse_block(block_text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for match in _STRING_FIELD_RE.finditer(block_text):
        fields[match.group(1)] = match.group(2)

    for match in _SCALAR_FIELD_RE.finditer(block_text):
        key, value = match.group(1), match.group(2)
        if key in fields:
            continue
        if key in {"optional", "allow_ragged_batch", "is_shape_tensor"}:
            fields[key] = _parse_bool(value)
        elif key == "max_batch_size":
            fields[key] = int(value)
        elif key not in {"data_type", "name"}:
            continue
        else:
            fields[key] = value

    dims_match = _DIMS_RE.search(block_text)
    if dims_match:
        fields["dims"] = _parse_dims(dims_match.group(1))

    return fields


def _extract_input_blocks(text: str) -> list[dict[str, Any]]:
    match = re.search(r"input\s*\[(.*)\]\s*(?:output|parameters|instance_group|$)", text, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    return [_parse_block(block.group(1)) for block in _BLOCK_RE.finditer(section)]


def parse_model_config_pbtxt(text: str) -> dict[str, Any]:
    """Parse name, max_batch_size, and input specs from config.pbtxt without protobuf."""
    name_match = re.search(r'name:\s*"([^"]+)"', text)
    if not name_match:
        raise ValueError("config.pbtxt is missing required field: name")

    max_batch_size = 0
    max_batch_match = re.search(r"max_batch_size:\s*(\d+)", text)
    if max_batch_match:
        max_batch_size = int(max_batch_match.group(1))

    inputs: list[dict[str, Any]] = []
    for block in _extract_input_blocks(text):
        if "name" not in block or "data_type" not in block:
            continue
        inputs.append(
            {
                "name": block["name"],
                "data_type": block["data_type"],
                "dims": block.get("dims", ()),
                "optional": bool(block.get("optional", False)),
                "allow_ragged_batch": bool(block.get("allow_ragged_batch", False)),
                "is_shape_tensor": bool(block.get("is_shape_tensor", False)),
            }
        )

    return {
        "name": name_match.group(1),
        "max_batch_size": max_batch_size,
        "input": inputs,
    }
