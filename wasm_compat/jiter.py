"""Minimal jiter projection for WASI.

``jiter`` is a Rust extension with no WASI build. DSPy imports ``openai``
eagerly, which imports ``jiter.from_json`` for streaming chat parsing. The
component never talks to OpenAI directly (the host owns provider calls), so
only complete-document parsing is projected; partial parsing traps.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["from_json"]


def from_json(
    json_data: bytes,
    *,
    allow_inf_nan: bool = True,
    cache_mode: Any = True,
    partial_mode: Any = False,
    catch_duplicate_keys: bool = False,
    float_mode: Any = "float",
) -> Any:
    if partial_mode not in (False, "off"):
        raise NotImplementedError("jiter partial parsing is unavailable inside dspy-wasm")
    return json.loads(json_data)
