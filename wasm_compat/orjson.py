"""Small orjson compatibility projection for WASI.

DSPy 3.4 uses dumps/loads plus three option flags on its eager import path.
Keep this module intentionally narrow: unsupported orjson surface should fail
rather than silently acquiring different semantics.
"""

from __future__ import annotations

import json
from typing import Any, Callable

OPT_SORT_KEYS = 1
OPT_INDENT_2 = 2
OPT_APPEND_NEWLINE = 4


def dumps(
    obj: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    option: int = 0,
) -> bytes:
    indent = 2 if option & OPT_INDENT_2 else None
    text = json.dumps(
        obj,
        default=default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=bool(option & OPT_SORT_KEYS),
        separators=None if indent else (",", ":"),
    )
    data = text.encode("utf-8")
    if option & OPT_APPEND_NEWLINE:
        data += b"\n"
    return data


def loads(data: str | bytes | bytearray) -> Any:
    return json.loads(data)
