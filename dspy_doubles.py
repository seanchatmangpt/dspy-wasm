"""Deterministic LM doubles speaking the host JSON protocol.

Used by the native test court and by the in-WASM self-tests, so both courts
exercise the same doubles through the same ``lm_call`` boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_OUTPUTS = re.compile(r"Your output fields are:\n(.*?)\n(?:All interactions|\Z)", re.DOTALL)
_FIELD = re.compile(r"^\d+\. `(\w+)` \((.*?)\)(?::|$)", re.MULTILINE)


def chat(**fields: Any) -> str:
    """Render a ChatAdapter completion for the given output fields."""
    body = "".join(
        f"[[ ## {name} ## ]]\n{json.dumps(v) if isinstance(v, (dict, list)) else v}\n\n"
        for name, v in fields.items()
    )
    return body + "[[ ## completed ## ]]"


def scripted(*texts: str) -> Callable[[str], str]:
    """Return the given completions in order; the last one repeats."""
    queue = list(texts)

    def lm_call(_request_json: str) -> str:
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return json.dumps({"text": text, "usage": {"input_tokens": 3, "output_tokens": 2}})

    return lm_call


def _value(name: str, annotation: str, overrides: dict[str, Any]) -> Any:
    if name in overrides:
        return overrides[name]
    lowered = annotation.lower()
    if lowered.startswith("literal["):
        return re.findall(r"'([^']*)'|\"([^\"]*)\"", annotation)[0][0] or annotation
    if lowered == "bool":
        return True
    if lowered == "int":
        return 1
    if lowered == "float":
        return 1.0
    if lowered.startswith(("list", "tuple")):
        return []
    if lowered.startswith("dict"):
        return {}
    return f"{name} value"


def schema_echo(**overrides: Any) -> Callable[[str], str]:
    """Answer *any* ChatAdapter signature with type-correct placeholder values.

    Optimizers (COPRO, MIPROv2, SIMBA, GEPA, InferRules) issue LM calls under
    signatures they construct internally; this double reads the output fields
    from the system prompt so those pipelines run end to end deterministically.
    """

    def lm_call(request_json: str) -> str:
        system = json.loads(request_json).get("system") or ""
        block = _OUTPUTS.search(system)
        fields = _FIELD.findall(block.group(1)) if block else [("answer", "str")]
        text = chat(**{name: _value(name, annotation, overrides) for name, annotation in fields})
        return json.dumps({"text": text, "usage": {"input_tokens": 1, "output_tokens": 1}})

    return lm_call
