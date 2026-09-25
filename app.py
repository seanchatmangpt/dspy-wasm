"""Operational DSPy WebAssembly component.

DSPy remains upstream code. WASM-specific policy lives at two boundaries:
1. native-wheel compatibility supplied by the build,
2. LM actuation supplied by the host through WIT.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
import traceback
from typing import Any, Callable

import dspy
import wit
from dspy.clients import configure_cache
from dspy.clients.cache import Cache
from dspy.dsp.utils.settings import DEFAULT_CONFIG
from dspy.lm15 import Message, Response, Usage
from dspy.utils.dummies import DummyLM
from wit.imports import host_lm


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "toDict"):
        return value.toDict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _reset() -> None:
    # Mirrors the material part of DSPy's tests/conftest.py fixture:
    # clean settings and memory-only cache, with no filesystem dependency.
    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG))
    configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=None,
    )


class HostEngine:
    """DSPy 3.4 custom engine backed by the WIT host capability."""

    def complete(self, request):
        payload = {
            "model": request.model,
            "system": request.system,
            "messages": [
                {
                    "role": message.role,
                    "text": message.text,
                }
                for message in request.messages
            ],
        }
        raw = json.loads(host_lm.complete(json.dumps(payload, sort_keys=True)))
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise TypeError("host LM must return a JSON object containing string field 'text'")
        return Response(
            id=raw.get("id"),
            model=raw.get("model") or request.model,
            message=Message.assistant(raw["text"]),
            finish_reason=raw.get("finish_reason", "stop"),
            usage=Usage(),
        )


def _host_lm() -> dspy.LM:
    return dspy.LM(
        "wasm-host",
        engine=HostEngine(),
        cache=False,
        num_retries=0,
    )


def _case_example() -> None:
    example = dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")
    assert example.inputs()["question"] == "Capital of France?"
    assert example.labels()["answer"] == "Paris"
    assert example.copy() == example


def _case_signature() -> None:
    signature = dspy.Signature("question -> answer")
    assert list(signature.input_fields) == ["question"]
    assert list(signature.output_fields) == ["answer"]

    typed = dspy.Signature("input1: int, input2: str -> output: float")
    assert typed.input_fields["input1"].annotation is int
    assert typed.input_fields["input2"].annotation is str
    assert typed.output_fields["output"].annotation is float


def _case_cache() -> None:
    cache = Cache(False, True, None)
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert cache.cache_key(left) == cache.cache_key(right)
    cache.put(left, {"value": 42})
    assert cache.get(right) == {"value": 42}


def _case_dummy_predict() -> None:
    _reset()
    lm = DummyLM([{"answer": "Paris"}])
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
    )
    prediction = dspy.Predict("question -> answer")(question="Capital of France?")
    assert prediction.answer == "Paris"


def _case_chain_of_thought() -> None:
    _reset()
    lm = DummyLM([{"reasoning": "Two plus two is four.", "answer": "4"}])
    dspy.configure(
        lm=lm,
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
    )
    prediction = dspy.ChainOfThought("question -> answer")(question="What is 2 + 2?")
    assert prediction.answer == "4"
    assert prediction.reasoning == "Two plus two is four."


def _case_host_engine_predict() -> None:
    _reset()
    dspy.configure(
        lm=_host_lm(),
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
    )
    prediction = dspy.Predict("question -> answer")(question="Capital of France?")
    assert prediction.answer == "Paris"


CASES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("example", _case_example),
    ("signature", _case_signature),
    ("memory-cache", _case_cache),
    ("dummy-predict", _case_dummy_predict),
    ("chain-of-thought", _case_chain_of_thought),
    ("host-engine-predict", _case_host_engine_predict),
)


def _self_test_report() -> dict[str, Any]:
    results = []
    for name, case in CASES:
        try:
            case()
            results.append({"name": name, "state": "ALIVE"})
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "state": "BUILD_BROKEN",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    failed = sum(item["state"] != "ALIVE" for item in results)
    return {
        "state": "ALIVE" if failed == 0 else "BUILD_BROKEN",
        "passed": len(results) - failed,
        "failed": failed,
        "cases": results,
    }


@wit.guest
class Dspy(wit.WorldExports):
    def component_version(self) -> str:
        return "0.1.0"

    def runtime_info(self) -> str:
        return json.dumps(
            {
                "component": "dspy-wasm",
                "python": sys.version.split()[0],
                "platform": platform.system(),
                "dspy": getattr(dspy, "__version__", "unknown"),
                "lm_boundary": "chatman:dspy/lm@0.1.0",
                "state": "DSPY_IMPORTED",
            },
            sort_keys=True,
        )

    def dspy_version(self) -> str:
        return json.dumps(
            {
                "state": "ALIVE",
                "version": getattr(dspy, "__version__", "unknown"),
            },
            sort_keys=True,
        )

    def run_self_tests(self) -> str:
        return json.dumps(_self_test_report(), sort_keys=True)

    def predict(self, signature: str, inputs_json: str) -> str:
        try:
            _reset()
            dspy.configure(
                lm=_host_lm(),
                adapter=dspy.ChatAdapter(use_json_adapter_fallback=False),
            )
            inputs = json.loads(inputs_json)
            if not isinstance(inputs, dict):
                raise TypeError("inputs-json must decode to an object")
            prediction = dspy.Predict(signature)(**inputs)
            return json.dumps(
                {
                    "state": "ALIVE",
                    "outputs": prediction.toDict(),
                },
                default=_json_default,
                sort_keys=True,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "state": "BUILD_BROKEN",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            )
