"""Operational DSPy WebAssembly component.

DSPy remains upstream code. WASM-specific policy lives at three boundaries:
1. native-wheel compatibility supplied by the build,
2. LM actuation supplied by the host through WIT (`chatman:dspy/lm`),
3. tool actuation supplied by the host through WIT (`chatman:dspy/tools`).
"""

from __future__ import annotations

import copy
import json
import linecache
import platform
import sys
import traceback
from typing import Any, Callable

import dspy
import dspy_bindings as wit
from dspy.clients import configure_cache
from dspy.clients.cache import Cache
from dspy.dsp.utils.settings import DEFAULT_CONFIG
from dspy.utils.dummies import DummyLM
from dspy_bindings.imports import host_lm, host_tools

import dspy_capabilities as caps
from dspy_doubles import chat as _chat
from dspy_doubles import schema_echo
from dspy_doubles import scripted as _scripted

# componentize-py snapshots the interpreter after module import, and the host
# grants no filesystem. Anything DSPy would import lazily on a call path must
# therefore be imported here, at build time.
import cachetools  # noqa: E402,F401
import cachetools.keys  # noqa: E402,F401
import dspy.clients.call_result  # noqa: E402,F401
import dspy.clients.costs  # noqa: E402,F401
import dspy.clients.engines.dummy_engine  # noqa: E402,F401
import dspy.clients.engines.streaming  # noqa: E402,F401
import dspy.clients.execution  # noqa: E402,F401
import dspy.utils.hasher  # noqa: E402,F401
import gepa.lm  # noqa: E402,F401

try:  # tqdm's write lock probes multiprocessing; absent under WASI is fine.
    import multiprocessing.synchronize  # noqa: E402,F401
except ImportError:
    pass


def _reset() -> None:
    # Mirrors the material part of DSPy's tests/conftest.py fixture:
    # clean settings and memory-only cache, with no filesystem dependency.
    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG))
    configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=None,
    )


def _host_lm() -> dspy.LM:
    return caps.build_lm(None, caps.HostEngine(host_lm.complete))


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


def _run(request: dict[str, Any], *texts: str) -> dict[str, Any]:
    _reset()
    return caps.run(request, _scripted(*texts), host_tools.call)


def _case_host_tool() -> None:
    # Every conforming host supplies the deterministic `calculator` tool.
    assert caps.HostTools(host_tools.call).call("calculator", {"expression": "6 * 7"}) == 42


def _case_react_host_tool() -> None:
    report = _run(
        {
            "module": "react",
            "signature": "question -> answer",
            "tools": [
                {
                    "name": "calculator",
                    "description": "Evaluate an arithmetic expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                }
            ],
            "max_iters": 3,
            "inputs": {"question": "What is 6 * 7?"},
        },
        _chat(
            next_thought="Compute.",
            next_tool_name="calculator",
            next_tool_args={"expression": "6 * 7"},
        ),
        _chat(next_thought="Done.", next_tool_name="finish", next_tool_args={}),
        _chat(reasoning="6 * 7 = 42", answer="42"),
    )
    assert report["outputs"]["trajectory"]["observation_0"] == 42
    assert report["outputs"]["answer"] == "42" and report["lm_calls"] == 3


def _case_structured_adapters() -> None:
    for adapter, text in (("json", '{"answer": "Paris"}'), ("xml", "<answer>Paris</answer>")):
        report = _run(
            {"adapter": adapter, "signature": "question -> answer", "inputs": {"question": "q"}},
            text,
        )
        assert report["outputs"] == {"answer": "Paris"}, adapter


def _case_refine() -> None:
    report = _run(
        {
            "module": "refine",
            "signature": "question -> answer",
            "n": 2,
            "reward": {"name": "exact_match", "expected": {"answer": "Paris"}},
            "inputs": {"question": "Capital of France?"},
        },
        _chat(answer="Paris"),
    )
    assert report["outputs"]["answer"] == "Paris"


def _case_best_of_n_host_reward() -> None:
    report = _run(
        {
            "module": "best-of-n",
            "signature": "question -> answer",
            "n": 3,
            "reward": {"tool": "grade_exact", "expected": {"answer": "Paris"}},
            "inputs": {"question": "Capital of France?"},
        },
        _chat(answer="Lyon"),
        _chat(answer="Paris"),
    )
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 2


def _case_multi_chain_comparison() -> None:
    report = _run(
        {"module": "multi-chain-comparison", "m": 2, "inputs": {"question": "q"}},
        _chat(reasoning="a", answer="Paris"),
        _chat(reasoning="b", answer="Lyon"),
        _chat(rationale="Compare.", answer="Paris"),
    )
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 3


def _case_render() -> None:
    _reset()
    report = caps.render(
        {"signature": "question -> answer", "inputs": {"question": "q"}},
        _scripted("unused"),
        host_tools.call,
    )
    assert [m["role"] for m in report["messages"]] == ["system", "user"]


def _case_evaluate() -> None:
    _reset()
    report = caps.evaluate(
        {
            "program": {"signature": "question -> answer"},
            "devset": [
                {"question": "France?", "answer": "Paris"},
                {"question": "Spain?", "answer": "Madrid"},
            ],
            "metric": "exact_match",
        },
        _scripted(_chat(answer="Paris"), _chat(answer="Rome")),
        host_tools.call,
    )
    assert report["score"] == 50.0 and [r["score"] for r in report["results"]] == [1.0, 0.0]


def _case_compile_round_trip() -> None:
    _reset()
    compiled = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "bootstrap-few-shot",
            "trainset": [
                {"question": "France?", "answer": "Paris"},
                {"question": "Peru?", "answer": "Lima"},
            ],
            "metric": "exact_match",
            "config": {"max_bootstrapped_demos": 2, "max_labeled_demos": 0},
        },
        _scripted(_chat(answer="Paris"), _chat(answer="wrong")),
        host_tools.call,
    )
    assert compiled["demos"] == {"self": 1}
    report = _run(
        {"program_state": compiled["program_state"], "inputs": {"question": "Italy?"}},
        _chat(answer="Rome"),
    )
    assert report["outputs"]["answer"] == "Rome"


CALCULATOR = {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
}


def _case_multihop_pipeline() -> None:
    # Host builtin `search` retrieves from its default corpus.
    step = {
        "name": "generate_query",
        "program": {
            "module": "chain-of-thought",
            "signature": "context: list[str], question -> query",
        },
    }
    report = _run(
        {
            "module": "pipeline",
            "retriever": "search",
            "steps": [
                {"set": {"context": []}},
                {
                    "repeat": 2,
                    "steps": [
                        step,
                        {
                            "retrieve": "$query",
                            "k": 1,
                            "output": "context",
                            "accumulate": ["context"],
                        },
                    ],
                },
                {
                    "name": "generate_answer",
                    "program": {"signature": "context: list[str], question -> answer"},
                },
            ],
            "outputs": ["answer", "context"],
            "inputs": {"question": "Where was the author of Hamlet born?"},
        },
        _chat(reasoning="a", query="Who wrote Hamlet?"),
        _chat(reasoning="b", query="Where was Shakespeare born?"),
        _chat(answer="Stratford-upon-Avon"),
    )
    assert report["outputs"]["context"] == [
        "Hamlet was written by William Shakespeare.",
        "Shakespeare was born in Stratford-upon-Avon.",
    ]
    assert report["outputs"]["answer"] == "Stratford-upon-Avon"


def _case_program_of_thought() -> None:
    report = _run(
        {"module": "program-of-thought", "inputs": {"question": "6 * 7?"}, "trace": True},
        _chat(reasoning="r", generated_code="```python\nresult = 6 * 7\nprint(result)\n```"),
        _chat(reasoning="printed 42", answer="42"),
    )
    assert report["outputs"]["answer"] == "42"
    assert "42" in json.dumps(report["history"][-1]["request"]["messages"])


def _case_code_act_host_tool() -> None:
    report = _run(
        {
            "module": "code-act",
            "tools": [CALCULATOR],
            "inputs": {"question": "6 * 7?"},
            "trace": True,
        },
        _chat(generated_code="```python\nprint(calculator('6 * 7'))\n```", finished=True),
        _chat(reasoning="calculator printed 42", answer="42"),
    )
    assert report["outputs"]["answer"] == "42"
    assert "42" in json.dumps(report["history"][-1]["request"]["messages"])


def _case_rlm() -> None:
    report = _run(
        {"module": "rlm", "inputs": {"question": "6 * 7?"}},
        _chat(reasoning="submit", code="```python\nSUBMIT(answer=str(6 * 7))\n```"),
    )
    assert report["outputs"]["answer"] == "42"


def _case_majority_and_async() -> None:
    report = _run(
        {"module": "majority", "n": 3, "inputs": {"question": "q"}},
        _chat(answer="Paris"),
        _chat(answer="Lyon"),
        _chat(answer="Paris"),
    )
    assert report["outputs"]["answer"] == "Paris"
    report = _run({"inputs": {"question": "q"}, "async": True}, _chat(answer="A"))
    assert report["outputs"] == {"answer": "A"}


def _case_multimodal_parts() -> None:
    captured: list[dict] = []

    def lm_call(request_json: str) -> str:
        captured.append(json.loads(request_json))
        return json.dumps({"text": _chat(answer="a cat")})

    _reset()
    caps.run(
        {
            "signature": "image: Image, question -> answer",
            "inputs": {"image": {"url": "https://example.com/cat.png"}, "question": "What?"},
        },
        lm_call,
        host_tools.call,
    )
    parts = [p for m in captured[0]["messages"] for p in m.get("parts", [])]
    assert any(p["type"] == "image" for p in parts)


def _case_knn_few_shot_host_embedder() -> None:
    _reset()
    report = caps.compile_program(
        {
            "program": {"signature": "question -> answer", "embedder": "embed"},
            "optimizer": "knn-few-shot",
            "trainset": [
                {"question": "Paris is the capital of?", "answer": "France"},
                {"question": "Lima is the capital of?", "answer": "Peru"},
            ],
            "config": {"k": 1, "max_bootstrapped_demos": 0, "max_labeled_demos": 1},
            "inputs": {"question": "Paris?"},
        },
        schema_echo(answer="France"),
        host_tools.call,
    )
    assert report["outputs"]["answer"] == "France"


_TRAINSET = [
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "French capital?", "answer": "Paris"},
    {"question": "Capital city of France?", "answer": "Paris"},
]


def _optimizer_case(name: str, config: dict[str, Any]) -> Callable[[], None]:
    def case() -> None:
        _reset()
        report = caps.compile_program(
            {
                "program": {"signature": "question -> answer"},
                "optimizer": name,
                "trainset": _TRAINSET,
                "valset": _TRAINSET[:2],
                "metric": "exact_match",
                "config": config,
                "inputs": {"question": "Capital of France?"},
            },
            schema_echo(answer="Paris"),
            host_tools.call,
        )
        assert report["outputs"]["answer"] == "Paris" and report["program_state"]

    return case


CASES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("example", _case_example),
    ("signature", _case_signature),
    ("memory-cache", _case_cache),
    ("dummy-predict", _case_dummy_predict),
    ("chain-of-thought", _case_chain_of_thought),
    ("host-engine-predict", _case_host_engine_predict),
    ("host-tool", _case_host_tool),
    ("react-host-tool", _case_react_host_tool),
    ("structured-adapters", _case_structured_adapters),
    ("refine", _case_refine),
    ("best-of-n-host-reward", _case_best_of_n_host_reward),
    ("multi-chain-comparison", _case_multi_chain_comparison),
    ("render", _case_render),
    ("evaluate", _case_evaluate),
    ("compile-round-trip", _case_compile_round_trip),
    ("multihop-pipeline", _case_multihop_pipeline),
    ("program-of-thought", _case_program_of_thought),
    ("code-act-host-tool", _case_code_act_host_tool),
    ("rlm", _case_rlm),
    ("majority-and-async", _case_majority_and_async),
    ("multimodal-parts", _case_multimodal_parts),
    ("knn-few-shot-host-embedder", _case_knn_few_shot_host_embedder),
    (
        "optimizer:bootstrap-random-search",
        _optimizer_case("bootstrap-random-search", {"num_candidate_programs": 1}),
    ),
    ("optimizer:copro", _optimizer_case("copro", {})),
    ("optimizer:mipro-v2", _optimizer_case("mipro-v2", {"max_bootstrapped_demos": 1})),
    ("optimizer:simba", _optimizer_case("simba", {})),
    ("optimizer:gepa", _optimizer_case("gepa", {})),
    (
        "optimizer:infer-rules",
        _optimizer_case("infer-rules", {"num_candidates": 1, "num_rules": 1}),
    ),
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


class DspyBindings(wit.DspyBindings):
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
                "tool_boundary": "chatman:dspy/tools@0.1.0",
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
                default=caps.json_default,
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

    def capabilities(self) -> str:
        return caps.guarded(caps.describe)

    def run(self, request_json: str) -> str:
        return self._invoke(caps.run, request_json)

    def render(self, request_json: str) -> str:
        return self._invoke(caps.render, request_json)

    def evaluate(self, request_json: str) -> str:
        return self._invoke(caps.evaluate, request_json)

    def compile(self, request_json: str) -> str:
        return self._invoke(caps.compile_program, request_json)

    @staticmethod
    def _invoke(capability, request_json: str) -> str:
        def call() -> dict[str, Any]:
            _reset()
            return capability(json.loads(request_json), host_lm.complete, host_tools.call)

        return caps.guarded(call)


def _freeze_sources(*prefixes: str) -> None:
    """Keep source text reachable after the snapshot, where no files exist.

    dspy.Refine calls inspect.getsource() on the wrapped module and reward
    function at construction. Entries with mtime=None are never invalidated
    by linecache.checkcache(), so source captured now survives at runtime.
    """
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            lines = linecache.getlines(filename)
            if lines:
                linecache.cache[filename] = (sum(map(len, lines)), None, lines, filename)


_freeze_sources("dspy.predict", "dspy.primitives", "dspy_capabilities", "dspy_runtime")
