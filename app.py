"""Operational DSPy WebAssembly component.

DSPy and every dependency are upstream code; native extensions are the real
libraries compiled to wasm32-wasip2 (see wasi/). Component policy lives at
three boundaries:
1. runtime policy for what a component lacks (dspy_runtime.py),
2. LM actuation supplied by the host through WIT (`chatman:dspy/lm`),
3. tool actuation supplied by the host through WIT (`chatman:dspy/tools`).
"""

from __future__ import annotations

import os

# A component has no .env files and fetches nothing at import: litellm skips
# dotenv in PRODUCTION mode and uses its bundled model cost map.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# Import numpy completely before dspy: dspy.utils.lazy_import parks a lazy
# proxy in sys.modules["numpy"], and numpy's own initialisation re-enters it.
import numpy
import numpy.fft
import numpy.linalg
import numpy.ma
import numpy.random

# isort: split

import copy
import json
import linecache
import platform
import sys
import traceback
from collections.abc import Callable
from typing import Any

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

# isort: split

# componentize-py snapshots the interpreter after module import, and the host
# grants no filesystem. Anything DSPy would import lazily on a call path must
# therefore be imported here, at build time.
import hashlib
import ssl

import cachetools
import cachetools.keys  # noqa: F401
import dspy.clients.call_result
import dspy.clients.costs
import dspy.clients.engines.dummy_engine
import dspy.clients.engines.streaming
import dspy.clients.execution
import dspy.utils.hasher
import gepa.lm  # noqa: F401
import litellm
import litellm.rust_bridge._native
import optuna
import optuna.samplers


def _materialize_litellm() -> None:
    """Resolve litellm's lazily imported providers/utilities at build time."""
    from litellm import _lazy_imports

    for name in list(_lazy_imports._get_lazy_import_registry()):
        try:
            getattr(litellm, name)
        except Exception:  # noqa: BLE001, S110 - an optional provider's extra dependency
            pass


_materialize_litellm()

try:  # tqdm's write lock probes multiprocessing; absent under WASI is fine.
    import multiprocessing.synchronize  # noqa: F401
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



def _case_boundary_refusals() -> None:
    """The request-boundary guards hold inside the component too."""

    def refusal(request: dict[str, Any]) -> str:
        _reset()
        report = json.loads(
            caps.guarded(lambda: caps.run(request, _scripted("unused"), host_tools.call))
        )
        assert report["state"] == "FAILED", report
        return report["message"]

    nested: list[dict[str, Any]] = [{"set": {"x": 1}}]
    for _ in range(3):
        nested = [{"repeat": caps.MAX_REPEAT, "steps": nested}]
    message = refusal({"module": "pipeline", "steps": nested, "inputs": {}})
    assert f"exceeded {caps.MAX_PIPELINE_STEPS} executed steps" in message
    for falsy in ([], 0, False, ""):
        message = refusal({"program_state": falsy, "inputs": {"question": "q"}})
        assert "program_state must be a JSON object" in message
    unbound = {"signature": {"fields": [{}, {}]}}
    message = refusal({"program_state": unbound, "inputs": {"question": "q"}})
    assert "unbound program_state" in message


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


def _case_native_dependencies() -> None:
    import orjson
    import regex
    import rpds
    import tiktoken  # noqa: F401
    import tokenizers
    import yaml

    for module in (orjson, regex, rpds, yaml, tokenizers, numpy):
        extensions = [m for m in sys.modules if m.startswith(module.__name__)]
        assert any(
            str(getattr(sys.modules[m], "__file__", "")).endswith(".so") for m in extensions
        ), f"{module.__name__} is not running its compiled extension"
    assert orjson.dumps({"b": 1, "a": 2}, option=orjson.OPT_SORT_KEYS) == b'{"a":2,"b":1}'
    assert regex.findall(r"\p{Greek}+", "alpha αβγ") == ["αβγ"]
    assert yaml.load("a: [1, 2]", Loader=yaml.CSafeLoader) == {"a": [1, 2]}


def _case_numpy() -> None:
    matrix = numpy.array([[4.0, 1.0], [2.0, 3.0]])
    assert sorted(numpy.linalg.eigvals(matrix).real.round(6).tolist()) == [2.0, 5.0]
    assert numpy.abs(numpy.fft.fft([1, 0, 0, 0])).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert float(numpy.percentile([1, 2, 3, 4], 10)) == 1.3


def _case_optuna_tpe() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(lambda trial: -((trial.suggest_float("x", -5, 5) - 2) ** 2), n_trials=20)
    assert abs(study.best_params["x"] - 2) < 1.0


def _case_tls_and_hashlib() -> None:
    import _hashlib

    context = ssl.create_default_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert ssl.OPENSSL_VERSION.startswith("OpenSSL 3.")
    assert hashlib.new("sha3_256", b"abc").hexdigest().startswith("3a985da7")
    assert _hashlib.__file__.endswith(".so")


def _case_litellm() -> None:
    from litellm.rust_bridge import _native

    assert _native.__file__.endswith(".so")
    assert litellm.model_cost["gpt-4o"]["max_input_tokens"] > 0
    assert litellm.get_llm_provider("claude-sonnet-4-5")[1] == "anthropic"


def _case_embeddings_retriever() -> None:
    report = _run(
        {
            "module": "retrieve",
            "retriever": {
                "corpus": ["Paris is the capital of France.", "Lima is the capital of Peru."],
                "embedder": "embed",
                "k": 1,
            },
            "k": 1,
            "inputs": {"query": "capital of Peru"},
        },
        "unused",
    )
    assert report["outputs"]["passages"] == ["Lima is the capital of Peru."]


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
    ("boundary-refusals", _case_boundary_refusals),
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
    (
        "optimizer:bootstrap-optuna",
        _optimizer_case("bootstrap-optuna", {"num_candidate_programs": 2}),
    ),
    ("native-dependencies", _case_native_dependencies),
    ("numpy", _case_numpy),
    ("optuna-tpe", _case_optuna_tpe),
    ("tls-and-hashlib", _case_tls_and_hashlib),
    ("litellm", _case_litellm),
    ("embeddings-retriever", _case_embeddings_retriever),
)


def _self_test_report() -> dict[str, Any]:
    results = []
    for name, case in CASES:
        try:
            case()
            results.append({"name": name, "state": "ALIVE"})
        except Exception as exc:  # noqa: BLE001 - reported as BUILD_BROKEN, never raised into the host
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
                "numpy": numpy.__version__,
                "openssl": ssl.OPENSSL_VERSION,
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
        except Exception as exc:  # noqa: BLE001 - reported as BUILD_BROKEN, never raised into the host
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


def _materialize_lazy_modules() -> None:
    """Import everything dspy's require() deferred, while a filesystem exists.

    inspect.getmodule() (via Module.__getattribute__ -> inspect.stack()) walks
    sys.modules and touches every entry, which would otherwise trigger those
    imports at runtime, where no source files exist.
    """
    from dspy.utils.lazy_import import _LazyModule

    for module in list(sys.modules.values()):
        if isinstance(module, _LazyModule):
            module._load()


_materialize_lazy_modules()
_freeze_sources("dspy.predict", "dspy.primitives", "dspy_capabilities", "dspy_runtime")
