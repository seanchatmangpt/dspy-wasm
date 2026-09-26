"""Deterministic timing benchmark for the dspy-wasm capability boundary.

Measures the exact code the component runs (dspy_capabilities over the JSON
lm/tool boundary, with the real host ToolProvider and deterministic LM
doubles), the host tools, and, for each component present under ``dist/``,
real Wasmtime calls: ``bootstrap.wasm`` (instantiate + export call, no DSPy)
and ``dspy.wasm`` (instantiate, and DSPy ``run``/``compile`` executing
inside the component over the host lm/tools imports).

    python bench/bench_capabilities.py                 # print the report
    python bench/bench_capabilities.py --write bench/receipt.json

``BOUNDS_MS`` are the regression ceilings (median, milliseconds) enforced by
tests/test_bench_bounds.py. Each is about 6-8x the committed median: room for
a slower CI runner, but a 10x regression on the recording machine trips it.
``tests/test_bench_bounds.py`` also refuses a ceiling more than
``MAX_HEADROOM`` times its committed median, so bounds cannot drift loose.

Component instantiation is reported under ``setup_ms`` and is not bounded:
it is dominated by wasmtime's on-disk compilation cache (a hit loads
bootstrap.wasm in ~0.3 s; a miss recompiles it in ~30 s, and dspy.wasm in
~60 s), which is host state rather than a property of the component.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOUNDS_MS: dict[str, float] = {
    "run:predict": 12.0,
    "run:chain-of-thought": 15.0,
    "run:pipeline-multihop": 50.0,
    "evaluate:2-rows": 18.0,
    "compile:bootstrap-few-shot": 20.0,
    "admit:stale-program-state-refusal": 4.0,
    "host:tool-envelope": 0.05,
    "host:calculator-dos-refusal": 0.1,
    "host:calculator-nonfinite-refusal": 0.06,
    # DSPy executing inside dspy.wasm (only measured when it has been built).
    "wasm:dspy-run-predict": 30.0,
    "wasm:dspy-compile-labeled-few-shot": 20.0,
}
# A ceiling may sit at most this many times above its committed median.
MAX_HEADROOM = 10.0


def _cases() -> dict[str, Callable[[], Any]]:
    import dspy_capabilities as caps
    import host
    from dspy_doubles import chat, scripted

    tools = host.ToolProvider()

    def tool_call(name: str, args_json: str) -> str:
        return tools.call(None, name, args_json)

    trainset = [
        {"question": "France?", "answer": "Paris"},
        {"question": "Peru?", "answer": "Lima"},
    ]
    stale_state = caps.compile_program(
        {"program": {"signature": "question -> answer"}, "trainset": trainset},
        scripted("unused"),
        tool_call,
    )["program_state"]

    def predict() -> None:
        report = caps.run(
            {"inputs": {"question": "Capital of France?"}},
            scripted(chat(answer="Paris")),
            tool_call,
        )
        assert report["outputs"] == {"answer": "Paris"}

    def chain_of_thought() -> None:
        report = caps.run(
            {"module": "chain-of-thought", "inputs": {"question": "2 + 2?"}},
            scripted(chat(reasoning="sum", answer="4")),
            tool_call,
        )
        assert report["outputs"]["answer"] == "4"

    def multihop() -> None:
        hop = {
            "name": "generate_query",
            "program": {
                "module": "chain-of-thought",
                "signature": "context: list[str], question -> query",
            },
        }
        report = caps.run(
            {
                "module": "pipeline",
                "retriever": "search",
                "steps": [
                    {"set": {"context": []}},
                    {
                        "repeat": 2,
                        "steps": [
                            hop,
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
                "outputs": ["answer"],
                "inputs": {"question": "Where was the author of Hamlet born?"},
            },
            scripted(
                chat(reasoning="a", query="Who wrote Hamlet?"),
                chat(reasoning="b", query="Where was Shakespeare born?"),
                chat(answer="Stratford-upon-Avon"),
            ),
            tool_call,
        )
        assert report["outputs"]["answer"] == "Stratford-upon-Avon"

    def evaluate() -> None:
        report = caps.evaluate(
            {
                "program": {"signature": "question -> answer"},
                "devset": trainset,
                "metric": "exact_match",
            },
            scripted(chat(answer="Paris"), chat(answer="Lima")),
            tool_call,
        )
        assert report["score"] == 100.0

    def compile_bootstrap() -> None:
        report = caps.compile_program(
            {
                "program": {"signature": "question -> answer"},
                "optimizer": "bootstrap-few-shot",
                "trainset": trainset,
                "metric": "exact_match",
                "config": {"max_bootstrapped_demos": 2, "max_labeled_demos": 0},
            },
            scripted(chat(answer="Paris"), chat(answer="wrong")),
            tool_call,
        )
        assert report["demos"] == {"self": 1}

    def stale_refusal() -> None:
        report = json.loads(
            caps.guarded(
                lambda: caps.run(
                    {
                        "signature": "context, question -> summary",
                        "program_state": stale_state,
                        "inputs": {"context": "c", "question": "q"},
                    },
                    scripted("unused"),
                    tool_call,
                )
            )
        )
        assert report["state"] == "FAILED" and "stale" in report["message"]

    def envelope() -> None:
        assert tools.call(None, "calculator", '{"expression": "6 * 7"}') == '{"result": 42}'

    def dos_refusal() -> None:
        envelope = json.loads(
            tools.call(None, "calculator", '{"expression": "(((10**64)**64)**64)**64"}')
        )
        assert "too large" in envelope["error"]

    def nonfinite_refusal() -> None:
        envelope = json.loads(tools.call(None, "calculator", '{"expression": "1e308 * 1e308"}'))
        assert "not finite" in envelope["error"]

    return {
        "run:predict": predict,
        "run:chain-of-thought": chain_of_thought,
        "run:pipeline-multihop": multihop,
        "evaluate:2-rows": evaluate,
        "compile:bootstrap-few-shot": compile_bootstrap,
        "admit:stale-program-state-refusal": stale_refusal,
        "host:tool-envelope": envelope,
        "host:calculator-dos-refusal": dos_refusal,
        "host:calculator-nonfinite-refusal": nonfinite_refusal,
    }


def _bootstrap_setup(component: Path) -> Callable[[], Any] | None:
    if not component.exists():
        return None
    import host

    def instantiate_and_call() -> None:
        provider = host.CompletionProvider(
            static_response=host.DEFAULT_RESPONSE,
            base_url=None,
            api_key=None,
            upstream_model=None,
        )
        store, instance = host.instantiate(component, provider)
        info = json.loads(instance.get_func(store, "runtime-info")(store))
        assert info["state"] == "BOOTSTRAP" and info["platform"] == "wasi"

    return instantiate_and_call


PARIS = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"


def _dspy_wasm_cases(
    component: Path,
) -> tuple[Callable[[], Any] | None, dict[str, Callable[[], Any]]]:
    """DSPy executing inside dspy.wasm; the host only answers lm/tools calls.

    Returns the one-time instantiation (setup) and the per-call cases.
    """
    if not component.exists():
        return None, {}
    import host

    def provider() -> host.CompletionProvider:
        return host.CompletionProvider(
            static_response=None,
            base_url=None,
            api_key=None,
            upstream_model=None,
            scripted_responses=[PARIS],
        )

    # One instance serves every case, created by the setup step; it is never
    # dropped while measuring, because tearing down a component this size
    # costs minutes on macOS (wasmtime deregisters its unwind tables one frame
    # at a time), which would swamp whichever case the collector ran in.
    shared: list[Any] = []

    def instantiate() -> None:
        assert not shared, "dspy.wasm is instantiated once per benchmark run"
        shared.extend(host.instantiate(component, provider()))
        info = host.call_json(*shared, "runtime-info")
        assert info["platform"] == "wasi" and info["state"] == "DSPY_IMPORTED", info

    def call(export: str, request: str) -> dict[str, Any]:
        store, instance = shared
        return host.call_json(store, instance, export, request)

    run_request = json.dumps({"inputs": {"question": "Capital of France?"}})
    compile_request = json.dumps(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "labeled-few-shot",
            "trainset": [{"question": "France?", "answer": "Paris"}],
        }
    )

    def run_predict() -> None:
        report = call("run", run_request)
        assert report["state"] == "ALIVE" and report["outputs"] == {"answer": "Paris"}, report

    def compile_labeled() -> None:
        report = call("compile", compile_request)
        assert report["state"] == "ALIVE" and report["program_state"]["__subject__"], report

    return instantiate, {
        "wasm:dspy-run-predict": run_predict,
        "wasm:dspy-compile-labeled-few-shot": compile_labeled,
    }


def measure(fn: Callable[[], Any], iterations: int, warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {
        "iterations": iterations,
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 4),
        "min_ms": round(samples[0], 4),
    }


def run_benchmarks(iterations: int = 30, wasm_iterations: int = 5) -> dict[str, Any]:
    import dspy

    results = {name: measure(fn, iterations) for name, fn in _cases().items()}
    # wasmtime-py stores sit in reference cycles; a collection that frees one
    # mid-measurement bills its (slow) teardown to an unrelated case.
    setup: dict[str, Any] = {}
    collecting = gc.isenabled()
    gc.disable()
    try:
        bootstrap = _bootstrap_setup(ROOT / "dist" / "bootstrap.wasm")
        if bootstrap is not None:
            setup["wasm:bootstrap-instantiate-and-call"] = measure(bootstrap, 1, warmup=0)
        instantiate, cases = _dspy_wasm_cases(ROOT / "dist" / "dspy.wasm")
        if instantiate is not None:
            setup["wasm:dspy-instantiate"] = measure(instantiate, 1, warmup=0)
        for name, fn in cases.items():
            results[name] = measure(fn, wasm_iterations, warmup=1)
    finally:
        if collecting:
            gc.enable()
    return {
        "schema": "dspy-wasm/bench-receipt/1",
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        # Host contention while measuring (1/5/15-minute load averages).
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "dspy": getattr(dspy, "__version__", "unknown"),
        "bounds_median_ms": BOUNDS_MS,
        "results": results,
        "setup_ms": setup,
        "within_bounds": all(results[name]["median_ms"] <= BOUNDS_MS[name] for name in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    report = run_benchmarks(args.iterations)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.write:
        args.write.write_text(text + "\n")
    raise SystemExit(0 if report["within_bounds"] else 1)


if __name__ == "__main__":
    main()
