"""Deterministic timing benchmark for the dspy-wasm capability boundary.

Measures the exact code the component runs (dspy_capabilities over the JSON
lm/tool boundary, with the real host ToolProvider and deterministic LM
doubles), the host tools, and, when ``dist/bootstrap.wasm`` exists, a real
Wasmtime instantiate + export call of the bootstrap component.

    python bench/bench_capabilities.py                 # print the report
    python bench/bench_capabilities.py --write bench/receipt.json

``BOUNDS_MS`` are the regression ceilings (median, milliseconds) enforced by
tests/test_bench_bounds.py; they sit well above the recorded medians so that
only an order-of-magnitude regression (or an unbounded path) trips them.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOUNDS_MS: dict[str, float] = {
    "run:predict": 50.0,
    "run:chain-of-thought": 50.0,
    "run:pipeline-multihop": 150.0,
    "evaluate:2-rows": 100.0,
    "compile:bootstrap-few-shot": 150.0,
    "admit:stale-program-state-refusal": 50.0,
    "host:tool-envelope": 2.0,
    "host:calculator-dos-refusal": 5.0,
    "wasm:bootstrap-instantiate-and-call": 2000.0,
}


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

    return {
        "run:predict": predict,
        "run:chain-of-thought": chain_of_thought,
        "run:pipeline-multihop": multihop,
        "evaluate:2-rows": evaluate,
        "compile:bootstrap-few-shot": compile_bootstrap,
        "admit:stale-program-state-refusal": stale_refusal,
        "host:tool-envelope": envelope,
        "host:calculator-dos-refusal": dos_refusal,
    }


def _wasm_case(component: Path) -> Callable[[], Any] | None:
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
    wasm = _wasm_case(ROOT / "dist" / "bootstrap.wasm")
    if wasm is not None:
        results["wasm:bootstrap-instantiate-and-call"] = measure(wasm, wasm_iterations, warmup=1)
    return {
        "schema": "dspy-wasm/bench-receipt/1",
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        "dspy": getattr(dspy, "__version__", "unknown"),
        "bounds_median_ms": BOUNDS_MS,
        "results": results,
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
