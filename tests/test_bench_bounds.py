"""Regression bounds for bench/bench_capabilities.py.

Runs every benchmark case for real (a few iterations) and fails if any
median exceeds its committed ceiling in ``BOUNDS_MS``; also checks that the
committed receipt names every bounded case and was within bounds when taken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("dspy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import bench_capabilities as bench  # noqa: E402 - bench/ is put on sys.path above


def test_benchmark_medians_stay_within_committed_bounds() -> None:
    report = bench.run_benchmarks(iterations=5, wasm_iterations=2)
    over = {
        name: (result["median_ms"], bench.BOUNDS_MS[name])
        for name, result in report["results"].items()
        if result["median_ms"] > bench.BOUNDS_MS[name]
    }
    assert not over, over
    assert report["within_bounds"]


def test_committed_receipt_covers_every_native_case() -> None:
    receipt = json.loads((ROOT / "bench" / "receipt.json").read_text())
    assert receipt["schema"] == "dspy-wasm/bench-receipt/1"
    assert receipt["within_bounds"] is True
    native = {name for name in bench.BOUNDS_MS if not name.startswith("wasm:")}
    assert native <= set(receipt["results"])
    for name, result in receipt["results"].items():
        assert result["median_ms"] <= receipt["bounds_median_ms"][name], name


def test_bounds_are_tight_against_the_committed_medians() -> None:
    # Before: ceilings sat 15-300x above the medians, so a 20x slowdown passed.
    receipt = json.loads((ROOT / "bench" / "receipt.json").read_text())
    assert receipt["bounds_median_ms"] == bench.BOUNDS_MS
    loose = {
        name: round(bench.BOUNDS_MS[name] / result["median_ms"], 1)
        for name, result in receipt["results"].items()
        if bench.BOUNDS_MS[name] > bench.MAX_HEADROOM * result["median_ms"]
    }
    assert not loose, loose


def test_committed_receipt_times_dspy_inside_the_component() -> None:
    # Before: the only wasm entry was bootstrap.wasm, which contains no DSPy.
    receipt = json.loads((ROOT / "bench" / "receipt.json").read_text())
    in_wasm = {name for name in bench.BOUNDS_MS if name.startswith("wasm:dspy-")}
    assert in_wasm and in_wasm <= set(receipt["results"])
    assert "wasm:dspy-instantiate" in receipt["setup_ms"]
