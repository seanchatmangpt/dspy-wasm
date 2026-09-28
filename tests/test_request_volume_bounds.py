"""Request volume is bounded in the component, not only by the host deadline/meters.

Falsifiers: before these guards a 1e9 optimizer count or a huge batch/dataset
ran until the host killed it. The hang reproductions run in a subprocess with a
timeout so the old behaviour is a clean failure, not a hung suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps
import limits
from dspy_doubles import chat, scripted

ROOT = Path(__file__).resolve().parents[1]
MAX_BATCH = int(limits.value("max_batch_items"))
MAX_DATASET = int(limits.value("max_dataset_items"))
MAX_COUNT = int(limits.value("max_optimizer_count"))

REPRO = """
import json, sys
import dspy_capabilities as caps
from dspy_doubles import chat, scripted
from host import ToolProvider
request = json.loads(sys.argv[1])
tools = ToolProvider()
report = json.loads(caps.guarded(lambda: caps.compile_program(
    request, scripted(chat(answer="Paris")), lambda n, a: tools.call(None, n, a))))
print(json.dumps(report))
"""

TRAIN = [{"question": f"q{i}", "answer": "Paris"} for i in range(3)]


def refused_isolated(**request) -> dict:
    payload = {
        "program": {"signature": "question -> answer"},
        "trainset": TRAIN,
        "metric": "exact_match",
        **request,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", REPRO, json.dumps(payload)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{request} ran past 30 s: no request-level bound")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    ("optimizer", "config", "compile_config"),
    [
        ("bootstrap-few-shot", {"max_rounds": 10**9}, None),
        ("bootstrap-random-search", {"num_candidate_programs": 10**6}, None),
        ("bootstrap-optuna", {"num_candidate_programs": 10**6}, None),
        ("copro", {"breadth": 10**9}, None),
        ("copro", {"depth": 10**9}, None),
        ("simba", {"max_steps": 10**9}, None),
        ("gepa", {"max_metric_calls": 10**9}, None),
        ("mipro-v2", {"num_candidates": 10**9}, None),
        ("mipro-v2", {}, {"num_trials": 10**9}),
    ],
)
def test_huge_optimizer_counts_are_refused_fast(optimizer, config, compile_config) -> None:
    extra = {"compile_config": compile_config} if compile_config else {}
    report = refused_isolated(optimizer=optimizer, config=config, **extra)
    assert report["state"] == "FAILED", report
    assert "integer in [" in report["message"], report


def _ok(**kwargs) -> None:
    caps._bound_counts("bootstrap-few-shot", kwargs, "config")


@pytest.mark.parametrize("bad", [True, False, 2.0, None, -1, "3", MAX_COUNT + 1, 10**9])
def test_count_keys_reject_non_integers_negatives_and_huge(bad) -> None:
    with pytest.raises(caps.RequestError, match=r"integer in \["):
        _ok(max_rounds=bad)


def test_count_keys_accept_the_ceiling_and_zero() -> None:
    _ok(max_rounds=MAX_COUNT, max_bootstrapped_demos=0)


def test_unknown_keys_pass_through_unbounded() -> None:
    _ok(metric_threshold=10**12, some_future_knob="x", seed=10**12)


def test_every_optimizer_has_an_allow_list_that_rejects_each_key() -> None:
    assert set(caps.OPTIMIZER_COUNT_KEYS) >= set(caps.OPTIMIZERS) - {"ensemble"}
    for name, keys in caps.OPTIMIZER_COUNT_KEYS.items():
        for key in keys:
            for where in ("config", "compile_config"):
                with pytest.raises(caps.RequestError, match=r"integer in \["):
                    caps._bound_counts(name, {key: MAX_COUNT + 1}, where)


def test_documented_hang_keys_are_on_the_allow_lists() -> None:
    keys = caps.OPTIMIZER_COUNT_KEYS
    assert "max_rounds" in keys["bootstrap-few-shot"]
    assert "num_candidate_programs" in keys["bootstrap-random-search"]
    assert "num_candidate_programs" in keys["bootstrap-optuna"]
    assert {"breadth", "depth"} <= keys["copro"]
    assert {"num_candidates", "num_trials"} <= keys["mipro-v2"]
    assert {"bsize", "num_candidates", "max_steps"} <= keys["simba"]
    assert "max_metric_calls" in keys["gepa"]
    assert "k" in keys["labeled-few-shot"] and "k" in keys["knn-few-shot"]


def test_labeled_few_shot_k_bad_values_refused_end_to_end() -> None:
    for bad in (True, 1.5, None, -1, "2", 10**9):
        report = json.loads(
            caps.guarded(
                lambda bad=bad: caps.compile_program(
                    {
                        "program": {"signature": "question -> answer"},
                        "trainset": TRAIN,
                        "config": {"k": bad},
                    },
                    scripted(chat(answer="x")),
                    lambda n, a: "",
                )
            )
        )
        assert report["state"] == "FAILED" and "integer in [" in report["message"], report


def test_in_limit_config_still_compiles() -> None:
    report = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "bootstrap-few-shot",
            "trainset": TRAIN,
            "metric": "exact_match",
            "config": {"max_rounds": 1, "max_bootstrapped_demos": 1, "max_labeled_demos": 0},
        },
        scripted(chat(answer="Paris")),
        lambda n, a: "",
    )
    assert report["state"] == "ALIVE"


# ---------------------------------------------------- batch and dataset lengths


def _guard(fn, request) -> dict:
    return json.loads(caps.guarded(lambda: fn(request, scripted(chat(answer="Paris")), _no_tools)))


def _no_tools(name: str, args: str) -> str:
    raise AssertionError("no tool should run")


def test_batch_over_the_ceiling_is_refused_before_any_lm_call() -> None:
    calls = []

    def lm(request_json: str) -> str:
        calls.append(request_json)
        return scripted(chat(answer="Paris"))(request_json)

    request = {
        "signature": "question -> answer",
        "inputs": [{"question": "q"}] * (MAX_BATCH + 1),
    }
    report = json.loads(caps.guarded(lambda: caps.run(request, lm, _no_tools)))
    assert report["state"] == "FAILED" and str(MAX_BATCH) in report["message"], report
    assert calls == []


def test_batch_at_the_ceiling_is_served() -> None:
    request = {"signature": "question -> answer", "inputs": [{"question": "q"}] * MAX_BATCH}
    report = _guard(caps.run, request)
    assert report["state"] == "ALIVE" and len(report["outputs"]) == MAX_BATCH


def test_evaluate_devset_over_the_ceiling_is_refused() -> None:
    rows = [{"question": "q", "answer": "Paris"}] * (MAX_DATASET + 1)
    report = _guard(
        caps.evaluate,
        {"program": {"signature": "question -> answer"}, "devset": rows, "metric": "exact_match"},
    )
    assert report["state"] == "FAILED" and str(MAX_DATASET) in report["message"], report


@pytest.mark.parametrize("field", ["trainset", "valset"])
def test_compile_datasets_over_the_ceiling_are_refused(field: str) -> None:
    rows = [{"question": "q", "answer": "Paris"}] * (MAX_DATASET + 1)
    request = {
        "program": {"signature": "question -> answer"},
        "optimizer": "labeled-few-shot",
        "trainset": TRAIN,
        "metric": "exact_match",
        field: rows,
    }
    report = _guard(caps.compile_program, request)
    assert report["state"] == "FAILED" and str(MAX_DATASET) in report["message"], report


def test_evaluate_max_errors_is_bounded() -> None:
    report = _guard(
        caps.evaluate,
        {
            "program": {"signature": "question -> answer"},
            "devset": TRAIN,
            "metric": "exact_match",
            "max_errors": 10**9,
        },
    )
    assert report["state"] == "FAILED" and "integer in [" in report["message"], report
