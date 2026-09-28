"""Conformance suite: does this host + component honour the dspy-wasm contract?

Run it against any host, not just host.py. A host supplies one callable::

    invoke(export: str, *json_args: str) -> dict     # the parsed JSON report

wired to a component whose imports behave as follows (the reference host does
this with ``--response``): ``chatman:dspy/lm.complete`` answers every request
with the text ``PREDICT_ANSWER``, and ``chatman:dspy/tools.call`` serves the
builtin ``calculator`` and ``embed`` tools with the limits in ``limits.py``.

    from conformance import run
    report = run(invoke)             # {"state": "ALIVE" | "FAILED", "cases": [...]}

The checks are data, not code: ``consumer/conformance.json`` holds every vector
(export, arguments, expectation), generated from ``limits.py`` by ``python
conformance.py --write``. A host in any language runs the same file: call
``export`` with ``args`` (JSON strings), parse the reply, and apply ``expect``
(the keys are defined by ``check_vector`` below).

Every check states the contract clause it pins. A host that returns ALIVE from
this suite refuses what the contract says to refuse; it does not prove the
host's provider, tools or deployment are authorised or safe.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import limits

ROOT = Path(__file__).resolve().parent

Invoke = Callable[..., dict[str, Any]]

PREDICT_ANSWER = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
_QA = {"signature": "question -> answer", "inputs": {"question": "capital of France?"}}


def _run(invoke: Invoke, **request: Any) -> dict[str, Any]:
    return invoke("run", json.dumps({**_QA, **request}))


def _refused(report: dict[str, Any], fragment: str) -> None:
    assert report.get("state") == "FAILED", report
    assert fragment in report.get("message", ""), report


def _pipeline(steps: list[dict[str, Any]], **request: Any) -> dict[str, Any]:
    return {"module": "pipeline", "steps": steps, "inputs": {}, **request}


# --------------------------------------------------------------------- vectors


def _v(group: str, name: str, export: str, args: list[str], **expect: Any) -> dict[str, Any]:
    return {"group": group, "name": name, "export": export, "args": args, "expect": expect}


def _run_args(**request: Any) -> list[str]:
    return [json.dumps({**_QA, **request})]


def _pipeline_args(steps: list[dict[str, Any]], **request: Any) -> list[str]:
    return [json.dumps({"module": "pipeline", "steps": steps, "inputs": {}, **request})]


def _int(name: str) -> int:
    return int(limits.value(name))


def vectors() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        _v(
            "capabilities_publish_the_contract_limits",
            "capabilities",
            "capabilities",
            [],
            state="ALIVE",
            includes={
                "pipeline_limits": {
                    "max_repeat": _int("max_repeat"),
                    "max_total_steps": _int("max_total_steps"),
                },
                "module_limits": {"max_fanout": _int("max_fanout"), "max_iters": _int("max_iters")},
            },
        )
    )
    out.append(
        _v(
            "predict_round_trips_through_the_host_lm",
            "predict",
            "predict",
            ["question -> answer", json.dumps(_QA["inputs"])],
            equals={"state": "ALIVE", "outputs": {"answer": "Paris"}},
        )
    )
    out.append(
        _v(
            "predict_refuses_non_object_inputs",
            "predict with an array",
            "predict",
            ["question -> answer", "[1]"],
            state_not="ALIVE",
            message_contains="object",
        )
    )
    for export in ("run", "render", "evaluate", "compile"):
        for payload in ("not json", "[]", '{"module": "no-such-module"}'):
            out.append(
                _v(
                    "request_exports_never_raise_into_the_host",
                    f"{export} {payload}",
                    export,
                    [payload],
                    state="FAILED",
                )
            )
    group = "fanout_beyond_the_ceiling_is_refused"
    for bad in (_int("max_fanout") + 1, 0, -1, True, 2.0, None, "3"):
        out.append(
            _v(
                group,
                f"majority n={bad!r}",
                "run",
                _run_args(module="majority", n=bad),
                state="FAILED",
                message_contains="integer in [",
            )
        )
    out.append(
        _v(group, "majority n=1 is served", "run", _run_args(module="majority", n=1), state="ALIVE")
    )
    group = "iteration_counts_beyond_the_ceiling_are_refused"
    for module in ("react", "code-act", "program-of-thought"):
        out.append(
            _v(
                group,
                f"{module} max_iters over the ceiling",
                "run",
                _run_args(module=module, max_iters=_int("max_iters") + 1),
                state="FAILED",
                message_contains="integer in [",
            )
        )
    out.append(
        _v(
            group,
            "rlm max_llm_calls=1e9",
            "run",
            _run_args(module="rlm", max_llm_calls=10**9),
            state="FAILED",
            message_contains="integer in [",
        )
    )
    for bad in (_int("max_repeat") + 1, -1, True):
        out.append(
            _v(
                "repeat_beyond_the_ceiling_is_refused",
                f"repeat={bad!r}",
                "run",
                _pipeline_args([{"repeat": bad, "steps": []}]),
                state="FAILED",
                message_contains="'repeat' must be",
            )
        )
    nested: list[dict[str, Any]] = []
    for _ in range(3):
        nested = [{"repeat": _int("max_repeat"), "steps": nested}]
    out.append(
        _v(
            "nested_repeat_product_is_refused_before_any_step_runs",
            "three nested max repeats",
            "run",
            _pipeline_args(nested),
            state="FAILED",
            message_contains="MAX_TOTAL_STEPS=",
        )
    )
    out.append(
        _v(
            "unbound_program_state_is_refused",
            "program_state without a subject",
            "run",
            _run_args(program_state={"demos": []}),
            state="FAILED",
            message_contains="unbound program_state",
        )
    )
    out.append(_v("batches", "empty batch is alive", "run", _run_args(inputs=[]), state="ALIVE"))
    out.append(
        _v("batches", "scalar batch item fails", "run", _run_args(inputs=[1]), state="FAILED")
    )
    group = "batch_and_dataset_lengths_beyond_the_ceiling_are_refused"
    over_batch = [{"question": "q"}] * (_int("max_batch_items") + 1)
    out.append(
        _v(
            group,
            "run batch over max_batch_items",
            "run",
            _run_args(inputs=over_batch),
            state="FAILED",
            message_contains="max_batch_items",
        )
    )
    over_rows = [{"question": "q", "answer": "Paris"}] * (_int("max_dataset_items") + 1)
    program = {"signature": "question -> answer"}
    for name, export, request in (
        ("evaluate devset", "evaluate", {"devset": over_rows}),
        ("compile trainset", "compile", {"trainset": over_rows}),
        ("compile valset", "compile", {"trainset": over_rows[:2], "valset": over_rows}),
    ):
        out.append(
            _v(
                group,
                f"{name} over max_dataset_items",
                export,
                [json.dumps({"program": program, "metric": "exact_match", **request})],
                state="FAILED",
                message_contains="max_dataset_items",
            )
        )
    group = "optimizer_counts_beyond_the_ceiling_are_refused"
    count_cases = [
        ("bootstrap-few-shot", "max_rounds", 10**9),
        ("bootstrap-random-search", "num_candidate_programs", 10**6),
        ("bootstrap-optuna", "num_candidate_programs", 10**6),
        ("copro", "breadth", _int("max_optimizer_count") + 1),
        ("simba", "max_steps", 10**9),
        ("gepa", "max_metric_calls", 10**9),
        ("labeled-few-shot", "k", True),
        ("labeled-few-shot", "k", 2.0),
        ("labeled-few-shot", "k", None),
        ("labeled-few-shot", "k", -1),
        ("labeled-few-shot", "k", "2"),
    ]
    for optimizer, key, bad in count_cases:
        out.append(
            _v(
                group,
                f"{optimizer} config {key}={bad!r}",
                "compile",
                [
                    json.dumps(
                        {
                            "program": program,
                            "optimizer": optimizer,
                            "metric": "exact_match",
                            "trainset": [{"question": "q", "answer": "Paris"}],
                            "config": {key: bad},
                        }
                    )
                ],
                state="FAILED",
                message_contains="integer in [",
            )
        )
    out.append(
        _v(
            group,
            "mipro-v2 compile_config num_trials=1e9",
            "compile",
            [
                json.dumps(
                    {
                        "program": program,
                        "optimizer": "mipro-v2",
                        "metric": "exact_match",
                        "trainset": [{"question": "q", "answer": "Paris"}],
                        "compile_config": {"num_trials": 10**9},
                    }
                )
            ],
            state="FAILED",
            message_contains="integer in [",
        )
    )
    dims, texts, values = (
        _int("max_embed_dimensions"),
        _int("max_embed_texts"),
        _int("max_embed_values"),
    )
    group = "tool_bounds_are_refused_by_the_host"
    cases = [
        ("embed dimensions over", "embed", {"texts": ["a"], "dimensions": dims + 1}),
        ("embed texts over", "embed", {"texts": ["a"] * (texts + 1), "dimensions": 1}),
        (
            "embed both axes at their ceilings",
            "embed",
            {"texts": ["a"] * texts, "dimensions": dims},
        ),
        (
            "embed product over",
            "embed",
            {"texts": ["a"] * (values // dims + 1), "dimensions": dims},
        ),
        ("calculator power tower", "calculator", {"expression": "9**9**9"}),
        (
            "calculator result over max_int_bits",
            "calculator",
            {"expression": f"2**{_int('max_int_bits') + 1}"},
        ),
    ]
    for name, tool, args in cases:
        out.append(
            _v(group, name, "run", _pipeline_args([{"tool": tool, "args": args}]), state="FAILED")
        )
    out.append(
        _v(
            "a_tool_at_its_ceiling_is_served",
            "embed at max dimensions",
            "run",
            _pipeline_args(
                [{"tool": "embed", "args": {"texts": ["a b"], "dimensions": dims}, "output": "e"}],
                outputs=["e"],
            ),
            state="ALIVE",
        )
    )
    return out


def check_vector(vector: dict[str, Any], report: dict[str, Any]) -> None:
    """Apply ``vector["expect"]`` to the parsed reply; raise AssertionError on mismatch.

    Keys: ``state`` (== reply.state), ``state_not`` (!=), ``message_contains``
    (substring of reply.message), ``equals`` (reply == value), ``includes``
    (each key's value == reply[key]).
    """
    expect = vector["expect"]
    if "state" in expect:
        assert report.get("state") == expect["state"], report
    if "state_not" in expect:
        assert report.get("state") != expect["state_not"], report
    if "message_contains" in expect:
        assert expect["message_contains"] in report.get("message", ""), report
    if "equals" in expect:
        assert report == expect["equals"], report
    for key, value in expect.get("includes", {}).items():
        assert report.get(key) == value, (key, report.get(key))


def write() -> None:
    (ROOT / "consumer" / "conformance.json").write_text(json.dumps(vectors(), indent=2) + "\n")


# ------------------------------------------------------------------- the runner


def run(invoke: Invoke, *, include_self_test: bool = False) -> dict[str, Any]:
    """Run every vector; never raises. ``include_self_test`` adds the in-component court."""
    cases: list[dict[str, Any]] = []
    todo: list[tuple[str, Callable[[], None]]] = []
    for vector in vectors():

        def attempt(v: dict[str, Any] = vector) -> None:
            check_vector(v, invoke(v["export"], *v["args"]))

        todo.append((f"{vector['group']}: {vector['name']}", attempt))
    if include_self_test:

        def self_test() -> None:
            report = invoke("run-self-tests")
            assert report["state"] == "ALIVE" and report["failed"] == 0, report

        todo.append(("in_component_self_test_is_alive", self_test))
    for name, attempt in todo:
        try:
            attempt()
            cases.append({"name": name, "state": "ALIVE"})
        except Exception as exc:  # noqa: BLE001 - a failing check is a result, not a crash
            cases.append(
                {
                    "name": name,
                    "state": "FAILED",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "traceback": traceback.format_exc(limit=3),
                }
            )
    failed = sum(case["state"] != "ALIVE" for case in cases)
    return {
        "state": "ALIVE" if failed == 0 else "FAILED",
        "passed": len(cases) - failed,
        "failed": failed,
        "cases": cases,
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write()
    else:
        raise SystemExit("usage: python conformance.py --write")
