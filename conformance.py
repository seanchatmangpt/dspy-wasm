"""Conformance suite: does this host + component honour the dspy-wasm contract?

Run it against any host, not just host.py. A host supplies one callable::

    invoke(export: str, *json_args: str) -> dict     # the parsed JSON report

wired to a component whose imports behave as follows (the reference host does
this with ``--response``): ``chatman:dspy/lm.complete`` answers every request
with the text ``PREDICT_ANSWER``, and ``chatman:dspy/tools.call`` serves the
builtin ``calculator`` and ``embed`` tools with the limits in ``limits.py``.

    from conformance import run
    report = run(invoke)             # {"state": "ALIVE" | "FAILED", "cases": [...]}

Every check states the contract clause it pins. A host that returns ALIVE from
this suite refuses what the contract says to refuse; it does not prove the
host's provider, tools or deployment are authorised or safe.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from typing import Any

import limits

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


# ------------------------------------------------------------------ the checks


def capabilities_publish_the_contract_limits(invoke: Invoke) -> None:
    report = invoke("capabilities")
    assert report["state"] == "ALIVE", report
    assert report["pipeline_limits"] == {
        "max_repeat": limits.value("max_repeat"),
        "max_total_steps": limits.value("max_total_steps"),
    }
    assert report["module_limits"] == {
        "max_fanout": limits.value("max_fanout"),
        "max_iters": limits.value("max_iters"),
    }


def predict_round_trips_through_the_host_lm(invoke: Invoke) -> None:
    report = invoke("predict", "question -> answer", json.dumps(_QA["inputs"]))
    assert report == {"state": "ALIVE", "outputs": {"answer": "Paris"}}, report


def predict_refuses_non_object_inputs(invoke: Invoke) -> None:
    report = invoke("predict", "question -> answer", "[1]")
    assert report["state"] != "ALIVE" and "object" in report["message"], report


def request_exports_never_raise_into_the_host(invoke: Invoke) -> None:
    for export in ("run", "render", "evaluate", "compile"):
        for payload in ("not json", "[]", '{"module": "no-such-module"}'):
            report = invoke(export, payload)
            assert report["state"] == "FAILED", (export, payload, report)


def fanout_beyond_the_ceiling_is_refused(invoke: Invoke) -> None:
    ceiling = int(limits.value("max_fanout"))
    for bad in (ceiling + 1, 0, -1, True, 2.0, None, "3"):
        _refused(_run(invoke, module="majority", n=bad), "integer in [")
    assert _run(invoke, module="majority", n=1)["state"] == "ALIVE"


def iteration_counts_beyond_the_ceiling_are_refused(invoke: Invoke) -> None:
    for module in ("react", "code-act", "program-of-thought"):
        _refused(
            _run(invoke, module=module, max_iters=int(limits.value("max_iters")) + 1),
            "integer in [",
        )
    _refused(_run(invoke, module="rlm", max_llm_calls=10**9), "integer in [")


def repeat_beyond_the_ceiling_is_refused(invoke: Invoke) -> None:
    ceiling = int(limits.value("max_repeat"))
    for bad in (ceiling + 1, -1, True):
        _refused(
            invoke("run", json.dumps(_pipeline([{"repeat": bad, "steps": []}]))), "'repeat' must be"
        )


def nested_repeat_product_is_refused_before_any_step_runs(invoke: Invoke) -> None:
    steps: list[dict[str, Any]] = []
    for _ in range(3):
        steps = [{"repeat": int(limits.value("max_repeat")), "steps": steps}]
    _refused(invoke("run", json.dumps(_pipeline(steps))), "MAX_TOTAL_STEPS=")


def unbound_program_state_is_refused(invoke: Invoke) -> None:
    _refused(_run(invoke, program_state={"demos": []}), "unbound program_state")


def empty_batch_is_alive_and_scalar_input_is_not_a_batch_item(invoke: Invoke) -> None:
    assert _run(invoke, inputs=[])["state"] == "ALIVE"
    assert _run(invoke, inputs=[1])["state"] == "FAILED"


def tool_bounds_are_refused_by_the_host(invoke: Invoke) -> None:
    dims = int(limits.value("max_embed_dimensions"))
    texts = int(limits.value("max_embed_texts"))
    values = int(limits.value("max_embed_values"))
    cases = [
        ("embed", {"texts": ["a"], "dimensions": dims + 1}),
        ("embed", {"texts": ["a"] * (texts + 1), "dimensions": 1}),
        ("embed", {"texts": ["a"] * texts, "dimensions": dims}),
        ("embed", {"texts": ["a"] * (values // dims + 1), "dimensions": dims}),
        ("calculator", {"expression": "9**9**9"}),
        ("calculator", {"expression": "2**" + str(int(limits.value("max_int_bits")) + 1)}),
    ]
    for tool, args in cases:
        report = invoke("run", json.dumps(_pipeline([{"tool": tool, "args": args}])))
        assert report["state"] == "FAILED", (tool, str(args)[:60], report)


def a_tool_at_its_ceiling_is_served(invoke: Invoke) -> None:
    dims = int(limits.value("max_embed_dimensions"))
    steps = [{"tool": "embed", "args": {"texts": ["a b"], "dimensions": dims}, "output": "e"}]
    assert invoke("run", json.dumps(_pipeline(steps, outputs=["e"])))["state"] == "ALIVE"


CHECKS: tuple[Callable[[Invoke], None], ...] = (
    capabilities_publish_the_contract_limits,
    predict_round_trips_through_the_host_lm,
    predict_refuses_non_object_inputs,
    request_exports_never_raise_into_the_host,
    fanout_beyond_the_ceiling_is_refused,
    iteration_counts_beyond_the_ceiling_are_refused,
    repeat_beyond_the_ceiling_is_refused,
    nested_repeat_product_is_refused_before_any_step_runs,
    unbound_program_state_is_refused,
    empty_batch_is_alive_and_scalar_input_is_not_a_batch_item,
    tool_bounds_are_refused_by_the_host,
    a_tool_at_its_ceiling_is_served,
)


def run(invoke: Invoke, *, include_self_test: bool = False) -> dict[str, Any]:
    """Run every check; never raises. ``include_self_test`` adds the in-component court."""
    cases: list[dict[str, Any]] = []
    checks: list[tuple[str, Callable[[Invoke], None]]] = [(c.__name__, c) for c in CHECKS]
    if include_self_test:

        def self_test(inv: Invoke) -> None:
            report = inv("run-self-tests")
            assert report["state"] == "ALIVE" and report["failed"] == 0, report

        checks.append(("in_component_self_test_is_alive", self_test))
    for name, check in checks:
        try:
            check(invoke)
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
