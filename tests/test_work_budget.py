"""Whole-pipeline total-work budget (court verdict on 37aa5c63, probe P1).

A per-level ``repeat`` bound is defeated by nesting: three nested
``repeat: 10000`` steps are 1e12 iterations, each level individually legal.
These courts pin the replacement: the product of nested repeat/foreach
multipliers is bounded by ``MAX_TOTAL_STEPS`` statically (refused before any
step runs) and dynamically (data-dependent ``foreach`` lengths and nested
pipelines charge one shared budget). Real collaborators throughout: the host
ToolProvider, the capability surface the component runs, and the
deterministic LM doubles in dspy_doubles.
"""

from __future__ import annotations

import json
import threading

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps  # noqa: E402
import host  # noqa: E402
from dspy_doubles import chat, scripted  # noqa: E402

TOOLS = host.ToolProvider()


def host_tools(name: str, args_json: str) -> str:
    return TOOLS.call(None, name, args_json)


def guarded(capability, request, lm, tools=host_tools) -> dict:
    return json.loads(caps.guarded(lambda: capability(request, lm, tools)))


def over_budget(report: dict) -> None:
    assert report["state"] == "FAILED", report
    assert report["error_type"] == "WorkBudgetError", report
    assert f"MAX_TOTAL_STEPS={caps.MAX_TOTAL_STEPS}" in report["message"], report


def pipeline(steps: list, inputs: dict | None = None) -> dict:
    return {"module": "pipeline", "steps": steps, "inputs": inputs or {}}


def nested_repeat(*counts: int) -> list:
    """repeat counts[0] { repeat counts[1] { ... { set x } } }"""
    steps: list = [{"set": {"x": 1}}]
    for count in reversed(counts):
        steps = [{"repeat": count, "steps": steps}]
    return steps


# ------------------------------------------------------------- court probe P1


def test_P1_nested_repeat_is_bounded_in_total_work() -> None:
    # Ported from the court probe: 3 nested repeats, each within MAX_REPEAT,
    # 1e12 iterations in total. The guarded run must return, refused.
    request = pipeline(nested_repeat(10_000, 10_000, 10_000))
    out: dict = {}
    thread = threading.Thread(
        target=lambda: out.setdefault("r", guarded(caps.run, request, scripted(chat(answer="A")))),
        daemon=True,
    )
    thread.start()
    thread.join(10.0)
    assert not thread.is_alive(), "nested repeat pinned the component > 10 s"
    over_budget(out["r"])


# ------------------------------------------------- static: before execution


@pytest.mark.parametrize(
    "counts",
    [
        (10_000, 10),  # depth 2: 1 + 10000 * 11 = 110001
        (100, 100, 10),  # depth 3: 110101
        (10, 10, 10, 100),  # depth 4: 101111
        (10_000, 10_000, 10_000),  # the court's 1e12 request
    ],
)
def test_over_budget_nesting_is_refused_at_every_depth(counts) -> None:
    assert all(0 <= count <= caps.MAX_REPEAT for count in counts)
    report = guarded(caps.run, pipeline(nested_repeat(*counts)), scripted("unused"))
    over_budget(report)


def test_refusal_happens_before_any_step_runs() -> None:
    # A real host tool records its invocations; the over-budget repeat comes
    # after it, so a dynamic-only budget would have run the tool first.
    marks: list[int] = []
    tools = host.ToolProvider({"mark": lambda: marks.append(1) or len(marks)})
    report = guarded(
        caps.run,
        pipeline([{"tool": "mark", "args": {}}, *nested_repeat(10_000, 10_000)]),
        scripted("unused"),
        lambda name, args_json: tools.call(None, name, args_json),
    )
    over_budget(report)
    assert marks == []


def test_budget_boundary_is_inclusive() -> None:
    # 1 + 369 * 271 == 100000 == MAX_TOTAL_STEPS: accepted and executed.
    assert 1 + 369 * 271 == caps.MAX_TOTAL_STEPS
    body = [{"set": {"x": "$x"}} for _ in range(270)] + [{"set": {"n": 1}}]
    steps = [{"repeat": 369, "steps": body}]
    assert caps.pipeline_work(steps) == caps.MAX_TOTAL_STEPS
    report = guarded(caps.run, pipeline(steps, {"x": 7}), scripted("unused"))
    assert report["state"] == "ALIVE", report
    assert report["outputs"] == {"n": 1}

    # One more step is refused.
    over_budget(
        guarded(caps.run, pipeline([*steps, {"set": {"y": 1}}], {"x": 7}), scripted("unused"))
    )


def test_pipeline_work_counts_every_multiplier() -> None:
    work = caps.pipeline_work
    assert work([{"set": {"a": 1}}]) == 1
    assert work(nested_repeat(3, 4)) == 1 + 3 * (1 + 4 * 1)
    assert work([{"repeat": 0, "steps": [{"set": {"a": 1}}]}]) == 1
    # A literal foreach list multiplies; a state reference counts once here
    # and is charged by its real length at run time.
    assert work([{"foreach": [1, 2, 3], "steps": [{"set": {"a": 1}}]}]) == 1 + 3
    assert work([{"foreach": "$items", "steps": [{"set": {"a": 1}}]}]) == 2
    # `when` does not discount a step: it is visited either way.
    assert work([{"when": "$flag", "repeat": 5, "steps": [{"set": {"a": 1}}]}]) == 6
    # A nested pipeline program contributes its own work.
    inner = {"module": "pipeline", "steps": nested_repeat(10)}
    assert work([{"repeat": 2, "steps": [{"name": "p", "program": inner}]}]) == 1 + 2 * (1 + 1 + 10)


def test_nested_pipeline_program_is_counted_statically() -> None:
    inner = {"module": "pipeline", "steps": nested_repeat(100)}
    steps = [{"repeat": 10_000, "steps": [{"name": "inner", "program": inner}]}]
    over_budget(guarded(caps.run, pipeline(steps), scripted("unused")))


@pytest.mark.parametrize("times", [-1, True, "3", caps.MAX_REPEAT + 1])
def test_malformed_repeat_is_refused_even_on_a_skipped_branch(times) -> None:
    steps = [{"when": "$never", "repeat": times, "steps": [{"set": {"x": 1}}]}]
    report = guarded(caps.run, pipeline(steps, {"never": False}), scripted("unused"))
    assert report["state"] == "FAILED" and "'repeat' must be an integer" in report["message"]


@pytest.mark.parametrize("steps", [{"set": {"x": 1}}, ["not-an-object"]])
def test_malformed_step_lists_are_typed_refusals(steps) -> None:
    with pytest.raises(caps.RequestError):
        caps.pipeline_work(steps)


# ------------------------------------------------------ dynamic: at run time


def test_foreach_over_state_is_charged_by_its_real_length() -> None:
    steps = [{"foreach": "$items", "steps": [{"set": {"y": "$item"}}]}]
    ok = guarded(caps.run, pipeline(steps, {"items": list(range(10))}), scripted("unused"))
    assert ok["state"] == "ALIVE", ok
    items = list(range(caps.MAX_TOTAL_STEPS))
    over_budget(guarded(caps.run, pipeline(steps, {"items": items}), scripted("unused")))


def test_nested_pipelines_share_one_budget() -> None:
    # Statically 1 + 1 * (1 + 1 + 1000) = 1003 steps; at run time 200 items
    # each enter a nested pipeline of 1001 steps. Per-pipeline budgets would
    # admit it (outer 201, each inner 1001); the shared budget refuses the
    # 200 * 1002 total.
    inner = {"module": "pipeline", "steps": nested_repeat(1_000), "outputs": ["x"]}
    steps = [
        {
            "foreach": "$items",
            "steps": [{"name": "inner", "program": inner}],
            "collect": {"xs": "x"},
        }
    ]
    assert caps.pipeline_work(steps) == 1_003
    small = guarded(caps.run, pipeline(steps, {"items": [1, 2]}), scripted("unused"))
    assert small["state"] == "ALIVE", small
    assert small["outputs"]["xs"] == [1, 1]
    items = list(range(200))
    over_budget(guarded(caps.run, pipeline(steps, {"items": items}), scripted("unused")))


def test_budget_is_per_call_not_cumulative() -> None:
    # Each top-level call starts a fresh budget: running a near-limit pipeline
    # twice in one batch-free sequence is admitted both times.
    steps = nested_repeat(9_999)
    for _ in range(3):
        report = guarded(caps.run, pipeline(steps), scripted("unused"))
        assert report["state"] == "ALIVE", report
    assert caps._WORK_BUDGET.get() is None


def test_describe_publishes_the_limits() -> None:
    limits = caps.describe()["pipeline_limits"]
    assert limits == {"max_repeat": caps.MAX_REPEAT, "max_total_steps": caps.MAX_TOTAL_STEPS}


def test_skipped_when_steps_are_still_charged() -> None:
    # A body whose every step is skipped by `when` still iterates: without a
    # charge per visited step, a long state list would loop unbudgeted.
    steps = [{"foreach": "$items", "steps": [{"when": "$never", "set": {"y": 1}}]}]
    ok = guarded(caps.run, pipeline(steps, {"items": [1, 2], "never": False}), scripted("unused"))
    assert ok["state"] == "ALIVE", ok
    items = list(range(caps.MAX_TOTAL_STEPS))
    over_budget(
        guarded(caps.run, pipeline(steps, {"items": items, "never": False}), scripted("unused"))
    )
