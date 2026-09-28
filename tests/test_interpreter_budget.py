"""Interpreted code cannot pin the guest: a trace-event budget stops pure-Python loops.

Under a host with no epoch interruption (Wasmex 0.15.1 exposes none) an
LM-authored `while True: pass` held an instance at ~290% CPU until killed.
The reproduction runs in a subprocess so that the old behaviour is a timeout,
not a hung suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("dspy")

import dspy_runtime
import limits

ROOT = Path(__file__).resolve().parents[1]

RUN_UNBOUNDED = """
import dspy_runtime
interp = dspy_runtime.ComponentInterpreter()
try:
    interp.execute({code!r})
except dspy_runtime.CodeExecutionError as exc:
    print("STOPPED", exc)
"""


def run_isolated(code: str, timeout: float = 60) -> str:
    result = subprocess.run(
        [sys.executable, "-c", RUN_UNBOUNDED.format(code=code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "code",
    [
        "while True:\n    pass",
        "while True:\n    try:\n        pass\n    except BaseException:\n        pass",
        "def f():\n    return f()\nwhile True:\n    x = 1",
        "[i for i in iter(int, 1)]",
    ],
)
def test_runaway_interpreted_code_is_stopped(code: str) -> None:
    out = run_isolated(code)
    assert "STOPPED" in out and "budget" in out, out


def test_ordinary_code_is_unaffected_and_the_budget_is_per_execute() -> None:
    interp = dspy_runtime.ComponentInterpreter()
    for _ in range(3):
        assert interp.execute("sum(i * i for i in range(1000))") == 332833500


def test_tracing_is_restored_after_execution_even_on_error() -> None:
    before = sys.gettrace()
    interp = dspy_runtime.ComponentInterpreter()
    with pytest.raises(dspy_runtime.CodeExecutionError):
        interp.execute("raise ValueError('x')")
    with pytest.raises(dspy_runtime.CodeExecutionError):
        interp.execute("while True:\n    pass")
    assert sys.gettrace() is before


def test_submit_still_ends_execution() -> None:
    interp = dspy_runtime.ComponentInterpreter()
    result = interp.execute("SUBMIT(7)\nwhile True:\n    pass")
    assert result.output == {"output": 7}


def test_the_budget_is_a_published_component_limit() -> None:
    limit = limits.LIMITS["max_interpreter_events"]
    assert limit.enforced_by == "component"
    assert dspy_runtime.MAX_INTERPRETER_EVENTS == limit.value
