"""Pytest plugin: with DSPY_WASM_FORBID_SKIPS=1 a skipped test is a failure.

A skip is silent coverage loss (the deadline tests skipped in CI for want of a
built bootstrap.wasm and nothing noticed). CI's full job sets the variable; a
job that legitimately lacks a dependency, such as one without dspy, does not.
"""

from __future__ import annotations

import os

import pytest

ENV = "DSPY_WASM_FORBID_SKIPS"


def _forbidden(report) -> bool:
    return os.environ.get(ENV) == "1" and report.skipped


def _fail(report, what: str) -> None:
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
    report.outcome = "failed"
    report.longrepr = f"{what} skipped, and {ENV}=1 forbids skips: {reason}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if _forbidden(report):
        _fail(report, item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    report = (yield).get_result()
    if _forbidden(report):
        _fail(report, collector.nodeid)
