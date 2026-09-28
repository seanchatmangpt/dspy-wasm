"""The conformance suite, run against the in-component court wired to the real host."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dspy")

import conformance


def _invoke(component):
    def invoke(export: str, *args: str) -> dict:
        return json.loads(getattr(component, export.replace("-", "_"))(*args))

    return invoke


def test_reference_host_passes_the_conformance_suite(component) -> None:
    report = conformance.run(_invoke(component))
    broken = [(c["name"], c["message"]) for c in report["cases"] if c["state"] != "ALIVE"]
    assert report["state"] == "ALIVE" and not broken, broken
    assert report["passed"] == len(conformance.CHECKS)


def test_a_host_that_skips_a_refusal_fails_the_suite(component) -> None:
    # A host whose component let fan-out through would answer ALIVE here.
    real = _invoke(component)

    def lax(export: str, *args: str) -> dict:
        if export == "run" and '"majority"' in args[0]:
            return {"state": "ALIVE"}
        return real(export, *args)

    report = conformance.run(lax)
    failed = {c["name"] for c in report["cases"] if c["state"] != "ALIVE"}
    assert report["state"] == "FAILED"
    assert failed == {"fanout_beyond_the_ceiling_is_refused"}


def test_a_host_that_raises_is_reported_not_propagated() -> None:
    def broken(export: str, *args: str) -> dict:
        raise RuntimeError("host down")

    report = conformance.run(broken)
    assert report["state"] == "FAILED" and report["failed"] == len(conformance.CHECKS)
    assert all(c["error_type"] == "RuntimeError" for c in report["cases"])
