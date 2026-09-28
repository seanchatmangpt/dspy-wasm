"""The in-component court (app.py), run natively against the real host.

app.py is what componentize-py packages into dspy.wasm; it imports the
generated ``dspy_bindings`` package, which only exists inside the component.
Here that package is wired exactly as host.py's Wasmtime linker wires it:
``chatman:dspy/lm.complete`` -> CompletionProvider.complete and
``chatman:dspy/tools.call`` -> ToolProvider.call. No behaviour is doubled; the
same exports and the same 36 self-test cases run, so a broken case or export
fails here in seconds instead of after the multi-hour WASI build.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

pytest.importorskip("dspy")

import host

ANSWER = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"


@pytest.fixture(scope="module")
def component():
    provider = host.CompletionProvider(
        static_response=ANSWER, base_url=None, api_key=None, upstream_model=None
    )
    tools = host.ToolProvider()

    bindings = types.ModuleType("dspy_bindings")
    bindings.DspyBindings = type("DspyBindings", (), {})
    imports = types.ModuleType("dspy_bindings.imports")
    imports.host_lm = types.SimpleNamespace(complete=lambda req: provider.complete(None, req))
    imports.host_tools = types.SimpleNamespace(call=lambda n, a: tools.call(None, n, a))
    bindings.imports = imports

    saved = {name: sys.modules.get(name) for name in ("dspy_bindings", "dspy_bindings.imports")}
    sys.modules["dspy_bindings"], sys.modules["dspy_bindings.imports"] = bindings, imports
    sys.modules.pop("app", None)
    # Inside the component app.py is the first importer, so its numpy imports
    # precede dspy's lazy proxy. pytest imported dspy first; dspy_capabilities
    # resolves that proxy (see its module header) before app imports numpy.
    importlib.import_module("dspy_capabilities")
    try:
        app = importlib.import_module("app")
        yield app.DspyBindings()
    finally:
        sys.modules.pop("app", None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_every_in_component_self_test_case_is_alive(component) -> None:
    report = json.loads(component.run_self_tests())
    broken = [
        (case["name"], case.get("message")) for case in report["cases"] if case["state"] != "ALIVE"
    ]
    assert report["state"] == "ALIVE", broken
    assert report["failed"] == 0 and report["passed"] == len(report["cases"]) == 36


def test_predict_export_round_trips_through_the_host_lm(component) -> None:
    report = json.loads(component.predict("question -> answer", '{"question": "capital?"}'))
    assert report == {"state": "ALIVE", "outputs": {"answer": "Paris"}}


def test_predict_export_refuses_non_object_inputs(component) -> None:
    report = json.loads(component.predict("question -> answer", "[1]"))
    assert report["state"] != "ALIVE"
    assert "object" in report["message"]


def test_request_exports_never_raise_into_the_host(component) -> None:
    for export in ("run", "render", "evaluate", "compile"):
        for payload in ("not json", "[]", '{"module": "no-such-module"}'):
            report = json.loads(getattr(component, export)(payload))
            assert report["state"] == "FAILED", (export, payload, report)


def test_capabilities_export_publishes_the_limits(component) -> None:
    report = json.loads(component.capabilities())
    assert report["state"] == "ALIVE"
    assert set(report["pipeline_limits"]) == {"max_repeat", "max_total_steps"}
    assert set(report["module_limits"]) == {"max_fanout", "max_iters"}
