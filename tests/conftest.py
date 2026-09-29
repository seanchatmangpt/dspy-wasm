"""Shared fixtures."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

import host

pytest_plugins = ["forbid_skips"]

ANSWER = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"


@pytest.fixture(scope="module")
def component():
    pytest.importorskip("dspy")
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
