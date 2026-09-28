"""Dependency-free component court for Python -> WebAssembly."""

import json
import platform
import sys

import dspy_bindings as wit

import dspy_wasm_version


class DspyBindings(wit.DspyBindings):
    def component_version(self) -> str:
        return dspy_wasm_version.VERSION

    def runtime_info(self) -> str:
        return json.dumps(
            {
                "component": "dspy-wasm",
                "python": sys.version.split()[0],
                "platform": platform.system(),
                "state": "BOOTSTRAP",
            },
            sort_keys=True,
        )

    def dspy_version(self) -> str:
        return json.dumps(
            {
                "state": "UNSUPPORTED",
                "reason": "bootstrap component intentionally excludes DSPy",
            },
            sort_keys=True,
        )
