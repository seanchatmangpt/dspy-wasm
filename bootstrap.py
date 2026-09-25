"""Dependency-free component court for Python -> WebAssembly."""

import json
import platform
import sys

import wit


@wit.guest
class Dspy(wit.WorldExports):
    def component_version(self) -> str:
        return "0.1.0"

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
