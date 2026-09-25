"""DSPy import court compiled as a WebAssembly component.

DSPy is imported at module scope because componentize-py resolves imports at
build time. A failure here is a dependency-closure finding, not a failure of the
bootstrap Python -> WebAssembly path.
"""

import json
import platform
import sys

import dspy
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
                "state": "DSPY_IMPORTED",
            },
            sort_keys=True,
        )

    def dspy_version(self) -> str:
        return json.dumps(
            {
                "state": "ALIVE",
                "version": getattr(dspy, "__version__", "unknown"),
            },
            sort_keys=True,
        )
