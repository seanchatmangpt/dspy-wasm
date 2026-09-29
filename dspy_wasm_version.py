"""The release version of the dspy-wasm component, in one place.

CalVer, ``YY.M.D`` with no zero padding (the date the release was cut). It is the
*component* version reported by the ``component-version`` export and published in
``consumer/contract.json``. It is independent of the WIT interface version
(``chatman:dspy@0.1.0``), which changes only when the WIT surface changes.
``python release.py write`` propagates it to every file that must carry it and
``python release.py check`` fails if any of them disagrees.
"""

VERSION = "26.9.28"
