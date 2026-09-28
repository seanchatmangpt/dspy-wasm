"""DSPY_WASM_FORBID_SKIPS=1 turns a skip into a failure."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

SKIPPING = "import pytest\n\ndef test_a():\n    pytest.skip('no artifact')\n"
MODULE_SKIP = (
    "import pytest\n\npytest.importorskip('no_such_module_xyz')\n\ndef test_a():\n    pass\n"
)


def run(tmp_path: Path, source: str, forbid: bool) -> subprocess.CompletedProcess:
    (tmp_path / "test_probe.py").write_text(source)
    env = {**os.environ, "PYTHONPATH": str(HERE), "PYTEST_ADDOPTS": ""}
    env.pop("DSPY_WASM_FORBID_SKIPS", None)
    if forbid:
        env["DSPY_WASM_FORBID_SKIPS"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "forbid_skips", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("source", [SKIPPING, MODULE_SKIP])
def test_skips_are_failures_when_forbidden(tmp_path, source) -> None:
    result = run(tmp_path, source, forbid=True)
    assert result.returncode in (1, 2), result.stdout  # 2: collection-time skip
    assert "forbids skips" in result.stdout


@pytest.mark.parametrize("source", [SKIPPING, MODULE_SKIP])
def test_skips_are_allowed_by_default(tmp_path, source) -> None:
    result = run(tmp_path, source, forbid=False)
    assert result.returncode in (0, 5), result.stdout  # 5: nothing left to run
