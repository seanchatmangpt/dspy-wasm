"""Host-side epoch deadline: the guest cannot pin the host.

The pipeline work budget is a request-level check inside the component; this
court pins the independent host-level bound (Wasmtime epoch interruption) that
still holds when such a check is bypassed. Real Wasmtime engines, stores,
core modules and components throughout (WAT sources compiled by Wasmtime),
plus the real ``host.py`` CLI in a subprocess against ``dist/bootstrap.wasm``
when it has been built (``make bootstrap``).
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
from wasmtime import Instance, Module, Trap
from wasmtime.component import Component, Linker

import host

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "dist" / "bootstrap.wasm"

CORE_WAT = """(module
  (func (export "spin") (loop br 0))
  (func (export "one") (result i32) i32.const 1)
  (func (export "div0") (result i32) i32.const 1 i32.const 0 i32.div_s))"""

COMPONENT_WAT = """(component
  (core module $m (func (export "spin") (loop br 0)))
  (core instance $i (instantiate $m))
  (func (export "spin") (canon lift (core func $i "spin"))))"""


def core_exports(deadline_s: float):
    engine = host.new_engine()
    store = host.new_store(engine, deadline_s)
    module = Module(engine, CORE_WAT)
    instance = host.guest_call(store, lambda s: Instance(s, module, []))
    return store, instance.exports(store)


def test_spinning_core_function_is_interrupted_at_its_deadline() -> None:
    store, exports = core_exports(0.3)
    started = time.perf_counter()
    with pytest.raises(host.DeadlineExceeded, match="0.3 s deadline"):
        host.guest_call(store, exports["spin"])
    elapsed = time.perf_counter() - started
    assert 0.25 <= elapsed < 5.0, elapsed


def test_spinning_component_export_is_interrupted_at_its_deadline() -> None:
    engine = host.new_engine()
    store = host.new_store(engine, 0.3)
    component = Component(engine, COMPONENT_WAT)
    instance = host.guest_call(store, Linker(engine).instantiate, component)
    spin = instance.get_func(store, "spin")
    started = time.perf_counter()
    with pytest.raises(host.DeadlineExceeded):
        host.guest_call(store, spin)
    assert time.perf_counter() - started < 5.0


def test_other_traps_are_not_reported_as_deadlines() -> None:
    store, exports = core_exports(5.0)
    with pytest.raises(Trap) as trap:
        host.guest_call(store, exports["div0"])
    assert not isinstance(trap.value, host.DeadlineExceeded)
    assert "divide by zero" in str(trap.value)


def test_deadline_is_rearmed_for_every_call() -> None:
    # The epoch keeps advancing between calls; a deadline armed only once
    # at store creation would already be spent when the second call starts.
    store, exports = core_exports(0.2)
    assert host.guest_call(store, exports["one"]) == 1
    time.sleep(0.5)
    assert host.guest_call(store, exports["one"]) == 1


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, True])
def test_invalid_deadlines_are_refused(bad) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        host.new_store(host.new_engine(), bad)


@pytest.mark.skipif(not BOOTSTRAP.exists(), reason="dist/bootstrap.wasm not built (make bootstrap)")
def test_bootstrap_component_runs_under_a_deadline() -> None:
    provider = host.CompletionProvider(
        static_response=host.DEFAULT_RESPONSE, base_url=None, api_key=None, upstream_model=None
    )
    store, instance = host.instantiate(BOOTSTRAP, provider, deadline_s=30.0)
    assert host.call_json(store, instance, "runtime-info")["state"] == "BOOTSTRAP"


@pytest.mark.skipif(not BOOTSTRAP.exists(), reason="dist/bootstrap.wasm not built (make bootstrap)")
def test_cli_deadline_flag() -> None:
    ok = subprocess.run(
        [sys.executable, "host.py", str(BOOTSTRAP), "--deadline", "30"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "component-version: 0.1.0" in ok.stdout
    refused = subprocess.run(
        [sys.executable, "host.py", str(BOOTSTRAP), "--deadline", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert refused.returncode != 0
    assert "positive finite" in refused.stderr
