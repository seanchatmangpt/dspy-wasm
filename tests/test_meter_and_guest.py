"""The host-side meter and the instance recycle policy.

Real Wasmtime engines, stores and components (a WAT component with a spinning
export), the real providers, no doubles. The meter aborting a real dspy.wasm
compile is exercised end to end by CI (`--max-lm-calls 1`, exit 3).
"""

from __future__ import annotations

import json
import time

import pytest

import host

_ALIVE = json.dumps({"state": "ALIVE"})
_DATA = "".join(f"\\{byte:02x}" for byte in _ALIVE.encode())
COMPONENT_WAT = f"""(component
  (core module $m
    (memory (export "memory") 1)
    (data (i32.const 8) "{_DATA}")
    (func (export "realloc") (param i32 i32 i32 i32) (result i32) i32.const 0)
    (func (export "ok") (result i32)
      (i32.store (i32.const 0) (i32.const 8))
      (i32.store (i32.const 4) (i32.const {len(_ALIVE)}))
      i32.const 0)
    (func (export "spin") (result i32) (loop br 0) i32.const 0))
  (core instance $i (instantiate $m))
  (func (export "ok") (result string)
    (canon lift (core func $i "ok") (memory $i "memory") (realloc (func $i "realloc"))))
  (func (export "spin") (result string)
    (canon lift (core func $i "spin") (memory $i "memory") (realloc (func $i "realloc")))))"""


@pytest.fixture
def component(tmp_path):
    path = tmp_path / "guest.wat"
    path.write_text(COMPONENT_WAT)
    return path


def provider() -> host.CompletionProvider:
    return host.CompletionProvider(
        static_response=host.DEFAULT_RESPONSE, base_url=None, api_key=None, upstream_model=None
    )


# ----------------------------------------------------------------------- meter


def test_lm_calls_past_the_budget_abort_the_call() -> None:
    store = host.new_store(host.new_engine(), budget=host.Budget(max_lm_calls=2))
    lm = provider()
    assert "text" in json.loads(lm.complete(store, "{}"))
    assert "text" in json.loads(lm.complete(store, "{}"))
    with pytest.raises(host.BudgetExceeded, match="2 LM calls"):
        lm.complete(store, "{}")
    assert store.dspy_wasm_meter.tripped is not None


def test_tool_calls_and_reply_bytes_are_metered() -> None:
    tools = host.ToolProvider()
    store = host.new_store(host.new_engine(), budget=host.Budget(max_tool_calls=1))
    tools.call(store, "echo", "{}")
    with pytest.raises(host.BudgetExceeded, match="1 tool calls"):
        tools.call(store, "echo", "{}")

    store = host.new_store(host.new_engine(), budget=host.Budget(max_reply_bytes=1_000))
    with pytest.raises(host.BudgetExceeded, match="1000 reply bytes"):
        tools.call(store, "embed", json.dumps({"texts": ["a"], "dimensions": 500}))


def test_counters_reset_for_every_guest_call(component) -> None:
    store, _ = host.instantiate(component, provider(), budget=host.Budget(max_lm_calls=1))
    lm = provider()
    for _ in range(3):  # each guest call has its own budget of one
        host.guest_call(store, lambda s: lm.complete(s, "{}"))


def test_unmetered_calls_still_work_without_a_store() -> None:
    assert "text" in json.loads(provider().complete(None, "{}"))
    assert json.loads(host.ToolProvider().call(None, "echo", '{"a": 1}'))["result"]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_guest_refuses_a_non_positive_recycle_count(component, bad) -> None:
    with pytest.raises(ValueError, match="max_calls"):
        host.Guest(component, provider(), max_calls=bad)


# ----------------------------------------------------------------------- guest


def test_a_healthy_instance_is_reused(component) -> None:
    guest = host.Guest(component, provider())
    for _ in range(3):
        assert guest.call_json("ok") == {"state": "ALIVE"}
    assert guest.instances_created == 1


def test_a_deadline_trap_is_replaced_by_a_fresh_instance(component) -> None:
    guest = host.Guest(component, provider(), deadline_s=0.3)
    assert guest.call_json("ok")["state"] == "ALIVE"
    with pytest.raises(host.DeadlineExceeded):
        guest.call_json("spin")
    assert guest.call_json("ok")["state"] == "ALIVE"  # a raw trapped instance cannot be re-entered
    assert guest.instances_created == 2


def test_the_instance_is_recycled_after_max_calls(component) -> None:
    guest = host.Guest(component, provider(), max_calls=2)
    for _ in range(5):
        guest.call_json("ok")
    assert guest.instances_created == 3


def test_prewarm_builds_the_replacement_off_the_callers_path(component) -> None:
    guest = host.Guest(component, provider(), deadline_s=0.3, prewarm=True)
    guest.call_json("ok")
    with pytest.raises(host.DeadlineExceeded):
        guest.call_json("spin")
    assert guest._warm is not None  # replacement already being built
    deadline = time.monotonic() + 30
    while not guest._warm_result and time.monotonic() < deadline:
        time.sleep(0.01)
    assert guest._warm_result, "prewarm never finished"
    started = time.perf_counter()
    assert guest.call_json("ok")["state"] == "ALIVE"
    assert time.perf_counter() - started < 1.0
    assert guest.instances_created == 2
