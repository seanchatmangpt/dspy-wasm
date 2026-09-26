"""Adversarial courts for the component boundary.

Each test is a falsifier for a defect found against d346ff9: it failed (or the
input was silently accepted) before the guard it pins. Real collaborators
throughout: the host ToolProvider/CompletionProvider, the capability surface
the component runs, and the deterministic LM doubles in dspy_doubles.
"""

from __future__ import annotations

import json
import threading
import time
import tracemalloc

import pytest

pytest.importorskip("dspy")

from dspy.primitives.code_interpreter import FinalOutput

import dspy_capabilities as caps
import dspy_runtime as rt
import host
from dspy_doubles import chat, scripted

TOOLS = host.ToolProvider()


def host_tools(name: str, args_json: str) -> str:
    return TOOLS.call(None, name, args_json)


def guarded(capability, request, lm) -> dict:
    return json.loads(caps.guarded(lambda: capability(request, lm, host_tools)))


def refused(report: dict, fragment: str) -> None:
    assert report["state"] == "FAILED", report
    assert fragment in report["message"], report["message"]


# ------------------------------------------------------------- pipeline names


def _two_step_pipeline(first: str, second: str) -> dict:
    return {
        "module": "pipeline",
        "steps": [
            {"name": first, "program": {"signature": "question -> answer"}},
            {"name": second, "program": {"signature": "question -> other"}},
        ],
        "inputs": {"question": "q"},
    }


@pytest.mark.parametrize("name", ["outputs", "steps", "_tools", "forward", "callbacks"])
def test_program_step_cannot_shadow_pipeline_attributes(name: str) -> None:
    # Before: 'forward' silently replaced the whole pipeline with its first
    # step (ALIVE, one output); 'outputs'/'steps'/'_tools' crashed later.
    report = guarded(
        caps.run, _two_step_pipeline(name, "second"), scripted(chat(answer="A"), chat(other="O"))
    )
    refused(report, "shadows pipeline attribute")


def test_distinct_programs_normalising_to_one_attribute_are_refused() -> None:
    # 'a-b' and 'a_b' both become attribute 'a_b'; before, the second step
    # silently re-ran the first program under the wrong signature.
    report = guarded(caps.run, _two_step_pipeline("a-b", "a_b"), scripted(chat(answer="A")))
    refused(report, "collides with a different program")


def test_same_program_reused_by_name_is_still_one_predictor() -> None:
    step = {"name": "hop", "program": {"signature": "question -> answer"}}
    report = guarded(
        caps.run,
        {"module": "pipeline", "steps": [step, dict(step)], "inputs": {"question": "q"}},
        scripted(chat(answer="A"), chat(answer="B")),
    )
    assert report["state"] == "ALIVE" and report["outputs"] == {"answer": "B"}
    assert report["lm_calls"] == 2


@pytest.mark.parametrize("times", [-1, True, "3", caps.MAX_REPEAT + 1])
def test_pipeline_repeat_is_bounded_and_typed(times) -> None:
    report = guarded(
        caps.run,
        {
            "module": "pipeline",
            "steps": [{"repeat": times, "steps": [{"set": {"x": 1}}]}],
            "inputs": {},
        },
        scripted("unused"),
    )
    refused(report, "'repeat' must be an integer")


def _nested_repeat(levels: int, body: list[dict]) -> list[dict]:
    steps = body
    for _ in range(levels):
        steps = [{"repeat": caps.MAX_REPEAT, "steps": steps}]
    return steps


@pytest.mark.parametrize(
    "steps",
    [
        _nested_repeat(3, [{"set": {"x": 1}}]),  # 1e12 iterations before the budget
        _nested_repeat(3, []),  # empty bodies still iterate: each iteration is charged
        [{"foreach": "$items", "steps": _nested_repeat(1, [])}],
    ],
    ids=["nested-set", "nested-empty", "foreach-x-repeat"],
)
def test_nested_iteration_is_bounded_in_total_work(steps) -> None:
    # Before: MAX_REPEAT held per level only; 3 nested repeats ran ~18 days.
    # A daemon thread makes a regression fail this test instead of hanging it.
    request = {"module": "pipeline", "steps": steps, "inputs": {"items": list(range(100))}}
    reports: list[dict] = []
    worker = threading.Thread(
        target=lambda: reports.append(guarded(caps.run, request, scripted("unused"))),
        daemon=True,
    )
    worker.start()
    worker.join(10.0)
    assert not worker.is_alive(), "nested iteration was not bounded in total work"
    refused(reports[0], f"exceeded {caps.MAX_PIPELINE_STEPS} executed steps")


def test_single_level_repeat_at_its_bound_still_runs() -> None:
    report = guarded(
        caps.run,
        {
            "module": "pipeline",
            "steps": [{"repeat": caps.MAX_REPEAT, "steps": [{"set": {"x": "$$done"}}]}],
            "inputs": {},
        },
        scripted("unused"),
    )
    assert report["state"] == "ALIVE" and report["outputs"] == {"x": "$done"}


# ------------------------------------------------------------ tool envelopes


@pytest.mark.parametrize("envelope", ["{}", '{"error": null}', '{"value": 1}'])
def test_tool_envelope_without_result_or_error_is_refused(envelope: str) -> None:
    tools = caps.HostTools(lambda _name, _args: envelope)
    with pytest.raises(caps.HostError, match="without 'result' or 'error'"):
        tools.call("anything", {})


def test_tool_envelope_with_null_result_is_a_value() -> None:
    assert caps.HostTools(lambda _n, _a: '{"result": null}').call("t", {}) is None


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "__host_tool__", "parameters": {"properties": {"x": {}}}},
        {"name": "SUBMIT", "parameters": {"properties": {"x": {}}}},
        {"name": "calculator", "parameters": {"properties": {"__host_tool__": {}}}},
        {"name": "calculator", "parameters": {"properties": {"__class__": {}}}},
        {"name": "class", "parameters": {"properties": {}}},
    ],
)
def test_tool_shim_names_cannot_shadow_the_bridge(spec: dict) -> None:
    # Before: a tool named __host_tool__ rebound the bridge to itself.
    with pytest.raises(caps.RequestError, match="must be a Python identifier"):
        caps.HostTools(host_tools).function(spec)


def test_tool_shim_round_trips_through_the_real_host() -> None:
    fn = caps.HostTools(host_tools).function(
        {"name": "calculator", "parameters": {"properties": {"expression": {}}}}
    )
    assert fn("6 * 7") == 42
    with pytest.raises(caps.HostError, match="unsupported expression element"):
        fn("__import__('os')")


# ------------------------------------------------------- program_state subject


def _compiled(signature: str = "question -> answer") -> dict:
    return caps.compile_program(
        {
            "program": {"signature": signature},
            "optimizer": "labeled-few-shot",
            "trainset": [{"question": "France?", "answer": "Paris"}],
        },
        scripted("unused"),
        host_tools,
    )["program_state"]


def test_compiled_state_is_bound_to_its_program_subject() -> None:
    state = _compiled()
    assert state[caps.SUBJECT_KEY].startswith("sha256:")
    report = guarded(
        caps.run, {"program_state": state, "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    assert report["state"] == "ALIVE" and report["outputs"] == {"answer": "x"}


def test_stale_program_state_for_another_signature_is_refused() -> None:
    # Before: loaded silently; 'Question:' prefix landed on 'context'.
    report = guarded(
        caps.run,
        {
            "signature": "context, question -> summary",
            "program_state": _compiled(),
            "inputs": {"context": "c", "question": "q"},
        },
        scripted(chat(summary="s")),
    )
    refused(report, "stale program_state")


def test_tampered_subject_digest_is_refused() -> None:
    state = {**_compiled(), caps.SUBJECT_KEY: "sha256:" + "0" * 64}
    report = guarded(
        caps.run, {"program_state": state, "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    refused(report, "stale program_state")


def test_stripped_subject_cannot_smuggle_a_stale_state() -> None:
    # Before: popping __subject__ made the state 'unbound' and it loaded
    # silently into any signature with the same field count.
    state = {k: v for k, v in _compiled().items() if k != caps.SUBJECT_KEY}
    report = guarded(
        caps.run,
        {"signature": "context -> summary", "program_state": state, "inputs": {"context": "c"}},
        scripted(chat(summary="S")),
    )
    refused(report, "unbound program_state")


def test_state_body_inconsistent_with_its_subject_is_refused() -> None:
    state = _compiled()
    state["signature"] = {**state["signature"], "fields": state["signature"]["fields"][:1]}
    report = guarded(
        caps.run, {"program_state": state, "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    refused(report, "fields; signature has 2")


@pytest.mark.parametrize("state", [[1], [], 0, False, ""])
def test_program_state_must_be_an_object(state) -> None:
    # Before: falsy values ([], 0, false, "") skipped admission and were ignored.
    report = guarded(
        caps.run, {"program_state": state, "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    refused(report, "program_state must be a JSON object")


def test_null_program_state_means_absent() -> None:
    report = guarded(
        caps.run, {"program_state": None, "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    assert report["state"] == "ALIVE" and report["outputs"] == {"answer": "x"}


def test_ensemble_refuses_a_stale_member() -> None:
    report = guarded(
        caps.compile_program,
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "ensemble",
            "programs": [_compiled(), _compiled("question -> verdict")],
        },
        scripted("unused"),
    )
    refused(report, "stale program_state")


# ------------------------------------------------------ replay and reordering


def test_replay_is_byte_identical() -> None:
    request = {
        "program": {"signature": "question -> answer"},
        "optimizer": "bootstrap-few-shot",
        "trainset": [
            {"question": "France?", "answer": "Paris"},
            {"question": "Peru?", "answer": "Lima"},
        ],
        "metric": "exact_match",
        "config": {"max_bootstrapped_demos": 2, "max_labeled_demos": 0},
    }
    responses = (chat(answer="Paris"), chat(answer="wrong"))
    first = caps.guarded(lambda: caps.compile_program(request, scripted(*responses), host_tools))
    second = caps.guarded(lambda: caps.compile_program(request, scripted(*responses), host_tools))
    assert first == second
    assert json.loads(first)["demos"] == {"self": 1}


def test_reordered_lm_responses_change_the_evaluation() -> None:
    request = {
        "program": {"signature": "question -> answer"},
        "devset": [
            {"question": "France?", "answer": "Paris"},
            {"question": "Spain?", "answer": "Madrid"},
        ],
        "metric": "exact_match",
    }
    ordered = guarded(caps.evaluate, request, scripted(chat(answer="Paris"), chat(answer="Madrid")))
    swapped = guarded(caps.evaluate, request, scripted(chat(answer="Madrid"), chat(answer="Paris")))
    assert ordered["score"] == 100.0 and swapped["score"] == 0.0


def test_duplicate_delivery_of_a_compiled_state_is_idempotent() -> None:
    state = _compiled()
    reports = [
        guarded(
            caps.run,
            {"program_state": state, "inputs": {"question": "q"}},
            scripted(chat(answer="x")),
        )
        for _ in range(2)
    ]
    assert reports[0] == reports[1] and reports[0]["state"] == "ALIVE"


# ------------------------------------------------------- malformed host replies


@pytest.mark.parametrize(
    ("reply", "fragment"),
    [
        ("not json", "Expecting value"),
        (json.dumps({"text": 5}), "string field 'text'"),
        (json.dumps({"error": "provider down"}), "provider down"),
        (json.dumps({"text": chat(answer="a"), "usage": {"input_tokens": "x"}}), "invalid literal"),
    ],
)
def test_malformed_lm_replies_fail_closed(reply: str, fragment: str) -> None:
    report = guarded(caps.run, {"inputs": {"question": "q"}}, lambda _request: reply)
    refused(report, fragment)


@pytest.mark.parametrize("responses", [{"a": 1}, ["ok", 3], "text"])
def test_scripted_provider_refuses_non_string_arrays(responses) -> None:
    with pytest.raises(TypeError, match="JSON array of strings"):
        host.CompletionProvider(
            static_response=None,
            base_url=None,
            api_key=None,
            upstream_model=None,
            scripted_responses=responses,
        )


# ------------------------------------------------------------ host tools


def test_calculator_refuses_doubly_exponential_growth_quickly() -> None:
    # Before: 4 nested **64 took ~37 s and 55.7M bits on the host; 5 would OOM.
    for expression in (
        "((10**64)**64)**64",
        "(((10**64)**64)**64)**64",
        "((((10**64)**64)**64)**64)**64",
        " * ".join(["9**64"] * 25),  # 25 x 203 bits > MAX_INT_BITS
    ):
        started = time.perf_counter()
        with pytest.raises(ValueError, match="too large"):
            host.calculator(expression)
        assert time.perf_counter() - started < 0.05, expression
    assert host.calculator("(10**64)**4") == 10**256
    assert host.calculator("2 ** 10 - (3 * 4) / 2") == 1018


def _peak_bytes(expression: str) -> int:
    tracemalloc.start()
    try:
        try:
            host.calculator(expression)
        except ValueError:
            pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_power_is_refused_before_it_is_computed() -> None:
    # (2**64)**63 has 4033 bits; raising it to 64 would allocate ~32 KiB
    # before any post-check could refuse it. The pre-check refuses first.
    _peak_bytes("1 + 1")  # warm caches so both measurements see the same overhead
    allowed = _peak_bytes("(2**64)**63")
    with pytest.raises(ValueError, match="too large"):
        host.calculator("((2**64)**63)**64")
    refused_peak = _peak_bytes("((2**64)**63)**64")
    assert refused_peak - allowed < 8 * 1024, (refused_peak, allowed)


@pytest.mark.parametrize("expression", ["(-1) ** 0.5", "True + 1", "1j"])
def test_calculator_is_real_arithmetic_only(expression: str) -> None:
    with pytest.raises(ValueError):
        host.calculator(expression)


@pytest.mark.parametrize("expression", ["1e308 * 1e308", "-1e308 * 10", "1e308 * 10 - 1e308 * 10"])
def test_calculator_refuses_non_finite_results(expression: str) -> None:
    # Before: '1e308*1e308' returned inf, serialised as non-standard 'Infinity'.
    with pytest.raises(ValueError, match="not finite"):
        host.calculator(expression)
    raw = TOOLS.call(None, "calculator", json.dumps({"expression": expression}))

    def strict(token: str):
        raise AssertionError(f"non-standard JSON constant {token}")

    assert "not finite" in json.loads(raw, parse_constant=strict)["error"]


def test_tool_envelope_is_strict_json_for_any_tool() -> None:
    provider = host.ToolProvider({"nan": lambda: float("nan")})
    envelope = json.loads(provider.call(None, "nan", "{}"))
    assert "Out of range float values" in envelope["error"]


def test_search_k_bounds() -> None:
    corpus = host.Corpus()
    assert corpus.search("Paris capital", k=0) == []
    assert corpus.search("Paris capital", k=1) == ["Paris is the capital of France."]
    for bad in (-1, True, "2", 1.5):
        with pytest.raises((ValueError, TypeError)):
            corpus.search("Paris capital", k=bad)
    envelope = json.loads(TOOLS.call(None, "search", '{"query": "Paris", "k": -1}'))
    assert "non-negative" in envelope["error"]


def test_embed_refuses_a_bare_string() -> None:
    # Before: "abc" embedded each character as its own text.
    envelope = json.loads(TOOLS.call(None, "embed", '{"texts": "abc"}'))
    assert "array of strings" in envelope["error"]
    assert len(host.Corpus.embed(["abc"])) == 1


# ------------------------------------------------------------ interpreter


def test_submit_cannot_be_swallowed_by_interpreted_code() -> None:
    interpreter = rt.ComponentInterpreter(output_fields=[{"name": "answer"}])
    swallowed = "try:\n    SUBMIT(answer=1)\nexcept Exception:\n    pass\n'escaped'"
    assert interpreter.execute(swallowed) == FinalOutput({"answer": 1})
    bare = "try:\n    SUBMIT(answer=2)\nexcept:\n    pass\n'escaped'"
    assert interpreter.execute(bare) == FinalOutput({"answer": 2})
    # SUBMIT ends execution even under 'except Exception': nothing after it runs.
    stops = "try:\n    SUBMIT(answer=3)\nexcept Exception:\n    pass\nran_past_submit = True"
    assert interpreter.execute(stops) == FinalOutput({"answer": 3})
    assert interpreter.execute("globals().get('ran_past_submit', False)") is False
    # A later execution without SUBMIT is unaffected by the earlier one.
    assert interpreter.execute("'next'") == "next"
