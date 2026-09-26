"""Adversarial courts for the component boundary.

Each test is a falsifier for a defect found against d346ff9: it failed (or the
input was silently accepted) before the guard it pins. Real collaborators
throughout: the host ToolProvider/CompletionProvider, the capability surface
the component runs, and the deterministic LM doubles in dspy_doubles.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps  # noqa: E402
import dspy_runtime as rt  # noqa: E402
import host  # noqa: E402
from dspy.primitives.code_interpreter import FinalOutput  # noqa: E402
from dspy_doubles import chat, scripted  # noqa: E402

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


def test_unbound_state_with_wrong_field_count_is_refused() -> None:
    state = {k: v for k, v in _compiled().items() if k != caps.SUBJECT_KEY}
    report = guarded(
        caps.run,
        {
            "signature": "context, question -> summary",
            "program_state": state,
            "inputs": {"context": "c", "question": "q"},
        },
        scripted(chat(summary="s")),
    )
    refused(report, "fields; signature has 3")


def test_program_state_must_be_an_object() -> None:
    report = guarded(
        caps.run, {"program_state": [1], "inputs": {"question": "q"}}, scripted(chat(answer="x"))
    )
    refused(report, "program_state must be a JSON object")


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


@pytest.mark.parametrize("expression", ["(-1) ** 0.5", "True + 1", "1j"])
def test_calculator_is_real_arithmetic_only(expression: str) -> None:
    with pytest.raises(ValueError):
        host.calculator(expression)


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
    # A later execution without SUBMIT is unaffected by the earlier one.
    assert interpreter.execute("'next'") == "next"
