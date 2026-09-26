"""Native court for the capability surface (the same code runs inside WASM)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps  # noqa: E402


def chat(**fields: object) -> str:
    body = "".join(
        f"[[ ## {name} ## ]]\n{json.dumps(v) if isinstance(v, (dict, list)) else v}\n\n"
        for name, v in fields.items()
    )
    return body + "[[ ## completed ## ]]"


class ScriptedLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, request_json: str) -> str:
        self.requests.append(json.loads(request_json))
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return json.dumps({"text": text, "usage": {"input_tokens": 3, "output_tokens": 2}})


def tools(name: str, args_json: str) -> str:
    args = json.loads(args_json)
    if name == "add":
        return json.dumps({"result": args["a"] + args["b"]})
    if name == "grader":
        expected = args["example"]["answer"]
        return json.dumps({"result": {"score": float(args["prediction"]["answer"] == expected)}})
    return json.dumps({"error": f"unknown tool {name}"})


def test_run_predict_forwards_lm_config_and_usage() -> None:
    lm = ScriptedLM(chat(answer="Paris"))
    report = caps.run(
        {
            "signature": "question -> answer",
            "inputs": {"question": "Capital of France?"},
            "lm": {"temperature": 0.25, "max_tokens": 64},
            "trace": True,
        },
        lm,
        tools,
    )
    assert report["outputs"] == {"answer": "Paris"}
    assert report["lm_calls"] == 1 and report["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert lm.requests[0]["config"] == {"max_tokens": 64, "temperature": 0.25}
    assert report["history"][0]["response"]["text"] == chat(answer="Paris")


def test_typed_signature_object_with_descriptions_and_batch() -> None:
    lm = ScriptedLM(chat(total=5))
    report = caps.run(
        {
            "signature": {
                "inputs": {"a": {"type": "int", "desc": "left"}, "b": "int"},
                "outputs": {"total": {"type": "int", "desc": "sum"}},
                "instructions": "Add the numbers.",
            },
            "inputs": [{"a": 2, "b": 3}, {"a": 1, "b": 4}],
        },
        lm,
        tools,
    )
    assert report["outputs"] == [{"total": 5}, {"total": 5}]
    assert "Add the numbers." in lm.requests[0]["system"]


def test_react_calls_host_tools() -> None:
    lm = ScriptedLM(
        chat(next_thought="Add them.", next_tool_name="add", next_tool_args={"a": 2, "b": 3}),
        chat(next_thought="Done.", next_tool_name="finish", next_tool_args={}),
        chat(reasoning="2 + 3 = 5", answer="5"),
    )
    report = caps.run(
        {
            "module": "react",
            "signature": "question -> answer",
            "tools": [
                {
                    "name": "add",
                    "description": "Add two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    },
                }
            ],
            "max_iters": 4,
            "inputs": {"question": "What is 2 + 3?"},
        },
        lm,
        tools,
    )
    assert report["outputs"]["answer"] == "5"
    assert report["outputs"]["trajectory"]["observation_0"] == 5


def test_best_of_n_uses_host_tool_reward() -> None:
    lm = ScriptedLM(chat(answer="Lyon"), chat(answer="Paris"))
    report = caps.run(
        {
            "module": "best-of-n",
            "signature": "question -> answer",
            "n": 3,
            "threshold": 1.0,
            "reward": {"tool": "grader", "expected": {"answer": "Paris"}},
            "inputs": {"question": "Capital of France?"},
        },
        lm,
        tools,
    )
    assert report["outputs"]["answer"] == "Paris"


def test_render_formats_without_calling_lm() -> None:
    def no_lm(_: str) -> str:
        raise AssertionError("render must not call the LM")

    report = caps.render(
        {
            "signature": "question -> answer",
            "demos": [{"question": "1+1?", "answer": "2"}],
            "inputs": {"question": "2+2?"},
        },
        no_lm,
        tools,
    )
    roles = [message["role"] for message in report["messages"]]
    assert roles[0] == "system" and roles[-1] == "user" and "assistant" in roles


def test_evaluate_uses_upstream_evaluate_with_failure_score() -> None:
    lm = ScriptedLM(chat(answer="Paris"), chat(answer="Berlin"), "garbage")
    report = caps.evaluate(
        {
            "program": {"signature": "question -> answer"},
            "devset": [
                {"question": "France?", "answer": "Paris"},
                {"question": "Spain?", "answer": "Madrid"},
                {"question": "Italy?", "answer": "Rome"},
            ],
            "metric": {"name": "exact_match", "field": "answer"},
        },
        lm,
        tools,
    )
    assert report["score"] == 33.33
    assert [row["score"] for row in report["results"]] == [1.0, 0.0, 0.0]


def test_compile_round_trips_program_state_into_run() -> None:
    trainset = [{"question": "France?", "answer": "Paris"}, {"question": "Peru?", "answer": "Lima"}]
    compiled = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "labeled-few-shot",
            "trainset": trainset,
            "config": {"k": 2},
        },
        ScriptedLM(chat(answer="unused")),
        tools,
    )
    assert compiled["demos"] == {"self": 2}

    lm = ScriptedLM(chat(answer="Rome"))
    report = caps.run(
        {
            "signature": "question -> answer",
            "program_state": compiled["program_state"],
            "inputs": {"question": "Italy?"},
        },
        lm,
        tools,
    )
    assert report["outputs"]["answer"] == "Rome"
    prompt = json.dumps(lm.requests[0]["messages"])
    assert "Lima" in prompt and "Paris" in prompt


def test_bootstrap_few_shot_keeps_only_metric_passing_traces() -> None:
    lm = ScriptedLM(chat(answer="Paris"), chat(answer="wrong"))
    compiled = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "bootstrap-few-shot",
            "trainset": [
                {"question": "France?", "answer": "Paris"},
                {"question": "Peru?", "answer": "Lima"},
            ],
            "metric": "exact_match",
            "config": {"max_bootstrapped_demos": 2, "max_labeled_demos": 0},
        },
        lm,
        tools,
    )
    demos = compiled["program_state"]["demos"]
    assert [demo["answer"] for demo in demos] == ["Paris"]


def test_failures_are_reported_not_raised() -> None:
    report = json.loads(caps.guarded(lambda: caps.run({"module": "nope"}, ScriptedLM(""), tools)))
    assert report["state"] == "FAILED" and report["error_type"] == "RequestError"

    def broken_lm(_: str) -> str:
        return json.dumps({"error": "quota exhausted"})

    report = json.loads(
        caps.guarded(lambda: caps.run({"inputs": {"question": "q"}}, broken_lm, tools))
    )
    assert report["state"] == "FAILED" and "quota exhausted" in report["message"]


def test_describe_lists_capabilities() -> None:
    described = caps.describe()
    assert "react" in described["modules"] and "bootstrap-few-shot" in described["optimizers"]
    assert "temperature" in described["lm_config"]


def test_refine_accepts_first_attempt_meeting_threshold() -> None:
    lm = ScriptedLM(chat(answer="Paris"))
    report = caps.run(
        {
            "module": "refine",
            "signature": "question -> answer",
            "n": 2,
            "reward": {"name": "exact_match", "expected": {"answer": "Paris"}},
            "inputs": {"question": "Capital of France?"},
        },
        lm,
        tools,
    )
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 1


def test_multi_chain_comparison_samples_m_chains_then_compares() -> None:
    lm = ScriptedLM(
        chat(reasoning="France's capital.", answer="Paris"),
        chat(reasoning="Maybe Lyon.", answer="Lyon"),
        chat(reasoning="Capital city.", answer="Paris"),
        chat(rationale="Two of three chains agree.", answer="Paris"),
    )
    report = caps.run(
        {
            "module": "multi-chain-comparison",
            "signature": "question -> answer",
            "m": 3,
            "inputs": {"question": "Capital of France?"},
        },
        lm,
        tools,
    )
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 4
    assert "Lyon" in json.dumps(lm.requests[-1]["messages"])


@pytest.mark.parametrize(
    ("adapter", "response"),
    [("json", '{"answer": "Paris"}'), ("xml", "<answer>Paris</answer>")],
)
def test_structured_adapters(adapter: str, response: str) -> None:
    report = caps.run(
        {"adapter": adapter, "signature": "question -> answer", "inputs": {"question": "q"}},
        ScriptedLM(response),
        tools,
    )
    assert report["outputs"] == {"answer": "Paris"}, report
