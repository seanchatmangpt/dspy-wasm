"""Native court for the full DSPy surface: pipelines, interpreters, optimizers, types."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps  # noqa: E402
from dspy_doubles import chat, schema_echo, scripted  # noqa: E402

PASSAGES = {
    "who wrote hamlet": ["Hamlet was written by William Shakespeare."],
    "where was shakespeare born": ["Shakespeare was born in Stratford-upon-Avon."],
}


def host_tools(name: str, args_json: str) -> str:
    args = json.loads(args_json)
    if name == "search":
        found = PASSAGES.get(args["query"].lower().rstrip("?"), ["No passage found."])
        return json.dumps({"result": found[: args.get("k", 3)]})
    if name == "add":
        return json.dumps({"result": args["a"] + args["b"]})
    if name == "embed":
        vectors = [
            [float("paris" in t.lower()), float("lima" in t.lower()), 1.0] for t in args["texts"]
        ]
        return json.dumps({"result": vectors})
    if name == "shout":
        return json.dumps({"result": args["text"].upper()})
    return json.dumps({"error": f"unknown tool {name}"})


ADD_TOOL = {
    "name": "add",
    "description": "Add two integers.",
    "parameters": {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    },
}


def multihop_spec() -> dict:
    return {
        "module": "pipeline",
        "retriever": "search",
        "inputs": ["question"],
        "steps": [
            {"set": {"context": []}},
            {
                "repeat": 2,
                "steps": [
                    {
                        "name": "generate_query",
                        "program": {
                            "module": "chain-of-thought",
                            "signature": "context: list[str], question -> query",
                        },
                    },
                    {"retrieve": "$query", "k": 1, "output": "context", "accumulate": ["context"]},
                ],
            },
            {
                "name": "generate_answer",
                "program": {
                    "module": "chain-of-thought",
                    "signature": "context: list[str], question -> answer",
                },
            },
        ],
        "outputs": ["answer", "context"],
    }


def test_multihop_pipeline_with_host_retrieval() -> None:
    lm = scripted(
        chat(reasoning="Find the author.", query="Who wrote Hamlet?"),
        chat(reasoning="Find birthplace.", query="Where was Shakespeare born?"),
        chat(reasoning="Combine.", answer="Stratford-upon-Avon"),
    )
    report = caps.run(
        {**multihop_spec(), "inputs": {"question": "Where was the author of Hamlet born?"}},
        lm,
        host_tools,
    )
    assert report["outputs"]["answer"] == "Stratford-upon-Avon"
    assert report["outputs"]["context"] == [
        "Hamlet was written by William Shakespeare.",
        "Shakespeare was born in Stratford-upon-Avon.",
    ]
    assert report["lm_calls"] == 3


def test_pipeline_is_optimizable_and_state_round_trips() -> None:
    compiled = caps.compile_program(
        {
            "program": multihop_spec(),
            "optimizer": "bootstrap-few-shot",
            "trainset": [
                {"question": "Where was the author of Hamlet born?", "answer": "Stratford"}
            ],
            "metric": "contains",
            "config": {"max_bootstrapped_demos": 1, "max_labeled_demos": 0},
        },
        scripted(
            chat(reasoning="a", query="Who wrote Hamlet?"),
            chat(reasoning="b", query="Where was Shakespeare born?"),
            chat(reasoning="c", answer="Stratford-upon-Avon"),
        ),
        host_tools,
    )
    assert compiled["demos"] == {"generate_query.predict": 1, "generate_answer.predict": 1}
    report = caps.render(
        {
            **multihop_spec(),
            "program_state": compiled["program_state"],
            "inputs": {"question": "q"},
        },
        scripted("unused"),
        host_tools,
    )
    assert [p["name"] for p in report["predictors"]] == [
        "generate_query.predict",
        "generate_answer.predict",
    ]
    assert "Who wrote Hamlet?" in json.dumps(report["messages"])


def test_pipeline_foreach_tool_and_when() -> None:
    report = caps.run(
        {
            "module": "pipeline",
            "steps": [
                {
                    "foreach": "$words",
                    "as": "text",
                    "steps": [{"name": "shout", "tool": "shout", "args": {"text": "$text"}}],
                    "collect": {"shouted": "shout"},
                },
                {"when": "$polite", "set": {"greeting": "Dear {{name}}"}},
                {
                    "name": "summarize",
                    "program": {"signature": "shouted: list[str] -> summary"},
                },
            ],
            "outputs": ["shouted", "summary"],
            "inputs": {"words": ["hi", "there"], "polite": False, "name": "Ada"},
        },
        scripted(chat(summary="HI THERE")),
        host_tools,
    )
    assert report["outputs"] == {"shouted": ["HI", "THERE"], "summary": "HI THERE"}


def test_program_of_thought_runs_code_in_component_interpreter() -> None:
    lm = scripted(
        chat(reasoning="Multiply.", generated_code="```python\nresult = 6 * 7\nprint(result)\n```"),
        chat(reasoning="The code printed 42.", answer="42"),
    )
    report = caps.run(
        {"module": "program-of-thought", "inputs": {"question": "6 * 7?"}, "trace": True},
        lm,
        host_tools,
    )
    assert report["outputs"]["answer"] == "42"
    assert "42" in json.dumps(report["history"][-1]["request"]["messages"])


def test_code_act_calls_host_tools_from_generated_code() -> None:
    lm = scripted(
        chat(generated_code="```python\nprint(add(2, 3))\n```", finished=True),
        chat(reasoning="add returned 5.", answer="5"),
    )
    report = caps.run(
        {
            "module": "code-act",
            "tools": [ADD_TOOL],
            "inputs": {"question": "2 + 3?"},
            "trace": True,
        },
        lm,
        host_tools,
    )
    assert report["outputs"]["answer"] == "5"
    assert "5" in json.dumps(report["history"][-1]["request"]["messages"])


def test_rlm_submits_from_interpreter() -> None:
    lm = scripted(
        chat(reasoning="Compute and submit.", code="```python\nSUBMIT(answer=str(6 * 7))\n```")
    )
    report = caps.run({"module": "rlm", "inputs": {"question": "6 * 7?"}}, lm, host_tools)
    assert report["outputs"]["answer"] == "42"


def test_majority_votes_over_samples() -> None:
    lm = scripted(chat(answer="Paris"), chat(answer="Lyon"), chat(answer="Paris"))
    report = caps.run(
        {"module": "majority", "n": 3, "inputs": {"question": "Capital?"}}, lm, host_tools
    )
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 3


def test_retrieve_module_and_async_run() -> None:
    report = caps.run(
        {
            "module": "retrieve",
            "retriever": "search",
            "k": 1,
            "inputs": {"query": "Who wrote Hamlet?"},
        },
        scripted("unused"),
        host_tools,
    )
    assert report["outputs"]["passages"] == ["Hamlet was written by William Shakespeare."]

    report = caps.run(
        {"inputs": {"question": "q"}, "async": True}, scripted(chat(answer="A")), host_tools
    )
    assert report["outputs"] == {"answer": "A"}


def test_two_step_adapter() -> None:
    lm = scripted("The answer is Paris.", chat(answer="Paris"))
    report = caps.run({"adapter": "two-step", "inputs": {"question": "Capital?"}}, lm, host_tools)
    assert report["outputs"]["answer"] == "Paris" and report["lm_calls"] == 2


def test_image_inputs_cross_as_message_parts() -> None:
    captured: list[dict] = []

    def lm(request_json: str) -> str:
        captured.append(json.loads(request_json))
        return json.dumps({"text": chat(answer="a cat")})

    report = caps.run(
        {
            "signature": "image: Image, question -> answer",
            "inputs": {"image": {"url": "https://example.com/cat.png"}, "question": "What is it?"},
        },
        lm,
        host_tools,
    )
    assert report["outputs"]["answer"] == "a cat"
    parts = [p for m in captured[0]["messages"] for p in m.get("parts", [])]
    assert any(
        p["type"] == "image" and p.get("url") == "https://example.com/cat.png" for p in parts
    )


def test_history_type_for_multi_turn() -> None:
    captured: list[dict] = []

    def lm(request_json: str) -> str:
        captured.append(json.loads(request_json))
        return json.dumps({"text": chat(answer="Your name is Ada.")})

    report = caps.run(
        {
            "signature": "history: History, question -> answer",
            "inputs": {
                "history": {"messages": [{"question": "I am Ada.", "answer": "Hello Ada."}]},
                "question": "What is my name?",
            },
        },
        lm,
        host_tools,
    )
    assert report["outputs"]["answer"] == "Your name is Ada."
    assert "I am Ada." in json.dumps(captured[0]["messages"])


def test_knn_few_shot_with_host_embedder() -> None:
    compiled = caps.compile_program(
        {
            "program": {"signature": "question -> answer", "embedder": "embed"},
            "optimizer": "knn-few-shot",
            "trainset": [
                {"question": "Paris is in?", "answer": "France"},
                {"question": "Lima is in?", "answer": "Peru"},
            ],
            "config": {"k": 1, "max_bootstrapped_demos": 0, "max_labeled_demos": 1},
            "inputs": {"question": "What country has Paris?"},
        },
        schema_echo(answer="France"),
        host_tools,
    )
    assert compiled["outputs"]["answer"] == "France"


TRAINSET = [
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "Capital of France, again?", "answer": "Paris"},
    {"question": "French capital?", "answer": "Paris"},
    {"question": "Paris is the capital of?", "answer": "Paris"},
]


@pytest.mark.parametrize(
    ("optimizer", "config"),
    [
        ("bootstrap-random-search", {"num_candidate_programs": 1, "max_bootstrapped_demos": 1}),
        ("copro", {}),
        ("mipro-v2", {"max_bootstrapped_demos": 1, "max_labeled_demos": 1}),
        ("simba", {}),
        ("gepa", {}),
        ("infer-rules", {"num_candidates": 1, "num_rules": 1, "max_bootstrapped_demos": 1}),
    ],
)
def test_optimizers_compile_end_to_end(optimizer: str, config: dict) -> None:
    report = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": optimizer,
            "trainset": TRAINSET,
            "valset": TRAINSET[:2],
            "metric": "exact_match",
            "config": config,
            "inputs": {"question": "Capital of France?"},
        },
        schema_echo(answer="Paris"),
        host_tools,
    )
    assert report["state"] == "ALIVE", report
    assert report["outputs"]["answer"] == "Paris"
    assert report["program_state"] and report["lm_calls"] > 0


def test_ensemble_majority_over_program_states() -> None:
    state = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "labeled-few-shot",
            "trainset": TRAINSET,
            "config": {"k": 1},
        },
        scripted("unused"),
        host_tools,
    )["program_state"]
    report = caps.compile_program(
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "ensemble",
            "programs": [state, state, state],
            "config": {"reduce": "majority"},
            "inputs": {"question": "Capital?"},
        },
        scripted(chat(answer="Paris"), chat(answer="Lyon"), chat(answer="Paris")),
        host_tools,
    )
    assert report["members"] == 3 and report["outputs"]["answer"] == "Paris"


def test_unsupported_features_fail_with_reasons() -> None:
    report = json.loads(
        caps.guarded(
            lambda: caps.compile_program(
                {"optimizer": "bootstrap-finetune", "trainset": TRAINSET}, scripted(""), host_tools
            )
        )
    )
    assert report["state"] == "FAILED" and "fine-tuning" in report["message"]
    described = caps.describe()
    assert "pipeline" in described["modules"] and "gepa" in described["optimizers"]
