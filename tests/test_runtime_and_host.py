"""Runtime projections (interpreter, executor, numeric, event loop) and host tools."""

from __future__ import annotations

import json
from concurrent.futures import as_completed

import pytest

pytest.importorskip("dspy")

import dspy  # noqa: E402
import dspy_runtime as rt  # noqa: E402
import host  # noqa: E402
from dspy.primitives.code_interpreter import CodeExecutionError, FinalOutput  # noqa: E402


def test_component_interpreter_matches_local_interpreter_outcomes() -> None:
    interpreter = rt.ComponentInterpreter(output_fields=[{"name": "answer"}])
    assert isinstance(interpreter, dspy.CodeInterpreter)
    assert interpreter.execute("x = 20\nx + 1") == 21
    assert interpreter.execute("print(x * 2)") == "40"  # state persists across calls
    assert interpreter.execute("SUBMIT(answer=x)") == FinalOutput({"answer": 20})
    assert interpreter.execute("y = 1") is None
    with pytest.raises(SyntaxError):
        interpreter.execute("def (")
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute("1 / 0")
    with pytest.raises(CodeExecutionError, match="SUBMIT fields"):
        interpreter.execute("SUBMIT(wrong=1)")


def test_interpreter_tools_and_host_bridge() -> None:
    calls = []
    interpreter = rt.ComponentInterpreter(
        tools={"double": lambda n: 2 * n}, bridge=lambda name, args: calls.append((name, args)) or 7
    )
    assert interpreter.execute("double(4)") == 8
    assert interpreter.execute("__host_tool__('lookup', {'q': 1})") == 7
    assert calls == [("lookup", {"q": 1})]


def test_sequential_executor_is_a_drop_in_pool() -> None:
    with rt.SequentialExecutor(max_workers=8) as pool:
        futures = [pool.submit(lambda i=i: i * i) for i in range(4)]
        failing = pool.submit(lambda: 1 / 0)
    assert sorted(f.result() for f in as_completed(futures)) == [0, 1, 4, 9]
    assert isinstance(failing.exception(), ZeroDivisionError)
    assert list(rt.SequentialExecutor().map(str, [1, 2])) == ["1", "2"]


def test_parallel_executor_pinned_to_sequential_path() -> None:
    rt.install_sequential_runtime()
    from dspy.utils.parallelizer import ParallelExecutor

    assert ParallelExecutor(num_threads=16).num_threads == 1


def test_numeric_projection_matches_numpy_semantics() -> None:
    np = rt.numeric
    matrix = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32).astype(np.float32)
    scores = np.dot(matrix, np.array([[1, 0]]).T).squeeze()
    assert scores.tolist() == [1.0, 0.0, 1.0]
    assert list(scores.argsort()[-2:][::-1]) == [2, 0]
    assert np.percentile([1, 2, 3, 4], 10) == pytest.approx(1.3)
    assert np.percentile([1, 2, 3, 4], 90) == pytest.approx(3.7)
    draws = [np.random.default_rng(0).poisson(0.5) for _ in range(3)]
    assert len(set(draws)) == 1  # seeded


def test_search_projection_drives_categorical_study() -> None:
    study = rt.search.create_study(
        direction="maximize", sampler=rt.search.samplers.TPESampler(seed=1)
    )
    study.add_trial(rt.search.trial.create_trial(params={}, distributions={}, value=0.1))
    study.optimize(
        lambda trial: {"a": 1.0, "b": 0.5}[trial.suggest_categorical("x", ["a", "b"])], 5
    )
    assert [t.number for t in study.trials] == list(range(6))
    assert study.best_trial.value == 1.0


def test_event_loop_runs_without_self_pipe() -> None:
    async def work() -> int:
        import asyncio

        await asyncio.sleep(0)
        return 5

    assert rt.run_async(work()) == 5


def test_host_calculator_is_arithmetic_only() -> None:
    assert host.calculator("2 ** 10 - (3 * 4) / 2") == 1018
    for hostile in ("__import__('os')", "open('x')", "(1).__class__", "9 ** 999"):
        with pytest.raises(ValueError):
            host.calculator(hostile)


def test_host_tool_provider_envelopes() -> None:
    tools = host.ToolProvider()
    assert json.loads(tools.call(None, "calculator", '{"expression": "6 * 7"}')) == {"result": 42}
    assert "unknown host tool" in json.loads(tools.call(None, "missing", "{}"))["error"]
    assert "error" in json.loads(tools.call(None, "calculator", "[1]"))
    graded = json.loads(
        tools.call(
            None,
            "grade_exact",
            json.dumps({"example": {"answer": "Paris"}, "prediction": {"answer": " paris "}}),
        )
    )
    assert graded == {"result": {"score": 1.0}}


def test_host_corpus_search_and_embed() -> None:
    corpus = host.Corpus()
    assert corpus.search("Where was Shakespeare born?", k=1) == [
        "Shakespeare was born in Stratford-upon-Avon."
    ]
    left, right = corpus.embed(["capital of France", "France capital"])
    assert sum(a * b for a, b in zip(left, right)) == pytest.approx(1.0)


def test_host_provider_errors_cross_as_envelopes() -> None:
    provider = host.CompletionProvider(
        static_response=None, base_url=None, api_key=None, upstream_model=None
    )
    assert "requires --response" in json.loads(provider.complete(None, "{}"))["error"]
    scripted = host.CompletionProvider(
        static_response=None,
        base_url=None,
        api_key=None,
        upstream_model=None,
        scripted_responses=["a", "b"],
    )
    assert [json.loads(scripted.complete(None, "{}"))["text"] for _ in range(3)] == ["a", "b", "b"]


def test_openai_content_projection_for_multimodal_parts() -> None:
    content = host._openai_content(
        {
            "text": "x",
            "parts": [
                {"type": "text", "text": "look"},
                {"type": "image", "url": "https://i/c.png"},
            ],
        }
    )
    assert content == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://i/c.png"}},
    ]
