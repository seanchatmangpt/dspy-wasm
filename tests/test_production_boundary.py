"""Falsifiers for boundary defects found against 8321947.

Each test failed (or the input was silently accepted) before the guard it
pins. Chicago-style: real collaborators throughout. The host's real
ToolProvider/CompletionProvider (the provider over a real local HTTP server),
the capability surface the component runs, the host CLI as a subprocess, and
the deterministic LM doubles in dspy_doubles.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps
import host
from dspy_doubles import chat, scripted

ROOT = Path(__file__).resolve().parents[1]
TOOLS = host.ToolProvider()


def host_tools(name: str, args_json: str) -> str:
    return TOOLS.call(None, name, args_json)


def guarded(capability, request, lm) -> dict:
    return json.loads(caps.guarded(lambda: capability(request, lm, host_tools)))


def refused(report: dict, fragment: str) -> None:
    assert report["state"] == "FAILED", report
    assert fragment in report["message"], report["message"]


def failing_lm(_request_json: str) -> str:
    return json.dumps({"error": "provider unavailable"})


QA_TRAINSET = [
    {"question": "capital of France?", "answer": "Paris"},
    {"question": "capital of Peru?", "answer": "Lima"},
]


# ------------------------------------------------------ optimizer options


def test_labeled_few_shot_sample_option_reaches_compile_not_constructor() -> None:
    # Before: {"sample": false} went to LabeledFewShot.__init__ -> TypeError.
    report = guarded(
        caps.compile_program,
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "labeled-few-shot",
            "config": {"k": 1, "sample": False},
            "trainset": QA_TRAINSET,
        },
        scripted(chat(answer="Paris")),
    )
    assert report["state"] == "ALIVE", report
    # sample=False takes the first k in order, deterministically.
    demos = report["program_state"]["demos"]
    assert [d["question"] for d in demos] == ["capital of France?"]


def test_bootstrap_optuna_max_demos_option_reaches_compile_not_constructor() -> None:
    # Before: {"max_demos": 1} went to the constructor -> TypeError.
    report = guarded(
        caps.compile_program,
        {
            "program": {"signature": "question -> answer"},
            "optimizer": "bootstrap-optuna",
            "config": {"max_demos": 1, "num_candidate_programs": 1},
            "metric": "exact_match",
            "trainset": QA_TRAINSET,
        },
        scripted(chat(answer="Paris")),
    )
    assert report["state"] == "ALIVE", report


# --------------------------------------------------------- batch failures


def test_batch_run_reports_failed_items_instead_of_alive_nulls() -> None:
    # Before: {"state": "ALIVE", "outputs": [null, null]} with no error text.
    report = guarded(
        caps.run,
        {"signature": "question -> answer", "inputs": [{"question": "a"}, {"question": "b"}]},
        failing_lm,
    )
    refused(report, "2 of 2 batch items failed")
    assert report["outputs"] == [None, None]
    assert [e["index"] for e in report["errors"]] == [0, 1]
    assert all("provider unavailable" in e["message"] for e in report["errors"])


def test_batch_run_all_success_stays_alive_without_errors() -> None:
    report = guarded(
        caps.run,
        {"signature": "question -> answer", "inputs": [{"question": "a"}, {"question": "b"}]},
        scripted(chat(answer="x")),
    )
    assert report["state"] == "ALIVE", report
    assert [o["answer"] for o in report["outputs"]] == ["x", "x"]
    assert "errors" not in report


# ---------------------------------------------------------- strict JSON out


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_outputs_never_cross_as_non_standard_json(value: str) -> None:
    # Before: the envelope carried bare NaN/Infinity, which strict JSON parsers
    # (and the host's own tool path) reject.
    text = caps.guarded(
        lambda: caps.run(
            {"signature": "question -> answer: float", "inputs": {"question": "q"}},
            scripted(chat(answer=value)),
            host_tools,
        )
    )
    report = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard {c}"))
    refused(report, "not JSON compliant")


# ---------------------------------------------------------- request shapes


@pytest.mark.parametrize("input_keys", ["question", [1], {"question": 1}])
def test_input_keys_must_be_a_list_of_strings(input_keys) -> None:
    # Before: "question" became with_inputs('q','u',...) and scored ALIVE.
    report = guarded(
        caps.evaluate,
        {
            "program": {"signature": "question -> answer"},
            "devset": QA_TRAINSET,
            "metric": "exact_match",
            "input_keys": input_keys,
        },
        scripted(chat(answer="Paris")),
    )
    refused(report, "'input_keys' must be an array of strings")


@pytest.mark.parametrize("step", [{"repeat": 2}, {"foreach": [1, 2]}, {"repeat": 2, "steps": None}])
def test_loop_steps_without_a_body_run_as_empty_loops(step) -> None:
    # Before: bare KeyError: 'steps' although the static check accepted it.
    report = guarded(
        caps.run,
        {"module": "pipeline", "steps": [step], "inputs": {"question": "q"}},
        scripted("unused"),
    )
    assert report["state"] == "ALIVE", report


# ------------------------------------------------------------ module bounds


@pytest.mark.parametrize(
    "spec",
    [
        {"module": "majority", "n": 10**9},
        {"module": "best-of-n", "n": 10**9},
        {"module": "refine", "n": 10**9},
        {"module": "multi-chain-comparison", "m": 10**9},
        {"module": "react", "max_iters": 10**9},
        {"module": "react-v2", "max_iters": 10**9},
        {"module": "program-of-thought", "max_iters": 10**9},
        {"module": "code-act", "max_iters": 10**9},
        {"module": "rlm", "max_iters": 10**9},
        {"module": "rlm", "max_llm_calls": 10**9},
        {"module": "retrieve", "k": 10**9},
        {"module": "majority", "n": 0},
        {"module": "react", "max_iters": "7"},
        {"module": "best-of-n", "n": True},
    ],
)
def test_module_fanout_is_bounded_before_any_lm_call(spec: dict) -> None:
    # Before: n=1e9 majority made ~800 LM calls/s until the host deadline.
    calls = []

    def counting_lm(request_json: str) -> str:
        calls.append(request_json)
        return scripted(chat(answer="x"))(request_json)

    report = guarded(
        caps.run,
        {"signature": "question -> answer", **spec, "inputs": {"question": "q"}},
        counting_lm,
    )
    refused(report, "integer in [")
    assert calls == []


def test_module_limits_are_published_in_capabilities() -> None:
    limits = caps.describe()["module_limits"]
    assert limits["max_fanout"] == caps.MAX_FANOUT
    assert limits["max_iters"] == caps.MAX_ITERS


def test_bounded_module_at_its_ceiling_is_still_accepted() -> None:
    report = guarded(
        caps.run,
        {
            "module": "majority",
            "n": 3,
            "signature": "question -> answer",
            "inputs": {"question": "q"},
        },
        scripted(chat(answer="x")),
    )
    assert report["state"] == "ALIVE", report


# ------------------------------------------------- in-component retriever k


def test_embeddings_retriever_k_zero_returns_no_passages() -> None:
    # Before: k=0 fell through `k or len(...)` and returned every passage,
    # disagreeing with the host `search` tool.
    session = caps.Session(
        {"retriever": {"corpus": list(host.DEFAULT_CORPUS), "embedder": "embed", "k": 3}},
        scripted("unused"),
        host_tools,
    )
    assert session.retriever("capital of France", k=0) == []
    assert len(session.retriever("capital of France", k=2)) == 2
    assert len(session.retriever("capital of France")) == 3


# ---------------------------------------------------------------- host tools


@pytest.mark.parametrize(
    "args",
    [
        {"texts": ["a"], "dimensions": caps_dims}
        for caps_dims in (host.MAX_EMBED_DIMENSIONS + 1, 5_000_000)
    ]
    + [{"texts": ["a"] * (host.MAX_EMBED_TEXTS + 1)}],
)
def test_embed_refuses_unbounded_output(args: dict) -> None:
    # Before: 3 texts x 5e6 dimensions took 32 s and 612 MB.
    envelope = json.loads(host_tools("embed", json.dumps(args)))
    assert "ValueError" in envelope["error"], envelope


def test_embed_at_its_ceiling_is_accepted() -> None:
    envelope = json.loads(
        host_tools(
            "embed", json.dumps({"texts": ["a b c"], "dimensions": host.MAX_EMBED_DIMENSIONS})
        )
    )
    assert len(envelope["result"][0]) == host.MAX_EMBED_DIMENSIONS


def test_embed_bounds_the_product_not_only_each_axis() -> None:
    # Before: each axis was capped, the product was not. The two ceilings
    # together made 41M floats: a 205 MB reply after 18 s.
    args = {"texts": ["a"] * host.MAX_EMBED_TEXTS, "dimensions": host.MAX_EMBED_DIMENSIONS}
    envelope = json.loads(host_tools("embed", json.dumps(args)))
    assert "ValueError" in envelope["error"], str(envelope)[:200]


def test_embed_at_its_value_ceiling_is_accepted() -> None:
    dimensions = host.MAX_EMBED_DIMENSIONS
    texts = ["a b c"] * (host.MAX_EMBED_VALUES // dimensions)
    envelope = json.loads(
        host_tools("embed", json.dumps({"texts": texts, "dimensions": dimensions}))
    )
    assert len(envelope["result"]) == len(texts)
    over = json.loads(
        host_tools("embed", json.dumps({"texts": texts + ["x"], "dimensions": dimensions}))
    )
    assert "ValueError" in over["error"], over


@pytest.mark.parametrize(
    "expression", ["(2**63)**64 * 1.0", "(2**63)**64 / 3", "(2**63)**64 ** 1.0", "10.0**400"]
)
def test_calculator_float_overflow_is_a_value_error(expression: str) -> None:
    # Before: OverflowError, against the documented "every refusal is ValueError".
    with pytest.raises(ValueError):
        host.calculator(expression)


# -------------------------------------------- real provider over real HTTP


class _Provider(BaseHTTPRequestHandler):
    status: ClassVar[int] = 200
    payload: ClassVar[dict] = {}
    seen: ClassVar[list] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        type(self).seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length)),
            }
        )
        data = json.dumps(type(self).payload).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def provider_server():
    handler = type("Handler", (_Provider,), {"status": 200, "payload": {}, "seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _complete(base_url: str, request: dict | None = None) -> dict:
    provider = host.CompletionProvider(
        static_response=None, base_url=base_url, api_key="sk-test", upstream_model="gpt-test"
    )
    request = request or {
        "model": "wasm-host",
        "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}],
        "config": {"temperature": 0.1, "max_tokens": 5},
    }
    return json.loads(provider.complete(None, json.dumps(request)))


def test_http_provider_round_trip_forwards_config_and_key(provider_server) -> None:
    handler, url = provider_server
    handler.payload = {
        "id": "c1",
        "model": "gpt-test-2026",
        "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
    }
    response = _complete(url)
    assert response["text"] == "Paris"
    assert response["usage"] == {"input_tokens": 7, "output_tokens": 1, "total_tokens": 8}
    (seen,) = handler.seen
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen["body"]["temperature"] == 0.1


def test_http_provider_null_usage_counts_do_not_fail_a_completion(provider_server) -> None:
    # Before: {"prompt_tokens": null} crossed as None and HostEngine's int(None)
    # turned a successful completion into FAILED.
    handler, url = provider_server
    handler.payload = {
        "choices": [{"message": {"content": "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"}}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }
    provider = host.CompletionProvider(
        static_response=None, base_url=url, api_key=None, upstream_model="gpt-test"
    )
    report = guarded(
        caps.run,
        {"signature": "question -> answer", "inputs": {"question": "q"}},
        lambda request_json: provider.complete(None, request_json),
    )
    assert report["state"] == "ALIVE", report
    assert report["outputs"]["answer"] == "Paris"


def test_component_engine_treats_null_usage_as_not_reported() -> None:
    # Malformed counts still fail closed (test_hardening); null is "unknown".
    def null_usage_lm(_request_json: str) -> str:
        return json.dumps(
            {"text": chat(answer="x"), "usage": {"input_tokens": None, "output_tokens": 2}}
        )

    report = guarded(
        caps.run, {"signature": "question -> answer", "inputs": {"question": "q"}}, null_usage_lm
    )
    assert report["state"] == "ALIVE", report


@pytest.mark.parametrize(
    ("status", "payload", "fragment"),
    [
        (500, {"error": {"message": "overloaded"}}, "HTTPError"),
        (200, {"choices": [{"message": {"content": None}}]}, "no text content"),
        (200, {"choices": []}, "no choices"),
    ],
)
def test_http_provider_failures_cross_as_error_envelopes(
    provider_server, status, payload, fragment
) -> None:
    handler, url = provider_server
    handler.status, handler.payload = status, payload
    response = _complete(url)
    assert set(response) == {"error"}
    assert fragment in response["error"], response


# ----------------------------------------------------------------- host CLI


def _cli(*args: str) -> subprocess.CompletedProcess:
    # A missing component path: every refusal below happens during argument
    # parsing, before the component would be loaded.
    return subprocess.run(
        [sys.executable, str(ROOT / "host.py"), "dist/absent.wasm", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (["--run", "@/nonexistent/request.json"], "cannot read"),
        (["--run", "{not json"], "not valid JSON"),
        (["--tool", "x=no_such_module_xyz:f"], "cannot import"),
        (["--tool", "x=json:no_such_function"], "has no attribute"),
        (["--tool", "x=json:__name__"], "not callable"),
        (["--corpus", '"Paris"'], "array of strings"),
        (["--corpus", "[1, 2]"], "array of strings"),
        (["--responses", '{"a": 1}'], "array of strings"),
        (["--deadline", "nan"], "positive finite"),
        (["--deadline", "0"], "positive finite"),
        (["--deadline", "-5"], "positive finite"),
    ],
)
def test_cli_refuses_bad_arguments_as_usage_errors(args: list[str], fragment: str) -> None:
    # Before: FileNotFoundError/ModuleNotFoundError/AttributeError/ValueError
    # tracebacks, and --corpus '"Paris"' silently became ['P','a','r','i','s'].
    result = _cli(*args)
    assert result.returncode == 2, result
    assert "Traceback" not in result.stderr, result.stderr
    assert fragment in result.stderr, result.stderr


def test_capabilities_module_imports_optuna_cleanly_in_a_fresh_process() -> None:
    # Before: dspy's lazy numpy proxy made the first optuna import in a fresh
    # native process fail with "data type 'bool' not understood", so a
    # bootstrap-optuna compile failed unless something imported optuna first.
    code = "import dspy, dspy_capabilities, optuna.samplers; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.stdout.strip() == "ok", result.stderr[-2000:]


def test_cli_deadline_accepts_positive_finite_seconds() -> None:
    assert host._deadline("2.5") == 2.5
    assert not math.isnan(host._deadline("1"))


# ------------------------------------------------------- runtime: forward check


def test_direct_forward_still_warns_and_call_does_not(caplog) -> None:
    import dspy

    module = dspy.Predict("question -> answer")
    with caplog.at_level("WARNING", logger="dspy.primitives.module"):
        module.forward  # noqa: B018 - the lookup is what warns
        assert "directly is discouraged" in caplog.text
        caplog.clear()
        with dspy.context(lm=dspy.utils.DummyLM([{"answer": "x"}])):
            module(question="q")
        assert "directly is discouraged" not in caplog.text


def test_module_call_work_does_not_scale_with_stack_depth() -> None:
    # Before: dspy's forward check ran inspect.stack(), resolving and reading
    # source for every frame on each module call: 2-3x slower from a
    # 200-frame stack (nested pipeline steps, optimizers). Counted, not
    # timed: the number of `inspect` calls one module call makes must not
    # depend on how deep the caller's stack is.
    import inspect

    request = {"signature": "question -> answer", "inputs": {"question": "q"}}

    def once() -> None:
        caps.run(request, scripted(chat(answer="x")), host_tools)

    def deep(n: int) -> None:
        return once() if n == 0 else deep(n - 1)

    def inspect_calls(fn) -> int:
        count = 0

        def profiler(frame, event, _arg) -> None:
            nonlocal count
            if event == "call" and frame.f_code.co_filename == inspect.__file__:
                count += 1

        fn()  # warm caches so both measurements see the same state
        sys.setprofile(profiler)
        try:
            fn()
        finally:
            sys.setprofile(None)
        return count

    shallow, deep_calls = inspect_calls(once), inspect_calls(lambda: deep(300))
    assert deep_calls == shallow, (shallow, deep_calls)
