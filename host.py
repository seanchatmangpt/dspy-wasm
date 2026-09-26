"""Wasmtime host for dspy-wasm.

The host owns provider/network authority and tool authority. The WASM
component owns DSPy program semantics and reaches the outside world only
through the typed WIT imports:

- ``chatman:dspy/lm@0.1.0``    ``complete(request-json) -> response-json``
- ``chatman:dspy/tools@0.1.0`` ``call(name, args-json) -> envelope-json``

Host callbacks never raise into the component: a Python exception inside a
Wasmtime callback is an unrecoverable trap, so failures cross the boundary as
``{"error": "..."}`` and surface in DSPy as ordinary exceptions.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import operator
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wasmtime import Config, Engine, Store, WasiConfig
from wasmtime.component import Component, Linker

DEFAULT_RESPONSE = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"

# Deterministic provider config fields forwarded to OpenAI-compatible APIs.
_OPENAI_CONFIG = {
    "max_tokens": "max_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "stop": "stop",
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "response_format": "response_format",
}


def _openai_content(message: dict[str, Any]) -> Any:
    """Plain text, or an OpenAI content array when multimodal parts are present."""
    parts = message.get("parts")
    if not parts:
        return message.get("text", "")
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image":
            url = part.get("url") or f"data:{part.get('media_type')};base64,{part.get('data')}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "audio" and part.get("data"):
            fmt = str(part.get("media_type", "audio/wav")).split("/")[-1]
            content.append(
                {"type": "input_audio", "input_audio": {"data": part["data"], "format": fmt}}
            )
        else:
            content.append({"type": "text", "text": json.dumps(part)})
    return content


class CompletionProvider:
    def __init__(
        self,
        *,
        static_response: str | None,
        base_url: str | None,
        api_key: str | None,
        upstream_model: str | None,
        scripted_responses: list[str] | None = None,
    ) -> None:
        if scripted_responses is not None and (
            not isinstance(scripted_responses, list)
            or not all(isinstance(text, str) for text in scripted_responses)
        ):
            raise TypeError("scripted responses must be a JSON array of strings")
        self.static_response = static_response
        self.scripted_responses = list(scripted_responses or [])
        self.base_url = base_url
        self.api_key = api_key
        self.upstream_model = upstream_model

    def complete(self, _store, request_json: str) -> str:
        try:
            request = json.loads(request_json)
            if self.scripted_responses:
                text = (
                    self.scripted_responses.pop(0)
                    if len(self.scripted_responses) > 1
                    else self.scripted_responses[0]
                )
                return json.dumps({"text": text, "model": "scripted", "finish_reason": "stop"})
            if self.static_response is not None:
                return json.dumps(
                    {
                        "text": self.static_response,
                        "model": self.upstream_model or request.get("model", "static"),
                        "finish_reason": "stop",
                    }
                )
            return json.dumps(self._http_complete(request))
        except Exception as exc:  # noqa: BLE001 - a callback must not trap the component
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _http_complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("host LM requires --response or --base-url/OPENAI_BASE_URL")

        model = self.upstream_model or request.get("model")
        if not model or model == "wasm-host":
            raise RuntimeError("real provider mode requires --upstream-model")

        messages: list[dict[str, str]] = []
        if request.get("system"):
            messages.append({"role": "system", "content": request["system"]})
        for message in request.get("messages", []):
            messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": _openai_content(message),
                }
            )

        body: dict[str, Any] = {"model": model, "messages": messages}
        for source, target in _OPENAI_CONFIG.items():
            if source in request.get("config", {}):
                body[target] = request["config"][source]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        http_request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(http_request, timeout=120) as response:
            payload = json.loads(response.read())

        choice = payload["choices"][0]
        usage = payload.get("usage") or {}
        return {
            "id": payload.get("id"),
            "model": payload.get("model", model),
            "text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }


# ------------------------------------------------------------------ host tools

_ARITHMETIC: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# Largest integer magnitude (in bits) the calculator will produce. Without it
# nested powers such as ((10**64)**64)**64 grow the host's memory and time
# doubly exponentially while every single exponent stays within bounds.
MAX_INT_BITS = 4096


def _bounded(value: Any) -> Any:
    if isinstance(value, complex):
        # Every calculator refusal is a ValueError (one error contract).
        raise ValueError("result is not a real number")  # noqa: TRY004
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError("result too large")
    # inf/nan have no JSON form: json.dumps would emit non-standard Infinity/NaN.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result is not finite")
    return value


def _arithmetic(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _bounded(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left, right = _arithmetic(node.left), _arithmetic(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 64:
                raise ValueError("exponent too large")
            # |left ** right| < 2 ** (bit_length(left) * right): refuse before computing.
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and left.bit_length() * right > MAX_INT_BITS
            ):
                raise ValueError("result too large")
        return _bounded(_ARITHMETIC[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC:
        return _bounded(_ARITHMETIC[type(node.op)](_arithmetic(node.operand)))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> Any:
    """Evaluate a pure arithmetic expression (no names, calls or attributes)."""
    return _arithmetic(ast.parse(expression, mode="eval"))


def echo(**kwargs: Any) -> Any:
    """Return the arguments unchanged."""
    return kwargs


def grade_exact(example: dict, prediction: dict, field: str = "answer") -> dict[str, float]:
    """Metric/reward tool: 1.0 when prediction[field] matches example[field]."""
    expected = str(example.get(field, "")).strip().lower()
    actual = str(prediction.get(field, "")).strip().lower()
    return {"score": float(expected == actual)}


DEFAULT_CORPUS = (
    "Hamlet was written by William Shakespeare.",
    "Shakespeare was born in Stratford-upon-Avon.",
    "Paris is the capital of France.",
    "Lima is the capital of Peru.",
    "WebAssembly components communicate through WIT interfaces.",
)


def _terms(text: str) -> list[str]:
    return [
        t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2
    ]


class Corpus:
    """Deterministic lexical retriever and embedder over an in-memory corpus."""

    def __init__(self, passages: list[str] | tuple[str, ...] = DEFAULT_CORPUS) -> None:
        self.passages = list(passages)

    def search(self, query: str, k: int = 3) -> list[str]:
        """Retriever tool: passages ranked by query-term overlap."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be a non-negative integer")
        if k < 0:
            raise ValueError("k must be a non-negative integer")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if k == 0:
            return []
        wanted = set(_terms(query))
        ranked = sorted(
            self.passages, key=lambda p: (-len(wanted & set(_terms(p))), self.passages.index(p))
        )
        return [p for p in ranked[: int(k)] if wanted & set(_terms(p))] or ranked[:1]

    @staticmethod
    def embed(texts: list[str], dimensions: int = 64) -> list[list[float]]:
        """Embedder tool: L2-normalised hashed bag-of-words vectors."""
        import hashlib

        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a JSON array of strings")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be a positive integer")
        if dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        vectors = []
        for text in texts:
            vector = [0.0] * dimensions
            for term in _terms(text):
                vector[int(hashlib.sha256(term.encode()).hexdigest(), 16) % dimensions] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def builtin_tools(corpus: Corpus | None = None) -> dict[str, Callable[..., Any]]:
    corpus = corpus or Corpus()
    return {
        "calculator": calculator,
        "echo": echo,
        "grade_exact": grade_exact,
        "search": corpus.search,
        "embed": corpus.embed,
    }


class ToolProvider:
    """Host-owned tool registry backing ``chatman:dspy/tools.call``."""

    def __init__(
        self, tools: dict[str, Callable[..., Any]] | None = None, corpus: Corpus | None = None
    ) -> None:
        self.tools = builtin_tools(corpus)
        self.tools.update(tools or {})

    def call(self, _store, name: str, args_json: str) -> str:
        try:
            tool = self.tools.get(name)
            if tool is None:
                raise KeyError(f"unknown host tool {name!r}; available: {sorted(self.tools)}")
            args = json.loads(args_json)
            if not isinstance(args, dict):
                raise TypeError("tool arguments must be a JSON object")
            # allow_nan=False: a non-finite tool result is an error, never Infinity/NaN.
            return json.dumps({"result": tool(**args)}, default=str, allow_nan=False)
        except Exception as exc:  # noqa: BLE001 - a callback must not trap the component
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def load_tool(spec: str) -> tuple[str, Callable[..., Any]]:
    """Parse ``NAME=package.module:function``."""
    name, _, target = spec.partition("=")
    module_name, _, attribute = target.partition(":")
    if not name or not module_name or not attribute:
        raise argparse.ArgumentTypeError("--tool expects NAME=package.module:function")
    return name, getattr(importlib.import_module(module_name), attribute)


# -------------------------------------------------------------------- runtime


def instantiate(
    component_path: Path,
    provider: CompletionProvider,
    tools: ToolProvider | None = None,
):
    config = Config()
    config.cache = True
    engine = Engine(config)
    store = Store(engine)
    # WASI grants clocks and entropy only: no preopened directories, no
    # environment, no argv, no network grants. Provider authority stays here.
    wasi = WasiConfig()
    wasi.inherit_stderr()
    store.set_wasi(wasi)
    component = Component.from_file(engine, str(component_path))
    linker = Linker(engine)
    linker.add_wasip2()
    tools = tools or ToolProvider()

    with linker.root() as root:
        with root.add_instance("chatman:dspy/lm@0.1.0") as lm:
            lm.add_func("complete", provider.complete)
        with root.add_instance("chatman:dspy/tools@0.1.0") as tool_instance:
            tool_instance.add_func("call", tools.call)

    instance = linker.instantiate(store, component)
    return store, instance


def call_json(store, instance, export_name: str, *args: str) -> dict[str, Any]:
    func = instance.get_func(store, export_name)
    if func is None:
        raise RuntimeError(f"missing export: {export_name}")
    result = func(store, *args)
    if not isinstance(result, str):
        raise TypeError(f"{export_name} returned non-string result")
    return json.loads(result)


def _request(value: str) -> str:
    """Accept inline JSON or ``@path/to/request.json``."""
    text = Path(value[1:]).read_text() if value.startswith("@") else value
    json.loads(text)
    return text


REQUEST_EXPORTS = ("run", "render", "evaluate", "compile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument("--predict-signature")
    parser.add_argument("--inputs", default="{}")
    for export_name in REQUEST_EXPORTS:
        parser.add_argument(
            f"--{export_name}",
            type=_request,
            metavar="JSON|@FILE",
            help=f"invoke the `{export_name}` export with a request object",
        )
    parser.add_argument("--response")
    parser.add_argument(
        "--responses",
        type=_request,
        metavar="JSON|@FILE",
        help="JSON array of LM responses returned in order (the last one repeats)",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        type=load_tool,
        metavar="NAME=MODULE:FUNC",
        help="grant the component a host Python function as a tool (repeatable)",
    )
    parser.add_argument(
        "--corpus",
        type=_request,
        metavar="JSON|@FILE",
        help="JSON array of passages for the builtin `search`/`embed` tools",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--upstream-model", default=os.environ.get("DSPY_WASM_MODEL"))
    args = parser.parse_args()

    scripted = json.loads(args.responses) if args.responses else None
    # Deterministic static response is the default only for tests/inspection.
    # Explicit real-provider configuration disables it.
    static_response = args.response
    if static_response is None and not args.base_url and not scripted:
        static_response = DEFAULT_RESPONSE

    provider = CompletionProvider(
        static_response=static_response,
        base_url=args.base_url,
        api_key=args.api_key,
        upstream_model=args.upstream_model,
        scripted_responses=scripted,
    )
    corpus = Corpus(json.loads(args.corpus)) if args.corpus else None
    store, instance = instantiate(args.component, provider, ToolProvider(dict(args.tool), corpus))

    def emit(report: dict[str, Any]) -> None:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report.get("state") == "ALIVE" else 1)

    if args.self_test:
        emit(call_json(store, instance, "run-self-tests"))

    if args.capabilities:
        emit(call_json(store, instance, "capabilities"))

    if args.predict_signature:
        emit(call_json(store, instance, "predict", args.predict_signature, args.inputs))

    for export_name in REQUEST_EXPORTS:
        request = getattr(args, export_name)
        if request is not None:
            emit(call_json(store, instance, export_name, request))

    for export_name in ("component-version", "runtime-info", "dspy-version"):
        func = instance.get_func(store, export_name)
        if func is None:
            raise RuntimeError(f"missing export: {export_name}")
        print(f"{export_name}: {func(store)}")


if __name__ == "__main__":
    main()
