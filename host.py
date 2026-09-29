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
import sys
import threading
import urllib.request
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from wasmtime import Config, Engine, Store, Trap, WasiConfig, WasmtimeError
from wasmtime.component import Component, Linker

import limits

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

    def complete(self, store, request_json: str) -> str:
        meter = _meter(store)
        if meter is not None:
            meter.charge_lm()  # over budget raises: the guest call is aborted, not answered
        reply = self._complete(request_json)
        return meter.charge_reply(reply) if meter is not None else reply

    def _complete(self, request_json: str) -> str:
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

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise RuntimeError("provider returned no choices")
        choice = choices[0]
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str):
            raise TypeError("provider returned no text content (refusal or tool-only reply)")
        usage = payload.get("usage") or {}
        return {
            "id": payload.get("id"),
            "model": payload.get("model", model),
            "text": text,
            "finish_reason": choice.get("finish_reason") or "stop",
            "usage": {
                # Providers may send null counts; the component only counts integers.
                "input_tokens": usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
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
MAX_INT_BITS = int(limits.value("max_int_bits"))


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
    try:
        return _arithmetic(ast.parse(expression, mode="eval"))
    except OverflowError as exc:  # float conversion of a large int, or float ** float
        raise ValueError(f"result too large: {exc}") from None


def echo(**kwargs: Any) -> Any:
    """Return the arguments unchanged."""
    return kwargs


def grade_exact(example: dict, prediction: dict, field: str = "answer") -> dict[str, float]:
    """Metric/reward tool: 1.0 when prediction[field] matches example[field]."""
    expected = str(example.get(field, "")).strip().lower()
    actual = str(prediction.get(field, "")).strip().lower()
    return {"score": float(expected == actual)}


# Ceilings on one `embed` call: its output is len(texts) * dimensions floats,
# so the product is bounded too (both axis ceilings together were 41M floats,
# a 205 MB reply).
MAX_EMBED_DIMENSIONS = int(limits.value("max_embed_dimensions"))
MAX_EMBED_TEXTS = int(limits.value("max_embed_texts"))
MAX_EMBED_VALUES = int(limits.value("max_embed_values"))

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
        if not 1 <= dimensions <= MAX_EMBED_DIMENSIONS:
            raise ValueError(f"dimensions must be an integer in [1, {MAX_EMBED_DIMENSIONS}]")
        if len(texts) > MAX_EMBED_TEXTS:
            raise ValueError(f"at most {MAX_EMBED_TEXTS} texts per call")
        if len(texts) * dimensions > MAX_EMBED_VALUES:
            raise ValueError(
                f"texts x dimensions must not exceed {MAX_EMBED_VALUES} values per call"
            )
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

    def call(self, store, name: str, args_json: str) -> str:
        meter = _meter(store)
        if meter is not None:
            meter.charge_tool()
        reply = self._call(name, args_json)
        return meter.charge_reply(reply) if meter is not None else reply

    def _call(self, name: str, args_json: str) -> str:
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
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise argparse.ArgumentTypeError(f"cannot import {module_name!r}: {exc}") from None
    function = getattr(module, attribute, None)
    if function is None:
        raise argparse.ArgumentTypeError(f"{module_name!r} has no attribute {attribute!r}")
    if not callable(function):
        raise argparse.ArgumentTypeError(f"{module_name}:{attribute} is not callable")
    return name, function


# -------------------------------------------------------------------- runtime


# Wall-clock ceiling on one guest call (instantiation or one export call).
# Enforced by Wasmtime epoch interruption, independently of any request-level
# check inside the component: a guest that loops (a bypassed pipeline budget,
# a pathological regex, interpreted code) traps instead of pinning the host.
DEFAULT_DEADLINE_S = float(limits.value("deadline_s"))
# Epoch tick period: deadlines are enforced with this granularity.
EPOCH_TICK_S = 0.01


class DeadlineExceeded(RuntimeError):
    """A guest call ran past its epoch deadline and was interrupted."""


class BudgetExceeded(RuntimeError):
    """A guest call spent more LM calls, tool calls or reply bytes than its budget."""


class Budget(NamedTuple):
    """What one guest call may spend crossing the WIT boundary. ``None`` is unlimited."""

    max_lm_calls: int | None = int(limits.value("max_lm_calls"))
    max_tool_calls: int | None = int(limits.value("max_tool_calls"))
    max_reply_bytes: int | None = int(limits.value("max_reply_bytes"))


class Meter:
    """Counts one guest call's boundary crossings and aborts it past its budget.

    The guest is the thing being contained, so its own request bounds are a
    courtesy; this counter lives in the host, where a guest that bypasses them
    (or a bound nobody thought of) cannot skip it. Exceeding a budget raises
    from the host callback, which traps the guest; ``guest_call`` then reports
    BudgetExceeded. Counters reset at the start of every guest call.
    """

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.lm_calls = self.tool_calls = self.reply_bytes = 0
        self.tripped: BudgetExceeded | None = None

    def reset(self) -> None:
        self.lm_calls = self.tool_calls = self.reply_bytes = 0
        self.tripped = None

    def _trip(self, message: str) -> None:
        self.tripped = BudgetExceeded(message)
        raise self.tripped

    def charge_lm(self) -> None:
        self.lm_calls += 1
        cap = self.budget.max_lm_calls
        if cap is not None and self.lm_calls > cap:
            self._trip(f"guest call exceeded its budget of {cap} LM calls")

    def charge_tool(self) -> None:
        self.tool_calls += 1
        cap = self.budget.max_tool_calls
        if cap is not None and self.tool_calls > cap:
            self._trip(f"guest call exceeded its budget of {cap} tool calls")

    def charge_reply(self, reply: str) -> str:
        self.reply_bytes += len(reply.encode())
        cap = self.budget.max_reply_bytes
        if cap is not None and self.reply_bytes > cap:
            self._trip(f"guest call exceeded its budget of {cap} reply bytes")
        return reply


class _EpochTicker:
    """One daemon thread advancing the epoch of every live engine."""

    def __init__(self, period: float) -> None:
        self.period = period
        self._engines: weakref.WeakSet[Engine] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def register(self, engine: Engine) -> None:
        with self._lock:
            self._engines.add(engine)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="dspy-wasm-epoch", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        event = threading.Event()
        while True:
            event.wait(self.period)
            with self._lock:
                engines = list(self._engines)
            for engine in engines:
                engine.increment_epoch()


_TICKER = _EpochTicker(EPOCH_TICK_S)


def _deadline_ticks(deadline_s: float) -> int:
    if isinstance(deadline_s, bool) or not deadline_s > 0 or not math.isfinite(deadline_s):
        raise ValueError(f"deadline must be a positive finite number of seconds: {deadline_s!r}")
    return max(1, math.ceil(deadline_s / EPOCH_TICK_S))


def new_engine() -> Engine:
    """Engine with epoch interruption on, ticked by the shared ticker."""
    config = Config()
    config.cache = True
    config.epoch_interruption = True
    engine = Engine(config)
    _TICKER.register(engine)
    return engine


def new_store(
    engine: Engine, deadline_s: float = DEFAULT_DEADLINE_S, budget: Budget | None = None
) -> Store:
    """Store whose every guest call is bounded by ``deadline_s`` seconds and ``budget``."""
    store = Store(engine)
    store.dspy_wasm_deadline_s = deadline_s
    store.dspy_wasm_meter = Meter(budget or Budget())
    arm_deadline(store)
    return store


def arm_deadline(store: Store) -> None:
    """Reset the store's epoch deadline to its full budget from now."""
    store.set_epoch_deadline(_deadline_ticks(store.dspy_wasm_deadline_s))


def guest_call(store: Store, func: Callable[..., Any], *args: Any) -> Any:
    """Call into the guest under a freshly armed deadline.

    An epoch interruption surfaces as DeadlineExceeded (core calls raise it
    as a ``Trap``, component calls as a ``WasmtimeError``); every other trap
    or error propagates unchanged.
    """
    arm_deadline(store)
    meter = _meter(store)
    if meter is not None:
        meter.reset()
    try:
        return func(store, *args)
    except (Trap, WasmtimeError) as error:
        if meter is not None and meter.tripped is not None:
            raise meter.tripped from error
        if _is_interrupt(error):
            raise DeadlineExceeded(
                f"guest call exceeded its {store.dspy_wasm_deadline_s:g} s deadline"
            ) from error
        raise


def _meter(store: Any) -> Meter | None:
    return getattr(store, "dspy_wasm_meter", None) if store is not None else None


# Wasmtime's rendering of TrapCode::Interrupt in a component call error.
_INTERRUPT_MESSAGE = "wasm trap: interrupt"


def _is_interrupt(error: Exception) -> bool:
    code = getattr(error, "trap_code", None)
    if code is not None:
        return getattr(code, "name", "") == "INTERRUPT"
    return _INTERRUPT_MESSAGE in str(error)


def instantiate(
    component_path: Path,
    provider: CompletionProvider,
    tools: ToolProvider | None = None,
    deadline_s: float = DEFAULT_DEADLINE_S,
    budget: Budget | None = None,
):
    engine = new_engine()
    store = new_store(engine, deadline_s, budget)
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
            # The callback receives Wasmtime's own context, not our Store: bind ours
            # so the provider can reach the meter.
            lm.add_func("complete", lambda _ctx, request: provider.complete(store, request))
        with root.add_instance("chatman:dspy/tools@0.1.0") as tool_instance:
            tool_instance.add_func("call", lambda _ctx, name, args: tools.call(store, name, args))

    instance = guest_call(store, linker.instantiate, component)
    return store, instance


def call_json(store, instance, export_name: str, *args: str) -> dict[str, Any]:
    func = instance.get_func(store, export_name)
    if func is None:
        raise RuntimeError(f"missing export: {export_name}")
    result = guest_call(store, func, *args)
    if not isinstance(result, str):
        raise TypeError(f"{export_name} returned non-string result")
    return json.loads(result)


class Guest:
    """A component instance with a recycle policy.

    A trapped instance cannot be re-entered (a deadline or budget abort leaves
    ``wasm trap: cannot enter component instance``), and a long-lived one only
    grows. ``Guest`` replaces the instance when a call traps, and after
    ``max_calls`` calls if set. With ``prewarm`` the replacement is instantiated
    on a background thread while the caller carries on, hiding the multi-second
    instantiation from the next request.

    Not thread-safe: one caller at a time, like the instance it wraps.
    """

    def __init__(
        self,
        component_path: Path,
        provider: CompletionProvider,
        tools: ToolProvider | None = None,
        *,
        deadline_s: float = DEFAULT_DEADLINE_S,
        budget: Budget | None = None,
        max_calls: int | None = None,
        prewarm: bool = False,
    ) -> None:
        if max_calls is not None and (
            isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1
        ):
            raise ValueError("max_calls must be a positive integer or None")
        self._make = lambda: instantiate(component_path, provider, tools, deadline_s, budget)
        self.max_calls = max_calls
        self.prewarm = prewarm
        self.instances_created = 0
        self._current: tuple[Any, Any] | None = None
        self._calls = 0
        self._warm: threading.Thread | None = None
        self._warm_result: list[Any] = []

    def _spawn(self) -> None:
        def build() -> None:
            try:
                self._warm_result.append(self._make())
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                self._warm_result.append(exc)

        self._warm_result.clear()
        self._warm = threading.Thread(target=build, name="dspy-wasm-prewarm", daemon=True)
        self._warm.start()

    def _instance(self) -> tuple[Any, Any]:
        if self._current is None:
            if self._warm is not None:
                self._warm.join()
                self._warm = None
                result = self._warm_result.pop()
                if isinstance(result, BaseException):
                    raise result
                self._current = result
            else:
                self._current = self._make()
            self.instances_created += 1
            self._calls = 0
        return self._current

    def _retire(self) -> None:
        self._current = None
        if self.prewarm and self._warm is None:
            self._spawn()

    def call_json(self, export_name: str, *args: str) -> dict[str, Any]:
        store, instance = self._instance()
        try:
            return call_json(store, instance, export_name, *args)
        except (DeadlineExceeded, BudgetExceeded, Trap, WasmtimeError):
            self._retire()  # the instance is dead; the next call gets a fresh one
            raise
        finally:
            self._calls += 1
            if self.max_calls is not None and self._calls >= self.max_calls and self._current:
                self._retire()


def _request(value: str) -> str:
    """Accept inline JSON or ``@path/to/request.json``."""
    try:
        text = Path(value[1:]).read_text() if value.startswith("@") else value
    except OSError as exc:
        raise argparse.ArgumentTypeError(f"cannot read {value[1:]!r}: {exc.strerror}") from None
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from None
    return text


def _strings(value: str) -> list[str]:
    """Inline JSON or ``@file`` holding an array of strings."""
    items = json.loads(_request(value))
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise argparse.ArgumentTypeError("expected a JSON array of strings")
    return items


def _deadline(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(
            f"deadline must be a positive finite number, got {value!r}"
        )
    return seconds


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


REQUEST_EXPORTS = ("run", "render", "evaluate", "compile")


def main() -> None:
    try:
        _main()
    except (DeadlineExceeded, BudgetExceeded) as exc:
        # A guest that ran away is an operational outcome, not a host bug: no traceback.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(3) from None


def _main() -> None:
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
        type=_strings,
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
        type=_strings,
        metavar="JSON|@FILE",
        help="JSON array of passages for the builtin `search`/`embed` tools",
    )
    parser.add_argument(
        "--deadline",
        type=_deadline,
        default=DEFAULT_DEADLINE_S,
        metavar="SECONDS",
        help="wall-clock ceiling on each guest call (Wasmtime epoch interruption)",
    )
    for flag, field, what in (
        ("--max-lm-calls", "max_lm_calls", "LM calls"),
        ("--max-tool-calls", "max_tool_calls", "tool calls"),
        ("--max-reply-bytes", "max_reply_bytes", "bytes returned to the guest"),
    ):
        parser.add_argument(
            flag,
            type=_positive_int,
            default=getattr(Budget(), field),
            metavar="N",
            help=f"budget of {what} per guest call; exceeding it aborts the call",
        )
    parser.add_argument(
        "--conformance",
        action="store_true",
        help="run the conformance suite against the component (needs --response PREDICT_ANSWER)",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--upstream-model", default=os.environ.get("DSPY_WASM_MODEL"))
    args = parser.parse_args()

    scripted = args.responses or None
    # Deterministic static response is the default only for tests/inspection.
    # Explicit real-provider configuration disables it.
    static_response = args.response
    if args.conformance:
        import conformance

        static_response, scripted = conformance.PREDICT_ANSWER, None
    if static_response is None and not args.base_url and not scripted:
        static_response = DEFAULT_RESPONSE

    provider = CompletionProvider(
        static_response=static_response,
        base_url=args.base_url,
        api_key=args.api_key,
        upstream_model=args.upstream_model,
        scripted_responses=scripted,
    )
    corpus = Corpus(args.corpus) if args.corpus else None
    budget = Budget(args.max_lm_calls, args.max_tool_calls, args.max_reply_bytes)
    tools = ToolProvider(dict(args.tool), corpus)

    if args.conformance:
        guest = Guest(args.component, provider, tools, deadline_s=args.deadline, budget=budget)
        report = conformance.run(guest.call_json, include_self_test=args.self_test)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["state"] == "ALIVE" else 1)

    store, instance = instantiate(args.component, provider, tools, args.deadline, budget)

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
        print(f"{export_name}: {guest_call(store, func)}")


if __name__ == "__main__":
    main()
