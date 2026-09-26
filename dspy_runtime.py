"""Runtime policy that lets the whole of DSPy run inside one WASM instance.

Every dependency is the real library compiled to WebAssembly (see wasi/). What
remains here is policy for what a component genuinely lacks: threads,
subprocesses, a filesystem and network. DSPy reaches for all four; this module
supplies faithful single-instance equivalents instead of disabling features:

- **Sequential execution.** ``ParallelExecutor`` already owns a sequential,
  context-propagating path for ``num_threads == 1``; every executor is pinned
  to it. Direct ``ThreadPoolExecutor`` users (GEPA, RLM, AvatarOptimizer) get
  ``SequentialExecutor``, which runs each submission to completion and returns
  a finished ``Future`` (so ``as_completed``/``wait``/``result`` all behave).
- **In-component code interpreter.** ProgramOfThought, CodeAct, RLM and Flex
  default to a Deno/Pyodide subprocess. Inside WASM the component *is* the
  sandbox, so ``ComponentInterpreter`` executes code in an isolated namespace
  with the exact outcome semantics of DSPy's ``LocalInterpreter`` worker:
  ``FinalOutput`` on ``SUBMIT``, last-expression value or captured stdout,
  ``SyntaxError`` and ``CodeExecutionError`` for failures.
- **Inline batching.** ``Unbatchify`` (behind ``dspy.retrievers.Embeddings``)
  coalesces calls on a worker thread; with one caller it runs inline.
- **Event loop without a self-pipe.** asyncio's cross-thread wake-up channel
  needs ``socketpair()``, which WASI lacks; with no threads it is unused.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import concurrent.futures
import contextlib
import io
import json
import keyword
import sys
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput

if TYPE_CHECKING:
    from typing_extensions import Self

# --------------------------------------------------------------- sequential


class SequentialExecutor(concurrent.futures.Executor):
    """``ThreadPoolExecutor`` projection for a runtime without threads."""

    def __init__(self, max_workers: int | None = None, *args: Any, **kwargs: Any) -> None:
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - Future carries it, as a pool would
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._shutdown = True


class SequentialUnbatchify:
    """``dspy.utils.unbatchify.Unbatchify`` without its worker thread.

    With a single caller the worker always collects exactly one pending item
    and calls ``batch_fn([item])``; doing that inline is equivalent.
    """

    def __init__(
        self,
        batch_fn: Callable[[list[Any]], list[Any]],
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
    ) -> None:
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._closed = False

    def __call__(self, input_item: Any) -> Any:
        if self._closed:
            raise RuntimeError("Unbatchify is closed")
        return self.batch_fn([input_item])[0]

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


_INSTALLED = False


def install_sequential_runtime() -> None:
    """Pin all DSPy parallelism to its sequential path. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    from dspy.utils import parallelizer

    original_init = parallelizer.ParallelExecutor.__init__

    def sequential_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.num_threads = 1

    parallelizer.ParallelExecutor.__init__ = sequential_init
    concurrent.futures.ThreadPoolExecutor = SequentialExecutor

    # Bind the submodules themselves (their Unbatchify globals are patched).
    import dspy.retrievers.embeddings as embeddings  # noqa: PLR0402
    import dspy.utils.unbatchify as unbatchify  # noqa: PLR0402

    unbatchify.Unbatchify = embeddings.Unbatchify = SequentialUnbatchify
    for module_name in (
        "dspy.predict.rlm",
        "dspy.teleprompt.avatar_optimizer",
        "dspy.teleprompt.gepa.gepa_utils",
        "dspy.streaming.messages",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "ThreadPoolExecutor"):
            module.ThreadPoolExecutor = SequentialExecutor
    _INSTALLED = True


# -------------------------------------------------------------- interpreter


class _Submission(BaseException):
    """Raised by SUBMIT. A ``BaseException`` so ``except Exception`` in
    interpreted code cannot swallow it; ``ComponentInterpreter`` also records
    the submission, so even a bare ``except:`` cannot discard it."""

    def __init__(self, value: Any) -> None:
        super().__init__("SUBMIT")
        self.value = value


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


class ComponentInterpreter:
    """``dspy.CodeInterpreter`` executing inside the component's own interpreter.

    ``bridge`` (name, kwargs) -> result is exposed to executed code as
    ``__host_tool__`` so host-tool source injected by CodeAct can call out.
    """

    execution_instructions = (
        "Code runs in a sandboxed CPython inside a WebAssembly component: the standard "
        "library is available, but there is no filesystem, network or subprocess access."
    )

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        bridge: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.tools: dict[str, Callable[..., Any]] = dict(tools or {})
        self.output_fields = output_fields
        self.bridge = bridge
        self._tools_registered = False
        self._namespace: dict[str, Any] | None = None
        self._submitted: dict[str, Any] | None = None

    def start(self) -> None:
        if self._namespace is None:
            self._namespace = {"__builtins__": vars(builtins).copy(), "__name__": "__interpreter__"}

    def shutdown(self) -> None:
        self._namespace = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    def _submit(self, *args: Any, **kwargs: Any) -> None:
        if self.output_fields is None:
            if len(args) != 1 or kwargs:
                raise TypeError("SUBMIT requires one output value")
            self._submitted = {"output": args[0]}
            raise _Submission(self._submitted)
        names = [field["name"] for field in self.output_fields]
        if args and kwargs:
            raise TypeError("SUBMIT accepts positional or keyword values, not both")
        values = dict(zip(names, args)) if args else dict(kwargs)
        if set(values) != set(names) or len(args) > len(names):
            raise TypeError("SUBMIT fields do not match the configured output fields")
        self._submitted = values
        raise _Submission(values)

    def _configure(self) -> dict[str, Any]:
        self.start()
        namespace = self._namespace
        assert namespace is not None
        for name in self.tools:
            if not name.isidentifier() or keyword.iskeyword(name) or name == "SUBMIT":
                raise CodeInterpreterError(f"tool name {name!r} is not a usable identifier")
        namespace.update(self.tools)
        namespace["SUBMIT"] = self._submit
        if self.bridge is not None:
            namespace["__host_tool__"] = self.bridge
        self._tools_registered = True
        return namespace

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        if not isinstance(code, str):
            raise CodeInterpreterError("code must be a string")
        variables = variables or {}
        reserved = {"SUBMIT", "__builtins__", "__host_tool__", *self.tools}
        clashes = [
            name
            for name in variables
            if not isinstance(name, str) or not name.isidentifier() or name in reserved
        ]
        if clashes:
            raise CodeInterpreterError(f"invalid variable names: {clashes!r}")
        namespace = self._configure()
        namespace.update(variables)

        tree = ast.parse(code, mode="exec")  # SyntaxError propagates, as the protocol requires
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        stdout = io.StringIO()
        self._submitted = None
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, "<interpreter>", "exec"), namespace)  # noqa: S102 - the interpreter
                value = (
                    eval(compile(ast.Expression(last.value), "<interpreter>", "eval"), namespace)
                    if last
                    else None
                )
        except _Submission as submission:
            return FinalOutput(_jsonable(submission.value))
        except SyntaxError:
            raise
        except BaseException as exc:  # interpreted code may raise anything
            if self._submitted is not None:  # SUBMIT ran, then code raised past it
                return FinalOutput(_jsonable(self._submitted))
            raise CodeExecutionError(f"{type(exc).__name__}: {exc}") from exc
        if self._submitted is not None:  # SUBMIT ran and interpreted code swallowed it
            return FinalOutput(_jsonable(self._submitted))
        captured = stdout.getvalue().rstrip("\n")
        value = _jsonable(value)
        return value if value is not None else (captured or None)


# ------------------------------------------------------------------ event loop


class ComponentEventLoop(asyncio.SelectorEventLoop):
    """asyncio loop without the self-pipe.

    The stock selector loop opens a ``socketpair()`` purely so other threads
    can wake it; WASI (rightly) grants no sockets and there are no threads, so
    the wake-up channel is omitted. Timers and ready callbacks run unchanged.
    """

    def _make_self_pipe(self) -> None:
        self._ssock = self._csock = None
        self._internal_fds = 0

    def _close_self_pipe(self) -> None:
        return None

    def _write_to_self(self) -> None:
        return None


def run_async(coroutine: Any) -> Any:
    loop = ComponentEventLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()
