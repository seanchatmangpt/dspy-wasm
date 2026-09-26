"""Runtime projections that let the whole of DSPy run inside one WASM instance.

The component has no threads, no subprocesses, no filesystem and no network.
DSPy reaches for all four; this module supplies faithful single-instance
equivalents instead of disabling features:

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
from concurrent.futures import Future
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput

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


class _Submission(Exception):
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

    def start(self) -> None:
        if self._namespace is None:
            self._namespace = {"__builtins__": vars(builtins).copy(), "__name__": "__interpreter__"}

    def shutdown(self) -> None:
        self._namespace = None

    def __enter__(self) -> ComponentInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    def _submit(self, *args: Any, **kwargs: Any) -> None:
        if self.output_fields is None:
            if len(args) != 1 or kwargs:
                raise TypeError("SUBMIT requires one output value")
            raise _Submission({"output": args[0]})
        names = [field["name"] for field in self.output_fields]
        if args and kwargs:
            raise TypeError("SUBMIT accepts positional or keyword values, not both")
        values = dict(zip(names, args)) if args else dict(kwargs)
        if set(values) != set(names) or len(args) > len(names):
            raise TypeError("SUBMIT fields do not match the configured output fields")
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
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, "<interpreter>", "exec"), namespace)
                value = (
                    eval(compile(ast.Expression(last.value), "<interpreter>", "eval"), namespace)
                    if last
                    else None
                )
        except _Submission as submission:
            return FinalOutput(_jsonable(submission.value))
        except SyntaxError:
            raise
        except BaseException as exc:  # noqa: BLE001 - interpreted code may raise anything
            raise CodeExecutionError(f"{type(exc).__name__}: {exc}") from exc
        captured = stdout.getvalue().rstrip("\n")
        value = _jsonable(value)
        return value if value is not None else (captured or None)


# ----------------------------------------------------------- numeric projection


class Array:
    """1-D/2-D float array: the numpy surface dspy.Embedder and dspy.KNN use.

    Injected only into ``dspy.clients.embedding`` and ``dspy.predict.knn`` (never
    registered as ``numpy``), so libraries probing for numpy are unaffected.
    """

    __slots__ = ("rows",)

    def __init__(self, data: Any) -> None:
        if isinstance(data, Array):
            data = data.tolist()
        data = list(data)
        self.rows = [
            [float(x) for x in row] if isinstance(row, (list, tuple, Array)) else float(row)
            for row in (r.tolist() if isinstance(r, Array) else r for r in data)
        ]

    @property
    def ndim(self) -> int:
        return 2 if self.rows and isinstance(self.rows[0], list) else 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.rows), len(self.rows[0])) if self.ndim == 2 else (len(self.rows),)

    @property
    def T(self) -> Array:  # noqa: N802 - numpy name
        if self.ndim == 1:
            return Array(self.rows)
        return Array([list(column) for column in zip(*self.rows)])

    def tolist(self) -> list:
        return [list(row) if isinstance(row, list) else row for row in self.rows]

    def astype(self, _dtype: Any) -> Array:
        return Array(self.rows)

    def squeeze(self) -> Array:
        if self.ndim == 2 and self.shape[1] == 1:
            return Array([row[0] for row in self.rows])
        if self.ndim == 2 and self.shape[0] == 1:
            return Array(self.rows[0])
        return Array(self.rows)

    def argsort(self) -> Array:
        order = sorted(range(len(self.rows)), key=lambda index: self.rows[index])
        return _IndexArray(order)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            yield Array(row) if isinstance(row, list) else row

    def __getitem__(self, key: Any) -> Any:
        value = self.rows[key]
        if isinstance(key, slice):
            return Array(value)
        return Array(value) if isinstance(value, list) else value

    def __repr__(self) -> str:
        return f"Array({self.rows!r})"


class _IndexArray(list):
    def __getitem__(self, key: Any) -> Any:
        value = list.__getitem__(self, key)
        return _IndexArray(value) if isinstance(key, slice) else value


def _dot(left: Array, right: Array) -> Array:
    left, right = Array(left), Array(right)
    if left.ndim == 1 and right.ndim == 1:
        return sum(a * b for a, b in zip(left.rows, right.rows))
    columns = right.T.rows if right.ndim == 2 else [right.rows]
    if left.ndim == 1:
        return Array([sum(a * b for a, b in zip(left.rows, column)) for column in columns])
    result = [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left.rows]
    return Array(result if right.ndim == 2 else [row[0] for row in result])


class _Generator:
    """``numpy.random.Generator`` surface SIMBA uses (seeded, deterministic)."""

    def __init__(self, seed: int | None = None) -> None:
        import random

        self._rng = random.Random(seed)

    def poisson(self, lam: float) -> int:
        # Knuth's algorithm; SIMBA draws small rates (demos / max_demos).
        import math

        limit, count, product = math.exp(-lam), 0, self._rng.random()
        while product > limit:
            count += 1
            product *= self._rng.random()
        return count

    def random(self) -> float:
        return self._rng.random()

    def choice(self, options: Any) -> Any:
        return self._rng.choice(list(options))


class _Random:
    def __init__(self) -> None:
        import random

        self._rng = random.Random(0)

    @staticmethod
    def default_rng(seed: int | None = None) -> _Generator:
        return _Generator(seed)

    def rand(self, *shape: int) -> Array:
        if len(shape) == 1:
            return Array([self._rng.random() for _ in range(shape[0])])
        return Array([[self._rng.random() for _ in range(shape[1])] for _ in range(shape[0])])


class numeric:  # noqa: N801 - stands in for the `np` module object
    float32 = "float32"
    ndarray = Array
    random = _Random()

    @staticmethod
    def array(data: Any, dtype: Any = None) -> Array:
        return Array(data)

    asarray = array
    dot = staticmethod(_dot)

    @staticmethod
    def exp(value: float) -> float:
        import math

        return math.exp(value)

    @staticmethod
    def percentile(values: Any, q: float) -> float:
        """numpy's default ('linear') interpolation."""
        ordered = sorted(float(v) for v in values)
        if not ordered:
            raise ValueError("percentile of empty sequence")
        position = (len(ordered) - 1) * q / 100
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ------------------------------------------------------------ search projection


class _Trial:
    def __init__(self, study: Any = None, params: dict | None = None, value: Any = None) -> None:
        self._study = study
        self.number = 0
        self.params = dict(params or {})
        self.value = value

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        choices = list(choices)
        value = self._study._rng.choice(choices)
        self.params[name] = value
        return value


class _Study:
    def __init__(self, direction: str = "maximize", sampler: Any = None) -> None:
        import random

        self.direction = direction
        self.trials: list[_Trial] = []
        self._rng = random.Random(getattr(sampler, "seed", None))

    def add_trial(self, trial: _Trial) -> None:
        trial.number = len(self.trials)
        self.trials.append(trial)

    def optimize(self, objective: Callable[[_Trial], Any], n_trials: int) -> None:
        for _ in range(n_trials):
            trial = _Trial(self)
            trial.number = len(self.trials)
            trial.value = objective(trial)
            self.trials.append(trial)

    @property
    def best_trial(self) -> _Trial:
        scored = [trial for trial in self.trials if trial.value is not None]
        pick = max if self.direction == "maximize" else min
        return pick(scored, key=lambda trial: trial.value)


class search:  # noqa: N801 - stands in for the `optuna` module object
    """Optuna surface MIPROv2 uses, with a *seeded random* categorical sampler.

    Optuna hard-depends on numpy (no WASI build). MIPROv2 only needs a
    categorical search over (instruction, demo-set) candidates; this projection
    keeps MIPROv2's proposal, minibatch and full-eval logic intact and swaps TPE
    for seeded random sampling. Capabilities report the substitution.
    """

    class logging:  # noqa: N801
        WARNING = 30

        @staticmethod
        def set_verbosity(_level: int) -> None:
            return None

    class samplers:  # noqa: N801
        class TPESampler:
            def __init__(self, seed: int | None = None, **_kwargs: Any) -> None:
                self.seed = seed

    class distributions:  # noqa: N801
        class CategoricalDistribution:
            def __init__(self, choices: Any) -> None:
                self.choices = tuple(choices)

    class trial:  # noqa: N801
        Trial = _Trial

        @staticmethod
        def create_trial(*, params: dict, distributions: dict, value: Any) -> _Trial:
            return _Trial(params=params, value=value)

    Study = _Study

    @staticmethod
    def create_study(direction: str = "maximize", sampler: Any = None, **_kwargs: Any) -> _Study:
        return _Study(direction, sampler)


def install_numeric_projections() -> None:
    """Point Embedder/KNN/SIMBA at ``numeric`` and MIPROv2 at ``search``."""
    import dspy.clients.embedding as embedding
    import dspy.predict.knn as knn
    import dspy.teleprompt.mipro_optimizer_v2 as mipro
    import dspy.teleprompt.simba as simba

    embedding.np = numeric
    knn.np = numeric
    simba.np = numeric
    mipro._import_optuna = lambda: search


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
