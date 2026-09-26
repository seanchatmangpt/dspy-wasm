"""DSPy capability surface exposed across the WIT boundary.

Every capability is driven by a JSON *program spec*, so the host can compose
and run any upstream DSPy program without Python-level access:

    {
      "module": one of MODULES (default "predict"),
      "signature": "question -> answer"  |  {"inputs": {...}, "outputs": {...}},
      "instructions": "...",
      "adapter": "chat" | "json" | "xml" | "two-step",
      "lm": {"model": "...", "temperature": 0.0, "max_tokens": 512, "cache": false, ...},
      "demos": [{...}],                 # few-shot demos for every predictor
      "program_state": {...},           # `compile` output, loaded verbatim
      "tools": [{"name", "description", "parameters"}],   # react/react-v2/code-act/rlm
      ...module-specific keys, see build_module()
    }

Pipelines compose programs, host tools and retrieval into one optimizable
``dspy.Module`` (see ``Pipeline``). Types such as ``Image``, ``Audio``,
``File``, ``History``, ``Code``, ``Reasoning`` and ``ToolCalls`` may be used
in signature strings.

Both external authorities stay with the host and are injected here as plain
callables, so this module runs identically in WASM and in native tests:

- ``lm_call(request_json) -> response_json``      (``chatman:dspy/lm``)
- ``tool_call(name, args_json) -> envelope_json`` (``chatman:dspy/tools``)

Retrieval (``dspy.Retrieve``) and embeddings (``dspy.Embedder``) are host
tools too: ``{"retriever": "<tool>"}`` / ``{"embedder": "<tool>"}``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import keyword
import linecache
import re
from typing import Any, Callable

import dspy
import pydantic
from dspy.dsp.utils import dotdict
from dspy.evaluate.metrics import EM, F1, normalize_text
from dspy.lm15 import Config, Message, Response, Usage

import dspy_runtime

LmCall = Callable[[str], str]
ToolCall = Callable[[str, str], str]

MODULES = (
    "predict",
    "chain-of-thought",
    "react",
    "react-v2",
    "program-of-thought",
    "code-act",
    "rlm",
    "best-of-n",
    "refine",
    "multi-chain-comparison",
    "majority",
    "retrieve",
    "pipeline",
)
ADAPTERS = ("chat", "json", "xml", "two-step")
METRICS = ("exact_match", "f1", "contains", "passage_match", "semantic-f1", "<host tool>")
OPTIMIZERS = (
    "labeled-few-shot",
    "bootstrap-few-shot",
    "bootstrap-random-search",
    "knn-few-shot",
    "copro",
    "mipro-v2",
    "simba",
    "gepa",
    "infer-rules",
    "ensemble",
)
UNSUPPORTED = {
    "optimizer:bootstrap-finetune": "weight training needs a provider fine-tuning authority",
    "optimizer:better-together": "composes bootstrap-finetune",
    "optimizer:grpo": "weight training needs a provider fine-tuning authority",
    "optimizer:bootstrap-optuna": "optuna TPE with numpy has no WASI build",
    "optimizer:avatar": "operates only on the deprecated Avatar module",
    "module:flex": "optimizer-authored sandbox programs; not yet projected",
    "retriever:embeddings": "dspy.retrievers.Embeddings needs numpy einsum/faiss; use a host retriever",
    "streaming": "WIT calls are synchronous; use run and read the full result",
}
CUSTOM_TYPES = {
    name: getattr(dspy, name)
    for name in ("Image", "Audio", "File", "History", "Code", "Reasoning", "ToolCalls")
    if hasattr(dspy, name)
}
_USAGE_FIELDS = tuple(field.name for field in dataclasses.fields(Usage))
_REF = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


class HostError(RuntimeError):
    """The host reported a failure through a WIT JSON envelope."""


class RequestError(ValueError):
    """The caller's request is malformed."""


def json_default(value: Any) -> Any:
    if isinstance(value, dspy_runtime.Array):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "toDict"):
        return value.toDict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, default=json_default, sort_keys=True)


# --------------------------------------------------------------------------- LM


def _config_payload(config: Any) -> dict[str, Any]:
    if config is None or not dataclasses.is_dataclass(config):
        return {}
    payload = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if value is None or value == () or value == {}:
            continue
        try:
            json.dumps(value)
        except TypeError:
            value = json.loads(json.dumps(value, default=json_default))
        payload[field.name] = value
    return payload


def _part_payload(part: Any) -> dict[str, Any]:
    payload = {"type": getattr(part, "type", type(part).__name__)}
    if dataclasses.is_dataclass(part):
        for field in dataclasses.fields(part):
            value = getattr(part, field.name)
            if field.name not in ("continuation", "type") and value not in (None, ()):
                payload[field.name] = value
    return json.loads(json.dumps(payload, default=json_default))


def _message_payload(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "text": message.text}
    parts = getattr(message, "parts", ()) or ()
    if any(getattr(part, "type", "text") != "text" for part in parts):
        payload["parts"] = [_part_payload(part) for part in parts]
    return payload


class HostEngine:
    """DSPy 3.4 custom engine backed by the WIT ``lm.complete`` capability.

    Request:  {"model", "system", "messages": [{"role", "text", "parts"?}], "config"}
    Response: {"text", "id"?, "model"?, "finish_reason"?, "usage"?} or {"error"}

    ``parts`` appears only for multimodal messages (image/audio/file parts).
    """

    def __init__(self, lm_call: LmCall) -> None:
        self.lm_call = lm_call
        self.ledger: list[dict[str, Any]] = []

    def __deepcopy__(self, memo: dict) -> HostEngine:
        # dspy.LM.copy() deep-copies (BestOfN, Refine, optimizer rounds). The
        # engine holds only the host capability and the boundary ledger, both
        # of which must stay shared so every crossing is accounted for.
        return self

    def complete(self, request):
        payload = {
            "model": request.model,
            "system": request.system,
            "messages": [_message_payload(message) for message in request.messages],
            "config": _config_payload(getattr(request, "config", None)),
        }
        raw = json.loads(self.lm_call(json.dumps(payload, sort_keys=True, default=str)))
        self.ledger.append({"request": payload, "response": raw})
        if isinstance(raw, dict) and raw.get("error") is not None:
            raise HostError(f"host LM failed: {raw['error']}")
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise TypeError("host LM must return a JSON object containing string field 'text'")
        usage = raw.get("usage") or {}
        return Response(
            id=raw.get("id"),
            model=raw.get("model") or request.model,
            message=Message.assistant(raw["text"]),
            finish_reason=raw.get("finish_reason", "stop"),
            usage=Usage(**{k: int(v) for k, v in usage.items() if k in _USAGE_FIELDS}),
        )


class AsyncHostEngine:
    """Async counterpart for ``acall``: the WIT call is synchronous, so the
    coroutine completes on the component's single event loop."""

    def __init__(self, engine: HostEngine) -> None:
        self.engine = engine

    def __deepcopy__(self, memo: dict) -> AsyncHostEngine:
        return self

    async def complete(self, request):
        return self.engine.complete(request)


def build_lm(lm_spec: dict[str, Any] | None, engine: HostEngine) -> dspy.LM:
    options = dict(lm_spec or {})
    model = options.pop("model", "wasm-host")
    options.setdefault("cache", False)
    options.setdefault("num_retries", 0)
    return dspy.LM(model, engine=engine, async_engine=AsyncHostEngine(engine), **options)


# ------------------------------------------------------------------------ tools


class HostTools:
    """The ``chatman:dspy/tools`` capability, shaped for DSPy's call sites."""

    def __init__(self, tool_call: ToolCall) -> None:
        self.tool_call = tool_call

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Envelope: {"result": <json>} or {"error": "<message>"}."""
        raw = json.loads(
            self.tool_call(name, json.dumps(args, default=json_default, sort_keys=True))
        )
        if not isinstance(raw, dict):
            raise HostError(f"host tool {name!r} returned a non-envelope value")
        if raw.get("error") is not None:
            raise HostError(f"host tool {name!r} failed: {raw['error']}")
        return raw.get("result")

    def function(self, spec: dict[str, Any]) -> Callable[..., Any]:
        """A real, named Python function whose source is registered.

        CodeAct re-executes ``inspect.getsource(tool.func)`` inside its
        interpreter, so the function body calls the ``__host_tool__`` bridge
        that ``ComponentInterpreter`` provides rather than a closure.
        """
        name = spec.get("name")
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise RequestError(f"tool name {name!r} must be a Python identifier")
        properties = (spec.get("parameters") or {}).get("properties", {})
        required = set((spec.get("parameters") or {}).get("required", properties))
        for arg in properties:
            if not arg.isidentifier() or keyword.iskeyword(arg):
                raise RequestError(f"tool {name!r} argument {arg!r} must be a Python identifier")
        params = [a for a in properties if a in required] + [
            f"{a}=None" for a in properties if a not in required
        ]
        packed = ", ".join(f"{a!r}: {a}" for a in properties)
        source = (
            f"def {name}({', '.join(params)}):\n"
            f"    {json.dumps(spec.get('description', ''))}\n"
            f"    args = {{{packed}}}\n"
            f"    return __host_tool__({name!r}, {{k: v for k, v in args.items() if v is not None}})\n"
        )
        filename = f"<host-tool:{name}>"
        lines = source.splitlines(keepends=True)
        linecache.cache[filename] = (len(source), None, lines, filename)
        namespace: dict[str, Any] = {"__host_tool__": self.call}
        exec(compile(source, filename, "exec"), namespace)
        return namespace[name]

    def tool(self, spec: dict[str, Any]) -> dspy.Tool:
        properties = (spec.get("parameters") or {}).get("properties", {})
        return dspy.Tool(
            self.function(spec),
            name=spec["name"],
            desc=spec.get("description", ""),
            args=properties,
            arg_types={arg: Any for arg in properties},
            arg_desc={
                arg: schema["description"]
                for arg, schema in properties.items()
                if isinstance(schema, dict) and "description" in schema
            },
        )

    def retriever(self, name: str) -> Callable[..., list]:
        """``dspy.settings.rm`` backed by a host tool returning passages."""

        def rm(query: str, k: int = 3, **kwargs: Any) -> list:
            passages = self.call(name, {"query": query, "k": k, **kwargs})
            return [
                dotdict(p if isinstance(p, dict) else {"long_text": str(p)}) for p in passages or []
            ]

        return rm

    def embedder(self, name: str) -> dspy.Embedder:
        """``dspy.Embedder`` backed by a host tool: {"texts": [...]} -> [[float]]."""

        def embed(texts: list[str]) -> list[list[float]]:
            return self.call(name, {"texts": list(texts)})

        return dspy.Embedder(embed, caching=False)


# ------------------------------------------------------------ signatures/types


def _field_type(field: Any) -> str:
    if isinstance(field, dict):
        return field.get("type", "str")
    return field if isinstance(field, str) else "str"


def build_signature(spec: Any, instructions: str | None = None) -> type[dspy.Signature]:
    if isinstance(spec, str):
        signature = dspy.Signature(spec, custom_types=CUSTOM_TYPES)
    elif isinstance(spec, dict):
        inputs, outputs = spec.get("inputs") or {}, spec.get("outputs") or {}
        if not inputs or not outputs:
            raise RequestError("signature objects need non-empty 'inputs' and 'outputs'")
        text = " -> ".join(
            ", ".join(f"{name}: {_field_type(field)}" for name, field in fields.items())
            for fields in (inputs, outputs)
        )
        signature = dspy.Signature(text, custom_types=CUSTOM_TYPES)
        for name, field in {**inputs, **outputs}.items():
            if isinstance(field, dict) and field.get("desc"):
                signature = signature.with_updated_fields(name, desc=field["desc"])
        instructions = instructions or spec.get("instructions")
    else:
        raise RequestError("'signature' must be a string or an object")
    if instructions:
        signature = signature.with_instructions(instructions)
    return signature


def coerce_inputs(signature: type[dspy.Signature] | None, inputs: dict[str, Any]) -> dict[str, Any]:
    """Validate JSON inputs into rich field types (dspy.Image, dspy.History, ...)."""
    if signature is None:
        return inputs
    coerced = dict(inputs)
    for name, field in signature.input_fields.items():
        annotation = field.annotation
        value = coerced.get(name)
        if (
            value is not None
            and isinstance(annotation, type)
            and issubclass(annotation, pydantic.BaseModel)
            and not isinstance(value, annotation)
        ):
            coerced[name] = annotation.model_validate(value)
    return coerced


# ---------------------------------------------------------------------- metrics


def make_metric(spec: Any, tools: HostTools) -> Callable[..., float]:
    """Metric registry. ``{"tool": name}`` delegates scoring to a host tool.

    Metrics accept DSPy's extended call shapes (``trace``, and GEPA's
    ``pred_name``/``pred_trace``) and ignore what they do not use.
    """
    if spec is None:
        spec = "exact_match"
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise RequestError("metric must be a name or an object")

    if "tool" in spec:
        tool = spec["tool"]

        def host_metric(example, prediction, *args: Any, **kwargs: Any) -> float:
            result = tools.call(
                tool,
                {
                    "example": dict(example.items()),
                    "prediction": prediction.toDict()
                    if hasattr(prediction, "toDict")
                    else prediction,
                },
            )
            if isinstance(result, dict):
                if "feedback" in result and kwargs.get("pred_name") is not None:
                    return dspy.Prediction(
                        score=float(result["score"]), feedback=result["feedback"]
                    )
                result = result.get("score")
            return float(result)

        return host_metric

    name, field = spec.get("name", "exact_match"), spec.get("field", "answer")

    def fields(example, prediction) -> tuple[str, str]:
        return str(prediction[field]), str(example[field])

    if name in ("exact_match", "f1"):
        score = EM if name == "exact_match" else F1

        def overlap(example, prediction, *args: Any, **kwargs: Any) -> float:
            predicted, expected = fields(example, prediction)
            return float(score(predicted, [expected]))

        return overlap
    if name == "contains":

        def contains(example, prediction, *args: Any, **kwargs: Any) -> float:
            predicted, expected = fields(example, prediction)
            return float(normalize_text(expected) in normalize_text(predicted))

        return contains
    if name == "passage_match":

        def passage_match(example, prediction, *args: Any, **kwargs: Any) -> float:
            return float(dspy.evaluate.answer_passage_match(example, prediction))

        return passage_match
    if name == "semantic-f1":
        judge = dspy.evaluate.SemanticF1(decompositional=bool(spec.get("decompositional", False)))

        def semantic_f1(example, prediction, *args: Any, **kwargs: Any) -> float:
            return float(judge(example, prediction))

        return semantic_f1
    raise RequestError(f"unknown metric {name!r}; expected one of {METRICS}")


def _reward(spec: dict[str, Any], tools: HostTools) -> Callable[[dict, Any], float]:
    reward = spec.get("reward")
    if reward is None:
        raise RequestError(f"{spec.get('module')} needs a 'reward' metric spec")
    metric = make_metric(reward, tools)
    expected = reward.get("expected", {}) if isinstance(reward, dict) else {}

    def reward_fn(args: dict, prediction) -> float:
        return metric(dspy.Example(**args, **expected), prediction)

    return reward_fn


# ---------------------------------------------------------------------- modules


class MultiChainProgram(dspy.Module):
    """Sample M reasoning chains, then compare them (dspy.MultiChainComparison)."""

    def __init__(self, signature: type[dspy.Signature], m: int = 3) -> None:
        super().__init__()
        self.m = m
        self.chain = dspy.ChainOfThought(signature)
        self.compare = dspy.MultiChainComparison(signature, M=m)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        completions = [
            self.chain(**kwargs, config={"rollout_id": index, "temperature": 1.0})
            for index in range(self.m)
        ]
        return self.compare(completions=completions, **kwargs)


class MajorityProgram(dspy.Module):
    """Sample a base module N times and return the majority answer (dspy.majority)."""

    def __init__(self, base: dspy.Module, n: int, field: str | None) -> None:
        super().__init__()
        self.base = base
        self.n = n
        self.field = field

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        samples = []
        for index in range(self.n):
            with dspy.context(lm=dspy.settings.lm.copy(rollout_id=index, temperature=1.0)):
                samples.append(self.base(**kwargs))
        completions = dspy.Prediction.from_completions([s.toDict() for s in samples])
        return dspy.majority(completions, field=self.field)


class RetrieveProgram(dspy.Module):
    """dspy.Retrieve over the session's host-backed retriever."""

    def __init__(self, k: int, query_field: str) -> None:
        super().__init__()
        self.query_field = query_field
        self.retrieve = dspy.Retrieve(k=k)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.retrieve(kwargs[self.query_field])


def _identifier(name: str) -> str:
    cleaned = re.sub(r"\W", "_", name)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"


class Pipeline(dspy.Module):
    """A composed program: modules, host tools and retrieval over shared state.

    Spec::

        {"module": "pipeline",
         "inputs": ["question"],                  # optional, informational
         "steps": [<step>, ...],
         "outputs": ["answer"]}                   # default: every produced key

    Steps (all accept ``"when": "$flag"`` to run conditionally):

    - program:  {"name", "program": <program spec>, "inputs"?: {field: expr},
                 "outputs"?: {state_key: field}, "accumulate"?: [state_key, ...]}
    - tool:     {"name", "tool": "<host tool>", "args": {arg: expr}, "output"?: key}
    - retrieve: {"name", "retrieve": expr, "k"?: 3, "output"?: "passages"}
    - set:      {"set": {state_key: expr}}
    - repeat:   {"repeat": n, "steps": [...], "until"?: expr}
    - foreach:  {"foreach": expr, "as": "item", "steps": [...],
                 "collect": {state_key: inner_key}}

    Expressions: ``"$key.path"`` references state; strings may interpolate
    ``{{key}}``; anything else is a literal. A program step reused across
    ``repeat`` iterations is one predictor, so optimizers tune it once (the
    multi-hop pattern). Every program step is a named sub-module, so the whole
    pipeline compiles, dumps and loads state like any DSPy program.
    """

    def __init__(self, spec: dict[str, Any], factory: Callable[[dict[str, Any]], dspy.Module]):
        super().__init__()
        self.outputs = spec.get("outputs")
        self.steps = spec.get("steps") or []
        if not self.steps:
            raise RequestError("pipeline needs non-empty 'steps'")
        self._inputs_for: dict[str, list[str] | None] = {}
        self._register(self.steps, factory)

    def _register(self, steps: list[dict[str, Any]], factory) -> None:
        for step in steps:
            if "program" in step:
                name = step.get("name")
                if not name:
                    raise RequestError("program steps need a 'name'")
                attribute = _identifier(name)
                if attribute not in self._inputs_for:
                    program = step["program"]
                    setattr(self, attribute, factory(program))
                    self._inputs_for[attribute] = program_inputs(program)
            for key in ("steps",):
                if key in step:
                    self._register(step[key], factory)

    @staticmethod
    def _lookup(state: dict[str, Any], path: str) -> Any:
        value: Any = state
        for part in path.split("."):
            if isinstance(value, dict):
                value = value[part]
            elif isinstance(value, (list, tuple)):
                value = value[int(part)]
            else:
                value = getattr(value, part)
        return value

    def _resolve(self, expr: Any, state: dict[str, Any]) -> Any:
        if isinstance(expr, str):
            if expr.startswith("$$"):
                return expr[1:]
            if expr.startswith("$"):
                return self._lookup(state, expr[1:])
            return _REF.sub(lambda m: str(self._lookup(state, m.group(1))), expr)
        if isinstance(expr, list):
            return [self._resolve(item, state) for item in expr]
        if isinstance(expr, dict):
            return {key: self._resolve(value, state) for key, value in expr.items()}
        return expr

    def _store(self, state: dict[str, Any], key: str, value: Any, accumulate: bool) -> None:
        if accumulate:
            existing = list(state.get(key) or [])
            state[key] = existing + (list(value) if isinstance(value, (list, tuple)) else [value])
        else:
            state[key] = value

    def _execute(self, steps: list[dict[str, Any]], state: dict[str, Any]) -> None:
        for step in steps:
            if "when" in step and not self._resolve(step["when"], state):
                continue
            accumulate = set(step.get("accumulate") or [])
            if "program" in step:
                attribute = _identifier(step["name"])
                module = getattr(self, attribute)
                if "inputs" in step:
                    kwargs = self._resolve(step["inputs"], state)
                else:
                    names = self._inputs_for[attribute]
                    names = list(state) if names is None else names
                    missing = [n for n in names if n not in state]
                    if missing:
                        raise RequestError(f"step {step['name']!r} is missing inputs {missing}")
                    kwargs = {n: state[n] for n in names}
                produced = module(**kwargs).toDict()
                renames = step.get("outputs") or {key: key for key in produced}
                for key, field in renames.items():
                    self._store(state, key, produced[field], key in accumulate)
            elif "tool" in step:
                result = self._tools.call(step["tool"], self._resolve(step.get("args", {}), state))
                key = step.get("output", step.get("name", step["tool"]))
                self._store(state, key, result, key in accumulate)
            elif "retrieve" in step:
                query = self._resolve(step["retrieve"], state)
                passages = dspy.Retrieve(k=int(step.get("k", 3)))(query).passages
                key = step.get("output", "passages")
                self._store(state, key, passages, key in accumulate)
            elif "set" in step:
                for key, expr in step["set"].items():
                    self._store(state, key, self._resolve(expr, state), key in accumulate)
            elif "repeat" in step:
                for _ in range(int(step["repeat"])):
                    self._execute(step["steps"], state)
                    if "until" in step and self._resolve(step["until"], state):
                        break
            elif "foreach" in step:
                collect = step.get("collect") or {}
                gathered: dict[str, list] = {key: [] for key in collect}
                for item in self._resolve(step["foreach"], state):
                    inner = {**state, step.get("as", "item"): item}
                    self._execute(step["steps"], inner)
                    for key, inner_key in collect.items():
                        gathered[key].append(inner[inner_key])
                state.update(gathered)
            else:
                raise RequestError(f"unrecognised pipeline step: {sorted(step)}")

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        state = dict(kwargs)
        self._execute(self.steps, state)
        keys = self.outputs or [key for key in state if key not in kwargs]
        return dspy.Prediction(**{key: state[key] for key in keys})


def program_inputs(spec: dict[str, Any]) -> list[str] | None:
    """Input field names a program spec consumes; None means 'whole state'."""
    kind = spec.get("module", "predict")
    if kind == "pipeline":
        return spec.get("inputs")
    if kind == "retrieve":
        return [spec.get("query_field", "query")]
    if kind == "majority":
        return program_inputs(spec.get("base") or {"module": "predict", **_signature_keys(spec)})
    return list(build_signature(spec.get("signature", "question -> answer")).input_fields)


def _signature_keys(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: spec[key] for key in ("signature", "instructions") if key in spec}


class Builder:
    """Turns program specs into DSPy modules against one set of host tools."""

    def __init__(self, tools: HostTools) -> None:
        self.tools = tools

    def __call__(self, spec: dict[str, Any]) -> dspy.Module:
        if not isinstance(spec, dict):
            raise RequestError("program specs must be JSON objects")
        kind = spec.get("module", "predict")
        builder = getattr(self, "_" + kind.replace("-", "_"), None)
        if kind not in MODULES or builder is None:
            reason = UNSUPPORTED.get(f"module:{kind}")
            raise RequestError(
                f"module {kind!r} is unsupported: {reason}"
                if reason
                else f"unknown module {kind!r}; expected one of {MODULES}"
            )
        module = builder(spec)
        if spec.get("demos"):
            demos = [dspy.Example(**demo) for demo in spec["demos"]]
            for predictor in module.predictors():
                predictor.demos = list(demos)
        if spec.get("program_state"):
            module.load_state(spec["program_state"])
        return module

    def _signature(self, spec: dict[str, Any]) -> type[dspy.Signature]:
        return build_signature(
            spec.get("signature", "question -> answer"), spec.get("instructions")
        )

    def _tools(self, spec: dict[str, Any]) -> list[dspy.Tool]:
        return [self.tools.tool(tool) for tool in spec.get("tools", [])]

    def _predict(self, spec):
        return dspy.Predict(self._signature(spec))

    def _chain_of_thought(self, spec):
        return dspy.ChainOfThought(self._signature(spec))

    def _react(self, spec):
        return dspy.ReAct(
            self._signature(spec), tools=self._tools(spec), max_iters=int(spec.get("max_iters", 10))
        )

    def _react_v2(self, spec):
        return dspy.ReActV2(
            self._signature(spec), tools=self._tools(spec), max_iters=int(spec.get("max_iters", 10))
        )

    def _program_of_thought(self, spec):
        return dspy.ProgramOfThought(self._signature(spec), max_iters=int(spec.get("max_iters", 3)))

    def _code_act(self, spec):
        return dspy.CodeAct(
            self._signature(spec),
            tools=[self.tools.function(t) for t in spec.get("tools", [])],
            max_iters=int(spec.get("max_iters", 5)),
        )

    def _rlm(self, spec):
        return dspy.RLM(
            self._signature(spec),
            max_iters=int(spec.get("max_iters", 20)),
            max_llm_calls=int(spec.get("max_llm_calls", 50)),
            tools=[self.tools.function(t) for t in spec.get("tools", [])] or None,
        )

    def _wrapped(self, spec, wrapper):
        base = spec.get("base", "predict")
        base_spec = base if isinstance(base, dict) else {"module": base, **_signature_keys(spec)}
        if base_spec.get("module") in ("best-of-n", "refine"):
            raise RequestError("best-of-n/refine 'base' must be a single-step module")
        return wrapper(
            module=self(base_spec),
            N=int(spec.get("n", 3)),
            reward_fn=_reward(spec, self.tools),
            threshold=float(spec.get("threshold", 1.0)),
            fail_count=spec.get("fail_count"),
        )

    def _best_of_n(self, spec):
        return self._wrapped(spec, dspy.BestOfN)

    def _refine(self, spec):
        return self._wrapped(spec, dspy.Refine)

    def _multi_chain_comparison(self, spec):
        return MultiChainProgram(self._signature(spec), m=int(spec.get("m", 3)))

    def _majority(self, spec):
        base = spec.get("base", "predict")
        base_spec = base if isinstance(base, dict) else {"module": base, **_signature_keys(spec)}
        return MajorityProgram(self(base_spec), n=int(spec.get("n", 5)), field=spec.get("field"))

    def _retrieve(self, spec):
        return RetrieveProgram(
            k=int(spec.get("k", 3)), query_field=spec.get("query_field", "query")
        )

    def _pipeline(self, spec):
        pipeline = Pipeline(spec, self)
        pipeline._tools = self.tools
        return pipeline


def build_adapter(name: str | None, lm: dspy.LM) -> dspy.Adapter:
    name = name or "chat"
    if name == "chat":
        return dspy.ChatAdapter(use_json_adapter_fallback=False)
    if name == "json":
        return dspy.JSONAdapter()
    if name == "xml":
        return dspy.XMLAdapter()
    if name == "two-step":
        return dspy.TwoStepAdapter(extraction_model=lm)
    raise RequestError(f"unknown adapter {name!r}; expected one of {ADAPTERS}")


# ---------------------------------------------------------------------- session


def _accounting(engine: HostEngine, trace: bool) -> dict[str, Any]:
    usage: dict[str, int] = {}
    for crossing in engine.ledger:
        response = crossing["response"]
        reported = response.get("usage") if isinstance(response, dict) else None
        for key, value in (reported or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + int(value)
    report: dict[str, Any] = {"lm_calls": len(engine.ledger), "usage": usage}
    if trace:
        report["history"] = list(engine.ledger)
    return report


class Session:
    """One capability invocation: host LM, adapter, tools and program."""

    def __init__(self, spec: dict[str, Any], lm_call: LmCall, tool_call: ToolCall) -> None:
        if not isinstance(spec, dict):
            raise RequestError("request must be a JSON object")
        self.spec = spec
        self.tools = HostTools(tool_call)
        self.engine = HostEngine(lm_call)
        self.lm = build_lm(spec.get("lm"), self.engine)
        self.adapter = build_adapter(spec.get("adapter"), self.lm)
        self.builder = Builder(self.tools)
        self.module = self.builder(spec)
        self.retriever = self.tools.retriever(spec["retriever"]) if spec.get("retriever") else None

    def context(self):
        tools = self.tools

        def interpreter() -> dspy_runtime.ComponentInterpreter:
            return dspy_runtime.ComponentInterpreter(bridge=tools.call)

        interpreter.execution_instructions = (
            dspy_runtime.ComponentInterpreter.execution_instructions
        )
        settings: dict[str, Any] = {
            "lm": self.lm,
            "adapter": self.adapter,
            "interpreter_factory": interpreter,
        }
        if self.retriever is not None:
            settings["rm"] = self.retriever
        return dspy.context(**settings)

    def signature(self) -> type[dspy.Signature] | None:
        predictors = self.module.predictors()
        if self.spec.get("module", "predict") in ("predict", "chain-of-thought") and predictors:
            return build_signature(self.spec.get("signature", "question -> answer"))
        return None

    def examples(self, rows: Any, input_keys: list[str] | None) -> list[dspy.Example]:
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise RequestError("datasets must be JSON arrays of objects")
        input_keys = input_keys or program_inputs(self.spec) or []
        return [dspy.Example(**row).with_inputs(*input_keys) for row in rows]


# ------------------------------------------------------------------ capabilities


def run(request: dict[str, Any], lm_call: LmCall, tool_call: ToolCall) -> dict[str, Any]:
    """Execute a program spec over one input object or a batch (``module.batch``)."""
    session = Session(request, lm_call, tool_call)
    inputs = request.get("inputs", {})
    signature = session.signature()
    with session.context():
        if isinstance(inputs, list):
            examples = [
                dspy.Example(**coerce_inputs(signature, item)).with_inputs(*item) for item in inputs
            ]
            outputs: Any = [
                p.toDict() if p is not None else None
                for p in session.module.batch(examples, num_threads=1)
            ]
        elif isinstance(inputs, dict):
            kwargs = coerce_inputs(signature, inputs)
            if request.get("async"):
                prediction = dspy_runtime.run_async(session.module.acall(**kwargs))
            else:
                prediction = session.module(**kwargs)
            outputs = prediction.toDict()
        else:
            raise RequestError("'inputs' must be an object or an array of objects")
    return {
        "state": "ALIVE",
        "outputs": outputs,
        **_accounting(session.engine, request.get("trace", False)),
    }


def render(request: dict[str, Any], lm_call: LmCall, tool_call: ToolCall) -> dict[str, Any]:
    """Format the first predictor's prompt without calling the LM."""
    session = Session(request, lm_call, tool_call)
    predictors = session.module.named_predictors()
    if not predictors:
        raise RequestError("program has no predictors to render")
    name, predictor = predictors[0]
    inputs = coerce_inputs(predictor.signature, dict(request.get("inputs") or {}))
    for field in predictor.signature.input_fields:
        inputs.setdefault(field, "")
    with session.context():
        messages = session.adapter.format(predictor.signature, predictor.demos, inputs)
    return {
        "state": "ALIVE",
        "predictor": name,
        "predictors": [
            {
                "name": n,
                "signature": p.signature.signature,
                "instructions": p.signature.instructions,
            }
            for n, p in predictors
        ],
        "messages": json.loads(json.dumps(messages, default=json_default)),
    }


def evaluate(request: dict[str, Any], lm_call: LmCall, tool_call: ToolCall) -> dict[str, Any]:
    """Upstream ``dspy.Evaluate`` (on DSPy's sequential executor path)."""
    program = request.get("program") or {}
    session = Session(program, lm_call, tool_call)
    devset = session.examples(request.get("devset"), request.get("input_keys"))
    metric = make_metric(request.get("metric"), session.tools)
    evaluator = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=1,
        failure_score=float(request.get("failure_score", 0.0)),
        max_errors=request.get("max_errors", len(devset) + 1),
        provide_traceback=False,
        display_progress=False,
    )
    with session.context():
        result = evaluator(session.module)
    rows = [
        {
            "example": dict(example.items()),
            "prediction": prediction.toDict() if hasattr(prediction, "toDict") else prediction,
            "score": float(score),
        }
        for example, prediction, score in result.results
    ]
    return {
        "state": "ALIVE",
        "score": result.score,
        "results": rows,
        **_accounting(session.engine, False),
    }


def _optimizer(name: str, config: dict[str, Any], metric, session: Session):
    lm = session.lm
    if name == "labeled-few-shot":
        return dspy.LabeledFewShot(**{"k": 16, **config}), {"sample": config.pop("sample", True)}
    if name == "bootstrap-few-shot":
        return dspy.BootstrapFewShot(metric=metric, **config), {}
    if name == "bootstrap-random-search":
        return dspy.BootstrapFewShotWithRandomSearch(metric=metric, num_threads=1, **config), {}
    if name == "knn-few-shot":
        embedder = config.pop("embedder", None) or session.spec.get("embedder")
        if not embedder:
            raise RequestError("knn-few-shot needs an 'embedder' host tool")
        return lambda trainset: dspy.KNNFewShot(
            k=int(config.pop("k", 3)),
            trainset=trainset,
            vectorizer=session.tools.embedder(embedder),
            **config,
        ), {}
    if name == "copro":
        return dspy.COPRO(metric=metric, prompt_model=lm, **{"breadth": 2, "depth": 1, **config}), {
            "eval_kwargs": {"num_threads": 1, "display_progress": False}
        }
    if name == "mipro-v2":
        config = {"auto": None, "num_candidates": 2, "num_threads": 1, **config}
        compile_kwargs = (
            {"num_trials": config.pop("num_trials", 2)} if config.get("auto") is None else {}
        )
        return dspy.MIPROv2(metric=metric, prompt_model=lm, task_model=lm, **config), compile_kwargs
    if name == "simba":
        return dspy.SIMBA(
            metric=metric,
            num_threads=1,
            **{"bsize": 2, "num_candidates": 2, "max_steps": 1, **config},
        ), {}
    if name == "gepa":
        config = {"max_metric_calls": 8, **config}
        return dspy.GEPA(metric=metric, reflection_lm=lm, num_threads=1, **config), {}
    if name == "infer-rules":
        return dspy.InferRules(metric=metric, num_threads=1, **config), {}
    if name == "ensemble":
        if config.pop("reduce", None) == "majority":
            config["reduce_fn"] = dspy.majority
        return dspy.Ensemble(**config), {}
    reason = UNSUPPORTED.get(f"optimizer:{name}")
    raise RequestError(
        f"optimizer {name!r} is unsupported: {reason}"
        if reason
        else f"unknown optimizer {name!r}; expected one of {OPTIMIZERS}"
    )


def compile_program(
    request: dict[str, Any], lm_call: LmCall, tool_call: ToolCall
) -> dict[str, Any]:
    """Run a DSPy optimizer; the returned program_state feeds back into run.

    ``config`` is passed through to the optimizer constructor and
    ``compile_config`` to ``compile()``, so every upstream knob is reachable. ``ensemble`` takes ``programs`` (a list of
    program_state objects for the same spec) instead of a trainset.
    """
    program = request.get("program") or {}
    session = Session(program, lm_call, tool_call)
    name = request.get("optimizer", "labeled-few-shot")
    config = dict(request.get("config") or {})
    metric = make_metric(request.get("metric"), session.tools)

    with session.context():
        if name == "ensemble":
            members = []
            for state in request.get("programs") or []:
                member = session.builder({**program, "program_state": state})
                members.append(member)
            optimizer, _ = _optimizer(name, config, metric, session)
            compiled = optimizer.compile(members)
            report = {"state": "ALIVE", "optimizer": name, "members": len(members)}
            if request.get("inputs"):
                report["outputs"] = compiled(**request["inputs"]).toDict()
            return {**report, **_accounting(session.engine, False)}

        trainset = session.examples(request.get("trainset"), request.get("input_keys"))
        valset = (
            session.examples(request["valset"], request.get("input_keys"))
            if request.get("valset")
            else None
        )
        optimizer, compile_kwargs = _optimizer(name, config, metric, session)
        if callable(optimizer) and not hasattr(optimizer, "compile"):
            optimizer = optimizer(trainset)
            compiled = optimizer.compile(session.module)
        else:
            accepted = inspect.signature(optimizer.compile).parameters
            if valset is not None and "valset" in accepted:
                compile_kwargs["valset"] = valset
            if name == "mipro-v2" and len(valset or trainset) < 35:
                compile_kwargs.setdefault("minibatch", False)
            compile_kwargs.update(request.get("compile_config") or {})
            compiled = optimizer.compile(session.module, trainset=trainset, **compile_kwargs)

        report = {
            "state": "ALIVE",
            "optimizer": name,
            "program_state": json.loads(json.dumps(compiled.dump_state(), default=json_default)),
            "demos": {n: len(p.demos) for n, p in compiled.named_predictors()},
            "instructions": {n: p.signature.instructions for n, p in compiled.named_predictors()},
            **_accounting(session.engine, False),
        }
        if request.get("inputs"):
            report["outputs"] = compiled(**request["inputs"]).toDict()
        return report


def describe() -> dict[str, Any]:
    return {
        "state": "ALIVE",
        "imports": {
            "chatman:dspy/lm@0.1.0": "complete(request-json) -> response-json",
            "chatman:dspy/tools@0.1.0": "call(name, args-json) -> envelope-json",
        },
        "exports": ["capabilities", "predict", "run", "render", "evaluate", "compile"],
        "modules": list(MODULES),
        "adapters": list(ADAPTERS),
        "types": sorted(CUSTOM_TYPES),
        "metrics": list(METRICS),
        "optimizers": list(OPTIMIZERS),
        "pipeline_steps": ["program", "tool", "retrieve", "set", "repeat", "foreach", "when"],
        "host_backed": {
            "retriever": "dspy.Retrieve / dspy.settings.rm via a host tool",
            "embedder": "dspy.Embedder (KNN, knn-few-shot) via a host tool",
            "interpreter": "in-component CodeInterpreter for program-of-thought, code-act, rlm",
        },
        "lm_config": [field.name for field in dataclasses.fields(Config)],
        "substitutions": {
            "parallelism": "sequential: ParallelExecutor pinned to 1 thread, pools run inline",
            "mipro-v2": "optuna TPE replaced by a seeded random categorical sampler",
            "numpy": "Embedder/KNN/SIMBA use a pure-Python array/statistics projection",
        },
        "unsupported": dict(UNSUPPORTED),
        "limits": {
            "threads": "none",
            "filesystem": "none",
            "network": "none; provider, tool, retrieval and embedding authority stay with the host",
        },
    }


def guarded(capability: Callable[[], dict[str, Any]]) -> str:
    try:
        return dumps(capability())
    except Exception as exc:
        return dumps({"state": "FAILED", "error_type": type(exc).__name__, "message": str(exc)})


dspy_runtime.install_sequential_runtime()
dspy_runtime.install_numeric_projections()
