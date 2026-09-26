# dspy-wasm

DSPy 3.4 packaged behind a WebAssembly Component Model boundary.

The component runs DSPy program semantics inside Wasmtime: signatures,
adapters, every inference module, composed pipelines, evaluation and the
prompt optimizers. Everything that touches the outside world (the LM
provider, tools, retrieval, embeddings) stays with the host, reached through
two typed WIT imports:

| import                     | function                                    |
|----------------------------|---------------------------------------------|
| `chatman:dspy/lm@0.1.0`    | `complete(request-json) -> response-json`   |
| `chatman:dspy/tools@0.1.0` | `call(name, args-json) -> envelope-json`    |

## Architecture

```text
 DSPy modules · pipelines · adapters · Evaluate · optimizers
                         |
                   componentize-py
                         |
                     dspy.wasm
                  /              \
   WIT lm.complete(JSON)     WIT tools.call(name, JSON)
                  \              /
              Wasmtime host / ash_dspy
             /          |            \
   any authorized    host tools    retriever / embedder
      provider      (ReAct, CodeAct,   (dspy.Retrieve,
                    metrics, rewards)   KNN, knn-few-shot)
```

This keeps LiteLLM/OpenAI/provider SDKs out of the WASM dependency surface.
DSPy 3.4's native custom-engine interface is the bridge.

## Build

```bash
python -m pip install -e ".[dev,dspy]"
make dspy
```

`componentize-py` is pinned to 0.17.2, the last release that embeds CPython
3.12; later releases embed 3.14, for which no WASI `pydantic-core` wheels
exist. `make dspy` installs a WASI overlay for `pydantic-core`, `regex` and
the pure-Python `charset-normalizer` from the WASI wheels index before
componentizing.

Native modules DSPy imports eagerly but without WASI builds are covered by
deliberately narrow projections in `wasm_compat/`:

| module   | reached via                    | projection                                   |
|----------|--------------------------------|----------------------------------------------|
| `orjson` | DSPy                           | `dumps`/`loads` + the 3 option flags used    |
| `rpds`   | `jsonschema -> referencing`    | persistent map/set/list, copy-on-write       |
| `jiter`  | `openai` (streaming helpers)   | complete-document `from_json`; partial traps |
| `zlib`   | `urllib3` (HTTP body decoding) | import-only; any compression call traps      |

The DSPy component is built **without** `--stub-wasi`. The host links WASI
with no preopened directories, environment, argv, or network grants: the
component receives clocks and entropy (DSPy timestamps LM history and mints
UUIDs) and nothing else.

The generated artifact is:

```text
dist/dspy.wasm
```

## Capability surface

| export         | request                                   | purpose                                          |
|----------------|-------------------------------------------|--------------------------------------------------|
| `capabilities` | -                                         | modules, adapters, types, metrics, optimizers, substitutions, unsupported |
| `run`          | program spec + `inputs` (object or batch) | execute any module or pipeline                   |
| `render`       | program spec + `inputs`                   | format prompt messages without calling the LM    |
| `evaluate`     | `program`, `devset`, `metric`             | upstream `dspy.Evaluate`                         |
| `compile`      | `program`, `optimizer`, `trainset`, ...   | run an optimizer, return `program_state`         |
| `predict`      | signature, inputs                         | original single-`Predict` entry point            |

A **program spec** is JSON:

```json
{
  "module": "chain-of-thought",
  "signature": "context: list[str], question -> answer",
  "instructions": "Answer from the context.",
  "adapter": "chat",
  "lm": {"temperature": 0.2, "max_tokens": 512},
  "demos": [{"context": ["..."], "question": "...", "answer": "..."}],
  "program_state": {"...": "output of compile"},
  "retriever": "search",
  "tools": [{"name": "calculator", "description": "...", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]
}
```

| area       | supported                                                                                                  |
|------------|------------------------------------------------------------------------------------------------------------|
| modules    | `predict`, `chain-of-thought`, `react`, `react-v2`, `program-of-thought`, `code-act`, `rlm`, `best-of-n`, `refine`, `multi-chain-comparison`, `majority`, `retrieve`, `pipeline` |
| adapters   | `chat`, `json`, `xml`, `two-step`                                                                          |
| types      | `Image`, `Audio`, `File`, `History`, `Code`, `Reasoning`, `ToolCalls` in signature strings; image/audio cross the LM boundary as message `parts` |
| metrics    | `exact_match`, `f1`, `contains`, `passage_match`, `semantic-f1`, or `{"tool": "<host tool>"}` (GEPA feedback supported) |
| optimizers | `labeled-few-shot`, `bootstrap-few-shot`, `bootstrap-random-search`, `knn-few-shot`, `copro`, `mipro-v2`, `simba`, `gepa`, `infer-rules`, `ensemble`; `config` goes to the constructor, `compile_config` to `compile()` |
| execution  | batches via `module.batch`, `async` via `acall`, LM `config` (temperature, max_tokens, stop, seed, ...) forwarded to the host, per-call usage ledger and `trace` |

### Pipelines

`"module": "pipeline"` composes programs, host tools and retrieval over shared
state into one `dspy.Module`. Every program step is a named sub-module, so a
pipeline evaluates, compiles and round-trips `program_state` like any DSPy
program. A step reused inside `repeat` is a single predictor (multi-hop).

```json
{"module": "pipeline", "retriever": "search",
 "steps": [
   {"set": {"context": []}},
   {"repeat": 2, "steps": [
     {"name": "generate_query", "program": {"module": "chain-of-thought", "signature": "context: list[str], question -> query"}},
     {"retrieve": "$query", "k": 1, "output": "context", "accumulate": ["context"]}]},
   {"name": "generate_answer", "program": {"module": "chain-of-thought", "signature": "context: list[str], question -> answer"}}],
 "outputs": ["answer", "context"]}
```

Step kinds: `program`, `tool`, `retrieve`, `set`, `repeat` (with `until`),
`foreach` (with `collect`); any step takes `when`. `"$key.path"` references
state, and `{{key}}` interpolates into strings.

### Runtime projections

The component has no threads, subprocesses, filesystem or network. Rather than
drop the features that assume them, `dspy_runtime.py` supplies single-instance
equivalents:

| DSPy expects                         | inside the component                                               |
|--------------------------------------|--------------------------------------------------------------------|
| thread pools (`Evaluate`, `Parallel`, optimizers) | `ParallelExecutor` pinned to its sequential path; `ThreadPoolExecutor` runs inline |
| Deno/Pyodide interpreter (PoT, CodeAct, RLM) | in-component `CodeInterpreter`: the component *is* the sandbox; host tools reachable via `__host_tool__` |
| numpy (`Embedder`, `KNN`, `SIMBA`)   | pure-Python array/statistics projection, injected only into those modules |
| optuna TPE (`MIPROv2`)               | seeded random categorical sampler (reported in `capabilities`)     |
| asyncio self-pipe socket             | event loop without the cross-thread wake-up channel               |

Not supported, with reasons reported by `capabilities`: weight-training
optimizers (`bootstrap-finetune`, `better-together`, `grpo`), `bootstrap-optuna`,
`avatar`, `flex`, `dspy.retrievers.Embeddings`, and streaming.

## Host

`host.py` is the reference host. It grants WASI clocks and entropy only, and
supplies deterministic builtin tools: `calculator` (arithmetic only),
`echo`, `grade_exact` (metric/reward), `search` and `embed` (over `--corpus`
or a small default corpus). `--tool NAME=module:function` grants any host
Python function as a tool. Host callbacks never raise into the component:
failures cross as `{"error": ...}` envelopes.

```bash
python host.py dist/dspy.wasm --capabilities
python host.py dist/dspy.wasm --run @examples/multihop_pipeline.json \
  --responses @examples/multihop_responses.json
python host.py dist/dspy.wasm --run @examples/code_act.json \
  --responses @examples/code_act_responses.json
python host.py dist/dspy.wasm --compile @examples/compile_bootstrap.json \
  --responses @examples/compile_responses.json
```

`--responses` scripts the LM for deterministic runs; omit it and pass
`--base-url`/`--upstream-model` to use a real OpenAI-compatible provider.

## WASM behavioral court

The test is executed **inside the WebAssembly component**:

```bash
make wasm-test
```

`run-self-tests()` covers the upstream seams (`Example`, typed `Signature`,
memory cache, `DummyLM -> Predict/ChainOfThought`), the WIT LM and tool
boundaries, and every capability above: ReAct over host tools, structured
adapters, Refine, BestOfN with a host reward, multi-chain comparison, render,
evaluate, compile round trips, a multi-hop retrieval pipeline,
ProgramOfThought, CodeAct, RLM, majority, async, multimodal parts, KNN
few-shot with a host embedder, and the BootstrapRS, COPRO, MIPROv2, SIMBA,
GEPA and InferRules optimizers.

A successful report has `"state": "ALIVE"` and every case `ALIVE`. The same
capability code is exercised natively by `make test`.

## Host-backed Predict

Deterministic host:

```bash
python host.py dist/dspy.wasm \
  --predict-signature "question -> answer" \
  --inputs '{"question":"What is the capital of France?"}' \
  --response $'[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]'
```

OpenAI-compatible provider:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1 \
OPENAI_API_KEY=... \
DSPY_WASM_MODEL=gpt-5-mini \
python host.py dist/dspy.wasm \
  --predict-signature "question -> answer" \
  --inputs '{"question":"What is the capital of France?"}'
```

The WASM component never receives the API key. The host owns the irreversible
provider call.

## Bootstrap court

The dependency-free Python -> Component Model court remains separate:

```bash
make bootstrap
make host-bootstrap
```

A DSPy dependency failure therefore cannot be misclassified as a generic
Python -> WASM failure.

## Evidence

Repository/CI execution establishes repository-local component behavior only.
It does not by itself establish production deployment, provider authority, or
external standing.
