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
make dspy        # toolchain -> WASI wheels -> build/wasi_deps -> dist/dspy.wasm
make wasm-test   # run the behavioral court inside the component
```

Requirements: a Rust toolchain with the `wasm32-wasip2` target (rustc >= 1.95),
`uv`, and a C toolchain for the native CPython bootstrap. Everything else is
downloaded and built by `wasi/toolchain.sh`.

Build hosts: Linux or macOS, on x86_64 or arm64. `wasi/toolchain.sh` picks
the matching wasi-sdk release (`wasi-sdk-33.0-{x86_64,arm64}-{linux,macos}`)
and refuses any other host with exit code 2; `JOBS` defaults to `nproc`, or
`sysctl -n hw.ncpu` where `nproc` is absent. CI builds on `ubuntu-latest`
(x86_64 Linux).

### Every dependency is real WebAssembly

The component runs CPython 3.14 via componentize-py 0.25.1. The full locked
dependency closure (`wasi/requirements.lock`, 70 packages: DSPy, litellm,
openai, optuna, numpy, ...) is installed into `build/wasi_deps`. Every package
with native code is **cross-compiled from its sdist** to a `cp314` WASI wheel
by `wasi/build_wheels.py`. There are no shims or projections.

| kind           | packages (compiled to `wasm32-wasip2`)                                                   |
|----------------|------------------------------------------------------------------------------------------|
| Rust / PyO3    | pydantic-core, rpds-py, jiter, orjson, tiktoken, tokenizers, fastuuid, hf-xet, litellm   |
| C / Cython     | regex, markupsafe, multidict, frozenlist, propcache, yarl, aiohttp, charset-normalizer, pyyaml (+libyaml), sqlalchemy |
| meson          | numpy (bundled lapack-lite)                                                               |
| CPython stdlib | `_ssl`, `_hashlib` against OpenSSL 3.5 (componentize-py's runtime omits them)            |

`wasi/toolchain.sh` mirrors componentize-py's own build: wasi-sdk 33, CPython
3.14.0 configured for `wasm32-wasip2` with `-fPIC` (headers, sysconfig), zlib,
libyaml, OpenSSL. Extensions are PIC shared libraries whose libpython symbols
are bound by componentize-py's linker.

Porting work lives in the recipes as data, each with its reason:

- **Toolchain facts:** wasi-libc's POSIX emulation libraries (signal,
  process-clocks, getpid, mman); `Py_ENABLE_SHARED` so PyAPI keeps default
  visibility; a wasm-ld shim so meson's GNU-style link lines work; imports
  resolved dynamically only for `-shared` links, so configure probes stay
  truthful.
- **C++ exceptions:** the component links the no-exceptions libc++. The few
  translation units compiled with `-fexceptions` (numpy's `unique`, pocketfft,
  esaxx) resolve `__cxa_throw` to a weak report-and-terminate archive, which is
  C++'s contract when exceptions are unavailable.
- **WASI is not the browser:** crates whose `wasm32` paths mean "browser"
  (reqwest 0.12, reqwest-middleware, hf-xet's runtime) are narrowed to
  `all(wasm32, not(wasi))`.
- **Small upstream-style ports:** a WASI platform module for
  `rustls-native-certs` (honours `SSL_CERT_FILE`/`SSL_CERT_DIR`),
  `gcp_auth`/`azure_identity` process paths, `os_str_bytes` on stable
  encoded-bytes APIs, the AWS-LC console and `AF_UNIX` guards, the llhttp
  wasm guard, numpy's CPU detection, `std::process::id()` (a component cannot
  fork), the tokio 1.53 and reqwest 0.13.5 bumps, and OpenSSL's thread pool
  and `socketpair` notifier.

### componentize-py build

`make dspy` uses componentize-py 0.25.1 built from its published crate, which
ships the same prebuilt runtime, libc and libpython as the PyPI wheel. The one
change raises wasmtime's 128 MiB per-call "hostcall fuel" for the build-time
pre-initialisation step. That step runs the (trusted) application and must
copy its whole linear memory out in one call, and litellm's import graph
exceeds 128 MiB. The bootstrap court uses the stock PyPI release.

### Host grants

The DSPy component is built without `--stub-wasi`. The host links WASI with no
preopened directories, environment, argv, or network grants: the component
receives clocks and entropy and nothing else. TLS, sockets and HTTP clients are
present in the component. They stay inert unless a host chooses to grant
`wasi:sockets`.

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
| optimizers | `labeled-few-shot`, `bootstrap-few-shot`, `bootstrap-random-search`, `knn-few-shot`, `bootstrap-optuna`, `copro`, `mipro-v2` (real optuna TPE), `simba`, `gepa`, `infer-rules`, `ensemble`; `config` goes to the constructor, `compile_config` to `compile()` |
| retrieval  | `"retriever": "<host tool>"` (via `dspy.Retrieve`), or `{"corpus": [...], "embedder": "<host tool>", "k": 3}` for `dspy.retrievers.Embeddings` indexed inside the component |
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

### Runtime policy

The component has no threads, subprocesses, filesystem or network. Rather than
drop the features that assume them, `dspy_runtime.py` supplies single-instance
equivalents:

| DSPy expects                         | inside the component                                               |
|--------------------------------------|--------------------------------------------------------------------|
| thread pools (`Evaluate`, `Parallel`, optimizers) | `ParallelExecutor` pinned to its sequential path; `ThreadPoolExecutor` runs inline |
| `Unbatchify` worker thread (`dspy.retrievers.Embeddings`) | single-caller batches run inline                  |
| Deno/Pyodide interpreter (PoT, CodeAct, RLM) | in-component `CodeInterpreter`: the component *is* the sandbox; host tools reachable via `__host_tool__` |
| asyncio self-pipe socket             | event loop without the cross-thread wake-up channel               |
| imports at call time                 | lazily imported modules (dspy `require()`, litellm's lazy providers, numpy submodules) are imported at build time |

Not supported, with reasons reported by `capabilities`: weight-training
optimizers (`bootstrap-finetune`, `better-together`, `grpo`) need a provider's
fine-tuning service. `avatar` works only with the deprecated Avatar module,
`flex` isn't wired up yet, and streaming can't cross a synchronous WIT call.

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
few-shot with a host embedder, the in-component embeddings retriever, the
BootstrapRS, COPRO, MIPROv2, SIMBA, GEPA, InferRules and BootstrapOptuna
optimizers, and the compiled dependencies themselves: every native extension
loaded from its `.so`, numpy linear algebra and FFT, optuna's TPE sampler,
OpenSSL TLS contexts and hashing, and litellm with its Rust bridge, plus the
request-boundary refusals (nested-repeat budget, `program_state` admission)
executed inside the component (36 cases).

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

## Boundary guards and benchmark

`tests/test_hardening.py` pins the refusals the boundary relies on:

- `compile` binds `program_state` to its program under `__subject__` (a
  digest of predictor names and signature fields); `run`/`compile` refuse a
  state without a subject, a stale or tampered subject, and a body whose field
  count does not match. `program_state` must be an object when present
  (`null` means absent). The digest is unkeyed: it catches a state meant for
  another program, not a deliberately forged one.
- Pipeline step names cannot shadow pipeline attributes (`forward`,
  `outputs`, ...), two different programs cannot share one normalised name,
  and `repeat` is an integer in `[0, 10000]`. One pipeline call executes at
  most 100000 steps in total (every step, `repeat` iteration and `foreach`
  item is charged), so nested repeats cannot multiply past the per-level bound.
- A tool envelope must carry `result` or `error`; generated tool shims refuse
  names that would shadow `__host_tool__`/`SUBMIT`.
- `SUBMIT` cannot be swallowed by `except` in interpreted code.
- Host tools: `calculator` refuses results above 4096 bits (powers are
  refused before they are computed), non-real and non-finite values; every
  tool envelope is strict JSON (no `Infinity`/`NaN`); `search` needs
  `k >= 0`; `embed` needs an array of strings.

```bash
make bench   # writes bench/receipt.json; medians bounded by BOUNDS_MS
```

`tests/test_bench_bounds.py` reruns every case and fails when a median
exceeds its ceiling, and refuses a ceiling more than 10x its committed
median. When `dist/dspy.wasm` is present the benchmark also times DSPy
executing inside the component (`wasm:dspy-run-predict`,
`wasm:dspy-compile-labeled-few-shot`: `run`/`compile` exports on one
instance); `bootstrap.wasm` alone contains no DSPy. Instantiation is
reported under `setup_ms`, unbounded: it measures wasmtime's compilation
cache (hit or miss) more than the component.

## Evidence

Repository/CI execution establishes repository-local component behavior only.
It does not by itself establish production deployment, provider authority, or
external standing.
