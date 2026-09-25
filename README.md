# dspy-wasm

DSPy 3.4 packaged behind a WebAssembly Component Model boundary.

The component runs DSPy program semantics inside Wasmtime. Provider/network
actuation stays outside the component and is supplied through the typed
`chatman:dspy/lm@0.1.0` WIT import.

## Architecture

```text
DSPy program + adapters + signatures
              |
        componentize-py
              |
          dspy.wasm
              |
      WIT lm.complete(JSON)
              |
     Wasmtime host / ash_dspy
              |
      any authorized provider
```

This keeps LiteLLM/OpenAI/provider SDKs out of the WASM dependency surface.
DSPy 3.4's native custom-engine interface is the bridge.

## Build

```bash
python -m pip install -e ".[dev,dspy]"
make dspy
```

`make dspy` installs a WASI overlay for `pydantic-core` and `regex` from
the WASI wheels index before componentizing. DSPy's eager `orjson` usage is
covered by `wasm_compat/orjson.py`, a deliberately narrow compatibility
projection implementing only the API DSPy 3.4 uses on this path.

The generated artifact is:

```text
dist/dspy.wasm
```

## WASM behavioral court

The test is executed **inside the WebAssembly component**:

```bash
make wasm-test
```

`run-self-tests()` mirrors representative seams from DSPy's standard test
suite:

- `Example` input/label semantics
- untyped and typed `Signature`
- memory cache / stable cache keys
- `DummyLM -> Predict`
- `DummyLM -> ChainOfThought`
- WIT host engine -> `Predict`

A successful report has `"state": "ALIVE"` and every case `ALIVE`.

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
