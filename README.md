# dspy-wasm

DSPy behind a WebAssembly Component Model boundary.

The repository deliberately separates two subjects:

1. **Python -> WebAssembly Component** via Bytecode Alliance `componentize-py`.
2. **DSPy dependency closure inside that component**.

A DSPy-native dependency failure must not be interpreted as evidence that Python componentization failed.

## Toolchain

- Python `>=3.10,<3.15`
- `componentize-py==0.25.0`
- `wasmtime==48.0.0`
- DSPy probe: `dspy==3.4.0`

## Contract

`wit/dspy.wit` is the source boundary:

```text
chatman:dspy/dspy
  component-version() -> string
  runtime-info()      -> string
  dspy-version()      -> string
```

The string payloads are JSON-compatible where structured state is required. That keeps the first ABI intentionally small while the dependency boundary is being qualified.

## First court: prove Python -> WASM

```bash
python -m pip install -e ".[dev]"
make bootstrap
make host-bootstrap
```

Expected meaning:

```text
bootstrap build + host execution => componentization path observed
```

This does **not** prove that DSPy itself can be packaged.

## Second court: qualify DSPy

```bash
python -m pip install -e ".[dev,dspy]"
make dspy
make host-dspy
```

`app.py` imports DSPy at module scope on purpose. Current `componentize-py` resolves application dependencies at build time, so the first incompatible dependency becomes an explicit boundary finding.

DSPy 3.4.0 currently depends on packages including `regex`, `orjson`, `pydantic`, `litellm`, and `openai`. Native-extension or platform-specific failures should be classified by exact package rather than worked around implicitly.

## State model

- `bootstrap.wasm`: **UNKNOWN** until build and host execution are observed.
- `dspy.wasm`: **UNKNOWN** until build and host execution are observed.
- A failed DSPy build is a typed dependency finding; it does not change the bootstrap subject's state.
- GitHub CI is evidence transport, not the definition of component liveness.

## Next extension

Once the import court identifies the exact incompatible dependency set:

```text
dependency
  -> existing WASI wheel?
  -> pure-Python replacement?
  -> capability moved across WIT boundary?
  -> irreducible port
```

The intended destination is not a Python RPC wrapper. It is a typed, language-neutral DSPy capability that Ash, SA2A, or another host can load through a WebAssembly component runtime.
