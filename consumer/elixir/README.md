# dspy_wasm: Elixir reference host

A host for the dspy-wasm component on Wasmex 0.15 (Wasmtime 47), and the
translation layer a consumer such as `ash_dspy` needs. Everything here was run
against the CI-built `dspy.wasm`.

```elixir
{:ok, host} =
  DspyWasm.Host.start_link(
    path: "dist/dspy.wasm",
    lm: fn request_json -> MyApp.LM.complete(request_json) end,   # {:ok, json} | {:error, msg}
    budget: [max_lm_calls: 500]
  )

:ok = DspyWasm.Host.await_ready(host)              # boots in the background
{:ok, spec} = DspyWasm.Signature.to_spec(%{inputs: [...], outputs: [...], description: "..."})
{:ok, report} = DspyWasm.run(host, Map.merge(spec, %{"module" => "chain-of-thought", "inputs" => %{...}}))
report["state"]   # "ALIVE" | "FAILED"
```

`mix test` with `DSPY_WASM_PATH=../../dist/dspy.wasm` runs the shared
conformance vectors (`consumer/conformance.json`), the budget abort and the
translation end to end. A missing path fails the suite; it does not skip.

## What a Wasmex host has to know

Measured, not assumed:

| finding | consequence |
|---|---|
| Wasmex exposes no epoch, fuel or memory limit for components | the host-side deadline of `host.py` has no Wasmex equivalent |
| Killing the component process does **not** stop a running guest: after `Process.exit(pid, :kill)` two OS threads kept burning CPU | a runaway guest is stopped only by restarting the VM or by a bound inside the component |
| An LM-authored `while True: pass` in `program-of-thought`/`code-act` ran forever | the component now stops interpreted code after `max_interpreter_events` trace events; a single long C call (`10**10**8`) is still not stopped |
| Raising from an import function aborts the guest call promptly (119 ms, 26th LM call) and traps the instance | budgets are enforced in `DspyWasm.Host`'s imports; the instance is replaced afterwards |
| Loading the component takes about a minute per boot and Wasmex compiles it each time | `DspyWasm.Host` boots on a separate process; calls return `{:error, :starting}` meanwhile |
| Wasmex's default call timeout is 5 s, and a timeout exits the caller | `DspyWasm.Host` defaults to 600 s and turns a timeout into `{:error, :timeout}` |
| Wasmex declares `elixir: "~> 1.18"` | a project that depends on it, such as `ash_dspy` (`~> 1.15` today), must raise its Elixir requirement |

Run the BEAM under an OS-level CPU/time limit (cgroup, `ulimit -t`, a container
limit) until Wasmex offers epoch interruption.

## Mapping an `ash_dspy` resource

| `ash_dspy` | dspy-wasm |
|---|---|
| `signature` `description` | `instructions` |
| `input`/`output` `name`, `type` | typed signature string, `question: str, passage: str -> answer: str` (`DspyWasm.Signature`); `{:array, t}` becomes `list[T]`; an unmappable type is an error |
| `input`/`output` `doc` | folded into `instructions` (a signature string has no per-field description) |
| `required: false` on an input | not expressible: dspy has no optional field, callers must supply it |
| `metric :exact_match, :accuracy` | `"metric": {"name": "exact_match", "field": "answer"}`; other names need a host tool (`{"tool": "..."}`) |
| `requirement :accuracy, :gte, bound: 90` | `DspyWasm.Requirement.satisfied?/2` on the `evaluate` report's percent `score` |
| `minimize :token_cost` | read `usage` from the report; nothing minimises it for you |
