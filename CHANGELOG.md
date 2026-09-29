# Changelog

Versions are CalVer, `YY.M.D` (see `dspy_wasm_version.py`). The WIT interface has
its own version (`chatman:dspy@0.1.0`) and did not change in any release below.

## [26.9.28]

The first release with explicit requirements (see `RELEASE.md`). Everything here
is on top of the production-boundary audit in #7.

### Added
- Host-side meter: LM calls, tool calls and reply bytes per guest call are
  counted in the host and abort the call past a `Budget` (`BudgetExceeded`, CLI
  exit 3; `--max-lm-calls`, `--max-tool-calls`, `--max-reply-bytes`).
- `host.Guest`: replaces an instance after a trap or `max_calls` calls, with
  optional background prewarm.
- Interpreter step budget (`max_interpreter_events`): interpreted code that
  loops or recurses is stopped inside the component, for hosts with no deadline.
- Request limits `max_batch_items`, `max_dataset_items`, `max_optimizer_count`,
  and a per-optimizer allow-list of count knobs.
- `limits.py`: every numeric limit in one table; the README table,
  `consumer/contract.json` (`limits`) and `capabilities` derive from it.
- Conformance vectors as data (`consumer/conformance.json`) and a runner
  (`conformance.py`, `host.py --conformance`) that any host can use.
- Elixir reference host on Wasmex (`consumer/elixir`): budgets in the imports,
  async boot, recycling, the vectors, `ash_dspy` signature translation.
- Release tooling (`release.py`): version consistency, `release.json` manifest,
  artifact verification.

### Fixed
- `embed` bounded each axis but not `texts x dimensions` (205 MB reply after
  18 s); now at most 1,048,576 values per call.
- The `dspy-wasm` CI job skipped the two epoch-deadline tests; it now builds
  `bootstrap.wasm` first, and a skipped test fails that job.
- Optimizer count knobs (`max_rounds`, `num_candidate_programs`, ...) and batch
  length had no request-level limit; three optimizers ran until killed.
- `dspy.wasm.sha256` recorded `dist/dspy.wasm`, so it could not be checked from
  inside the artifact directory; it now records `dspy.wasm`.
- The Elixir host read its contract from outside `consumer/elixir`, so it could
  not build as a sparse git dependency.

### Changed
- `component-version` reports `26.9.28` (was `0.1.0`).

### Known limitations
- One long C call (`10**10**8`) in interpreted code is not stopped by the
  interpreter budget; only a host deadline bounds it. Wasmex has no deadline and
  cannot stop a running guest: run the BEAM under an OS CPU/time limit.
- Not verified: a real OpenAI-compatible provider end to end; a clean-cache WASI
  build (the release workflow performs one); concurrent load.
