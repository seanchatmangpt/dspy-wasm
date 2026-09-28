# Release requirements

What a dspy-wasm release must satisfy, and the check that enforces each one.
`python release.py check` covers R1 and R7 locally; the rest are CI gates.
A release with an unmet or waived requirement lists it under *Known limitations*
in `CHANGELOG.md`.

| # | Requirement | Enforced by |
|---|---|---|
| R1 | One CalVer `YY.M.D` version (`dspy_wasm_version.py`), reported by `component-version` and carried by `pyproject.toml`, `consumer/contract.json`, `consumer/elixir/mix.exs` and its `priv` copy | `python release.py check`, `tests/test_release.py` |
| R2 | The consumer contract is stable: the WIT surface, imports and exports are unchanged unless the WIT package version changes; the contract and its Elixir copy are generated, not edited | `tests/test_consumer_contract.py`, `tests/test_limits_derived.py` |
| R3 | Every numeric limit is in `limits.py` and published (README, contract, `capabilities`); no request can start work that only a deadline stops | `tests/test_limits_derived.py`, `tests/test_request_volume_bounds.py`, the conformance vectors |
| R4 | The artifact is verifiable: `dspy.wasm`, `dspy.wit`, `contract.json`, `conformance.json`, a relative-path `dspy.wasm.sha256` and a `release.json` manifest (hashes, version, source commit, limits) | `python release.py verify dist` in CI and in the release workflow |
| R5 | The behaviour gates pass on the built component: zero skipped tests, `--self-test` 36 of 36, every conformance vector, the host budget abort | the `dspy-wasm` CI job |
| R6 | Both reference hosts pass the same conformance vectors: `host.py` (Python) and `consumer/elixir` (Wasmex) | `dspy-wasm` and `elixir-host` CI jobs |
| R7 | Docs match code and the version has a changelog entry | `tests/test_readme_matches_code.py`, `python release.py check` |
| R8 | The release is built from a clean cache with rustc >= 1.95 | `.github/workflows/release.yml` (no cache steps, prints `rustc --version`) |
| R9 | Known limitations are disclosed, not hidden | `CHANGELOG.md` |
| R10 | Consumers pin the release, not a branch (`ash_dspy` depends on the tag) | manual, after the tag exists |

## Cutting a release

1. Set `dspy_wasm_version.VERSION`, run `python release.py write`, add the
   `CHANGELOG.md` entry, and get CI green on the branch.
2. Tag `vYY.M.D` on that commit and push the tag. `release.yml` builds from a
   clean cache, runs the gates above, verifies the artifact and publishes a
   GitHub release with the five artifact files and `release.json`.
3. Repoint consumers (R10).

## Open for 26.9.28

- R8 has not run yet: no clean-cache build has been observed. The workflow can
  also be started by hand (`workflow_dispatch`) as a dry run that uploads the
  artifact without publishing.
- R10 waits on the tag.
- A real OpenAI-compatible provider has not been exercised end to end (waived,
  listed as a known limitation).
