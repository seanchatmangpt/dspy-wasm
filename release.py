"""Release tooling: version consistency, the release manifest, artifact verification.

    python release.py write            # propagate dspy_wasm_version.VERSION to derived files
    python release.py check            # exit 1 listing every requirement that is not met
    python release.py manifest DIST    # write DIST/release.json (hashes, version, limits, commit)
    python release.py verify DIST      # check a packed artifact directory against the repo

Requirements for a release are in RELEASE.md; each names the check that enforces it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import limits
from dspy_wasm_version import VERSION

ROOT = Path(__file__).resolve().parent
ARTIFACT_FILES = ("dspy.wasm", "dspy.wit", "contract.json", "conformance.json")
_CALVER = re.compile(r"^(\d{2})\.([1-9]|1[0-2])\.([1-9]|[12]\d|3[01])$")

PYPROJECT = ROOT / "pyproject.toml"
MIX_EXS = ROOT / "consumer" / "elixir" / "mix.exs"
CONTRACT = ROOT / "consumer" / "contract.json"
CHANGELOG = ROOT / "CHANGELOG.md"


def calver_problem(version: str) -> str | None:
    match = _CALVER.match(version)
    if not match:
        return f"{version!r} is not CalVer YY.M.D (no zero padding)"
    year, month, day = (int(part) for part in match.groups())
    try:
        datetime.date(2000 + year, month, day)
    except ValueError:
        return f"{version!r} is not a real date"
    return None


def _sub_version(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"{path}: no version line matching {pattern!r}")
    path.write_text(new)


def write() -> None:
    problem = calver_problem(VERSION)
    if problem:
        raise SystemExit(problem)
    _sub_version(PYPROJECT, r'^version = ".*"$', f'version = "{VERSION}"')
    _sub_version(MIX_EXS, r'^(\s*)version: ".*",$', rf'\g<1>version: "{VERSION}",')
    data = json.loads(CONTRACT.read_text())
    data["component_version"] = VERSION
    CONTRACT.write_text(json.dumps(data, indent=2) + "\n")
    limits.write()  # README table, contract limits and the Elixir priv copies


def _find(path: Path, pattern: str) -> str | None:
    match = re.search(pattern, path.read_text(), flags=re.MULTILINE)
    return match.group(1) if match else None


def check() -> list[str]:
    """Every unmet version-identity requirement; empty when the repo is releasable."""
    problems: list[str] = []
    if (bad := calver_problem(VERSION)) is not None:
        problems.append(bad)
    sites = {
        "pyproject.toml": _find(PYPROJECT, r'^version = "(.*)"$'),
        "consumer/elixir/mix.exs": _find(MIX_EXS, r'^\s*version: "(.*)",$'),
        "consumer/contract.json": json.loads(CONTRACT.read_text()).get("component_version"),
        "consumer/elixir/priv/contract.json": json.loads(
            (ROOT / "consumer" / "elixir" / "priv" / "contract.json").read_text()
        ).get("component_version"),
    }
    problems += [
        f"{name} says {found!r}, expected {VERSION!r}"
        for name, found in sites.items()
        if found != VERSION
    ]
    for source in ("app.py", "bootstrap_app.py"):
        text = (ROOT / source).read_text()
        if "return dspy_wasm_version.VERSION" not in text:
            problems.append(f"{source} does not return dspy_wasm_version.VERSION")
    if not CHANGELOG.exists() or not re.search(
        rf"^## \[{re.escape(VERSION)}\]", CHANGELOG.read_text(), re.MULTILINE
    ):
        problems.append(f"CHANGELOG.md has no '## [{VERSION}]' entry")
    return problems


# ---------------------------------------------------------- artifact manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


def manifest(dist: Path) -> dict:
    """Write dist/release.json and the relative-path dist/dspy.wasm.sha256."""
    for name in ARTIFACT_FILES:
        if not (dist / name).is_file():
            raise SystemExit(f"{dist / name} is missing")
    contract = json.loads((dist / "contract.json").read_text())
    data = {
        "component": contract["component"],
        "version": VERSION,
        "source_commit": _commit(),
        "wit_package": contract["wit_package"],
        "target": contract["target"],
        "files": {
            name: {"sha256": sha256(dist / name), "bytes": (dist / name).stat().st_size}
            for name in ARTIFACT_FILES
        },
        "limits": limits.contract_limits(),
    }
    (dist / "release.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    # Relative path, so `sha256sum -c dspy.wasm.sha256` works from inside the directory.
    (dist / "dspy.wasm.sha256").write_text(f"{data['files']['dspy.wasm']['sha256']}  dspy.wasm\n")
    return data


def verify(dist: Path) -> list[str]:
    """Problems with a packed artifact directory: hashes, version and contract drift."""
    problems: list[str] = []
    try:
        recorded = json.loads((dist / "release.json").read_text())
    except (OSError, ValueError) as exc:
        return [f"release.json unreadable: {exc}"]
    if recorded.get("version") != VERSION:
        problems.append(f"release.json version {recorded.get('version')!r} != {VERSION!r}")
    for name in ARTIFACT_FILES:
        path = dist / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif recorded["files"].get(name, {}).get("sha256") != sha256(path):
            problems.append(f"{name} does not match its recorded sha256")
    sums = dist / "dspy.wasm.sha256"
    if not sums.is_file() or sums.read_text().split()[:2] != [
        recorded["files"]["dspy.wasm"]["sha256"],
        "dspy.wasm",
    ]:
        problems.append("dspy.wasm.sha256 is missing or not a relative-path checksum")
    for name, repo in (
        ("contract.json", CONTRACT),
        ("conformance.json", ROOT / "consumer" / "conformance.json"),
    ):
        if (dist / name).is_file() and (dist / name).read_text() != repo.read_text():
            problems.append(f"{name} in the artifact differs from the repo's {repo.name}")
    if recorded.get("limits") != limits.contract_limits():
        problems.append("release.json limits differ from limits.py")
    return problems


def main(argv: list[str]) -> int:
    match argv:
        case ["write"]:
            write()
            return 0
        case ["check"]:
            problems = check()
        case ["manifest", dist]:
            print(json.dumps(manifest(Path(dist)), indent=2, sort_keys=True))
            return 0
        case ["verify", dist]:
            problems = verify(Path(dist))
        case _:
            print(__doc__)
            return 2
    for problem in problems:
        print(f"release: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
