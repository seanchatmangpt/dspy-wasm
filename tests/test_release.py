"""Release requirements R1 (version identity) and R4 (verifiable artifact)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import dspy_wasm_version
import release

ROOT = Path(__file__).resolve().parents[1]


def test_the_repo_meets_the_version_identity_requirements() -> None:
    assert release.check() == []


@pytest.mark.parametrize("good", ["26.9.28", "26.10.1", "27.1.31"])
def test_calver_accepts_real_dates(good: str) -> None:
    assert release.calver_problem(good) is None


@pytest.mark.parametrize(
    "bad", ["0.1.0", "26.09.28", "26.9", "26.13.1", "26.2.30", "v26.9.28", "2026.9.28", ""]
)
def test_calver_refuses_everything_else(bad: str) -> None:
    assert release.calver_problem(bad) is not None


def test_the_component_reports_the_release_version() -> None:
    assert (
        dspy_wasm_version.VERSION == json.loads(release.CONTRACT.read_text())["component_version"]
    )
    for source in ("app.py", "bootstrap_app.py"):
        assert "return dspy_wasm_version.VERSION" in (ROOT / source).read_text()


def test_a_disagreeing_version_site_is_reported(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "pyproject.toml"
    fake.write_text(release.PYPROJECT.read_text().replace(dspy_wasm_version.VERSION, "0.1.0"))
    monkeypatch.setattr(release, "PYPROJECT", fake)
    assert any("pyproject.toml says '0.1.0'" in problem for problem in release.check())


def test_a_missing_changelog_entry_is_reported(monkeypatch, tmp_path) -> None:
    empty = tmp_path / "CHANGELOG.md"
    empty.write_text("# Changelog\n")
    monkeypatch.setattr(release, "CHANGELOG", empty)
    assert any("CHANGELOG.md has no" in problem for problem in release.check())


# ---------------------------------------------------------------- the artifact


@pytest.fixture
def dist(tmp_path) -> Path:
    directory = tmp_path / "dist"
    directory.mkdir()
    (directory / "dspy.wasm").write_bytes(b"\0asm-not-really" * 100)
    shutil.copy(ROOT / "wit" / "dspy.wit", directory / "dspy.wit")
    shutil.copy(release.CONTRACT, directory / "contract.json")
    shutil.copy(ROOT / "consumer" / "conformance.json", directory / "conformance.json")
    return directory


def test_a_packed_artifact_verifies_and_its_checksum_is_relative(dist) -> None:
    data = release.manifest(dist)
    assert data["version"] == dspy_wasm_version.VERSION
    assert set(data["files"]) == set(release.ARTIFACT_FILES)
    assert release.verify(dist) == []
    line = (dist / "dspy.wasm.sha256").read_text().split()
    assert line == [data["files"]["dspy.wasm"]["sha256"], "dspy.wasm"]


def test_a_tampered_component_fails_verification(dist) -> None:
    release.manifest(dist)
    (dist / "dspy.wasm").write_bytes(b"tampered")
    assert any("dspy.wasm does not match" in p for p in release.verify(dist))


def test_a_contract_that_drifted_from_the_repo_fails_verification(dist) -> None:
    release.manifest(dist)
    contract = json.loads((dist / "contract.json").read_text())
    contract["limits"]["max_repeat"] += 1
    (dist / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    problems = release.verify(dist)
    assert any("contract.json" in p for p in problems)


def test_a_wrong_version_in_the_manifest_fails_verification(dist) -> None:
    release.manifest(dist)
    recorded = json.loads((dist / "release.json").read_text())
    recorded["version"] = "0.1.0"
    (dist / "release.json").write_text(json.dumps(recorded))
    assert any("version" in p for p in release.verify(dist))


def test_a_missing_manifest_is_a_problem_not_a_crash(dist) -> None:
    assert release.verify(dist)[0].startswith("release.json unreadable")
