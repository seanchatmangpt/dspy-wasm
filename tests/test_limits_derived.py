"""README's limits table, the consumer contract and the code all come from limits.py."""

from __future__ import annotations

import json
from pathlib import Path

import limits

ROOT = Path(__file__).resolve().parents[1]


def test_readme_limits_block_is_generated_from_the_table() -> None:
    assert limits.readme_block() in (ROOT / "README.md").read_text(), (
        "run `python limits.py --write`"
    )


def test_contract_limits_are_generated_from_the_table() -> None:
    contract = json.loads((ROOT / "consumer" / "contract.json").read_text())
    assert contract["limits"] == limits.contract_limits(), "run `python limits.py --write`"


def test_every_limit_names_who_enforces_it() -> None:
    assert {limit.enforced_by for limit in limits.LIMITS.values()} <= {"component", "host"}
