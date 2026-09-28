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


def test_elixir_host_carries_identical_copies_of_the_contract_and_vectors() -> None:
    # consumer/elixir is a sparse git dependency: it may not read files outside itself.
    consumer = ROOT / "consumer"
    priv = consumer / "elixir" / "priv"
    assert (priv / "contract.json").read_text() == (consumer / "contract.json").read_text()
    assert (priv / "conformance.json").read_text() == (consumer / "conformance.json").read_text()
    for source in (ROOT / "consumer" / "elixir" / "lib").rglob("*.ex"):
        assert "../../../" not in source.read_text(), f"{source} reads outside consumer/elixir"
