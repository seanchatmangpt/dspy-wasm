"""The numbers README.md states are the numbers the code enforces."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("dspy")

import dspy_capabilities as caps
import host

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


@pytest.mark.parametrize(
    ("pattern", "actual"),
    [
        (r"`MAX_REPEAT` \((\d+)\)", caps.MAX_REPEAT),
        (r"`MAX_TOTAL_STEPS` \((\d+)\)", caps.MAX_TOTAL_STEPS),
        (r"`MAX_FANOUT` \((\d+)\)", caps.MAX_FANOUT),
        (r"`MAX_ITERS` \((\d+)\)", caps.MAX_ITERS),
        (r"`--deadline SECONDS` \(default (\d+)\)", host.DEFAULT_DEADLINE_S),
        (r"at most (\d+) strings", host.MAX_EMBED_TEXTS),
        (r"`dimensions` in `\[1, (\d+)\]`", host.MAX_EMBED_DIMENSIONS),
        (r"at most (\d+) values", host.MAX_EMBED_VALUES),
        (r"above (\d+) bits", host.MAX_INT_BITS),
    ],
)
def test_stated_limit_is_the_enforced_limit(pattern: str, actual: float) -> None:
    match = re.search(pattern, README)
    assert match, f"README no longer states {pattern!r}"
    assert float(match.group(1)) == actual
