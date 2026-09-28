"""Portable execution fabric: capability."""
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Record:
    subject: str
    data: dict[str, Any] = field(default_factory=dict)

    def valid(self) -> bool:
        return bool(self.subject)

    def snapshot(self) -> dict[str, Any]:
        return {"kind": "capability", "subject": self.subject, "data": dict(self.data)}
