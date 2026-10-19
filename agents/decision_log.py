from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionLogEntry:
    agent: str
    action: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionLog:
    entries: list[DecisionLogEntry] = field(default_factory=list)

    def add(
        self,
        agent: str,
        action: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            DecisionLogEntry(
                agent=agent,
                action=action,
                reason=reason,
                details=details or {},
            )
        )

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "Agent": entry.agent,
                "动作": entry.action,
                "依据": entry.reason,
                "细节": entry.details,
            }
            for entry in self.entries
        ]
