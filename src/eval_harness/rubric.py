"""Rubric loading and channel filtering.

A rubric is a versioned set of compliance criteria. Each criterion has a
severity and an optional list of channels it applies to (empty = all channels).
The judge only evaluates criteria applicable to a message's channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml

Severity = Literal["critical", "major", "minor"]


@dataclass
class Criterion:
    """A single compliance criterion within a rubric."""

    id: str
    title: str
    description: str
    severity: Severity
    # Channels this criterion applies to. Empty means "all channels".
    applies_to: list[str] = field(default_factory=list)

    def applies_to_channel(self, channel: str) -> bool:
        return not self.applies_to or channel in self.applies_to


@dataclass
class Rubric:
    """A named, versioned collection of compliance criteria."""

    id: str
    name: str
    description: str
    criteria: list[Criterion] = field(default_factory=list)

    def applicable(self, channel: str) -> list[Criterion]:
        """Return the criteria that apply to ``channel``."""
        return [c for c in self.criteria if c.applies_to_channel(channel)]

    @classmethod
    def load(cls, path: str) -> "Rubric":
        """Load a rubric from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        criteria = [
            Criterion(
                id=c["id"],
                title=c["title"],
                description=c["description"],
                severity=c["severity"],
                applies_to=list(c.get("applies_to", []) or []),
            )
            for c in data.get("criteria", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            criteria=criteria,
        )
