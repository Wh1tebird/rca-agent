from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QueryBudget:
    initial: int
    remaining: int

    @classmethod
    def create(cls, units: int) -> QueryBudget:
        if units < 0:
            raise ValueError("query budget cannot be negative")
        return cls(initial=units, remaining=units)

    def reset(self) -> None:
        self.remaining = self.initial

    def can_spend(self, units: int) -> bool:
        return units >= 0 and self.remaining >= units

    def spend(self, units: int) -> tuple[int, int]:
        if not self.can_spend(units):
            raise RuntimeError("query budget exhausted")
        before = self.remaining
        self.remaining -= units
        return before, self.remaining
