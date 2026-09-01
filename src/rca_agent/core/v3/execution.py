from __future__ import annotations

from typing import Protocol

from rca_agent.core.v3.models import Observation, TypedQueryPlan


class ObservationProvider(Protocol):
    def scout(self) -> list[Observation]: ...

    def execute(self, plans: list[TypedQueryPlan]) -> tuple[list[Observation], list[str]]: ...


class DictionaryObservationProvider:
    """Deterministic provider used by tests and dataset-specific adapters."""

    def __init__(self, values: dict[str, float | str], entity: str = "test-entity") -> None:
        self._values = values
        self._entity = entity
        self.executed_plans: list[TypedQueryPlan] = []

    def scout(self) -> list[Observation]:
        return [
            Observation(
                signal=signal,
                value=value,
                query_ref=f"scout:{signal}",
                observed_entity=self._entity,
            )
            for signal, value in sorted(self._values.items())
            if signal in {
                "node_cpu_peak",
                "node_cpu_high_fraction",
                "node_memory_peak",
                "node_disk_peak",
                "service_error_ratio",
                "service_latency_ratio",
                "service_workload_ratio",
            }
        ]

    def execute(self, plans: list[TypedQueryPlan]) -> tuple[list[Observation], list[str]]:
        self.executed_plans.extend(plans)
        observations = []
        missing = []
        for plan in plans:
            if plan.signal not in self._values:
                missing.append(f"{plan.condition_id}: observation unavailable")
                continue
            observations.append(
                Observation(
                    signal=plan.signal,
                    value=self._values[plan.signal],
                    query_ref=f"typed:{plan.query_type}:{plan.signal}",
                    observed_entity=self._entity,
                )
            )
        return observations, missing
