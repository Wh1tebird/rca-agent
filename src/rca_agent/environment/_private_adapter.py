from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from rca_agent.core.cases import BlindCase, FlatBlindCaseRepository
from rca_agent.core.rca100_adapter import (
    AdapterQueryError,
    Rca100ObservationAdapter,
    SlowServiceTraceQuery,
    SlowTraceQuery,
    TopologyQuery,
    TraceByIdQuery,
    WindowComparisonQuery,
)
from rca_agent.core.trace_analysis import extract_trace_stage_observation

PUBLIC_METRIC_MAP: dict[str, dict[str, tuple[str, str, str | None]]] = {
    "apm.operation": {
        "request_latency_seconds": ("apm.operation", "avg_request_latency_seconds", "s"),
        "request_count": ("apm.operation", "request_count", "count"),
        "slow_count": ("apm.operation", "slow_count", "count"),
        "error_count": ("apm.operation", "error_count", "count"),
        "error_rate": ("apm.operation", "error_rate", "ratio"),
    },
    "apm.service": {
        "request_latency_seconds": ("apm.service.legacy", "latency", "s"),
        "request_count": ("apm.service.legacy", "workload", "count"),
        "slow_count": ("apm.service.legacy", "slow", "count"),
        "error_count": ("apm.service.legacy", "error", "count"),
        "cpu_usage": ("apm.service.legacy", "cpu", None),
        "memory_usage": ("apm.service.legacy", "mem", None),
        "thread_pool_active": (
            "apm.metric.jvm", "arms_thread_pool_active_thread_count", "count"
        ),
        "thread_pool_max": (
            "apm.metric.jvm", "arms_thread_pool_max_thread_count", "count"
        ),
        "thread_pool_queue_depth": (
            "apm.metric.jvm", "arms_thread_pool_queue_size", "count"
        ),
        "thread_pool_rejected_tasks": (
            "apm.metric.jvm", "arms_thread_pool_rejected_task_count", "count"
        ),
        "jvm_runnable_threads": (
            "apm.metric.thread", "arms_grouped_thread_runnable_count", "count"
        ),
        "jvm_thread_count": ("apm.metric.jvm", "arms_jvm_threads_count", "count"),
        "jvm_heap_used_bytes": ("apm.metric.jvm", "arms_jvm_mem_used_bytes", "bytes"),
        "jvm_heap_max_bytes": ("apm.metric.jvm", "arms_jvm_mem_max_bytes", "bytes"),
        "jvm_gc_collection_delta": ("apm.metric.jvm", "arms_jvm_gc_delta", "count"),
        "jvm_gc_pause_seconds_delta": (
            "apm.metric.jvm", "arms_jvm_gc_seconds_delta", "s"
        ),
    },
    "k8s.node": {
        "cpu_usage_rate": ("k8s.node", "node_cpu_usage_rate", "percent"),
        "memory_usage_rate": ("k8s.node", "node_memory_usage_rate", "percent"),
        "disk_usage_rate": ("k8s.node", "node_disk_usage_rate", "percent"),
        "ready_status": ("k8s.node", "node_ready_status", "state"),
        "not_ready_status": ("k8s.node", "node_not_ready_status", "state"),
    },
    "k8s.deployment": {
        "cpu_usage_total": ("k8s.deployment", "deployment_cpu_usage_total", None),
        "memory_usage_total": ("k8s.deployment", "deployment_memory_usage_total", None),
        "ready_replicas": ("k8s.deployment", "deployment_ready_replicas", "count"),
    },
}


@dataclass(frozen=True, slots=True)
class EntityBinding:
    public_ref: str
    name: str
    entity_type: str


@dataclass(frozen=True, slots=True)
class MetricTarget:
    entity_set: str
    raw_signal: str
    unit: str | None


@dataclass(frozen=True, slots=True)
class TraceSelection:
    raw_trace_id: str
    public_trace_ref: str
    service: str
    operation: str
    source_entity_ref: str


class PrivateRca100Adapter:
    """Host-side data adapter. Instances and return types never cross the Agent contract."""

    def __init__(self, cases_root: Path, case_selector: str) -> None:
        resolved = cases_root.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError("configured RCA100 cases directory does not exist")
        self.__case: BlindCase = FlatBlindCaseRepository(resolved).load(case_selector)
        self.__entities = self._build_entity_registry()

    @property
    def event_time(self) -> datetime:
        return datetime.fromisoformat(self.__case.task.alert_trigger_time)

    @property
    def alert_summary(self) -> str:
        return self.__case.task.alert_title

    @property
    def incident_start(self) -> datetime:
        return datetime.fromisoformat(self.__case.task.alert_window.start)

    @property
    def incident_end(self) -> datetime:
        return datetime.fromisoformat(self.__case.task.alert_window.end)

    @property
    def alert_entity(self) -> EntityBinding:
        task_entity = self.__case.task.alert_entity
        return self._binding(task_entity.entity_name, task_entity.entity_type)

    def metric_target(self, entity: EntityBinding, signal: str) -> MetricTarget | None:
        spec = PUBLIC_METRIC_MAP.get(entity.entity_type, {}).get(signal)
        return MetricTarget(*spec) if spec is not None else None

    def metric_series(
        self,
        entity: EntityBinding,
        target: MetricTarget,
        start_us: int,
        end_us: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        connection = duckdb.connect(":memory:")
        try:
            rows = connection.execute(
                """
                SELECT time, value
                FROM read_parquet(?)
                WHERE entity_name = ? AND entity_set = ? AND metric = ?
                  AND time >= ? AND time <= ?
                ORDER BY time, value
                LIMIT ?
                """,
                [
                    str(self.__case.observations["metrics"]),
                    entity.name,
                    target.entity_set,
                    target.raw_signal,
                    start_us,
                    end_us,
                    limit + 1,
                ],
            ).fetchall()
        finally:
            connection.close()
        truncated = len(rows) > limit
        return [
            {
                "time": datetime.fromtimestamp(row[0] / 1_000_000, tz=UTC),
                "value": row[1],
            }
            for row in rows[:limit]
        ], truncated

    def compare_metric_windows(
        self,
        entity: EntityBinding,
        target: MetricTarget,
        incident_start_us: int,
        incident_end_us: int,
    ) -> dict[str, Any] | None:
        with Rca100ObservationAdapter(self.__case) as adapter:
            try:
                result = adapter.compare_metric_windows(
                    WindowComparisonQuery(
                        entity_name=entity.name,
                        entity_set=target.entity_set,
                        signal=target.raw_signal,
                        incident_start_us=incident_start_us,
                        incident_end_us=incident_end_us,
                    )
                )
            except AdapterQueryError:
                return None
        return dict(result.rows[0]) if result.rows else None

    def related_entities(self, entity: EntityBinding) -> list[dict[str, Any]]:
        if entity.entity_type == "apm.operation" and "::" in entity.name:
            service = entity.name.partition("::")[0]
            binding = self._binding(service, "apm.service")
            return [{"binding": binding, "direction": "parent", "relation": "operation_of"}]
        with Rca100ObservationAdapter(self.__case) as adapter:
            try:
                result = adapter.topology_neighbors(TopologyQuery(entity_name=entity.name))
            except AdapterQueryError:
                return []
        related = []
        for row in result.rows:
            binding = self._binding(str(row["neighbor_name"]), str(row["neighbor_type"]))
            related.append(
                {
                    "binding": binding,
                    "direction": row["direction"],
                    "relation": row["relation"],
                }
            )
        return related

    def trace_summary(
        self,
        entity: EntityBinding,
        start_ns: int,
        end_ns: int,
        min_duration_ms: float,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[TraceSelection]]:
        if entity.entity_type == "apm.operation":
            service, separator, operation = entity.name.partition("::")
            if not separator:
                operation = None
        elif entity.entity_type == "apm.service":
            service, operation = entity.name, None
        else:
            raise ValueError("trace summaries require an operation or service entity")
        with Rca100ObservationAdapter(self.__case) as adapter:
            if operation is None:
                result = adapter.find_slow_service_traces(SlowServiceTraceQuery(
                    service=service,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    min_duration_ms=min_duration_ms,
                    limit=limit,
                ))
            else:
                result = adapter.find_slow_traces(SlowTraceQuery(
                    service=service,
                    operation=operation,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    min_duration_ms=min_duration_ms,
                    limit=limit,
                ))
        rows: list[dict[str, Any]] = []
        selections: list[TraceSelection] = []
        for row in result.rows:
            trace_ref = f"trace-{_digest(row['trace_id'])}"
            selections.append(
                TraceSelection(
                    raw_trace_id=row["trace_id"],
                    public_trace_ref=trace_ref,
                    service=service,
                    operation=row["span_name"],
                    source_entity_ref=entity.public_ref,
                )
            )
            rows.append(
                {
                    "trace_ref": trace_ref,
                    "start_time": datetime.fromtimestamp(
                        row["start_time_ns"] / 1_000_000_000, tz=UTC
                    ),
                    "duration_ms": row["duration_ms"],
                    "status_code": row["status_code"],
                }
            )
        return rows, selections

    def trace_stage_breakdown(
        self, selection: TraceSelection
    ) -> tuple[dict[str, Any], EntityBinding, EntityBinding, str]:
        with Rca100ObservationAdapter(self.__case) as adapter:
            result = adapter.get_trace(TraceByIdQuery(trace_id=selection.raw_trace_id, limit=200))
        extracted = extract_trace_stage_observation(
            result.rows,
            selection.service,
            selection.operation,
            ["private-query-ref"],
        )
        entity = self._binding(
            f"{extracted.service}::{extracted.operation}", "apm.operation"
        )
        service = self._binding(extracted.service, "apm.service")
        return {
            "operation_duration_ms": extracted.operation_duration_ms,
            "connection_duration_ms": extracted.connection_duration_ms,
            "connection_acquisition_ms": extracted.connection_duration_ms,
            "sql_duration_ms": extracted.sql_duration_ms,
            "db_client_span_duration_ms": extracted.sql_duration_ms,
            "sql_duration_share": extracted.sql_duration_share,
            "sql_shape": extracted.sql_shape.value,
        }, entity, service, extracted.resolution

    def _build_entity_registry(self) -> dict[tuple[str, str], EntityBinding]:
        topology = json.loads(self.__case.observations["topology"].read_text())
        registry: dict[tuple[str, str], EntityBinding] = {}
        for raw in topology.get("entities", []):
            name = str(raw.get("name", ""))
            entity_type = str(raw.get("type", "unknown"))
            if name:
                binding = _new_binding(name, entity_type)
                registry[(name, entity_type)] = binding
        task_entity = self.__case.task.alert_entity
        registry[(task_entity.entity_name, task_entity.entity_type)] = _new_binding(
            task_entity.entity_name, task_entity.entity_type
        )
        if "::" in task_entity.entity_name:
            service = task_entity.entity_name.partition("::")[0]
            registry.setdefault((service, "apm.service"), _new_binding(service, "apm.service"))
        return registry

    def _binding(self, name: str, entity_type: str) -> EntityBinding:
        return self.__entities.get((name, entity_type), _new_binding(name, entity_type))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _new_binding(name: str, entity_type: str) -> EntityBinding:
    identity = f"{entity_type}\0{name}"
    return EntityBinding(
        public_ref=f"ent-{_digest(identity)}",
        name=name,
        entity_type=entity_type,
    )
