from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rca_agent.core.cases import BlindCase

MAX_QUERY_WINDOW_SECONDS = 3600
TRACE_ID_PATTERN = r"^[0-9a-f]{32}$"


class AdapterQueryError(ValueError):
    pass


class WindowComparisonQuery(BaseModel):
    """Compare one metric across equal before and incident windows."""

    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1, max_length=300)
    entity_set: str = Field(min_length=1, max_length=120)
    signal: str = Field(min_length=1, max_length=120)
    incident_start_us: int = Field(gt=0)
    incident_end_us: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> WindowComparisonQuery:
        _validate_range(self.incident_start_us, self.incident_end_us, 1_000_000)
        return self


class SlowTraceQuery(BaseModel):
    """Find bounded slow operation spans overlapping an incident window."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1, max_length=200)
    operation: str = Field(min_length=1, max_length=500)
    start_ns: int = Field(gt=0)
    end_ns: int = Field(gt=0)
    min_duration_ms: float = Field(default=100.0, ge=0.0, le=3_600_000.0)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def validate_time_range(self) -> SlowTraceQuery:
        _validate_range(self.start_ns, self.end_ns, 1_000_000_000)
        return self


class SlowServiceTraceQuery(BaseModel):
    """Find bounded slow server spans for one issued service."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1, max_length=200)
    start_ns: int = Field(gt=0)
    end_ns: int = Field(gt=0)
    min_duration_ms: float = Field(default=100.0, ge=0.0, le=3_600_000.0)
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def validate_time_range(self) -> SlowServiceTraceQuery:
        _validate_range(self.start_ns, self.end_ns, 1_000_000_000)
        return self


class TraceByIdQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    limit: int = Field(default=200, ge=1, le=500)


class TopologyQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(min_length=1)
    relation: str | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_index: int
    query_ref: str
    rows: tuple[dict[str, Any], ...]


def _validate_range(start: int, end: int, units_per_second: int) -> None:
    if end <= start:
        raise ValueError("query end must be after start")
    if end - start > MAX_QUERY_WINDOW_SECONDS * units_per_second:
        raise ValueError(f"query window cannot exceed {MAX_QUERY_WINDOW_SECONDS} seconds")


class Rca100ObservationAdapter:
    """Read-only typed access to the bounded RCA100 queries used by the Agent.

    This is the minimal runtime slice extracted from the frozen ``rca-core``
    adapter. SQL and paths remain private; callers provide validated filters.
    """

    def __init__(self, case: BlindCase) -> None:
        self._case = case
        self._connection = duckdb.connect(":memory:")
        self._query_count = 0

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def compare_metric_windows(self, query: WindowComparisonQuery) -> QueryResult:
        self._query_count += 1
        duration = query.incident_end_us - query.incident_start_us
        before_start = query.incident_start_us - duration
        row = self._connection.execute(
            """
            SELECT avg(value) FILTER (WHERE time >= ? AND time < ?),
                   avg(value) FILTER (WHERE time >= ? AND time < ?),
                   min(value) FILTER (WHERE time >= ? AND time < ?),
                   max(value) FILTER (WHERE time >= ? AND time < ?),
                   count(*) FILTER (WHERE time >= ? AND time < ?),
                   count(*) FILTER (WHERE time >= ? AND time < ?)
            FROM read_parquet(?)
            WHERE entity_name = ? AND entity_set = ? AND metric = ?
            """,
            [
                before_start,
                query.incident_start_us,
                query.incident_start_us,
                query.incident_end_us,
                query.incident_start_us,
                query.incident_end_us,
                query.incident_start_us,
                query.incident_end_us,
                before_start,
                query.incident_start_us,
                query.incident_start_us,
                query.incident_end_us,
                str(self._case.observations["metrics"]),
                query.entity_name,
                query.entity_set,
                query.signal,
            ],
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            raise AdapterQueryError(
                f"comparison windows unavailable for {query.entity_set}/"
                f"{query.entity_name}/{query.signal}"
            )
        data = ({
            "signal": query.signal,
            "before_mean": row[0],
            "incident_mean": row[1],
            "incident_min": row[2],
            "incident_max": row[3],
            "before_count": row[4],
            "incident_count": row[5],
            "ratio": row[1] / row[0] if row[0] != 0 else None,
        },)
        return QueryResult(
            self._query_count,
            self._ref(
                "window_comparison",
                {
                    "entity_name": query.entity_name,
                    "entity_set": query.entity_set,
                    "signal": query.signal,
                    "incident_start_us": query.incident_start_us,
                    "incident_end_us": query.incident_end_us,
                },
            ),
            data,
        )

    def find_slow_traces(self, query: SlowTraceQuery) -> QueryResult:
        self._query_count += 1
        self._require_trace_value("serviceName", query.service)
        self._require_trace_value("spanName", query.operation, service=query.service)
        rows = self._connection.execute(
            """
            SELECT traceId, spanId, parentSpanId, serviceName, spanName,
                   try_cast(startTime AS UBIGINT) AS start_time_ns,
                   try_cast(duration AS DOUBLE) / 1000000.0 AS duration_ms,
                   statusCode, statusMessage
            FROM read_parquet(?)
            WHERE serviceName = ? AND spanName = ?
              AND try_cast(startTime AS UBIGINT) < ?
              AND try_cast(startTime AS UBIGINT) + try_cast(duration AS UBIGINT) > ?
              AND try_cast(duration AS DOUBLE) / 1000000.0 >= ?
            ORDER BY duration_ms DESC, traceId, spanId
            LIMIT ?
            """,
            [
                str(self._case.observations["traces"]),
                query.service,
                query.operation,
                query.end_ns,
                query.start_ns,
                query.min_duration_ms,
                query.limit,
            ],
        ).fetchall()
        columns = (
            "trace_id",
            "span_id",
            "parent_span_id",
            "service",
            "span_name",
            "start_time_ns",
            "duration_ms",
            "status_code",
            "status_message",
        )
        data = tuple(dict(zip(columns, row, strict=True)) for row in rows)
        return QueryResult(
            self._query_count,
            self._ref(
                "slow_traces",
                {
                    "service": query.service,
                    "operation": query.operation,
                    "start_ns": query.start_ns,
                    "end_ns": query.end_ns,
                    "min_duration_ms": query.min_duration_ms,
                    "limit": query.limit,
                    "time_match": "overlap",
                },
            ),
            data,
        )

    def find_slow_service_traces(self, query: SlowServiceTraceQuery) -> QueryResult:
        self._query_count += 1
        rows = self._connection.execute(
            """
            SELECT traceId, spanId, parentSpanId, serviceName, spanName,
                   try_cast(startTime AS UBIGINT) AS start_time_ns,
                   try_cast(duration AS DOUBLE) / 1000000.0 AS duration_ms,
                   statusCode, statusMessage
            FROM read_parquet(?)
            WHERE serviceName = ?
              AND try_cast(startTime AS UBIGINT) < ?
              AND try_cast(startTime AS UBIGINT) + try_cast(duration AS UBIGINT) > ?
              AND try_cast(duration AS DOUBLE) / 1000000.0 >= ?
              AND try_cast(json_extract_string(attributes, '$."call.kind"') AS VARCHAR)
                  NOT IN ('sql', 'internal')
            ORDER BY duration_ms DESC, traceId, spanId
            LIMIT ?
            """,
            [
                str(self._case.observations["traces"]), query.service,
                query.end_ns, query.start_ns, query.min_duration_ms, query.limit,
            ],
        ).fetchall()
        columns = (
            "trace_id", "span_id", "parent_span_id", "service", "span_name",
            "start_time_ns", "duration_ms", "status_code", "status_message",
        )
        data = tuple(dict(zip(columns, row, strict=True)) for row in rows)
        return QueryResult(
            self._query_count,
            self._ref(
                "slow_service_traces",
                {
                    "service": query.service,
                    "start_ns": query.start_ns,
                    "end_ns": query.end_ns,
                    "min_duration_ms": query.min_duration_ms,
                    "limit": query.limit,
                    "time_match": "overlap",
                },
            ),
            data,
        )

    def get_trace(self, query: TraceByIdQuery) -> QueryResult:
        self._query_count += 1
        rows = self._connection.execute(
            """
            SELECT traceId, spanId, parentSpanId, serviceName, spanName,
                   try_cast(startTime AS UBIGINT),
                   try_cast(duration AS DOUBLE) / 1000000.0,
                   statusCode, statusMessage, attributes
            FROM read_parquet(?)
            WHERE traceId = ?
            ORDER BY try_cast(startTime AS UBIGINT), try_cast(duration AS UBIGINT) DESC
            LIMIT ?
            """,
            [str(self._case.observations["traces"]), query.trace_id, query.limit + 1],
        ).fetchall()
        if not rows:
            raise AdapterQueryError(f"trace_id is not present in this case: {query.trace_id}")
        if len(rows) > query.limit:
            raise AdapterQueryError(
                f"trace has more than the bounded limit of {query.limit} spans"
            )
        data = tuple(self._safe_span(row) for row in rows)
        return QueryResult(
            self._query_count,
            self._ref("trace", {"trace_id": query.trace_id, "limit": query.limit}),
            data,
        )

    def topology_neighbors(self, query: TopologyQuery) -> QueryResult:
        self._query_count += 1
        topology = json.loads(self._case.observations["topology"].read_text())
        entities = {item["id"]: item for item in topology["entities"]}
        entity_ids = {
            entity_id
            for entity_id, item in entities.items()
            if item.get("name") == query.entity_name
        }
        if not entity_ids:
            raise AdapterQueryError(
                f"topology entity is not present in this case: {query.entity_name}"
            )
        rows: list[dict[str, Any]] = []
        for edge in topology["edges"]:
            if query.relation is not None and edge.get("relation") != query.relation:
                continue
            if edge["src"] in entity_ids:
                neighbor = entities[edge["dst"]]
                rows.append(
                    {
                        "direction": "outgoing",
                        "relation": edge["relation"],
                        "neighbor_id": edge["dst"],
                        "neighbor_name": neighbor.get("name", ""),
                        "neighbor_type": neighbor.get("type", ""),
                    }
                )
            if edge["dst"] in entity_ids:
                neighbor = entities[edge["src"]]
                rows.append(
                    {
                        "direction": "incoming",
                        "relation": edge["relation"],
                        "neighbor_id": edge["src"],
                        "neighbor_name": neighbor.get("name", ""),
                        "neighbor_type": neighbor.get("type", ""),
                    }
                )
        rows.sort(key=lambda item: (item["direction"], item["relation"], item["neighbor_name"]))
        return QueryResult(
            self._query_count,
            self._ref(
                "topology",
                {"entity_name": query.entity_name, "relation": query.relation},
            ),
            tuple(rows),
        )

    def _require_trace_value(
        self,
        column: Literal["serviceName", "spanName"],
        value: str,
        service: str | None = None,
    ) -> None:
        path = str(self._case.observations["traces"])
        if column == "serviceName":
            row = self._connection.execute(
                "SELECT 1 FROM read_parquet(?) WHERE serviceName = ? LIMIT 1",
                [path, value],
            ).fetchone()
        else:
            row = self._connection.execute(
                """
                SELECT 1 FROM read_parquet(?)
                WHERE spanName = ? AND serviceName = ? LIMIT 1
                """,
                [path, value, service],
            ).fetchone()
        if row is None:
            label = "service" if column == "serviceName" else "operation"
            raise AdapterQueryError(f"trace {label} is not present in this case: {value}")

    @staticmethod
    def _safe_span(row: tuple[Any, ...]) -> dict[str, Any]:
        try:
            attributes = json.loads(row[9] or "{}")
        except json.JSONDecodeError:
            attributes = {}
        allowed_attributes = {
            key: attributes[key]
            for key in (
                "call.kind",
                "component.name",
                "db.name",
                "db.statement.id",
                "endpoint",
                "http.status_code",
                "opType",
                "rpc",
                "sql",
                "tableName",
            )
            if key in attributes
        }
        sql = allowed_attributes.get("sql")
        if isinstance(sql, str) and len(sql) > 4000:
            allowed_attributes["sql"] = sql[:4000]
        return {
            "trace_id": row[0],
            "span_id": row[1],
            "parent_span_id": row[2],
            "service": row[3],
            "span_name": row[4],
            "start_time_ns": row[5],
            "duration_ms": row[6],
            "status_code": row[7],
            "status_message": row[8],
            "attributes": allowed_attributes,
        }

    def _ref(self, source: str, arguments: dict[str, Any]) -> str:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{self._case.task.task_id}:q{self._query_count}:{source}:{encoded}"
