from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SqlShape(StrEnum):
    SIMPLE_POINT_LOOKUP = "simple_point_lookup"
    COMPLEX_CROSS_JOIN = "complex_cross_join"
    OTHER = "other"


class TraceStageObservation(BaseModel):
    """Query-derived fact independent of candidate interpretation."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    operation_duration_ms: float = Field(gt=0.0)
    connection_duration_ms: float = Field(ge=0.0)
    sql_duration_ms: float = Field(ge=0.0)
    sql_duration_share: float = Field(ge=0.0)
    sql_shape: SqlShape
    sql_span_name: str
    sql_preview: str
    stage_parent_ids: dict[str, str]
    query_refs: list[str] = Field(min_length=1)


def extract_trace_stage_observation(
    rows: tuple[dict[str, Any], ...],
    service: str,
    operation: str,
    query_refs: list[str],
) -> TraceStageObservation:
    """Derive bounded stage facts from a trace returned by the typed adapter."""

    operation_spans = [
        row for row in rows if row["service"] == service and row["span_name"] == operation
    ]
    connection_spans = [
        row
        for row in rows
        if row["service"] == service and row["span_name"] == "HikariDataSource.getConnection"
    ]
    sql_spans = [
        row
        for row in rows
        if row["service"] == service and row["attributes"].get("call.kind") == "sql"
    ]
    if not operation_spans or not connection_spans or not sql_spans:
        raise RuntimeError("trace does not contain operation, connection, and SQL stages")

    root = max(operation_spans, key=lambda row: row["duration_ms"])
    connection = max(connection_spans, key=lambda row: row["duration_ms"])
    sql_span = max(sql_spans, key=lambda row: row["duration_ms"])
    sql_text = str(sql_span["attributes"].get("sql", ""))
    normalized_sql = " ".join(sql_text.casefold().split())
    complex_cross_join = normalized_sql.count(" cross join ") >= 2
    simple_point_lookup = (
        "where" in normalized_sql
        and "product_id=?" in normalized_sql.replace(" ", "")
        and not complex_cross_join
    )
    if complex_cross_join:
        sql_shape = SqlShape.COMPLEX_CROSS_JOIN
    elif simple_point_lookup:
        sql_shape = SqlShape.SIMPLE_POINT_LOOKUP
    else:
        sql_shape = SqlShape.OTHER

    return TraceStageObservation(
        trace_id=root["trace_id"],
        operation_duration_ms=root["duration_ms"],
        connection_duration_ms=connection["duration_ms"],
        sql_duration_ms=sql_span["duration_ms"],
        sql_duration_share=sql_span["duration_ms"] / root["duration_ms"],
        sql_shape=sql_shape,
        sql_span_name=sql_span["span_name"],
        sql_preview=sql_text[:500],
        stage_parent_ids={
            "connection_parent": connection["parent_span_id"],
            "sql_parent": sql_span["parent_span_id"],
            "operation_span": root["span_id"],
        },
        query_refs=query_refs,
    )
