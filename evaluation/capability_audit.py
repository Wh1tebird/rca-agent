"""Generate an evaluation-only observability capability matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from rca_agent.core.runtime_registry import (
    BASELINE_SIGNAL_CONTRACTS,
    METRIC_SIGNAL_CONTRACTS,
    NON_CAUSAL_SUPPORT_TRACE_RULES,
    TRACE_SIGNAL_CONTRACTS,
)
from rca_agent.environment._private_adapter import PUBLIC_METRIC_MAP
from rca_agent.environment.rca100 import Rca100Environment

RUNTIME_METRIC_RULES = frozenset({
    "cpu_usage_rate", "memory_usage_rate", "disk_usage_rate",
})

MECHANISM_REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "mechanism": "slowSQL",
        "support": ("sql_duration_share", "sql_shape"),
        "refutation": ("connection_acquisition_ms",),
        "distinguishing": ("sql_shape", "connection_acquisition_ms"),
        "missing": (),
        "queryability": "full",
        "note": "Full only when one issued trace contains a complete DB stage bundle.",
    },
    {
        "mechanism": "dbNetworkLatency",
        "support": ("operation_duration_ms", "connection_acquisition_ms"),
        "refutation": ("sql_duration_share", "sql_shape"),
        "distinguishing": (),
        "missing": ("db_network_rtt_ms", "db_server_execution_ms"),
        "queryability": "partial",
        "note": "Hikari/JDBC client spans cannot distinguish network delay from pool wait.",
    },
    {
        "mechanism": "connectionPoolExhaustion",
        "support": ("connection_acquisition_ms",),
        "refutation": ("sql_duration_share",),
        "distinguishing": (),
        "missing": ("connection_pool_active", "connection_pool_max", "connection_pool_wait_ms"),
        "queryability": "partial",
        "note": "Acquisition duration exists, but pool capacity and wait telemetry do not.",
    },
    {
        "mechanism": "threadExhaustion",
        "support": (
            "thread_pool_active_baseline_ratio",
            "thread_pool_queue_depth_baseline_ratio", "thread_pool_rejected_tasks",
            "jvm_runnable_threads_baseline_ratio", "jvm_thread_count_baseline_ratio",
            "jvm_thread_cpu_time_ns_baseline_ratio",
        ),
        "refutation": (
            "sql_duration_share", "connection_acquisition_ms",
            "jvm_gc_pause_seconds_delta_baseline_ratio",
        ),
        "distinguishing": (
            "sql_duration_share", "thread_pool_rejected_tasks",
            "jvm_gc_pause_seconds_delta_baseline_ratio",
        ),
        "missing": ("thread_pool_queue_wait_ms",),
        "queryability": "partial",
        "note": "Aggregate thread/JVM recipes exist; thresholds still need holdout validation.",
    },
    {
        "mechanism": "memoryPressure",
        "support": (
            "jvm_heap_used_bytes_baseline_ratio",
            "jvm_gc_collection_delta_baseline_ratio",
            "jvm_gc_pause_seconds_delta_baseline_ratio",
        ),
        "refutation": ("thread_pool_rejected_tasks",),
        "distinguishing": (
            "jvm_heap_used_bytes_baseline_ratio", "thread_pool_rejected_tasks",
        ),
        "missing": ("old_gen_usage", "allocation_rate", "heap_after_gc"),
        "queryability": "partial",
        "note": "Heap and generic GC telemetry exist, but memory-specific refutation is incomplete.",
    },
    {
        "mechanism": "fullGC",
        "support": (),
        "refutation": (),
        "distinguishing": (),
        "missing": ("full_gc_count", "full_gc_pause_duration", "heap_after_gc"),
        "queryability": "unsupported",
        "note": (
            "Generic GC count/pause is useful context but is not Full-GC-specific support; "
            "specific Full GC verification is blocked."
        ),
    },
    {
        "mechanism": "nodeCpuHigh",
        "support": ("cpu_usage_rate",),
        "refutation": (),
        "distinguishing": (),
        "missing": ("cpu_throttling", "run_queue"),
        "queryability": "partial",
        "note": "One utilization dimension cannot satisfy generic dynamic verification.",
    },
    {
        "mechanism": "diskIOHigh",
        "support": ("disk_usage_rate",),
        "refutation": (),
        "distinguishing": (),
        "missing": ("disk_io_wait", "disk_latency_ms", "disk_queue_depth"),
        "queryability": "partial",
        "note": "Disk usage is capacity, not evidence of IO saturation by itself.",
    },
)


def _metric_coverage(cases_root: Path) -> dict[tuple[str, str], dict[str, int]]:
    parquet_glob = str(cases_root / "*" / "metrics.parquet")
    with duckdb.connect(":memory:") as connection:
        rows = connection.execute(
            """
            SELECT entity_set, metric, count(*) AS samples,
                   count(DISTINCT entity_name) AS entities,
                   count(DISTINCT regexp_extract(filename, '/cases/([^/]+)/', 1)) AS cases
            FROM read_parquet(?, union_by_name=true, filename=true)
            GROUP BY entity_set, metric
            """,
            [parquet_glob],
        ).fetchall()
    return {
        (entity_set, metric): {"sample_count": samples, "entity_count": entities,
                               "case_count": cases}
        for entity_set, metric, samples, entities, cases in rows
    }


def _trace_coverage(cases_root: Path) -> dict[str, int]:
    parquet_glob = str(cases_root / "*" / "traces.parquet")
    with duckdb.connect(":memory:") as connection:
        row = connection.execute(
            """
            WITH spans AS (
                SELECT regexp_extract(filename, '/cases/([^/]+)/', 1) AS case_id,
                       traceId, serviceName,
                       max(CASE WHEN spanName = 'HikariDataSource.getConnection'
                                THEN 1 ELSE 0 END) AS has_connection,
                       max(CASE WHEN json_extract_string(attributes, '$."call.kind"') = 'sql'
                                THEN 1 ELSE 0 END) AS has_sql
                FROM read_parquet(?, union_by_name=true, filename=true)
                GROUP BY case_id, traceId, serviceName
            )
            SELECT count(DISTINCT case_id),
                   count(DISTINCT case_id) FILTER (WHERE has_connection = 1),
                   count(DISTINCT case_id) FILTER (WHERE has_sql = 1),
                   count(DISTINCT case_id)
                       FILTER (WHERE has_connection = 1 AND has_sql = 1)
            FROM spans
            """,
            [parquet_glob],
        ).fetchone()
    return dict(zip(
        ("trace_cases", "connection_span_cases", "sql_span_cases", "db_bundle_cases"),
        row,
        strict=True,
    ))


def _topology_entity_types(cases_root: Path) -> list[str]:
    entity_types = set()
    for path in cases_root.glob("*/topology.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entity_types.update(
            item["type"] for item in payload.get("entities", ()) if item.get("type")
        )
    return sorted(entity_types)


def _tool_inventory(cases_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capabilities = None
    valid_cases = 0
    failures: Counter[str] = Counter()
    for case_path in sorted(path.parent for path in cases_root.glob("*/task.json")):
        try:
            current = Rca100Environment.from_case(
                cases_root, case_path.name
            ).reset().capabilities
            capabilities = capabilities or current
            valid_cases += 1
        except (ValueError, FileNotFoundError) as exc:
            failures[type(exc).__name__] += 1
    if capabilities is None:
        raise ValueError("no RCA100 case can initialize the Environment capability contract")
    topology_types = _topology_entity_types(cases_root)
    entity_types = {
        "get_metric_series": sorted(PUBLIC_METRIC_MAP),
        "compare_metric_windows": sorted(PUBLIC_METRIC_MAP),
        "list_related_entities": topology_types,
        "get_service_trace_summary": ["apm.operation", "apm.service"],
        "get_trace_stage_breakdown": ["issued_trace_ref"],
        "inspect_topology": topology_types,
        "search_log_markers": topology_types,
    }
    tools = [
        {
            **item.model_dump(mode="json"),
            "supported_entity_types": entity_types[str(item.tool_name)],
        }
        for item in capabilities
    ]
    return tools, {
        "initializable_case_count": valid_cases,
        "uninitializable_case_count": sum(failures.values()),
        "failure_types": dict(sorted(failures.items())),
    }


def _signal_inventory(
    cases_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_coverage = _metric_coverage(cases_root)
    trace_coverage = _trace_coverage(cases_root)
    signals = []
    for signal, (unit, group, description, _, _) in TRACE_SIGNAL_CONTRACTS.items():
        rules = [signal]
        if signal == "sql_duration_share":
            rules.append("sql_duration_share_low")
        signals.append({
            "signal": signal,
            "entity_type": "apm.operation",
            "tool": "get_trace_stage_breakdown",
            "unit": unit,
            "data_type": "string" if signal == "sql_shape" else "number",
            "independent_group": group,
            "rule_ids": rules,
            "support_eligible": signal not in NON_CAUSAL_SUPPORT_TRACE_RULES,
            "runtime_registered": True,
            "raw_data_case_count": trace_coverage["db_bundle_cases"],
            "description": description,
        })
    for entity_type, mappings in PUBLIC_METRIC_MAP.items():
        for signal, (entity_set, raw_metric, unit) in mappings.items():
            coverage = metric_coverage.get((entity_set, raw_metric), {})
            contract = METRIC_SIGNAL_CONTRACTS.get(signal)
            runtime_registered = True  # Neutral descriptors are registered for issued entities.
            has_rule = signal in RUNTIME_METRIC_RULES or (
                contract is not None and "comparison" in contract
            )
            signals.append({
                "signal": signal,
                "entity_type": entity_type,
                "tool": "get_metric_series",
                "unit": unit,
                "data_type": "number",
                "independent_group": (
                    contract["group"] if contract is not None
                    else signal if signal in RUNTIME_METRIC_RULES else None
                ),
                "rule_ids": [signal] if has_rule else [],
                "support_eligible": has_rule,
                "runtime_registered": runtime_registered,
                "raw_metric": raw_metric,
                "raw_entity_set": entity_set,
                "raw_data_case_count": coverage.get("case_count", 0),
                "raw_sample_count": coverage.get("sample_count", 0),
                "raw_entity_count": coverage.get("entity_count", 0),
            })
    for entity_type, signal, contract in (
        (entity_type, signal, contract)
        for entity_type in ("apm.service", "apm.operation", "apm.instance")
        for signal, contract in BASELINE_SIGNAL_CONTRACTS.items()
        if contract["base_signal"] in PUBLIC_METRIC_MAP[entity_type]
    ):
        entity_set, raw_metric, _ = PUBLIC_METRIC_MAP[entity_type][contract["base_signal"]]
        coverage = metric_coverage.get((entity_set, raw_metric), {})
        signals.append({
            "signal": signal,
            "entity_type": entity_type,
            "tool": "compare_metric_windows",
            "unit": "ratio",
            "data_type": "number",
            "independent_group": contract["group"],
            "rule_ids": [signal],
            "support_eligible": contract.get("support_eligible", True),
            "runtime_registered": True,
            "raw_metric": raw_metric,
            "raw_entity_set": entity_set,
            "raw_data_case_count": coverage.get("case_count", 0),
            "raw_sample_count": coverage.get("sample_count", 0),
            "raw_entity_count": coverage.get("entity_count", 0),
            "description": contract["description"],
        })
    mapped = {
        (entity_set, raw_metric)
        for mappings in PUBLIC_METRIC_MAP.values()
        for entity_set, raw_metric, _ in mappings.values()
    }
    unmapped = [
        {"raw_entity_set": entity_set, "raw_metric": raw_metric, **coverage}
        for (entity_set, raw_metric), coverage in metric_coverage.items()
        if (entity_set, raw_metric) not in mapped
    ]
    return (
        sorted(signals, key=lambda item: (item["signal"], item["entity_type"])),
        sorted(unmapped, key=lambda item: (item["raw_entity_set"], item["raw_metric"])),
    )


def build_matrix(answer_key_root: Path, cases_root: Path | None = None) -> dict[str, Any]:
    cases_root = cases_root or answer_key_root.parent / "cases"
    counts: Counter[str] = Counter()
    for path in answer_key_root.glob("*.gt.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts.update(payload.get("root_cause_types", ()))
    signals, unmapped_raw_metrics = _signal_inventory(cases_root)
    tools, environment_coverage = _tool_inventory(cases_root)
    return {
        "classification": {
            "full": "two independent support groups plus executable refutation and distinction",
            "partial": "some executable evidence exists but generic verification cannot be completed",
            "unsupported": "no executable mechanism-specific evidence contract exists",
        },
        "dataset_case_count": len(tuple(cases_root.glob("*/task.json"))),
        "environment_coverage": environment_coverage,
        "tools": tools,
        "signals": signals,
        "unmapped_raw_metrics": unmapped_raw_metrics,
        "declared_trace_signals": sorted(TRACE_SIGNAL_CONTRACTS),
        "declared_metric_signals": sorted({
            signal for entity in PUBLIC_METRIC_MAP.values() for signal in entity
        }),
        "mechanisms": [
            {
                "available_but_unmapped": (),
                **row,
                "rca100_case_count": counts.get(row["mechanism"], 0),
            }
            for row in MECHANISM_REQUIREMENTS
        ],
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# RCA Observability Capability Matrix",
        "",
        "This report describes executable observation contracts, not a root-cause Catalog. Ground",
        "Truth counts are loaded only by this evaluation-only audit and never enter Agent inference.",
        "",
        "| Mechanism | Support | Refutation | Distinguishing | Unmapped raw evidence | Missing | Status | Cases |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    symbols = {"full": "FULL", "partial": "PARTIAL", "unsupported": "UNSUPPORTED"}
    for row in matrix["mechanisms"]:
        values = [", ".join(row[key]) or "-" for key in (
            "support", "refutation", "distinguishing", "available_but_unmapped", "missing"
        )]
        lines.append(
            f"| {row['mechanism']} | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {values[4]} | {symbols[row['queryability']]} | "
            f"{row['rca100_case_count']} |"
        )
    lines.extend(("", "## Signal semantics", ""))
    for signal, (_, group, description, _, _) in TRACE_SIGNAL_CONTRACTS.items():
        lines.append(f"- `{signal}` ({group}): {description}.")
    lines.extend((
        "",
        "`connection_acquisition_ms` is derived from the same Hikari span as",
        "`connection_duration_ms`; it is not an independent evidence group and is not network RTT.",
        "Likewise, `db_client_span_duration_ms` is the observed JDBC client span, not database-server",
        "execution time. Missing transport, pool-capacity and server-side signals remain explicit.",
        "",
        "## Mechanism notes",
        "",
    ))
    for row in matrix["mechanisms"]:
        lines.append(f"- **{row['mechanism']}**: {row['note']}")
    lines.extend(("", "## Automated tool inventory", ""))
    coverage = matrix["environment_coverage"]
    lines.append(
        f"Environment initialization: {coverage['initializable_case_count']} / "
        f"{matrix['dataset_case_count']} cases; failures are retained as data-quality gaps."
    )
    lines.append("")
    lines.append("| Tool | Available | Entity types | Window limit | Result limit |")
    lines.append("|---|---:|---|---:|---:|")
    for tool in matrix["tools"]:
        lines.append(
            f"| `{tool['tool_name']}` | {str(tool['available']).lower()} | "
            f"{', '.join(tool['supported_entity_types'])} | "
            f"{tool['max_window_seconds']} s | {tool['max_results']} |"
        )
    lines.extend(("", "## Automated signal inventory", ""))
    lines.append("| Signal | Entity | Tool | Unit | Group | Rules | Causal support | Raw cases |")
    lines.append("|---|---|---|---|---|---|---:|---:|")
    for signal in matrix["signals"]:
        lines.append(
            f"| `{signal['signal']}` | {signal['entity_type']} | `{signal['tool']}` | "
            f"{signal['unit'] or '-'} | {signal['independent_group'] or '-'} | "
            f"{', '.join(signal['rule_ids']) or '-'} | "
            f"{str(signal['support_eligible']).lower()} | {signal['raw_data_case_count']} |"
        )
    lines.extend((
        "",
        "Raw-case coverage proves that a source field exists somewhere in a Case. It does not prove",
        "that the relevant entity is discoverable or that enough independent evidence exists for",
        "verification. Those are reported separately instead of being inferred from file presence.",
    ))
    lines.extend(("", "## Raw metrics not exposed by the Environment", ""))
    lines.append("| Raw metric | Entity set | RCA100 cases | Samples |")
    lines.append("|---|---|---:|---:|")
    for metric in matrix["unmapped_raw_metrics"]:
        lines.append(
            f"| `{metric['raw_metric']}` | {metric['raw_entity_set']} | "
            f"{metric['case_count']} | {metric['sample_count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    default_dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key-root", type=Path,
                        default=default_dataset / "answer_key")
    parser.add_argument("--cases-root", type=Path, default=default_dataset / "cases")
    parser.add_argument("--json-output", type=Path,
                        default=Path("analysis/capability_matrix.json"))
    parser.add_argument("--markdown-output", type=Path,
                        default=Path("analysis/capability_matrix.md"))
    args = parser.parse_args()
    matrix = build_matrix(args.answer_key_root, args.cases_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(matrix), encoding="utf-8")


if __name__ == "__main__":
    main()
