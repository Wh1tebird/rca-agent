"""Generate an evaluation-only observability capability matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rca_agent.core.runtime_registry import TRACE_SIGNAL_CONTRACTS
from rca_agent.environment._private_adapter import PUBLIC_METRIC_MAP

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
        "support": (),
        "refutation": (),
        "distinguishing": (),
        "missing": ("thread_pool_active", "thread_pool_max", "thread_pool_queue_wait_ms"),
        "queryability": "unsupported",
        "note": "Synthetic tests exist; the RCA100 runtime registry has no production recipe.",
    },
    {
        "mechanism": "memoryPressure",
        "support": ("memory_usage", "memory_usage_rate"),
        "refutation": (),
        "distinguishing": (),
        "missing": ("heap_usage", "gc_pause_duration", "allocation_rate"),
        "queryability": "partial",
        "note": "Usage is queryable, but independent JVM/GC mechanism evidence is absent.",
    },
    {
        "mechanism": "fullGC",
        "support": (),
        "refutation": (),
        "distinguishing": (),
        "missing": ("full_gc_count", "gc_pause_duration", "heap_after_gc"),
        "queryability": "unsupported",
        "note": "No executable GC-specific signal contract is registered.",
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


def build_matrix(answer_key_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for path in answer_key_root.glob("*.gt.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts.update(payload.get("root_cause_types", ()))
    trace_signals = sorted(TRACE_SIGNAL_CONTRACTS)
    metric_signals = sorted({
        signal for entity in PUBLIC_METRIC_MAP.values() for signal in entity
    })
    return {
        "classification": {
            "full": "two independent support groups plus executable refutation and distinction",
            "partial": "some executable evidence exists but generic verification cannot be completed",
            "unsupported": "no executable mechanism-specific evidence contract exists",
        },
        "declared_trace_signals": trace_signals,
        "declared_metric_signals": metric_signals,
        "mechanisms": [
            {**row, "rca100_case_count": counts.get(row["mechanism"], 0)}
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
        "| Mechanism | Support signals | Refutation | Distinguishing | Missing | Status | Cases |",
        "|---|---|---|---|---|---|---:|",
    ]
    symbols = {"full": "FULL", "partial": "PARTIAL", "unsupported": "UNSUPPORTED"}
    for row in matrix["mechanisms"]:
        values = [", ".join(row[key]) or "-" for key in (
            "support", "refutation", "distinguishing", "missing"
        )]
        lines.append(
            f"| {row['mechanism']} | {values[0]} | {values[1]} | {values[2]} | "
            f"{values[3]} | {symbols[row['queryability']]} | {row['rca100_case_count']} |"
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
    return "\n".join(lines) + "\n"


def main() -> None:
    default_dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key-root", type=Path,
                        default=default_dataset / "answer_key")
    parser.add_argument("--json-output", type=Path,
                        default=Path("analysis/capability_matrix.json"))
    parser.add_argument("--markdown-output", type=Path,
                        default=Path("analysis/capability_matrix.md"))
    args = parser.parse_args()
    matrix = build_matrix(args.answer_key_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(matrix), encoding="utf-8")


if __name__ == "__main__":
    main()
