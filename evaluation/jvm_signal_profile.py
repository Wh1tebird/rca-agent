"""Profile JVM/thread signals against RCA100 Ground Truth outside inference runtime."""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import duckdb

from rca_agent.core.runtime_registry import BASELINE_SIGNAL_CONTRACTS
from rca_agent.environment._private_adapter import PUBLIC_METRIC_MAP

FAMILIES = frozenset({"threadExhaustion", "memoryPressure", "fullGC"})
POD_SUFFIX = re.compile(r"-[a-f0-9]{9,10}-[a-z0-9]{5}$")


def _quantile(values: list[float], fraction: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def _service_targets(payload: dict[str, Any]) -> set[str]:
    return {POD_SUFFIX.sub("", str(name)) for name in payload.get("root_cause_entities", ())}


def build_profile(answer_key_root: Path, cases_root: Path) -> dict[str, Any]:
    raw_by_public = {
        public: raw
        for public, (entity_set, raw, _) in PUBLIC_METRIC_MAP["apm.service"].items()
        if entity_set in {"apm.metric.jvm", "apm.metric.thread"}
    }
    base_signals = sorted({item["base_signal"] for item in BASELINE_SIGNAL_CONTRACTS.values()})
    raw_metrics = [raw_by_public[signal] for signal in base_signals]
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for task_path in sorted(cases_root.glob("*/task.json")):
        case_id = task_path.parent.name
        answer_path = answer_key_root / f"{case_id}.gt.json"
        if not answer_path.exists():
            continue
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            truth = json.loads(answer_path.read_text(encoding="utf-8"))
            start = int(datetime.fromisoformat(task["alert_window"]["start"]).timestamp() * 1e6)
            end = int(datetime.fromisoformat(task["alert_window"]["end"]).timestamp() * 1e6)
            if end <= start:
                raise ValueError("invalid incident window")
            duration = end - start
            targets = _service_targets(truth)
            with duckdb.connect(":memory:") as connection:
                rows = connection.execute(
                    """
                    SELECT entity_name, metric,
                           avg(value) FILTER (WHERE time >= ? AND time < ?) AS before_mean,
                           avg(value) FILTER (WHERE time >= ? AND time <= ?) AS incident_mean,
                           max(value) FILTER (WHERE time >= ? AND time <= ?) AS incident_max,
                           count(*) FILTER (WHERE time >= ? AND time < ?) AS before_count,
                           count(*) FILTER (WHERE time >= ? AND time <= ?) AS incident_count
                    FROM read_parquet(?)
                    WHERE metric IN (SELECT unnest(?))
                    GROUP BY entity_name, metric
                    """,
                    [
                        start - duration, start, start, end, start, end,
                        start - duration, start, start, end,
                        str(task_path.parent / "metrics.parquet"), raw_metrics,
                    ],
                ).fetchall()
            by_raw = {value: key for key, value in raw_by_public.items()}
            root_types = tuple(truth.get("root_cause_types", ()))
            family = next((item for item in root_types if item in FAMILIES), None)
            for entity, raw, before, incident, incident_max, before_count, incident_count in rows:
                if before is None or incident is None:
                    continue
                records.append({
                    "case_id": case_id,
                    "entity": entity,
                    "signal": by_raw[raw],
                    "family": family,
                    "is_gt_entity": entity in targets,
                    "before_mean": before,
                    "incident_mean": incident,
                    "incident_max": incident_max,
                    "baseline_ratio": incident / before if before > 0 else None,
                    "zero_or_negative_baseline": before <= 0,
                    "before_count": before_count,
                    "incident_count": incident_count,
                })
        except (ValueError, KeyError, duckdb.Error) as exc:
            skipped.append({"case_id": case_id, "error_type": type(exc).__name__})

    summaries = []
    for derived, contract in BASELINE_SIGNAL_CONTRACTS.items():
        signal = contract["base_signal"]
        positives = [
            row for row in records
            if row["signal"] == signal and row["family"] in FAMILIES and row["is_gt_entity"]
        ]
        negatives = [
            row for row in records
            if row["signal"] == signal and not (
                row["family"] in FAMILIES and row["is_gt_entity"]
            )
        ]
        positive_ratios = [row["baseline_ratio"] for row in positives if row["baseline_ratio"]]
        negative_ratios = [row["baseline_ratio"] for row in negatives if row["baseline_ratio"]]
        threshold = contract["threshold"]
        summaries.append({
            "derived_signal": derived,
            "base_signal": signal,
            "threshold": threshold,
            "positive_records": len(positives),
            "negative_records": len(negatives),
            "positive_zero_or_negative_baseline": sum(
                row["zero_or_negative_baseline"] for row in positives
            ),
            "negative_zero_or_negative_baseline": sum(
                row["zero_or_negative_baseline"] for row in negatives
            ),
            "positive_ratio_median": median(positive_ratios) if positive_ratios else None,
            "positive_ratio_min": min(positive_ratios, default=None),
            "positive_threshold_match_rate": (
                sum(value >= threshold for value in positive_ratios) / len(positive_ratios)
                if positive_ratios else None
            ),
            "negative_ratio_p50": _quantile(negative_ratios, 0.5),
            "negative_ratio_p90": _quantile(negative_ratios, 0.9),
            "negative_ratio_p95": _quantile(negative_ratios, 0.95),
            "negative_ratio_max": max(negative_ratios, default=None),
            "negative_threshold_match_rate": (
                sum(value >= threshold for value in negative_ratios) / len(negative_ratios)
                if negative_ratios else None
            ),
        })
    positive_rows = [
        row for row in records if row["family"] in FAMILIES and row["is_gt_entity"]
    ]
    return {
        "scope": "evaluation_only",
        "case_count": len(tuple(cases_root.glob("*/task.json"))),
        "profiled_record_count": len(records),
        "positive_case_count": len({row["case_id"] for row in positive_rows}),
        "skipped_cases": skipped,
        "signal_summaries": summaries,
        "positive_case_records": positive_rows,
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# JVM and Thread Signal Profile",
        "",
        "This is an evaluation-only signal profile. Ground Truth is never loaded by inference.",
        "Ratios compare the incident-window mean with the immediately preceding equal window.",
        "A zero or missing baseline remains unavailable; it is never converted into positive evidence.",
        "",
        (
            f"Profiled {profile['profiled_record_count']} entity/signal records across "
            f"{profile['case_count']} cases; {profile['positive_case_count']} JVM/thread/GC cases "
            "had a resolvable service-level Ground-Truth entity."
        ),
        "",
        (
            "| Derived signal | Threshold | Positive n | Positive median | Positive match | "
            "Negative p95 | Negative match |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profile["signal_summaries"]:
        lines.append(
            f"| `{row['derived_signal']}` | {row['threshold']:.3g} | "
            f"{row['positive_records']} | {_format(row['positive_ratio_median'])} | "
            f"{_format(row['positive_threshold_match_rate'])} | "
            f"{_format(row['negative_ratio_p95'])} | "
            f"{_format(row['negative_threshold_match_rate'])} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- These distributions diagnose signal behavior; they are not an accuracy result.",
        (
            "- Threshold matches are never sufficient alone: Core still requires two independent "
            "support groups, refutation, alternative distinction, fresh verification evidence and "
            "arbitration."
        ),
        "- Generic GC frequency and pause signals describe GC pressure but cannot identify Full GC.",
        (
            "- `thread_pool_active` has inconsistent aggregate scale across cases. It is exposed as "
            "a neutral statistic and used only relative to its own baseline, never as a universal "
            "capacity ratio."
        ),
        "",
    ])
    return "\n".join(lines)


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"


def parser() -> argparse.ArgumentParser:
    dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--cases-root", type=Path, default=dataset / "cases")
    value.add_argument("--answer-key-root", type=Path, default=dataset / "answer_key")
    value.add_argument("--json-output", type=Path, default=Path("analysis/jvm_signal_profile.json"))
    value.add_argument("--markdown-output", type=Path, default=Path("analysis/jvm_signal_profile.md"))
    return value


def main() -> None:
    args = parser().parse_args()
    profile = build_profile(args.answer_key_root, args.cases_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(profile), encoding="utf-8")
    print(json.dumps({key: profile[key] for key in (
        "case_count", "profiled_record_count", "positive_case_count", "skipped_cases"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
