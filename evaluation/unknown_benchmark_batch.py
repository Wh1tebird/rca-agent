"""Run a preregistered set of real-model Catalog-holdout experiments."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluation.unknown_benchmark import parser as case_parser
from evaluation.unknown_benchmark import run_case


def mechanism_matches(root_cause: str, mechanism: str) -> bool:
    """Conservative post-freeze lexical scorer; ambiguous cases still require manual review."""
    normalized = mechanism.casefold()
    if root_cause == "dbNetworkLatency":
        return (
            "network" in normalized
            and "latenc" in normalized
            and ("database" in normalized or "db" in normalized)
        )
    if root_cause == "slowSQL":
        return any(
            re.search(pattern, normalized)
            for pattern in (
                r"\bslow\s+sql\b",
                r"\bcomplex\b.{0,60}\bsql\b",
                r"\bsql\s+(execution|query)\b.{0,80}\b(slow|long|dominat|high|latenc)",
            )
        )
    raise ValueError("unsupported Ground-Truth mechanism in holdout manifest")


def score_case(result: dict[str, Any]) -> dict[str, Any]:
    root_causes = result["ground_truth"]["root_cause_types"]
    mechanisms = result["dynamic_mechanisms"]
    matched = [
        mechanism for mechanism in mechanisms
        if any(mechanism_matches(root_cause, mechanism) for root_cause in root_causes)
    ]
    status = result["status"].strip()
    verified = status == "discovered_verified"
    return {
        "case_id": result["case_id"],
        "holdout_candidate": result["holdout_candidate"],
        "status": status,
        "discovery_completed": (result.get("discovery") or {}).get("status") == "completed",
        "compiled": bool(mechanisms),
        "gt_mechanism_proposed": bool(matched),
        "matched_mechanisms": matched,
        "discovered_verified": verified,
        "correct_verified": verified and bool(result.get("winner")) and any(
            mechanism_matches(root_cause, result["winner"]["mechanism"])
            for root_cause in root_causes
        ),
        "wrong_verified": verified and (
            not result.get("winner") or not any(
                mechanism_matches(root_cause, result["winner"]["mechanism"])
                for root_cause in root_causes
            )
        ),
        "replay_valid": result["replay_valid"],
    }


def summarize(scores: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    total = len(scores) + len(failures)
    status_counts: dict[str, int] = {}
    for item in scores:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "unknown_cases": total,
        "completed_cases": len(scores),
        "execution_failures": len(failures),
        "discovery_completed": sum(item["discovery_completed"] for item in scores),
        "compiled": sum(item["compiled"] for item in scores),
        "gt_mechanism_proposed": sum(item["gt_mechanism_proposed"] for item in scores),
        "discovered_verified": sum(item["discovered_verified"] for item in scores),
        "correct_verified": sum(item["correct_verified"] for item in scores),
        "wrong_verified": sum(item["wrong_verified"] for item in scores),
        "replay_valid": sum(item["replay_valid"] for item in scores),
        "status_counts": status_counts,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("unknown_holdout_manifest.json"),
    )
    value.add_argument("--output-dir", type=Path, default=Path("analysis/unknown_benchmark_batch"))
    value.add_argument("--summary-output", type=Path,
                       default=Path("analysis/unknown_benchmark_batch_summary.json"))
    value.add_argument("--model", default="deepseek-chat")
    value.add_argument("--thinking-enabled", action="store_true")
    value.add_argument("--rerun-existing", action="store_true")
    value.add_argument("--model-timeout-seconds", type=float, default=115)
    value.add_argument("--discovery-timeout-seconds", type=float, default=120)
    return value


def main() -> None:
    args = parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("holdout manifest must be a nonempty list")
    case_ids = [item.get("case_id") for item in manifest if isinstance(item, dict)]
    if len(case_ids) != len(manifest) or len(set(case_ids)) != len(case_ids):
        raise ValueError("holdout manifest cases must be objects with unique case_id values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    defaults = case_parser().parse_args([])
    scores: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in manifest:
        case_id = item["case_id"]
        try:
            output = args.output_dir / f"{case_id}.json"
            result = None
            if output.exists() and not args.rerun_existing:
                cached = json.loads(output.read_text(encoding="utf-8"))
                if (
                    cached.get("model") == args.model
                    and cached.get("thinking_enabled") == args.thinking_enabled
                    and cached.get("holdout_candidate") == item["holdout_candidate"]
                ):
                    result = cached
            if result is None:
                case_values = vars(defaults) | {
                    "case_id": case_id,
                    "holdout_candidate": item["holdout_candidate"],
                    "model": args.model,
                    "disable_thinking": not args.thinking_enabled,
                    "model_timeout_seconds": args.model_timeout_seconds,
                    "discovery_timeout_seconds": args.discovery_timeout_seconds,
                    "output": None,
                }
                case_args = argparse.Namespace(**case_values)
                result = run_case(case_args)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
            scores.append(score_case(result))
        except Exception as exc:  # noqa: BLE001 - batch continues; provider text is not retained
            failures.append({"case_id": case_id, "error_type": type(exc).__name__})
        summary = {
            "model": args.model,
            "thinking_enabled": args.thinking_enabled,
            "summary": summarize(scores, failures),
            "cases": scores,
        }
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
