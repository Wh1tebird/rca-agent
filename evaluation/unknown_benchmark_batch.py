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
    if root_cause == "threadExhaustion":
        return "thread" in normalized and any(
            word in normalized for word in ("exhaust", "saturat", "pool", "reject", "starv")
        )
    if root_cause == "memoryPressure":
        return any(word in normalized for word in ("memory", "heap")) and any(
            word in normalized for word in ("pressure", "leak", "exhaust", "saturat", "high")
        )
    if root_cause == "fullGC":
        return (
            "full gc" in normalized
            or "full garbage collection" in normalized
            or ("garbage collection" in normalized and "pause" in normalized)
        )
    if root_cause in {"cpuFullLoad", "nodeCpuHigh", "cpuDeadLoop"}:
        return "cpu" in normalized and any(
            word in normalized for word in ("high", "load", "saturat", "loop")
        )
    if root_cause == "nodeMemoryOOM":
        return any(word in normalized for word in ("oom", "out of memory"))
    if root_cause in {"trafficSurge", "trafficHotspot"}:
        return "traffic" in normalized and any(
            word in normalized for word in ("surge", "spike", "hotspot", "load")
        )
    if root_cause == "codeDefect":
        return any(word in normalized for word in ("code defect", "application bug", "code bug"))
    raise ValueError("unsupported Ground-Truth mechanism in holdout manifest")


def score_case(result: dict[str, Any]) -> dict[str, Any]:
    root_causes = result["ground_truth"]["root_cause_types"]
    mechanisms = result["dynamic_mechanisms"]
    drafts = result.get("raw_discovery_drafts", ())
    raw_matched = [
        item for item in drafts
        if any(mechanism_matches(root_cause, item["raw_draft"]["mechanism"])
               for root_cause in root_causes)
    ]
    initially_accepted = [item for item in drafts if item["initial_status"] == "accepted"]
    finally_accepted = [item for item in drafts if item["final_status"] == "accepted"]
    repaired = [item for item in drafts if item.get("repair_actions")]
    matched = [
        mechanism for mechanism in mechanisms
        if any(mechanism_matches(root_cause, mechanism) for root_cause in root_causes)
    ]
    status = result["status"].strip()
    verified = status == "discovered_verified"
    diagnostics = result.get("diagnostics", {})
    blockers = diagnostics.get("verification_blockers", ())
    rejections = diagnostics.get("discovery_rejections", ())
    return {
        "case_id": result["case_id"],
        "holdout_candidate": result["holdout_candidate"],
        "status": status,
        "discovery_completed": (result.get("discovery") or {}).get("status") == "completed",
        "compiled": bool(mechanisms),
        "raw_draft_count": len(drafts),
        "raw_gt_mechanism_proposed": bool(raw_matched),
        "initial_contract_passed": bool(initially_accepted),
        "gt_initial_contract_passed": any(
            item in initially_accepted for item in raw_matched
        ),
        "mechanical_repair_attempted": bool(repaired),
        "mechanical_repair_succeeded": any(
            item["final_status"] == "accepted" for item in repaired
        ),
        "gt_contract_passed_after_repair": any(
            item in finally_accepted for item in raw_matched
        ),
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
        "target_service_issued": diagnostics.get("target_service_issued"),
        "missing_observations": diagnostics.get("missing_observations", 0),
        "query_failed": bool(diagnostics.get("tool_status_counts", {}).get("query_failed")),
        "rule_or_compile_failure": bool(rejections),
        "alternative_resolution_failure": "alternative_explanations_unresolved" in blockers,
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
        "raw_drafts": sum(item.get("raw_draft_count", 0) for item in scores),
        "raw_gt_mechanism_proposed": sum(
            item.get("raw_gt_mechanism_proposed", False) for item in scores
        ),
        "initial_contract_passed": sum(
            item.get("initial_contract_passed", False) for item in scores
        ),
        "gt_initial_contract_passed": sum(
            item.get("gt_initial_contract_passed", False) for item in scores
        ),
        "mechanical_repair_attempted": sum(
            item.get("mechanical_repair_attempted", False) for item in scores
        ),
        "mechanical_repair_succeeded": sum(
            item.get("mechanical_repair_succeeded", False) for item in scores
        ),
        "gt_contract_passed_after_repair": sum(
            item.get("gt_contract_passed_after_repair", False) for item in scores
        ),
        "gt_mechanism_proposed": sum(item["gt_mechanism_proposed"] for item in scores),
        "discovered_verified": sum(item["discovered_verified"] for item in scores),
        "correct_verified": sum(item["correct_verified"] for item in scores),
        "wrong_verified": sum(item["wrong_verified"] for item in scores),
        "target_service_issued": sum(item.get("target_service_issued") is True for item in scores),
        "query_failed_cases": sum(item.get("query_failed", False) for item in scores),
        "rule_or_compile_failure_cases": sum(
            item.get("rule_or_compile_failure", False) for item in scores
        ),
        "alternative_resolution_failure_cases": sum(
            item.get("alternative_resolution_failure", False) for item in scores
        ),
        "replay_valid": sum(item["replay_valid"] for item in scores),
        "status_counts": status_counts,
        "failures": failures,
        "roles": {
            role: {
                "cases": sum(item.get("role") == role for item in scores),
                "correct_verified": sum(
                    item.get("role") == role and item["correct_verified"] for item in scores
                ),
                "wrong_verified": sum(
                    item.get("role") == role and item["wrong_verified"] for item in scores
                ),
            }
            for role in sorted({item.get("role") for item in scores if item.get("role")})
        },
    }


def render_funnel_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = [
        "# Unknown RCA discovery funnel",
        "",
        "This diagnostic separates raw semantic recall from contract compilation and evidence",
        "verification. Ground Truth is loaded only after the answer is frozen.",
        "",
        "## Funnel",
        "",
        "| Stage | Cases |",
        "|---|---:|",
        f"| Completed | {summary['completed_cases']} |",
        f"| Raw GT-like mechanism proposed | {summary['raw_gt_mechanism_proposed']} |",
        f"| Any initial contract passed | {summary['initial_contract_passed']} |",
        f"| GT-like draft initially passed | {summary['gt_initial_contract_passed']} |",
        f"| Mechanical repair attempted | {summary['mechanical_repair_attempted']} |",
        f"| Mechanical repair succeeded | {summary['mechanical_repair_succeeded']} |",
        f"| GT-like draft passed after repair | {summary['gt_contract_passed_after_repair']} |",
        f"| Correct `DISCOVERED_VERIFIED` | {summary['correct_verified']} |",
        f"| Wrong `DISCOVERED_VERIFIED` | {summary['wrong_verified']} |",
        "",
        "## Cases",
        "",
        "| Case | Role | Holdout GT | Raw GT | Initial GT pass | Repaired | Final status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for item in payload["cases"]:
        rows.append(
            "| {case_id} | {role} | {holdout_candidate} | {raw} | {initial} | "
            "{repaired} | {status} |".format(
                case_id=item["case_id"],
                role=item.get("role") or "-",
                holdout_candidate=item["holdout_candidate"],
                raw="yes" if item["raw_gt_mechanism_proposed"] else "no",
                initial="yes" if item["gt_initial_contract_passed"] else "no",
                repaired="yes" if item["mechanical_repair_succeeded"] else "no",
                status=item["status"],
            )
        )
    rows.extend((
        "",
        "`Raw GT` uses the same conservative post-freeze lexical scorer as the existing benchmark.",
        "Mechanical repair only canonicalizes case and whitespace in references to an already",
        "declared alternative. It never changes mechanism, family, rules, entities, polarity, or",
        "causal explanation.",
        "",
    ))
    return "\n".join(rows)


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
    value.add_argument("--report-output", type=Path,
                       default=Path("analysis/unknown_benchmark_batch.md"))
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
            score = score_case(result)
            score["role"] = item.get("role")
            scores.append(score)
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
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(render_funnel_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
