"""Post-freeze comparison of the fixed capability expansion diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

from rca_agent.environment.metric_catalog import MISSING_SIGNALS


def main() -> None:
    root = Path("analysis")
    before = json.loads((root / "jvm_refutation_final_summary.json").read_text())
    after = json.loads((root / "capability_expansion_summary.json").read_text())
    old_cases = {item["case_id"]: item for item in before["cases"]}
    rows = [
        "# Capability expansion: fixed 12-case comparison", "",
        ("Same manifest, deepseek-chat, thinking disabled, 20-query budget, 30 rounds, "
        "unchanged five-signal acquisition screen. GT is evaluated only after answer freeze. "
        "This run measures new verification capability under the existing screen, not an "
        "autonomous query-planner improvement. One run cannot separate model variance from "
        "capability effects. Semantic recall uses the existing lexical scorer, not human labels."),
        "", "| Metric | Before | After |", "|---|---:|---:|",
    ]
    for key in (
        "completed_cases", "raw_gt_mechanism_proposed", "compiled",
        "query_failed_cases", "alternative_resolution_failure_cases",
        "correct_verified", "wrong_verified", "replay_valid",
    ):
        rows.append(f"| {key} | {before['summary'][key]} | {after['summary'][key]} |")
    rows += ["", "## Per-case evidence and residual blockers", "",
             "| Case | Raw GT before/after | Status | New capability selected | Blockers |",
             "|---|---|---|---|---|"]
    selected_count = 0
    for case in after["cases"]:
        result = json.loads((root / "capability_expansion_run" / f"{case['case_id']}.json")
                            .read_text())
        selected = sorted({
            prediction["rule_id"]
            for plan in result["dynamic_plans"] for prediction in plan["predictions"]
            if prediction["rule_id"] in {
                "request_count_baseline_ratio", "request_latency_seconds_baseline_ratio",
                "error_rate_baseline_ratio",
            }
        })
        selected_count += bool(selected)
        blockers = result["diagnostics"]["verification_blockers"]
        rows.append(
            f"| {case['case_id']} | {old_cases[case['case_id']]['raw_gt_mechanism_proposed']} / "
            f"{case['raw_gt_mechanism_proposed']} | {case['status']} | "
            f"{', '.join(selected) or '-'} | {', '.join(blockers) or '-'} |"
        )
    rows += ["", f"Cases selecting new baseline capabilities: {selected_count}.", "",
             "## Interpretation boundaries", "",
             ("- Query success is not evidence closure; missing samples, zero baselines and "
             "unresolved alternatives remain visible in the per-case JSON."),
             ("- Waiting threads, aggregate pool sizes/task statistics, restart counts and CPU "
             "relative-to-allocation metrics are queryable context, not new causal rules."),
             ("- Task completion/restart rate derivations are deferred: raw aggregate counters "
             "lack validated reset and monotonicity semantics. No rate is fabricated."),
             ("- Request-volume growth provides one support group. Absence of mean growth "
             "does not exclude brief bursts or request-mix changes, so it has no refutes scope."),
             ("- Error-rate comparison exists on operation/instance entities; service raw data "
             "provides error counts rather than a documented error ratio. No zero-fill or "
             "unvalidated denominator alignment was introduced."),
             ("- The expanded neutral tool inventory is visible to the general Policy. The "
             "fixed benchmark acquisition policy does not autonomously explore those new "
             "context metrics; testing that strategy is a separate experiment."),
             "", "## Missing source capabilities", "",
             ("These are unavailable in the current adapter/source contract; no substitute "
             "is presented as the missing measurement."), ""]
    rows.extend(f"- `{signal}`: {reason}" for signal, reason in MISSING_SIGNALS.items())
    (root / "capability_expansion_comparison.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
