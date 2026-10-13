"""Offline scoring of frozen paired Qwen runs; never imported by inference."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from evaluation.ground_truth import FlatGroundTruthRepository
from evaluation.record_active_formalizer_qwen import _evidence, _write_json
from evaluation.unknown_benchmark_batch import mechanism_matches
from rca_agent.models import AnswerArtifact
from rca_agent.runner import RecordedReplay
from rca_agent.runner.transcript import load_jsonl_events


def direction_match(root, text, *, exact=False):
    text = text.casefold()
    # A symptom in the consequence clause is not the proposed causal mechanism.
    # Keep broad GC-family proximity separate from exact Full-GC diagnosis below.
    causal_head = re.split(r"\b(?:causing|leading to|resulting in)\b", text, maxsplit=1)[0]
    if root == "fullGC":
        return bool(re.search(r"full\s*gc|full garbage collection", text)) if exact else bool(
            re.search(r"\bgc\b|garbage collection", text))
    if root == "cpuDeadLoop":
        return ("cpu" in text and bool(re.search(r"loop|spin|循环", text))) if exact else "cpu" in text
    if root == "nodeMemoryOOM":
        return bool(re.search(r"\boom\b|out.of.memory", text)) if exact else bool(
            re.search(r"memory|heap", text))
    if root == "slowSQL":
        return bool(re.search(r"slow sql|cross[- ]?join|complex.{0,100}(sql|query)|sql.{0,40}execution|query execution", causal_head))
    try:
        return mechanism_matches(root, causal_head)
    except ValueError:
        return False


def _nonempty(value):
    return value is not None and (not isinstance(value, (list, dict, str)) or bool(value))


def query_metrics(events):
    responses = [e.payload.get("response", e.payload) for e in events
                 if e.event_type.value == "tool_response"]
    trusted = lambda o: o["data_status"] in {"observed", "derived"} and _nonempty(o.get("value"))
    registry = {}
    for e in events:
        if e.event_type.value != "capability_registry_resolved":
            continue
        for c in e.payload["capabilities"]:
            if c["support_eligible"]:
                registry.setdefault((c["entity_ref"], c["signal"]), set()).add(c["independent_group"])
    groups, eligible = set(), set()
    observations = [o for r in responses for o in r["observations"]]
    for o in observations:
        c = registry.get((o["entity_ref"], o["signal"]))
        if trusted(o) and c:
            groups.update(c)
            eligible.add(o["observation_id"])
    fingerprints = [r["query_ref"] for r in responses]
    data_queries = [r for r in responses if any(trusted(o) for o in r["observations"])]
    measurement_queries = [r for r in responses if any(
        trusted(o) and o["modality"] in {"metric", "trace"} for o in r["observations"])]
    rounds = [e for e in events if e.event_type.value == "inquiry_round_recorded"]
    handoffs = [e for e in events if e.event_type.value == "discovery_started"]
    boundary_sequence = handoffs[0].sequence if handoffs else float("inf")
    inquiry_metrics = {}
    # Shadow intentionally stores no selected_actions. Recover exact executed inquiry
    # refs from the transcript-validated handoff, not a guessed tool-request interval.
    if rounds and not any("selected_actions" in e.payload for e in rounds):
        offered = sum(len(e.payload["draft"]["choices"]) for e in rounds)
        executed = None
        if handoffs:
            boundary = handoffs[0]
            refs = {l["query_ref"] for l in (boundary.payload.get("inquiry_handoff") or {}).get("query_links", ())}
            executed = sum(e.payload["query_ref"] in refs for e in events
                           if e.event_type.value == "tool_request" and e.sequence < boundary.sequence)
        inquiry_metrics = {"inquiry_queries_offered": offered, "inquiry_queries_executed": executed,
                           "inquiry_queries_not_executed": offered - executed if executed is not None else None,
                           "inquiry_count_source": "validated_handoff" if executed is not None else "unavailable"}
    return {
        "query_count": len(responses), "trusted_nonempty_queries": len(data_queries),
        "unique_trusted_data_queries": len({r["query_ref"] for r in data_queries}),
        "metric_trace_data_queries": len(measurement_queries),
        "status_success_queries": sum(r["status"] == "success" for r in responses),
        "query_statuses": dict(Counter(r["status"] for r in responses)),
        "duplicate_queries": len(fingerprints) - len(set(fingerprints)),
        "trusted_observations": sum(trusted(o) for o in observations),
        "missing_unknown_observations": sum(o["data_status"] in {"missing", "unknown"} for o in observations),
        "eligible_observation_count": len(eligible), "eligible_groups": sorted(groups),
        "planner_rejections": dict(Counter(e.payload.get("error_code", "unknown") for e in events
                                            if e.event_type.value == "verification_plan_rejected")),
        "elapsed_seconds": (events[-1].recorded_at - events[0].recorded_at).total_seconds(),
        "exploration_query_sequence": [e.payload["query_ref"] for e in events
                                       if e.event_type.value == "tool_request" and e.sequence < boundary_sequence],
        "active_multi_option_selections": sum(
            len(set(e.payload.get("offered_query_fingerprints", ()))
                | set(e.payload.get("pending_query_fingerprints", ()))) > 1 for e in events
            if e.event_type.value == "discrimination_active_selection_recorded"),
        "active_nonfirst_selections": sum(
            bool(e.payload.get("offered_query_fingerprints"))
            and e.payload["selected_query_fingerprint"] != e.payload["offered_query_fingerprints"][0]
            for e in events if e.event_type.value == "discrimination_active_selection_recorded"),
        **inquiry_metrics,
    }


def score_run(run_dir, repository):
    config = json.loads((run_dir / "config.json").read_text())
    answer = AnswerArtifact.model_validate_json((run_dir / "answer.json").read_text())
    events = load_jsonl_events(run_dir / "transcript.jsonl")
    if not answer.verify_digest() or not RecordedReplay.verify(events).valid:
        raise ValueError("invalid frozen run: GT must not be loaded")
    freeze = next(e.payload for e in events if e.event_type.value == "answer_frozen")
    if freeze["artifact_digest"] != answer.artifact_digest:
        raise ValueError("Answer is not this Transcript's frozen artifact")
    # First and only GT access, strictly after freeze binding and Replay verification.
    truth = repository.load(config["case_id"])
    roots = truth.root_cause_types
    rounds = [e.payload["draft"]["hypotheses"] for e in events
              if e.event_type.value == "inquiry_round_recorded"]
    accepted_rounds = sum(e.payload["status"] == "accepted" for e in events
                         if e.event_type.value == "inquiry_attempt_recorded")
    inquiry_complete = accepted_rounds == len(rounds)
    any_texts = [h["mechanism"] for r in rounds for h in r]
    last_texts = [h["mechanism"] for h in rounds[-1]] if rounds else []
    returned = [c.mechanism for c in answer.inquiry_candidates]
    formal = [e.payload["mechanism"] for e in events if e.event_type.value == "mechanism_draft_frozen"]
    compiled = {e.payload["mechanism_id"] for e in events if e.event_type.value == "verification_plan_compiled"}
    match = lambda text, exact=False: any(direction_match(root, text, exact=exact) for root in roots)
    matched_returned = [c.model_dump(mode="json") for c in answer.inquiry_candidates if match(c.mechanism)]
    formal_match = lambda m, exact=False: any(
        direction_match(root, m["mechanism"], exact=exact)
        and (root != "fullGC" or m["mechanism_class"] == "gc_pause") for root in roots)
    matched_formal = [m for m in formal if formal_match(m)]
    states = [e.payload for e in events if e.event_type.value == "dynamic_verification"]
    rejected_actions = [e.payload.get("error_type", "unknown") for e in events
                        if e.event_type.value == "action_rejected"]
    failure_stage = (
        "runtime_action_or_recording_rejection" if rejected_actions else
        "inquiry_no_valid_round" if not rounds else
        "inquiry_no_gt_direction" if not any(map(match, any_texts)) else
        "candidate_return_loss" if not any(map(match, returned)) else
        "formalization_gap" if not matched_formal else
        "planner_compile_gap" if not any(m["mechanism_id"] in compiled for m in matched_formal) else
        "verification_evidence_or_alternative_gap" if answer.authoritative_decision.status.value != "discovered_verified" else
        "verified_needs_winner_review"
    )
    return {
        "case_id": config["case_id"], "mode": config["investigation_workspace_mode"],
        "run_id": run_dir.name, "implementation_digest": config["implementation_digest"],
        "replay_valid": True, "gt": truth.model_dump(mode="json", exclude={"steps"}),
        "inquiry_recording_complete": inquiry_complete,
        "inquiry_similar": True if any(map(match, any_texts)) else (False if inquiry_complete else None),
        "inquiry_exact_mechanism": True if any(match(t, True) for t in any_texts) else (False if inquiry_complete else None),
        "last_inquiry_similar": any(map(match, last_texts)),
        "returned_similar": any(map(match, returned)),
        "proposed_and_retained": any(map(match, any_texts)) and any(map(match, returned)),
        "returned_exact_mechanism": any(match(t, True) for t in returned),
        "formal_similar": bool(matched_formal),
        "formal_exact_mechanism": any(formal_match(m, True) for m in formal),
        "compiled_similar": any(m["mechanism_id"] in compiled for m in matched_formal),
        "compiled_exact_mechanism": any(m["mechanism_id"] in compiled and formal_match(m, True)
                                         for m in formal),
        "candidate_texts": any_texts, "last_candidate_texts": last_texts,
        "matched_returned_candidates": matched_returned,
        "formal_mechanisms": formal,
        "entity_text_hint": any(any(entity.split("::")[0].casefold() in c["mechanism"].casefold()
                                    for entity in truth.root_cause_entities) for c in matched_returned),
        "core_states": states[-1] if states else None,
        "first_primary_failure_stage": failure_stage,
        "status": answer.authoritative_decision.status.value,
        "termination_reason": answer.termination_reason.value,
        "action_rejection_types": rejected_actions,
        **_evidence(events), **query_metrics(events),
    }


def evaluate(root, answer_key_root):
    # Batch-wide gate: no offline GT access while any paired inference is running.
    preflight = json.loads((root / "preflight.json").read_text())
    batch = json.loads((root / "batch_results.json").read_text())
    expected = {(r["case_id"], mode) for r in preflight if r["valid"]
                for mode in ("shadow", "active")}
    completed = {(r["case_id"], r["workspace_mode"]) for r in batch}
    if len(batch) != len(expected) or completed != expected:
        raise ValueError("paired batch is incomplete: GT must not be loaded")
    configs = [json.loads(p.read_text()) for p in root.glob("*/run-*/config.json")]
    variants = {json.dumps({k: v for k, v in c.items()
                           if k not in {"case_id", "investigation_workspace_mode"}}, sort_keys=True)
                for c in configs}
    if len(configs) != len(expected) or len(variants) != 1:
        raise ValueError("paired inference configurations are not frozen and identical")
    repository = FlatGroundTruthRepository(answer_key_root)
    rows, errors = [], []
    for path in sorted(root.glob("*/run-*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] != "succeeded":
            errors.append(result)
            continue
        try:
            rows.append(score_run(path.parent, repository))
        except Exception as exc:  # noqa: BLE001 - diagnostic class only
            errors.append({"run_id": path.parent.name, "error_code": type(exc).__name__})
    digests = {row["implementation_digest"] for row in rows}
    if len(digests) > 1:
        raise ValueError("inference implementation changed during comparison")
    summary = {}
    for mode in ("shadow", "active"):
        arm = [r for r in rows if r["mode"] == mode]
        keys = ["inquiry_similar", "inquiry_exact_mechanism", "last_inquiry_similar", "returned_similar", "proposed_and_retained",
                "returned_exact_mechanism", "formal_similar", "formal_exact_mechanism", "compiled_similar",
                "query_count", "trusted_nonempty_queries", "metric_trace_data_queries", "duplicate_queries",
                "inquiry_queries_offered", "inquiry_queries_executed", "bound_originals_formalized",
                "bound_originals_deferred", "candidates_compiled", "missing_unknown_observations",
                "compiled_exact_mechanism", "active_multi_option_selections", "active_nonfirst_selections"]
        keys.append("unique_trusted_data_queries")
        summary[mode] = {"completed_replay_valid": len(arm), **{k: sum(r[k] or 0 for r in arm) for k in keys},
                         "inquiry_execution_count_unknown_cases": sum(r["inquiry_queries_executed"] is None for r in arm),
                         "inquiry_recall_unknown_cases": sum(r["inquiry_similar"] is None for r in arm),
                         "inquiry_recording_incomplete_cases": sum(not r["inquiry_recording_complete"] for r in arm),
                         "status_counts": dict(Counter(r["status"] for r in arm)),
                         "verified_runs": sum(r["status"] in {"discovered_verified", "confirmed"} for r in arm),
                         "failure_stages": dict(Counter(r["first_primary_failure_stage"] for r in arm)),
                         "runtime_action_rejection_runs": sum(bool(r["action_rejection_types"]) for r in arm),
                         "planner_rejections": dict(sum((Counter(r["planner_rejections"]) for r in arm), Counter())),
                         "errors": sum(e.get("workspace_mode") == mode for e in errors)}
    summary["paired_completed_cases"] = len(
        {r["case_id"] for r in rows if r["mode"] == "shadow"}
        & {r["case_id"] for r in rows if r["mode"] == "active"})
    paired = {}
    for r in rows:
        paired.setdefault(r["case_id"], {})[r["mode"]] = r
    comparison = []
    for case_id, arms in paired.items():
        if set(arms) != {"shadow", "active"}:
            continue
        old, new = arms["shadow"], arms["active"]
        comparison.append({"case_id": case_id,
            "same_exploration_order": old["exploration_query_sequence"] == new["exploration_query_sequence"],
            "same_exploration_set": set(old["exploration_query_sequence"]) == set(new["exploration_query_sequence"]),
            "trusted_data_query_delta": new["trusted_nonempty_queries"] - old["trusted_nonempty_queries"],
            "inquiry_direction_delta": (int(new["inquiry_similar"]) - int(old["inquiry_similar"])
                                        if new["inquiry_similar"] is not None and old["inquiry_similar"] is not None else None),
        })
    summary["same_exploration_order_cases"] = sum(c["same_exploration_order"] for c in comparison)
    summary["same_exploration_set_cases"] = sum(c["same_exploration_set"] for c in comparison)
    common = [arms for arms in paired.values() if set(arms) == {"shadow", "active"}]
    uninterrupted = [arms for arms in common if all(r["inquiry_recording_complete"] for r in arms.values())]
    summary["matched_cohorts"] = {}
    for name, cohort in (("frozen_pairs", common), ("uninterrupted_inquiry_pairs", uninterrupted)):
        summary["matched_cohorts"][name] = {
            "cases": len(cohort),
            **{mode: {key: sum(p[mode][key] or 0 for p in cohort) for key in (
                "query_count", "trusted_nonempty_queries", "duplicate_queries",
                "inquiry_similar", "compiled_similar")}
               for mode in ("shadow", "active")},
        }
    _write_json(root / "offline_scores.json", {"summary": summary, "runs": rows, "errors": errors, "paired": comparison,
               "scorer": "causal-head-aware lexical matching; GC proximity is family-only, formal GC requires gc_pause; entity hint is text-only"})
    lines = ["# Active 查询首批对照结果", "", "一次成对诊断；非统计显著性或泛化准确率证明。", "",
             "GT 仅在冻结 Answer 验证和 Replay 完成后离线读取。相近/精确为保守文本评分，需结合逐例机制文本复核。", "",
             "| Case | 查询策略 | 提出相近/精确 | 记录最后一轮相近 | 最终返回相近 | 正式化相近 | 编译相近 | 有数据/查询 | Core |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["case_id"], r["mode"])):
        yes = lambda value: "未知" if value is None else ("是" if value else "否")
        lines.append(f"| {r['case_id']} | {r['mode']} | {yes(r['inquiry_similar'])}/{yes(r['inquiry_exact_mechanism'])} "
                     f"| {yes(r['last_inquiry_similar'])} | {yes(r['returned_similar'])} | {yes(r['formal_similar'])} "
                     f"| {yes(r['compiled_similar'])} | {r['trusted_nonempty_queries']}/{r['query_count']} | {r['status']} |")
    lines.extend(["", "## 汇总", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
                  f"运行/评分错误：{len(errors)}。完整候选文本、GT、证据和 Core 阻塞见 offline_scores.json。",
                  "查询拿到数据不等价于证据支持或成功排除 alternative；不能仅根据成功率判定查询有区分力。"])
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--answer-key-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.runs_root, args.answer_key_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
