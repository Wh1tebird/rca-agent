"""Offline audit of saved, frozen Runs for MechanismCoveragePolicy shadow.

No provider is called. Ground Truth is loaded only after Answer digest and
Recorded Replay checks. Legacy Runs without a Workspace are never imputed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.ground_truth import FlatGroundTruthRepository
from evaluation.unknown_benchmark_batch import mechanism_matches
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.investigation import (
    MECHANISM_COVERAGE_POLICY_VERSION,
    InvestigationWorkspace,
    MechanismCoveragePolicy,
    MechanismCoverageRecord,
    build_mechanism_coverage_input,
    project_formal_coverage_candidates,
    reduce_investigation_workspace,
)
from rca_agent.investigation.coverage import _NEED_SIGNALS
from rca_agent.models import AnswerArtifact, DataStatus, ResetResponse, ToolResponse
from rca_agent.runner import EventType, IssuedObservation, RecordedReplay
from rca_agent.runner.transcript import load_jsonl_events

GT_FAMILY = {
    "slowSQL": "database_execution",
    "dbNetworkLatency": "database_transport",
    "threadExhaustion": "jvm_thread_queue_pressure",
    "fullGC": "jvm_gc_pressure",
    "memoryPressure": "jvm_memory_pressure",
    "cpuFullLoad": "jvm_cpu_load_pressure",
}
DB_FAMILIES = frozenset({"database_execution", "database_transport", "database_connection_acquisition"})
JVM_FAMILIES = frozenset({
    "jvm_thread_queue_pressure", "jvm_gc_pressure", "jvm_memory_pressure",
    "jvm_cpu_load_pressure",
})


def _truth_family(root_cause_types: list[str]) -> str | None:
    mapped = {GT_FAMILY[item] for item in root_cause_types if item in GT_FAMILY}
    return next(iter(mapped)) if len(mapped) == 1 else None


def _domain(family: str | None) -> str:
    if family in DB_FAMILIES:
        return "database"
    if family in JVM_FAMILIES:
        return "jvm"
    return "other"


def _issued(events: tuple) -> tuple[IssuedObservation, ...]:
    ledger: dict[str, IssuedObservation] = {}
    for event in events:
        if event.event_type != EventType.TOOL_RESPONSE:
            continue
        response = ToolResponse.model_validate(event.payload)
        for observation in response.observations:
            ledger.setdefault(observation.observation_id, IssuedObservation(
                observation, response.query_ref,
                response.status.value == "success" and not response.truncated,
            ))
    return tuple(ledger.values())


def _capability_diagnostics(
    assessment: Any, events: tuple,
    *, executable_override: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    registry_events = [event for event in events
                       if event.event_type == EventType.CAPABILITY_REGISTRY_RESOLVED]
    registry = registry_events[-1].payload.get("capabilities", ()) if registry_events else ()
    executable_pairs = (executable_override if executable_override is not None
                        else {(item["entity_ref"], item["signal"])
                              for item in registry if item["executable"]})
    advertised = {item["signal"] for event in events
                  if event.event_type == EventType.ENVIRONMENT_RESET
                  for capability in event.payload["bootstrap"]["capabilities"]
                  if capability["available"] for item in capability.get("metrics", ())}
    statuses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in _issued(events):
        statuses[(item.observation.entity_ref, item.observation.signal)].add(
            item.observation.data_status.value
        )
    result = []
    for gap in assessment.gaps:
        for need in gap.discriminator_needs:
            required = set(_NEED_SIGNALS[need.missing_family])
            target_options = sorted({entity for entity, _ in executable_pairs})
            target = min(target_options, key=lambda entity: (
                -len(required & {signal for candidate, signal in executable_pairs
                                  if candidate == entity}), entity,
            )) if target_options else None
            resolved = (required & {signal for entity, signal in executable_pairs
                                    if entity == target})
            all_executable = {signal for _, signal in executable_pairs}
            kind = ("exact" if resolved == required else "partial" if resolved
                    else "unresolvable")
            result.append({
                "family": need.missing_family.value,
                "need_code": need.need_code,
                "resolution": kind,
                "target_entity_ref": target,
                "required_signals": sorted(required),
                "executable_signals": sorted(resolved),
                "registry_only_on_other_entity": sorted((required & all_executable) - resolved),
                "advertised_but_not_registry": sorted((required & advertised)
                                                      - all_executable),
                "unsupported_signals": sorted(required - advertised - all_executable),
                "capability_exists_but_entity_no_data": sorted(
                    signal for signal in resolved
                    if statuses[(target, signal)]
                    and not statuses[(target, signal)].intersection({
                        DataStatus.OBSERVED.value, DataStatus.DERIVED.value,
                    })
                ),
                "capability_exists_but_not_observed_in_run": sorted(
                    signal for signal in resolved if not statuses[(target, signal)]
                ),
            })
    return result


def _observability_probes(runs_root: Path, valid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two legal-observation probes, with empty synthetic Workspace and no GT selection."""

    db_signals = {
        "connection_acquisition_ms", "connection_duration_ms", "db_client_span_duration_ms",
        "sql_duration_ms", "sql_duration_share", "sql_shape",
    }
    jvm_signals = {
        "thread_pool_active", "thread_pool_queue_depth", "thread_pool_rejected_tasks",
        "jvm_runnable_threads", "jvm_gc_collection_delta", "jvm_gc_pause_seconds_delta",
        "jvm_heap_used_bytes", "jvm_thread_cpu_time_ns", "cpu_usage_rate",
    }
    probes = []
    for domain, trigger_signals in (("database", db_signals), ("jvm", jvm_signals)):
        for row in valid:
            events = load_jsonl_events(runs_root / row["run_id"] / "transcript.jsonl")
            observations = _issued(events)
            if not any(item.observation.signal in trigger_signals
                       and item.observation.data_status in {
                           DataStatus.OBSERVED, DataStatus.DERIVED,
                       } for item in observations):
                continue
            reset = next(event for event in events
                         if event.event_type == EventType.ENVIRONMENT_RESET)
            bootstrap = ResetResponse.model_validate(reset.payload["bootstrap"])
            answer_event = next(event for event in events
                                if event.event_type == EventType.ANSWER_FROZEN)
            workspace = InvestigationWorkspace(answer_event.episode_id).snapshot(
                transcript_head_sequence=answer_event.sequence - 1,
                transcript_head_digest=answer_event.previous_digest,
            )
            coverage_input = build_mechanism_coverage_input(workspace, observations, bootstrap)
            assessment = MechanismCoveragePolicy().evaluate(coverage_input)
            registry = rca100_evidence_registry(
                bootstrap, tuple(item.observation for item in observations)
            )
            executable = {(item.entity_ref, item.signal) for item in registry.signals}
            needs = _capability_diagnostics(
                assessment, events, executable_override=executable,
            )
            probes.append({
                "domain": domain, "source_run_id": row["run_id"],
                "synthetic_workspace": "empty",
                "gap_families": [family.value for gap in assessment.gaps
                                 if gap.domain.value == domain
                                 for family in gap.missing_families],
                "needs": [item for item in needs if item["family"] in (
                    DB_FAMILIES if domain == "database" else JVM_FAMILIES
                )],
                "repeat_digest_equal": assessment.assessment_digest
                == MechanismCoveragePolicy().evaluate(coverage_input).assessment_digest,
            })
            break
    return probes


def _assess_saved_run(run_dir: Path, truth_repo: FlatGroundTruthRepository) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "run_id": run_dir.name, "case_id": manifest["case_id"],
        "replay_valid": False, "answer_valid": False,
        "coverage_reconstructable": False, "reason": None,
        "gt_family": None, "gt_domain": None, "gaps": [], "needs": [],
    }
    answer_path, transcript_path = run_dir / "answer.json", run_dir / "transcript.jsonl"
    if not answer_path.is_file() or not transcript_path.is_file():
        row["reason"] = "missing_frozen_answer_or_transcript"
        return row
    try:
        answer = AnswerArtifact.model_validate_json(answer_path.read_text(encoding="utf-8"))
        events = load_jsonl_events(transcript_path)
        row["answer_valid"] = answer.verify_digest()
        if not row["answer_valid"] or not any(
            event.event_type == EventType.ANSWER_FROZEN for event in events
        ):
            row["reason"] = "invalid_or_unfrozen_answer"
            return row
        replay = RecordedReplay.verify(events)
        row["replay_valid"] = replay.valid
        if not replay.valid:
            row["reason"] = "recorded_replay_invalid"
            return row
        row["replay_repeat_equal"] = RecordedReplay.verify(events) == replay
        row["tool_requests"] = sum(event.event_type == EventType.TOOL_REQUEST for event in events)
        row["query_cost_units"] = sum(event.payload["units"] for event in events
                                      if event.event_type == EventType.QUERY_COST)
        row["inquiry_candidate_count"] = len(answer.inquiry_candidates)
        row["formal_mechanism_count"] = sum(
            event.event_type == EventType.MECHANISM_DRAFT_FROZEN for event in events
        )
        row["compiled_candidate_count"] = sum(
            event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED for event in events
        )
        row["core_status"] = answer.authoritative_decision.status.value
        row["wrong_verified"] = None
    except (ValueError, KeyError, OSError) as exc:
        row["reason"] = "replay_or_transcript_incompatible"
        row["error_type"] = type(exc).__name__
        return row

    # This is the only point at which evaluation-only Ground Truth is opened.
    truth = truth_repo.load(row["case_id"])
    family = _truth_family(truth.root_cause_types)
    row["gt_family"] = family
    row["gt_domain"] = _domain(family)
    if answer.authoritative_decision.status.value == "discovered_verified":
        winner = next((proposal for proposal in answer.policy_proposals
                       if proposal.hypothesis_id
                       == answer.authoritative_decision.winning_hypothesis_id), None)
        row["wrong_verified"] = winner is None or not any(
            mechanism_matches(cause, winner.mechanism)
            for cause in truth.root_cause_types if cause in GT_FAMILY
        )
    if answer.investigation_workspace is None:
        row["reason"] = "legacy_transcript_has_no_workspace_snapshot"
        return row
    reset = next((event for event in events
                  if event.event_type == EventType.ENVIRONMENT_RESET), None)
    if reset is None:
        row["reason"] = "missing_reset"
        return row
    try:
        coverage_event = next((event for event in events
                               if event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED),
                              None)
        if coverage_event is not None:
            prior_events = events[:coverage_event.sequence - 1]
            workspace = reduce_investigation_workspace(prior_events, answer.episode_id).snapshot(
                transcript_head_sequence=coverage_event.sequence - 1,
                transcript_head_digest=coverage_event.previous_digest,
            )
        else:
            workspace = answer.investigation_workspace
        coverage_input = build_mechanism_coverage_input(
            workspace, _issued(events), ResetResponse.model_validate(reset.payload["bootstrap"]),
            policy_version=(MechanismCoverageRecord.model_validate(
                coverage_event.payload
            ).input.policy_version if coverage_event is not None else None)
            or MECHANISM_COVERAGE_POLICY_VERSION,
            formal_candidates=project_formal_coverage_candidates(
                events[:coverage_event.sequence - 1] if coverage_event is not None else events
            ),
        )
        first = MechanismCoveragePolicy().evaluate(coverage_input)
        second = MechanismCoveragePolicy().evaluate(coverage_input)
        row["stability"] = {
            "gap_set_equal": {f.value for gap in first.gaps for f in gap.missing_families}
            == {f.value for gap in second.gaps for f in gap.missing_families},
            "order_equal": first.gaps == second.gaps,
            "digest_equal": first.assessment_digest == second.assessment_digest,
            "replay_equal": RecordedReplay.verify(events) == replay,
        }
        if (coverage_event is not None
                and MechanismCoverageRecord.model_validate(coverage_event.payload).assessment != first):
            row["reason"] = "recorded_coverage_differs"
            return row
        row["coverage_reconstructable"] = True
        row["covered_families"] = [family.value for family in first.covered_families]
        row["gaps"] = [{
            "domain": gap.domain.value,
            "families": [item.value for item in gap.missing_families],
            "candidate_triggers": len(gap.trigger_basis.candidate_track_ids),
            "observation_triggers": len(gap.trigger_basis.observation_refs),
        } for gap in first.gaps]
        row["needs"] = _capability_diagnostics(first, events)
        row["gt_already_covered"] = family in row["covered_families"] if family else None
        row["gt_gap_hit"] = any(family in gap["families"] for gap in row["gaps"])
        row["capability_only_trigger"] = any(
            gap["candidate_triggers"] == gap["observation_triggers"] == 0
            for gap in row["gaps"]
        )
    except (ValueError, KeyError, TypeError) as exc:
        row["reason"] = "coverage_input_incompatible"
        row["error_type"] = type(exc).__name__
    return row


def evaluate(*, cases_root: Path, answer_key_root: Path, runs_root: Path) -> dict[str, Any]:
    truth_repo = FlatGroundTruthRepository(answer_key_root)
    runs = [_assess_saved_run(path, truth_repo) for path in sorted(runs_root.iterdir())
            if path.is_dir() and (path / "manifest.json").is_file()]
    from rca_agent.environment import Rca100Environment

    case_rows = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_case[run["case_id"]].append(run)
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_runs = by_case.get(case_dir.name, ())
        try:
            Rca100Environment.from_case(cases_root, case_dir.name).reset()
            state, error_type = "initializable", None
        except (ValueError, OSError) as exc:
            state, error_type = "cannot_initialize", type(exc).__name__
        case_rows.append({
            "case_id": case_dir.name, "initialization": state,
            "initialization_error_type": error_type,
            "run_ids": [item["run_id"] for item in case_runs],
            "replay_valid_runs": sum(item["replay_valid"] for item in case_runs),
            "coverage_reconstructable_runs": sum(
                item["coverage_reconstructable"] for item in case_runs
            ),
            "gt_domain_after_replay": next((item["gt_domain"] for item in case_runs
                                            if item["replay_valid"]), None),
        })
    valid = [item for item in runs if item["replay_valid"] and item["answer_valid"]]
    eligible = [item for item in valid if item["coverage_reconstructable"]]
    needs = [need for item in eligible for need in item["needs"]]
    probes = _observability_probes(runs_root, valid)
    probe_needs = [need for probe in probes for need in probe["needs"]]
    suggestions = [family for item in eligible for gap in item["gaps"]
                   for family in gap["families"]]
    related = sum(item["gt_family"] == family for item in eligible
                  for gap in item["gaps"] for family in gap["families"])
    missing = [item for item in eligible if item["gt_family"] is not None
               and not item["gt_already_covered"]]
    return {
        "schema_version": "mechanism-coverage-phase5-v1",
        "definitions": {
            "valid_run": "frozen Answer digest and independent Recorded Replay both pass",
            "reconstructable": "valid Run with frozen Workspace and legal reset/Observation input",
            "incremental_recall": "GT-family gap hits / eligible Runs with GT family absent from live candidates",
            "noise_rate": "suggested missing families other than exact GT family / all suggested missing families",
            "capability_resolution": "all, some or none of each need's policy signal set has an executable entity-bound Registry recipe",
            "no_data": "Registry recipe exists but no issued trusted Observation of that signal",
        },
        "summary": {
            "historical_runs": len(runs),
            "distinct_run_cases": len(by_case),
            "dataset_cases": len(case_rows),
            "cannot_initialize_cases": sum(item["initialization"] == "cannot_initialize"
                                           for item in case_rows),
            "replay_valid_runs": len(valid),
            "replay_repeat_equal_runs": sum(item.get("replay_repeat_equal") is True
                                            for item in valid),
            "coverage_reconstructable_runs": len(eligible),
            "coverage_metric_denominator_runs": len(eligible),
            "transcript_incompatible_runs": sum(item["reason"] in {
                "replay_or_transcript_incompatible", "coverage_input_incompatible",
                "recorded_coverage_differs",
            } for item in runs),
            "insufficient_data_runs": sum(item["reason"] in {
                "legacy_transcript_has_no_workspace_snapshot", "missing_reset",
                "missing_frozen_answer_or_transcript", "invalid_or_unfrozen_answer",
            } for item in runs),
            "gt_domain_run_counts_after_replay": dict(sorted(Counter(
                item["gt_domain"] for item in valid
            ).items())),
            "gt_domain_distinct_case_counts_after_replay": dict(sorted(Counter(
                next(item["gt_domain"] for item in valid if item["case_id"] == case_id)
                for case_id in {item["case_id"] for item in valid}
            ).items())),
            "gt_already_candidate_covered": sum(item.get("gt_already_covered") is True
                                                for item in eligible),
            "gt_completely_omitted": len(missing),
            "gt_omission_gap_hits": sum(item["gt_gap_hit"] for item in missing),
            "gt_omission_without_help": sum(not item["gt_gap_hit"] for item in missing),
            "incremental_recall_rate": (sum(item["gt_gap_hit"] for item in missing) / len(missing)
                                        if missing else None),
            "gap_count": sum(len(item["gaps"]) for item in eligible),
            "suggestion_count": len(suggestions),
            "gt_related_suggestion_count": related,
            "unrelated_or_broad_suggestion_count": len(suggestions) - related,
            "noise_rate": ((len(suggestions) - related) / len(suggestions)
                           if suggestions else None),
            "suggestions_per_eligible_run": (len(suggestions) / len(eligible)
                                             if eligible else None),
            "capability_resolution_counts": dict(sorted(Counter(
                item["resolution"] for item in needs
            ).items())),
            "capability_resolution_rates": {
                kind: sum(item["resolution"] == kind for item in needs) / len(needs)
                if needs else None for kind in ("exact", "partial", "unresolvable")
            },
            "observation_only_probe_count": len(probes),
            "observation_only_probe_gap_count": len(probes),
            "observation_only_probe_suggestion_count": sum(
                len(probe["gap_families"]) for probe in probes
            ),
            "observation_only_probe_capability_resolution_counts": {
                kind: sum(item["resolution"] == kind for item in probe_needs)
                for kind in ("exact", "partial", "unresolvable")
            },
            "observation_only_probe_capability_resolution_rates": {
                kind: sum(item["resolution"] == kind for item in probe_needs)
                / len(probe_needs) if probe_needs else None
                for kind in ("exact", "partial", "unresolvable")
            },
            "stability_all_equal": (all(all(item["stability"].values()) for item in eligible)
                                    if eligible else None),
            "capability_only_trigger_count": sum(item.get("capability_only_trigger", False)
                                                 for item in eligible),
            "wrong_verified_in_valid_runs": sum(item["wrong_verified"] is True for item in valid),
            "historical_off_shadow_pairs": 0,
        },
        "runs": runs,
        "cases": case_rows,
        "observability_probes": probes,
    }


def render_case_table(payload: dict[str, Any]) -> str:
    lines = [
        "# MechanismCoveragePolicy Phase 5 — 逐 Case 诊断表",
        "",
        "只计保存了 Manifest、完整 Transcript 与冻结 Answer 的 Run。GT 机制族只在",
        "Answer digest 和 Recorded Replay 均通过后读取。`—` 表示没有合格 Run；",
        "`数据不足` 表示旧 Run 缺冻结 Workspace，不能重建 shadow。",
        "",
        "| Case | 初始化 | 历史 Run | Replay 有效 | shadow 可重建 | GT 范围 | 诊断 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for case in payload["cases"]:
        run_count = len(case["run_ids"])
        diagnosis = ("无法初始化" if case["initialization"] == "cannot_initialize"
                     else "数据不足" if run_count and not case["coverage_reconstructable_runs"]
                     else "无保存 Run" if not run_count else "可重建")
        lines.append(
            f"| {case['case_id']} | {case['initialization']} | {run_count} | "
            f"{case['replay_valid_runs']} | {case['coverage_reconstructable_runs']} | "
            f"{case['gt_domain_after_replay'] or '—'} | {diagnosis} |"
        )
    lines.extend((
        "", "## 保存的 Run 明细", "",
        "| Run | Case | Replay | 冻结 Answer | shadow 可重建 | 原因 |",
        "|---|---|---|---|---|---|",
    ))
    for run in payload["runs"]:
        lines.append(
            f"| {run['run_id']} | {run['case_id']} | {run['replay_valid']} | "
            f"{run['answer_valid']} | {run['coverage_reconstructable']} | "
            f"{run['reason'] or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    parser.add_argument("--cases-root", type=Path, default=dataset / "cases")
    parser.add_argument("--answer-key-root", type=Path, default=dataset / "answer_key")
    parser.add_argument("--runs-root", type=Path, default=Path("backend/var/runs"))
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/mechanism_coverage_phase5_raw.json"))
    parser.add_argument("--case-table", type=Path,
                        default=Path("analysis/mechanism_coverage_phase5_cases.md"))
    args = parser.parse_args()
    result = evaluate(cases_root=args.cases_root, answer_key_root=args.answer_key_root,
                      runs_root=args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    args.case_table.parent.mkdir(parents=True, exist_ok=True)
    args.case_table.write_text(render_case_table(result), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
