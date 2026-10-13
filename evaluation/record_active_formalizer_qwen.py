"""Run selected GT-free Qwen Cases through active Inquiry and a chosen mechanism stage.

Each Case gets independent artifacts. The command records safe failure types and
continues. Ground Truth is deliberately absent from this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from backend.app.config import Settings
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.environment import Rca100Environment
from rca_agent.inquiry_policy import InquiryInvestigationPolicy
from rca_agent.langchain_acquisition import LangChainInquiryAdapter
from rca_agent.langchain_discovery import LangChainMechanismDiscoveryAdapter
from rca_agent.langchain_formalizer import (
    LangChainDiscoveryFormatFormalizerAdapter,
    LangChainMechanismFormalizerAdapter,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langchain_verification_planner import LangChainVerificationPlannerAdapter
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.local_model import create_local_chat_model
from rca_agent.models import CallToolAction
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, EventType, RecordedReplay
from rca_agent.runner.transcript import load_jsonl_events
from rca_agent.tools.fingerprint import query_fingerprint

MechanismStage = Literal["formalizer", "discovery", "discovery-formalizer"]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _config(
    settings: Settings, case_id: str, mechanism_stage: MechanismStage,
    workspace_mode: str = "active",
) -> dict[str, object]:
    endpoint = urlsplit(settings.qwen_base_url)
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("base_url contains private fields")
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in [*sorted((repo / "src").rglob("*.py")), Path(__file__).resolve()]:
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return {
        "implementation_digest": digest.hexdigest(),
        "catalog_policy": "unchanged-default-no-gt-specific-exclusion",
        "case_id": case_id,
        "model": settings.qwen_model,
        "base_url": settings.qwen_base_url,
        "provider_timeout_seconds": settings.qwen_timeout_seconds,
        "discovery_timeout_seconds": settings.qwen_discovery_timeout_seconds,
        "max_completion_tokens": 16384,
        "query_budget": settings.qwen_query_budget,
        "max_rounds": settings.qwen_max_rounds,
        "recursion_limit": settings.qwen_recursion_limit,
        "inquiry_query_limit": 10,
        "structured_method": "json_schema",
        "investigation_workspace_mode": workspace_mode,
        "mechanism_stage": mechanism_stage,
        "discovery_handoff_protocol": (
            "host-bound-v6" if mechanism_stage != "formalizer" else "legacy-formalizer"
        ),
        "mechanism_formalizer": (
            "LangChainMechanismFormalizerAdapter"
            if mechanism_stage == "formalizer" else None
        ),
        "discovery": (
            "LangChainMechanismDiscoveryAdapter"
            if mechanism_stage != "formalizer" else None
        ),
        "discovery_format_formalizer": (
            "LangChainDiscoveryFormatFormalizerAdapter"
            if mechanism_stage == "discovery-formalizer" else None
        ),
        "mechanism_coverage_mode": "off",
        "core_decision_provider": "CoreDecisionProvider",
        "evidence_registry": "rca100_evidence_registry",
        "gt_available_to_inference": False,
    }


def _run_case(
    settings: Settings, case_id: str, transcript_path: Path,
    mechanism_stage: MechanismStage,
    workspace_mode: str = "active",
):
    environment = Rca100Environment.from_case(
        settings.cases_root, case_id, query_budget=settings.qwen_query_budget,
    )
    model = create_local_chat_model(
        model=settings.qwen_model,
        base_url=settings.qwen_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
        max_completion_tokens=16384,
        allow_insecure_http=settings.qwen_allow_insecure_http,
    )
    inquiry = LangChainInquiryAdapter.from_chat_model(model, structured_method="json_schema")
    acquisition = InquiryInvestigationPolicy(inquiry)
    runner = EpisodeRunner(
        environment,
        max_rounds=settings.qwen_max_rounds,
        transcript_path=transcript_path,
        decision_provider=CoreDecisionProvider(),
        discovery=(
            LangChainMechanismDiscoveryAdapter.from_chat_model(
                model, structured_method="json_schema",
            ) if mechanism_stage != "formalizer" else None
        ),
        mechanism_formalizer=(
            LangChainMechanismFormalizerAdapter.from_chat_model(
                model, structured_method="json_schema",
            ) if mechanism_stage == "formalizer" else None
        ),
        discovery_format_formalizer=(
            LangChainDiscoveryFormatFormalizerAdapter.from_chat_model(
                model, structured_method="json_schema",
            ) if mechanism_stage == "discovery-formalizer" else None
        ),
        verification_planner=LangChainVerificationPlannerAdapter.from_chat_model(
            model, structured_method="json_schema",
        ),
        evidence_registry=rca100_evidence_registry,
        discovery_timeout_seconds=settings.qwen_discovery_timeout_seconds,
        investigation_workspace_mode=workspace_mode,
        mechanism_coverage_mode="off",
    )
    report = LangGraphInvestigation(
        runner,
        LangChainPolicyAdapter(acquisition),
        discovery_notes_provider=acquisition.discovery_handoff,
        inquiry_rounds_provider=lambda: tuple(acquisition.inquiry_drafts),
        enable_core_feedback=True,
    ).run(recursion_limit=settings.qwen_recursion_limit)
    return report.answer


def _evidence(events: tuple) -> dict[str, object]:
    counts = Counter(event.event_type.value for event in events)
    round_events = [event for event in events
                    if event.event_type == EventType.INQUIRY_ROUND_RECORDED]
    offered = sum(len(event.payload.get("selected_actions", ())) for event in round_events)
    active = [event for event in events if event.event_type
              == EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED]
    requests = [event for event in events if event.event_type == EventType.TOOL_REQUEST]
    discovery_sequence = min((event.sequence for event in events
                              if event.event_type == EventType.DISCOVERY_STARTED), default=float("inf"))
    offered_actions = [CallToolAction.model_validate(item["action"])
                       for event in round_events for item in event.payload.get("selected_actions", ())]
    offered_refs = {query_fingerprint(action.tool_name, action.request) for action in offered_actions}
    inquiry_executed = sum(request.sequence < discovery_sequence
                           and request.payload.get("query_ref") in offered_refs for request in requests)
    active_executed = sum(any(
        request.sequence > selection.sequence
        and request.payload.get("query_ref")
        == selection.payload["selected_query_fingerprint"]
        for request in requests
    ) for selection in active)
    explicitly_suppressed = sum(
        len(event.payload.get("suppressed_query_fingerprints", ())) for event in active
    )
    formalized = [event for event in events if event.event_type
                  == EventType.MECHANISM_FORMALIZATION_COMPLETED]
    bound = [event for event in events if event.event_type
             == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED]
    final_bound = {event.payload["binding"]["hypothesis_id"]: event.payload for event in bound}
    return {
        "inquiry_attempts": [event.payload for event in events
                             if event.event_type == EventType.INQUIRY_ATTEMPT_RECORDED],
        "inquiry_rounds": len(round_events),
        "inquiry_queries_offered": offered,
        "inquiry_queries_executed": inquiry_executed,
        "inquiry_queries_not_executed": offered - inquiry_executed,
        "inquiry_queries_explicitly_suppressed": explicitly_suppressed,
        "inquiry_queries_without_selection": (
            offered - len(active) - explicitly_suppressed if active else offered - inquiry_executed
        ),
        "tool_requests_executed": counts[EventType.TOOL_REQUEST.value],
        "active_selections": len(active),
        "active_executed": active_executed > 0,
        "active_tasks_completed": counts[EventType.DISCRIMINATION_TASK_COMPLETED.value],
        "formalizer_started": counts[EventType.MECHANISM_FORMALIZATION_STARTED.value],
        "formalizer_completed": len(formalized),
        "formalizer_rejected": counts[EventType.MECHANISM_FORMALIZATION_REJECTED.value],
        "formalizer_retried": counts[EventType.MECHANISM_FORMALIZATION_RETRIED.value],
        "formalizer_executed": bool(formalized),
        "bound_formalization_attempts": sum(event.payload["attempt"] > 0 for event in bound),
        "bound_originals_formalized": sum(
            (item.get("response") or {}).get("status") == "formalized"
            for item in final_bound.values()
        ),
        "bound_originals_deferred": sum(
            (item.get("response") or {}).get("status") != "formalized"
            for item in final_bound.values()
        ),
        "formalized_mechanisms": sum(
            len(event.payload["payload"]["mechanisms"]) for event in formalized
        ),
        "formalizer_skipped_candidates": sum(
            len(event.payload["payload"].get("skipped_candidates", ()))
            for event in formalized
        ),
        "discovery_format_repair_requested": counts[
            EventType.DISCOVERY_FORMAT_REPAIR_REQUESTED.value
        ],
        "discovery_format_repair_completed": counts[
            EventType.DISCOVERY_FORMAT_REPAIR_COMPLETED.value
        ],
        "discovery_format_repair_rejected": counts[
            EventType.DISCOVERY_FORMAT_REPAIR_REJECTED.value
        ],
        "mechanisms_frozen": counts[EventType.MECHANISM_DRAFT_FROZEN.value],
        "candidates_compiled": counts[EventType.DYNAMIC_CANDIDATE_REGISTERED.value],
    }


def record_cases(
    case_ids: tuple[str, ...], runs_root: Path,
    *, mechanism_stage: MechanismStage = "formalizer",
    workspace_mode: str = "active",
) -> list[dict[str, object]]:
    if not case_ids or any(not re.fullmatch(r"t\d{3}", item) for item in case_ids):
        raise ValueError("Case IDs must match tNNN")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate Case ID")
    if workspace_mode not in {"active", "shadow"}:
        raise ValueError("unsupported query comparison mode")
    settings = Settings.from_environment()
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        run_id = f"run-{workspace_mode}-{mechanism_stage}-{case_id}-{uuid.uuid4().hex[:12]}"
        run_dir = runs_root / run_id
        run_dir.mkdir()
        row: dict[str, object] = {
            "case_id": case_id, "run_id": run_id, "mechanism_stage": mechanism_stage,
            "workspace_mode": workspace_mode,
        }
        transcript_path = run_dir / "transcript.jsonl"
        try:
            _write_json(run_dir / "config.json", _config(settings, case_id, mechanism_stage, workspace_mode))
            answer = _run_case(settings, case_id, transcript_path, mechanism_stage, workspace_mode)
            _write_json(run_dir / "answer.json", answer.model_dump(mode="json"))
            replay = RecordedReplay.from_path(transcript_path)
            replay_result = {
                "valid": replay.valid,
                "answer_digest_valid": answer.verify_digest(),
            }
            _write_json(run_dir / "replay.json", replay_result)
            if not all(replay_result.values()):
                raise RuntimeError("frozen Answer or Recorded Replay is invalid")
            evidence = _evidence(load_jsonl_events(transcript_path))
            evidence["retained_candidate_tracks"] = len(
                answer.investigation_workspace.candidates
            ) if answer.investigation_workspace is not None else 0
            _write_json(run_dir / "evidence.json", evidence)
            row.update({
                "status": "succeeded",
                "replay_valid": True,
                "final_status": answer.authoritative_decision.status.value,
                **evidence,
            })
        except Exception as exc:  # noqa: BLE001 - one Case failure must not stop the next
            row.update({"status": "failed", "error_code": type(exc).__name__})
            if transcript_path.exists():
                try:
                    _write_json(run_dir / "evidence.json", _evidence(
                        load_jsonl_events(transcript_path)
                    ))
                except (OSError, ValueError):
                    pass
        _write_json(run_dir / "result.json", row)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def record_paired(case_ids, runs_root, *, mechanism_stage="discovery-formalizer", workers=2):
    """Preflight without GT, then reuse the same recorder for both query arms."""
    if workers not in {1, 2} or len(set(case_ids)) != len(case_ids):
        raise ValueError("invalid paired batch bounds")
    settings = Settings.from_environment()
    valid, preflight = [], []
    for case_id in case_ids:
        if not re.fullmatch(r"t\d{3}", case_id):
            raise ValueError("invalid Case ID")
        try:
            Rca100Environment.from_case(settings.cases_root, case_id,
                                       query_budget=settings.qwen_query_budget).reset()
            valid.append(case_id)
            preflight.append({"case_id": case_id, "valid": True})
        except Exception as exc:  # noqa: BLE001 - no private exception text
            preflight.append({"case_id": case_id, "valid": False, "error_code": type(exc).__name__})
    runs_root.mkdir(parents=True, exist_ok=True)
    _write_json(runs_root / "preflight.json", preflight)
    jobs = [(case_id, mode) for case_id in valid for mode in ("shadow", "active")]
    def execute(job):
        case_id, mode = job
        return record_cases((case_id,), runs_root / mode, mechanism_stage=mechanism_stage,
                            workspace_mode=mode)[0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(execute, jobs))
    _write_json(runs_root / "batch_results.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="+", help="Selected Cases, e.g. t007 t030")
    parser.add_argument("--workspace-mode", choices=("active", "shadow"), default="active")
    parser.add_argument("--paired", action="store_true", help="Compare legacy order vs active ranking")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--mechanism-stage", choices=("formalizer", "discovery", "discovery-formalizer"),
        default="formalizer", help="Choose the mechanism-stage experiment",
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("analysis/investigation_active_qwen_20260924/runs"),
    )
    args = parser.parse_args()
    if args.paired:
        record_paired(tuple(args.case_ids), args.runs_root,
                      mechanism_stage=args.mechanism_stage, workers=args.workers)
    else:
        record_cases(tuple(args.case_ids), args.runs_root, mechanism_stage=args.mechanism_stage,
                     workspace_mode=args.workspace_mode)


if __name__ == "__main__":
    main()
