"""Bounded background execution for server-approved, GT-free run profiles."""
from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from backend.app.config import Settings
from backend.app.schemas import RecoveryStatus, RunCreateResponse, RunStatus
from backend.app.services.run_store import RUN_ID_RE, RunManifest
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.environment import Rca100Environment
from rca_agent.inquiry_policy import InquiryInvestigationPolicy
from rca_agent.langchain_acquisition import LangChainInquiryAdapter
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.local_model import create_local_chat_model
from rca_agent.models import (
    CallToolAction,
    GetServiceTraceSummaryRequest,
    GetTraceStageBreakdownRequest,
    RequestStopAction,
    ToolName,
    ToolResponse,
)
from rca_agent.qwen_policy import create_qwen_discovery_pipeline
from rca_agent.runner import (
    CoreDecisionProvider,
    EpisodeRunner,
    OperationalMetricName,
    OperationalMetrics,
    RecordedReplay,
)

SCRIPTED_PROFILE = "scripted-trace-demo"
QWEN_PROFILE = "qwen-investigation"
APPROVED_PROFILES = frozenset({SCRIPTED_PROFILE, QWEN_PROFILE})


class RunProfileUnavailable(RuntimeError):
    """A server-owned profile is known but its host-side credential is absent."""


class RunJobManager:
    """Small local job manager for the first live-console increment.

    It is intentionally not a durable distributed queue. Every durable state transition is
    written to the Run manifest, and interrupted local jobs become explicit failures on restart.
    """

    def __init__(
        self,
        cases_root: Path,
        runs_root: Path,
        *,
        max_workers: int = 1,
        settings: Settings | None = None,
    ) -> None:
        self._cases_root = cases_root
        self._runs_root = runs_root
        self._settings = settings or Settings(cases_root=cases_root, runs_root=runs_root)
        self._settings.validate()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rca-live")
        self._lock = Lock()
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted_runs()

    def submit(self, case_id: str, profile_id: str) -> RunCreateResponse:
        if profile_id not in APPROVED_PROFILES:
            raise ValueError("unsupported run profile")
        if profile_id == QWEN_PROFILE and not os.environ.get("RCA_LOCAL_API_KEY"):
            raise RunProfileUnavailable("RCA_LOCAL_API_KEY is not configured")
        run_id = f"run-live-{uuid.uuid4().hex[:16]}"
        if not RUN_ID_RE.fullmatch(run_id):
            raise AssertionError("generated run ID is invalid")
        run_dir = self._runs_root / run_id
        with self._lock:
            run_dir.mkdir(parents=False, exist_ok=False)
            manifest = RunManifest(
                schema_version=1,
                run_id=run_id,
                case_id=case_id,
                status=RunStatus.QUEUED,
                started_at=datetime.now(UTC).isoformat(),
                profile_id=profile_id,
                model_label=(
                    "No model — scripted live profile"
                    if profile_id == SCRIPTED_PROFILE
                    else self._settings.qwen_model
                ),
            )
            _write_manifest(run_dir, manifest)
            self._pool.submit(self._execute, run_dir, manifest)
        return RunCreateResponse(
            run_id=run_id,
            case_id=case_id,
            status=RunStatus.QUEUED,
            profile_id=profile_id,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)

    def _execute(self, run_dir: Path, queued: RunManifest) -> None:
        running = queued.model_copy(update={"status": RunStatus.RUNNING})
        _write_manifest(run_dir, running)
        operational_metrics = OperationalMetrics()
        try:
            if queued.profile_id == SCRIPTED_PROFILE:
                answer = _run_scripted_trace_profile(
                    self._cases_root,
                    queued.case_id,
                    run_dir / "transcript.jsonl",
                    operational_metrics=operational_metrics,
                )
            else:
                answer = _run_qwen_investigation_profile(
                    self._settings,
                    queued.case_id,
                    run_dir / "transcript.jsonl",
                    operational_metrics=operational_metrics,
                )
            replay = RecordedReplay.from_path(run_dir / "transcript.jsonl")
            if not replay.valid:
                raise RuntimeError("recorded replay is invalid")
            _atomic_write(run_dir / "answer.json", answer.model_dump_json(indent=2) + "\n")
            finished = running.model_copy(update={
                "status": RunStatus.SUCCEEDED,
                "finished_at": datetime.now(UTC).isoformat(),
            })
            _write_manifest(run_dir, finished)
        except Exception as exc:  # noqa: BLE001 - persist safe failure type, never exception text
            operational_metrics.count(
                OperationalMetricName.EPISODE_FAILED,
                failure_category="EXECUTION",
            )
            failed = running.model_copy(update={
                "status": RunStatus.FAILED,
                "finished_at": datetime.now(UTC).isoformat(),
                "error_code": type(exc).__name__,
            })
            _write_manifest(run_dir, failed)
        finally:
            _atomic_write(
                run_dir / "metrics.json",
                json.dumps(operational_metrics.snapshot().model_dump(mode="json"), ensure_ascii=False) + "\n",
            )

    def _recover_interrupted_runs(self) -> None:
        for directory in self._runs_root.iterdir():
            manifest_path = directory / "manifest.json"
            if not directory.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = RunManifest.model_validate_json(manifest_path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if manifest.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                continue
            recovered = manifest.model_copy(update={
                "status": RunStatus.FAILED,
                "finished_at": datetime.now(UTC).isoformat(),
                "error_code": "HostRestarted",
                "recovery_status": RecoveryStatus.RETRYABLE,
                "recovery_failure": "host_restart_pending_operation",
            })
            _write_manifest(directory, recovered)


def _run_scripted_trace_profile(
    cases_root: Path,
    case_id: str,
    transcript_path: Path,
    *,
    operational_metrics: OperationalMetrics | None = None,
):
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, case_id),
        transcript_path=transcript_path,
        operational_metrics=operational_metrics,
    )
    bootstrap = runner.start()
    response = runner.handle(CallToolAction(
        tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
        request=GetServiceTraceSummaryRequest(
            entity_ref=bootstrap.alert_entity.entity_ref,
            window=bootstrap.observation_horizon,
            min_duration_ms=100,
            max_traces=4,
        ),
    ))
    trace_ref = _first_trace_ref(response)
    if trace_ref is not None:
        runner.handle(CallToolAction(
            tool_name=ToolName.GET_TRACE_STAGE_BREAKDOWN,
            request=GetTraceStageBreakdownRequest(trace_ref=trace_ref, max_stages=12),
        ))
    answer = runner.handle(RequestStopAction(
        summary=(
            "Completed the server-approved scripted trace investigation. "
            "No model or Ground Truth was used."
        ),
    ))
    if answer is None:
        raise RuntimeError("runner did not freeze an answer")
    return answer


def _run_qwen_investigation_profile(
    settings: Settings,
    case_id: str,
    transcript_path: Path,
    *,
    mechanism_coverage_mode: str = "off",
    operational_metrics: OperationalMetrics | None = None,
):
    """Run the provider-backed Agent without evaluator or Ground Truth access."""
    environment = Rca100Environment.from_case(
        settings.cases_root,
        case_id,
        query_budget=settings.qwen_query_budget,
    )
    discovery, verification_planner = create_qwen_discovery_pipeline(
        model=settings.qwen_model,
        base_url=settings.qwen_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
        allow_insecure_http=settings.qwen_allow_insecure_http,
    )
    runner = EpisodeRunner(
        environment,
        max_rounds=settings.qwen_max_rounds,
        transcript_path=transcript_path,
        decision_provider=CoreDecisionProvider(),
        discovery=discovery,
        verification_planner=verification_planner,
        evidence_registry=rca100_evidence_registry,
        discovery_timeout_seconds=settings.qwen_discovery_timeout_seconds,
        mechanism_coverage_mode=mechanism_coverage_mode,
        operational_metrics=operational_metrics,
    )
    chat_model = create_local_chat_model(
        model=settings.qwen_model,
        base_url=settings.qwen_base_url,
        timeout_seconds=settings.qwen_timeout_seconds,
        # The Qwen endpoint has a 65,536-token context window.  Investigation
        # prompts can grow beyond 57k tokens after several observations, so
        # reserving 8k output tokens can still exceed the 65,536-token context
        # window before generation starts.  The Inquiry schema is bounded and
        # fits within a 4k completion budget while leaving room for long rounds.
        max_completion_tokens=4096,
        allow_insecure_http=settings.qwen_allow_insecure_http,
    )
    inquiry = LangChainInquiryAdapter.from_chat_model(
        chat_model,
        structured_method="json_schema",
    )
    acquisition = InquiryInvestigationPolicy(inquiry)
    policy = LangChainPolicyAdapter(acquisition)
    report = LangGraphInvestigation(
        runner,
        policy,
        discovery_notes_provider=acquisition.discovery_handoff,
        inquiry_rounds_provider=lambda: tuple(acquisition.inquiry_drafts),
        enable_core_feedback=True,
    ).run(
        recursion_limit=settings.qwen_recursion_limit
    )
    return report.answer


def _first_trace_ref(response) -> str | None:
    if not isinstance(response, ToolResponse):
        return None
    for observation in response.observations:
        trace_ref = observation.details.get("trace_ref")
        if isinstance(trace_ref, str):
            return trace_ref
    return None


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write(run_dir / "manifest.json", payload)


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
