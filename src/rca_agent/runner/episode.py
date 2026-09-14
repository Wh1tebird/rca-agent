from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from rca_agent.core.dynamic import DiscoveryPayload, EvidenceRegistry
from rca_agent.core.v3.catalog import mechanism_catalog
from rca_agent.environment.contract import RcaEnvironment
from rca_agent.models import (
    AnswerArtifact,
    AuthoritativeDecision,
    CallToolAction,
    CoreAssessmentPreview,
    DecisionSource,
    DecisionStatus,
    EpisodeStatus,
    HypothesisProposal,
    PolicyAction,
    RequestStopAction,
    ResetResponse,
    StopReason,
    SubmitHypothesisAction,
    ToolResponse,
    ToolStatus,
    UpdateCandidateEvidenceAction,
)
from rca_agent.runner.core_decision import CoreDecisionProvider, IssuedObservation
from rca_agent.runner.transcript import EventType, JsonlTranscriptStore, Transcript
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

ENTITY_REF_RE = re.compile(r"^ent-[a-z0-9-]{3,40}$")


class EpisodeStateError(RuntimeError):
    pass


class ActionRejectedError(ValueError):
    pass


class DecisionProvider(Protocol):
    def decide(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        termination_reason: StopReason,
    ) -> AuthoritativeDecision: ...


class LifecycleDefaultDecisionProvider:
    """Phase 2 lifecycle decision only; it has no root-cause confirmation authority."""

    def decide(
        self,
        proposals: tuple[HypothesisProposal, ...],
        observations: tuple[IssuedObservation, ...],
        termination_reason: StopReason,
    ) -> AuthoritativeDecision:
        del proposals, observations, termination_reason
        return AuthoritativeDecision(
            status=DecisionStatus.INCONCLUSIVE,
            source=DecisionSource.PHASE2_LIFECYCLE_DEFAULT,
            reason="phase 2 validates lifecycle and isolation without a diagnosis policy",
        )


class EpisodeRunner:
    def __init__(
        self,
        environment: RcaEnvironment,
        *,
        max_rounds: int = 20,
        transcript_path: Path | None = None,
        decision_provider: DecisionProvider | None = None,
        discovery: Any | None = None,
        evidence_registry: EvidenceRegistry | Callable[..., EvidenceRegistry] | None = None,
        discovery_timeout_seconds: float = 30,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if discovery_timeout_seconds <= 0 or discovery_timeout_seconds > 120:
            raise ValueError("discovery timeout must be in (0, 120]")
        if discovery is not None and evidence_registry is None:
            raise ValueError("discovery requires host-owned signal and rule contracts")
        if isinstance(decision_provider, CoreDecisionProvider) and decision_provider.has_dynamic_candidates:
            raise ValueError("a runner requires a fresh Core decision provider")
        self.episode_id = f"ep-{uuid.uuid4().hex}"
        store = JsonlTranscriptStore(transcript_path) if transcript_path is not None else None
        self.transcript = Transcript(self.episode_id, store)
        self.status = EpisodeStatus.CREATED
        self.__environment = environment
        self.__max_rounds = max_rounds
        self.__rounds = 0
        self.__decision_provider = (deepcopy(decision_provider)
                                   if isinstance(decision_provider, CoreDecisionProvider)
                                   else decision_provider or CoreDecisionProvider())
        self.__discovery = discovery
        self.__registry = evidence_registry
        self.__discovery_timeout = discovery_timeout_seconds
        self.__discovery_attempted = False
        self.__dynamic_options: list[dict[str, Any]] = []
        self.__verification_queries: list[tuple[str, CallToolAction]] = []
        self.__bootstrap: ResetResponse | None = None
        self.__budget_remaining = 0
        self.__proposals: list[HypothesisProposal] = []
        self.__observation_refs: set[str] = set()
        self.__observation_ledger: dict[str, IssuedObservation] = {}
        self.__issued_entities: set[str] = set()
        self.__answer: AnswerArtifact | None = None
        self.transcript.append(
            EventType.EPISODE_CREATED,
            {"status": self.status, "max_rounds": max_rounds},
        )

    @property
    def answer(self) -> AnswerArtifact | None:
        return self.__answer

    @property
    def rounds(self) -> int:
        return self.__rounds

    @property
    def bootstrap(self) -> ResetResponse | None:
        return self.__bootstrap

    @property
    def dynamic_candidate_options(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.__dynamic_options))

    def should_discover(self, *, stopping: bool = False) -> bool:
        if (self.status != EpisodeStatus.ACTIVE or self.__discovery is None
                or self.__discovery_attempted or self.__rounds >= self.__max_rounds
                or self.__budget_remaining <= 0 or not self.__observation_ledger
                or not isinstance(self.__decision_provider, CoreDecisionProvider)):
            return False
        preview = self.preview_core_assessment()
        if preview.overall_decision.status == DecisionStatus.CONFIRMED_ROOT_CAUSE:
            return False
        resolved = {h for c in preview.candidate_assessments for h in c.hypothesis_ids}
        if any(p.hypothesis_id not in resolved for p in self.__proposals):
            return True
        return bool(self.__proposals) and (stopping or all(
            c.evidence_status.value == "contradicted" for c in preview.candidate_assessments
        ))

    def discover_candidates(self) -> dict[str, Any]:
        """One bounded model call; only the Runner registers and binds compiled candidates."""
        from queue import Empty, Queue
        from threading import Thread

        self._require_status(EpisodeStatus.ACTIVE)
        if not self.should_discover(stopping=True):
            return {"status": "skipped", "registered": 0}
        self.__discovery_attempted = True
        self.__rounds += 1
        registry = self.__registry
        assert registry is not None and self.__bootstrap is not None
        if callable(registry):
            try:
                registry = registry(self.__bootstrap, tuple(
                    r.observation for r in self.__observation_ledger.values()
                ))
                registry = EvidenceRegistry.model_validate(registry)
            except (ValueError, TypeError):
                self.transcript.append(EventType.DISCOVERY_STARTED, {"registry_unavailable": True})
                result = {"status": "registry_unavailable", "registered": 0}
                self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
                return result
        proposals = {p.hypothesis_id: p for p in self.__proposals}
        context = {
            "alert_summary": self.__bootstrap.alert_summary,
            "hypotheses": [p.model_dump(mode="json") for p in proposals.values()],
            "signals": [{"signal": s.signal, "entity_ref": s.entity_ref,
                         "data_type": s.data_type, "unit": s.unit, "description": s.description,
                         "available": self._signal_executable(s)} for s in registry.signals],
            "rules": [{"rule_id": r.rule_id, "signal": r.signal,
                       "phenomenon": r.description, "aggregation": r.aggregation}
                      for r in registry.rules],
            "observations": [r.observation.model_dump(mode="json")
                             for r in self.__observation_ledger.values()
                             if any(s.accepts(r.observation) for s in registry.signals)],
            "existing_candidates": [{"candidate_id": c.candidate_id, "mechanism": c.fault_type}
                                    for c in mechanism_catalog(
                                        self.__bootstrap.alert_entity.entity_ref)],
        }
        self.transcript.append(EventType.DISCOVERY_STARTED, {"context_digest": sha256_digest(context)})
        mailbox: Queue = Queue(maxsize=1)
        generator = self.__discovery

        def invoke():
            try:
                mailbox.put((True, generator.discover(deepcopy(context))))
            except Exception as exc:  # noqa: BLE001 - no provider text crosses the boundary
                mailbox.put((False, type(exc).__name__))

        Thread(target=invoke, daemon=True).start()
        registered = 0
        aligned = 0
        rejections = []
        try:
            ok, raw = mailbox.get(timeout=self.__discovery_timeout)
            if not ok:
                raise ValueError("discovery failed")
            payload = DiscoveryPayload.model_validate(raw)
            seen = set()
            for draft in payload.hypotheses:
                try:
                    original = proposals.get(draft.source_hypothesis_id)
                    if original is None or draft.source_hypothesis_id in seen:
                        raise ValueError("unknown or duplicate source hypothesis")
                    seen.add(draft.source_hypothesis_id)
                    compiled = registry.compile(draft, original.causal_entity_ref)
                    if not all(self._signal_executable(s) for s in compiled.signals):
                        raise ValueError("query recipe is not executable in this episode")
                    existing = next((c for c in mechanism_catalog(original.causal_entity_ref)
                                     if draft.mechanism.casefold() in {
                                         c.candidate_id.casefold(), c.fault_type.casefold()
                                     } and c.family == draft.family), None)
                    if existing is not None:
                        required = {c.signal for c in existing.conditions}
                        if not required <= {s.signal for s in compiled.signals}:
                            raise ValueError("catalog alignment lacks executable signal recipes")
                        hypothesis_id = "h-aligned-" + compiled.version[:20]
                        self.transcript.append(EventType.CATALOG_CANDIDATE_ALIGNED, {
                            "candidate_id": existing.candidate_id,
                            "hypothesis_id": hypothesis_id,
                            "source_hypothesis_id": original.hypothesis_id,
                        })
                        self._submit_hypothesis(HypothesisProposal(
                            hypothesis_id=hypothesis_id, candidate_id=existing.candidate_id,
                            mechanism=existing.fault_type, causal_entity_ref=original.causal_entity_ref,
                            rationale=draft.causal_explanation,
                        ))
                        queries = {query_fingerprint(s.query.tool_name, s.query.request): s.query
                                   for s in compiled.signals}
                        self.__verification_queries.extend((hypothesis_id, q) for q in queries.values())
                        aligned += 1
                        continue
                    # Preserve the original hypothesis and its evidence; discovery gets its own ID.
                    hypothesis_id = "h-discovered-" + compiled.version[:20]
                    rebound = compiled.model_copy(update={"source_hypothesis_id": hypothesis_id})
                    self.__decision_provider.register_dynamic(rebound)
                    self.transcript.append(EventType.DYNAMIC_CANDIDATE_REGISTERED, {
                        "compiled": rebound.model_dump(mode="json"),
                        "source_hypothesis_id": original.hypothesis_id,
                    })
                    self._submit_hypothesis(HypothesisProposal(
                        hypothesis_id=hypothesis_id, candidate_id=compiled.candidate.candidate_id,
                        mechanism=draft.mechanism, causal_entity_ref=original.causal_entity_ref,
                        rationale=draft.causal_explanation,
                    ))
                    candidate = compiled.candidate
                    self.__dynamic_options.append({
                        "candidate_id": candidate.candidate_id, "fault_type": candidate.fault_type,
                        "family": candidate.family.value,
                        "required_signals": tuple(s.signal for s in compiled.signals),
                        "required_capabilities": tuple(candidate.required_capabilities),
                    })
                    queries = {query_fingerprint(s.query.tool_name, s.query.request): s.query
                               for s in compiled.signals}
                    self.__verification_queries.extend((hypothesis_id, q) for q in queries.values())
                    registered += 1
                except ValueError as exc:
                    rejections.append(str(exc) if type(exc) is ValueError
                                      else "invalid compiled candidate contract")
            result = {"status": "completed", "registered": registered,
                      "aligned": aligned, "rejections": rejections}
        except Empty:
            result = {"status": "timeout", "registered": 0}
        except (ValueError, TypeError):
            result = {"status": "failed", "registered": 0}
        self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
        return result

    def _signal_executable(self, spec) -> bool:
        assert self.__bootstrap is not None
        if not spec.available or spec.entity_ref not in self.__issued_entities:
            return False
        capability = next((c for c in self.__bootstrap.capabilities
                           if c.tool_name == spec.query.tool_name and c.available), None)
        if capability is None:
            return False
        request = spec.query.request
        entity = getattr(request, "entity_ref", None)
        if entity is not None and entity != spec.entity_ref:
            return False
        trace = getattr(request, "trace_ref", None)
        if trace is not None and not any(
            r.observation.entity_ref == spec.entity_ref
            and r.observation.details.get("trace_ref") == trace
            for r in self.__observation_ledger.values()
        ):
            return False
        window = getattr(request, "window", getattr(request, "incident_window", None))
        if window is not None:
            horizon = self.__bootstrap.observation_horizon
            if (window.start < horizon.start or window.end > horizon.end
                    or window.seconds > capability.max_window_seconds):
                return False
        limit = getattr(request, "max_points", getattr(request, "max_stages", 1))
        return limit <= capability.max_results

    @property
    def has_verification_queries(self) -> bool:
        return bool(self.__verification_queries)

    def verify_next_discovered_query(self) -> ToolResponse | AnswerArtifact | None:
        self._require_status(EpisodeStatus.ACTIVE)
        if not self.__verification_queries:
            return None
        hypothesis_id, action = self.__verification_queries.pop(0)
        result = self.handle(action)
        if isinstance(result, AnswerArtifact):
            return result
        if isinstance(result, ToolResponse) and result.observations:
            update = self.handle(UpdateCandidateEvidenceAction(
                hypothesis_id=hypothesis_id,
                observation_refs=tuple(o.observation_id for o in result.observations),
            ))
            if isinstance(update, AnswerArtifact):
                return update
        return result

    def preview_core_assessment(self) -> CoreAssessmentPreview:
        """Return a read-only Core assessment of the current submitted evidence."""

        self._require_status(EpisodeStatus.ACTIVE)
        proposals = tuple(self.__proposals)
        issued_observations = tuple(
            self.__observation_ledger[item]
            for item in sorted(self.__observation_ledger)
        )
        provider = self.__decision_provider
        if isinstance(provider, CoreDecisionProvider):
            assert self.__bootstrap is not None
            return provider.preview(
                proposals,
                issued_observations,
                self.__bootstrap.capabilities,
            )
        return CoreAssessmentPreview(
            overall_decision=provider.decide(
                proposals,
                issued_observations,
                StopReason.POLICY_REQUEST,
            )
        )

    def start(self) -> ResetResponse:
        self._require_status(EpisodeStatus.CREATED)
        bootstrap = self.__environment.reset()
        self.__bootstrap = bootstrap
        self.__budget_remaining = bootstrap.query_budget.remaining
        self.__issued_entities = {bootstrap.alert_entity.entity_ref}
        self.status = EpisodeStatus.ACTIVE
        self.transcript.append(
            EventType.ENVIRONMENT_RESET,
            {"status": self.status, "bootstrap": bootstrap.model_dump(mode="json")},
        )
        return bootstrap

    def handle(self, action: PolicyAction) -> ToolResponse | AnswerArtifact | None:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(EventType.POLICY_ACTION, action)
        if isinstance(action, RequestStopAction):
            self.transcript.append(EventType.STOP_REQUESTED, action)
            return self.freeze_answer(StopReason.POLICY_REQUEST)
        if self.__rounds >= self.__max_rounds:
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.MAX_ROUNDS, "rounds": self.__rounds},
            )
            return self.freeze_answer(StopReason.MAX_ROUNDS)
        self.__rounds += 1
        if isinstance(action, CallToolAction):
            return self._call_tool(action)
        if isinstance(action, SubmitHypothesisAction):
            self._submit_hypothesis(action.proposal)
            return None
        if isinstance(action, UpdateCandidateEvidenceAction):
            self._update_candidate_evidence(action)
            return None
        raise TypeError(f"unsupported typed action: {type(action).__name__}")

    def reject_raw_action(self, payload: Any, error_type: str = "invalid_policy_action") -> None:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.ACTION_REJECTED,
            {"error_type": error_type, "payload_digest": sha256_digest(payload)},
        )

    def record_policy_process_started(self, visible_context: dict[str, Any]) -> None:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.POLICY_PROCESS_STARTED,
            {
                "context_fields": sorted(visible_context),
                "context_digest": sha256_digest(visible_context),
            },
        )

    def record_policy_process_exited(self, exit_kind: str) -> None:
        self.transcript.append(EventType.POLICY_PROCESS_EXITED, {"exit_kind": exit_kind})

    def terminate_policy_failure(self, reason: StopReason, error_type: str) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        event_type = (
            EventType.POLICY_TIMEOUT
            if reason == StopReason.POLICY_TIMEOUT
            else EventType.POLICY_EXCEPTION
        )
        self.transcript.append(event_type, {"error_type": error_type})
        self.transcript.append(EventType.STOP_REQUESTED, {"reason": reason})
        return self.freeze_answer(reason)

    def finish_script(self) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        self.transcript.append(
            EventType.STOP_REQUESTED,
            {"reason": StopReason.SCRIPT_COMPLETED},
        )
        return self.freeze_answer(StopReason.SCRIPT_COMPLETED)

    def freeze_answer(self, termination_reason: StopReason) -> AnswerArtifact:
        self._require_status(EpisodeStatus.ACTIVE)
        proposals = tuple(self.__proposals)
        used_observations = tuple(
            sorted({item for proposal in proposals for item in proposal.observation_refs})
        )
        missing = tuple(
            sorted({item for proposal in proposals for item in proposal.missing_evidence})
        )
        decision = self.__decision_provider.decide(
            proposals,
            tuple(self.__observation_ledger[item] for item in used_observations),
            termination_reason,
        )
        frozen_at = datetime.now(UTC)
        fields = {
            "episode_id": self.episode_id,
            "termination_reason": termination_reason,
            "policy_proposals": proposals,
            "authoritative_decision": decision,
            "observation_refs": used_observations,
            "missing_evidence": missing,
            "transcript_digest": self.transcript.head_digest,
            "component_versions": {
                "rca-agent": _package_version("rca-agent"),
                "rca-agent-domain": _package_version("rca-agent"),
            },
            "frozen_at": frozen_at,
        }
        artifact = AnswerArtifact(
            **fields,
            artifact_digest=sha256_digest(_jsonable(fields)),
        )
        self.__answer = artifact
        self.status = EpisodeStatus.ANSWER_FROZEN
        self.transcript.append(EventType.ANSWER_FROZEN, artifact)
        return artifact

    def begin_evaluation(self) -> None:
        self._require_status(EpisodeStatus.ANSWER_FROZEN)
        if self.__answer is None or not self.__answer.verify_digest():
            raise EpisodeStateError("a valid frozen answer is required for evaluation")
        if any(item.event_type == EventType.EVALUATION_STARTED for item in self.transcript.events):
            raise EpisodeStateError("evaluation was already started")
        self.transcript.append(
            EventType.EVALUATION_STARTED,
            {"answer_artifact_digest": self.__answer.artifact_digest},
        )

    def complete_evaluation(self, evaluation_artifact_digest: str) -> None:
        self._require_status(EpisodeStatus.ANSWER_FROZEN)
        if not self.transcript.events or (
            self.transcript.events[-1].event_type != EventType.EVALUATION_STARTED
        ):
            raise EpisodeStateError("evaluation must be started before completion")
        self.transcript.append(
            EventType.EVALUATION_COMPLETED,
            {"evaluation_artifact_digest": evaluation_artifact_digest},
        )
        self.status = EpisodeStatus.EVALUATED

    def _call_tool(self, action: CallToolAction) -> ToolResponse | AnswerArtifact:
        query_ref = query_fingerprint(action.tool_name, action.request)
        self.transcript.append(
            EventType.TOOL_REQUEST,
            {
                "tool_name": action.tool_name,
                "request": action.request.model_dump(mode="json"),
                "query_ref": query_ref,
                "budget_before": self.__budget_remaining,
            },
        )
        tool = getattr(self.__environment, action.tool_name.value)
        try:
            response: ToolResponse = tool(action.request)
        except Exception as exc:  # noqa: BLE001 - exception type only crosses the transcript
            self.transcript.append(
                EventType.TOOL_EXCEPTION,
                {"tool_name": action.tool_name, "error_type": type(exc).__name__},
            )
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.TOOL_FAILURE},
            )
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        self.transcript.append(EventType.TOOL_RESPONSE, response)
        self.transcript.append(
            EventType.QUERY_COST,
            {
                "query_ref": response.query_ref,
                **response.cost.model_dump(mode="json"),
            },
        )
        self.__budget_remaining = response.cost.budget_after
        if any(
            item.observation_id in self.__observation_ledger
            and self.__observation_ledger[item.observation_id] != IssuedObservation(
                item, response.query_ref, not response.truncated and response.status == ToolStatus.SUCCESS
            )
            for item in response.observations
        ):
            self.transcript.append(EventType.TOOL_EXCEPTION, {
                "error_type": "issued_observation_changed", "tool_name": action.tool_name,
            })
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        self.__observation_refs.update(item.observation_id for item in response.observations)
        self.__observation_ledger.update(
            {
                item.observation_id: IssuedObservation(
                    item, response.query_ref,
                    not response.truncated and response.status == ToolStatus.SUCCESS,
                )
                for item in response.observations
            }
        )
        self.__issued_entities.update(_find_entity_refs(response.model_dump(mode="json")))
        if response.query_ref != query_ref:
            raise RuntimeError("environment query fingerprint diverged from the request")
        if response.status == ToolStatus.BUDGET_EXHAUSTED:
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.BUDGET_EXHAUSTED},
            )
            return self.freeze_answer(StopReason.BUDGET_EXHAUSTED)
        if response.status == ToolStatus.QUERY_FAILED:
            self.transcript.append(
                EventType.TOOL_EXCEPTION,
                {"tool_name": action.tool_name, "error_type": "modeled_query_failure"},
            )
            self.transcript.append(
                EventType.STOP_REQUESTED,
                {"reason": StopReason.TOOL_FAILURE},
            )
            return self.freeze_answer(StopReason.TOOL_FAILURE)
        return response

    def _submit_hypothesis(self, proposal: HypothesisProposal) -> None:
        unknown_observations = sorted(set(proposal.observation_refs) - self.__observation_refs)
        if unknown_observations or proposal.causal_entity_ref not in self.__issued_entities:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unissued_reference",
                    "proposal_digest": sha256_digest(proposal),
                },
            )
            raise ActionRejectedError("hypothesis references values not issued by this episode")
        if proposal.hypothesis_id in {item.hypothesis_id for item in self.__proposals}:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {"error_type": "duplicate_hypothesis_id", "hypothesis_id": proposal.hypothesis_id},
            )
            raise ActionRejectedError("hypothesis ID was already submitted")
        self.__proposals.append(proposal)
        self.transcript.append(EventType.HYPOTHESIS_SUBMITTED, proposal)

    def _update_candidate_evidence(self, action: UpdateCandidateEvidenceAction) -> None:
        unknown_observations = sorted(set(action.observation_refs) - self.__observation_refs)
        if unknown_observations:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unissued_reference",
                    "hypothesis_id": action.hypothesis_id,
                    "action_digest": sha256_digest(action),
                },
            )
            raise ActionRejectedError("candidate evidence references values not issued by this episode")
        index = next(
            (
                item_index
                for item_index, proposal in enumerate(self.__proposals)
                if proposal.hypothesis_id == action.hypothesis_id
            ),
            None,
        )
        if index is None:
            self.transcript.append(
                EventType.ACTION_REJECTED,
                {
                    "error_type": "unknown_hypothesis_id",
                    "hypothesis_id": action.hypothesis_id,
                },
            )
            raise ActionRejectedError("candidate evidence update requires a submitted hypothesis")
        proposal = self.__proposals[index]
        updated = proposal.model_copy(
            update={
                "observation_refs": tuple(
                    sorted({*proposal.observation_refs, *action.observation_refs})
                ),
                "missing_evidence": tuple(
                    sorted({*proposal.missing_evidence, *action.missing_evidence})
                ),
            }
        )
        self.__proposals[index] = updated
        self.transcript.append(
            EventType.CANDIDATE_EVIDENCE_UPDATED,
            {
                "hypothesis_id": action.hypothesis_id,
                "observation_refs": action.observation_refs,
                "missing_evidence": action.missing_evidence,
            },
        )

    def _require_status(self, expected: EpisodeStatus) -> None:
        if self.status != expected:
            raise EpisodeStateError(
                f"operation requires {expected.value}, current status is {self.status.value}"
            )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _jsonable(value: dict[str, Any]) -> dict[str, Any]:
    from pydantic import TypeAdapter

    return TypeAdapter(dict[str, Any]).dump_python(value, mode="json")


def _find_entity_refs(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {item for nested in value.values() for item in _find_entity_refs(nested)}
    if isinstance(value, list | tuple):
        return {item for nested in value for item in _find_entity_refs(nested)}
    if isinstance(value, str) and ENTITY_REF_RE.fullmatch(value):
        return {value}
    return set()
