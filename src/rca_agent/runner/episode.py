from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from rca_agent.environment.contract import RcaEnvironment
from rca_agent.models import (
    AnswerArtifact,
    AuthoritativeDecision,
    CallToolAction,
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
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.episode_id = f"ep-{uuid.uuid4().hex}"
        store = JsonlTranscriptStore(transcript_path) if transcript_path is not None else None
        self.transcript = Transcript(self.episode_id, store)
        self.status = EpisodeStatus.CREATED
        self.__environment = environment
        self.__max_rounds = max_rounds
        self.__rounds = 0
        self.__decision_provider = decision_provider or CoreDecisionProvider()
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

    def preview_core_assessment(self) -> AuthoritativeDecision:
        """Return a read-only Core assessment of the current submitted evidence."""

        self._require_status(EpisodeStatus.ACTIVE)
        proposals = tuple(self.__proposals)
        used_observations = tuple(
            sorted({item for proposal in proposals for item in proposal.observation_refs})
        )
        provider = self.__decision_provider
        if isinstance(provider, CoreDecisionProvider):
            return provider.preview(
                proposals,
                tuple(self.__observation_ledger[item] for item in used_observations),
            )
        return provider.decide(
            proposals,
            tuple(self.__observation_ledger[item] for item in used_observations),
            StopReason.POLICY_REQUEST,
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
        self.__observation_refs.update(item.observation_id for item in response.observations)
        self.__observation_ledger.update(
            {
                item.observation_id: IssuedObservation(item, response.query_ref)
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
