from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.models import (
    AnswerArtifact,
    DecisionStatus,
    EpisodeStatus,
    HypothesisProposal,
    ToolResponse,
)
from rca_agent.runner.transcript import (
    GENESIS_DIGEST,
    EventType,
    TranscriptEvent,
    load_jsonl_events,
)
from rca_agent.tools.fingerprint import query_fingerprint


class ReplayValidationError(ValueError):
    pass


class ReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    episode_id: str
    event_count: int = Field(ge=1)
    final_status: EpisodeStatus
    head_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    tool_calls: int = Field(ge=0)


class RecordedReplay:
    """Offline transcript verification. It never opens RCA100 or evaluation data."""

    @classmethod
    def from_path(cls, path: Path) -> ReplayReport:
        return cls.verify(load_jsonl_events(path))

    @classmethod
    def verify(cls, events: tuple[TranscriptEvent, ...] | list[TranscriptEvent]) -> ReplayReport:
        if not events:
            raise ReplayValidationError("transcript is empty")
        episode_id = events[0].episode_id
        previous = GENESIS_DIGEST
        status: EpisodeStatus | None = None
        evaluation_started = False
        budget_remaining: int | None = None
        pending_request: dict | None = None
        pending_response: ToolResponse | None = None
        tool_calls = 0
        discovery_started = False
        discovery_completed = False
        dynamic_ids: set[str] = set()
        from rca_agent.runner.core_decision import CoreDecisionProvider, IssuedObservation

        provider = CoreDecisionProvider()
        observations = {}
        proposals = {}
        verification_report = None
        discovery_refs = ()
        discovery_queries = ()
        for expected_sequence, event in enumerate(events, start=1):
            if event.episode_id != episode_id:
                raise ReplayValidationError("episode ID changed inside transcript")
            if event.sequence != expected_sequence:
                raise ReplayValidationError("event sequence is not contiguous")
            if event.previous_digest != previous:
                raise ReplayValidationError("previous-event digest chain is broken")
            if not event.verify_digest():
                raise ReplayValidationError("event payload digest is invalid")
            previous = event.digest

            if expected_sequence == 1:
                if event.event_type != EventType.EPISODE_CREATED:
                    raise ReplayValidationError("first event must create the episode")
                status = EpisodeStatus.CREATED
                continue
            if event.event_type == EventType.ENVIRONMENT_RESET:
                if status != EpisodeStatus.CREATED:
                    raise ReplayValidationError("environment reset is out of order")
                status = EpisodeStatus.ACTIVE
                budget_remaining = int(event.payload["bootstrap"]["query_budget"]["remaining"])
            elif event.event_type == EventType.DISCOVERY_STARTED:
                if status != EpisodeStatus.ACTIVE or discovery_started:
                    raise ReplayValidationError("discovery start is out of order")
                discovery_started = True
                discovery_refs = tuple(sorted(observations))
                discovery_queries = tuple(sorted({r.query_ref for r in observations.values()}))
            elif event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED:
                from rca_agent.core.dynamic import CompiledCandidate, observation_fact_digest

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("candidate registration is out of order")
                raw_compiled = dict(event.payload["compiled"])
                # Legacy v1 records carried a per-mechanism confirmation configuration.
                raw_compiled.pop("confirmation", None)
                compiled = CompiledCandidate.model_validate(raw_compiled)
                if compiled.candidate.candidate_id in dynamic_ids:
                    raise ReplayValidationError("dynamic candidate registered twice")
                dynamic_ids.add(compiled.candidate.candidate_id)
                if compiled.predictions and (
                    compiled.discovery_observation_refs != discovery_refs
                    or compiled.discovery_query_refs != discovery_queries
                    or compiled.discovery_fact_digests != tuple(sorted({
                        observation_fact_digest(r.observation) for r in observations.values()
                    }))
                ):
                    raise ReplayValidationError("discovery evidence snapshot mismatch")
                provider.register_dynamic(compiled)
            elif event.event_type == EventType.DISCOVERY_COMPLETED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("discovery completion is out of order")
                if event.payload["registered"] != len(dynamic_ids):
                    raise ReplayValidationError("discovery registration count mismatch")
                discovery_completed = True
            elif event.event_type == EventType.CATALOG_CANDIDATE_ALIGNED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("catalog alignment is out of order")
            elif event.event_type == EventType.HYPOTHESIS_SUBMITTED:
                if status != EpisodeStatus.ACTIVE:
                    raise ReplayValidationError("hypothesis submitted outside active episode")
                proposal = HypothesisProposal.model_validate(event.payload)
                if proposal.hypothesis_id in proposals:
                    raise ReplayValidationError("duplicate hypothesis")
                proposals[proposal.hypothesis_id] = proposal
            elif event.event_type == EventType.CANDIDATE_EVIDENCE_UPDATED:
                if status != EpisodeStatus.ACTIVE:
                    raise ReplayValidationError("evidence updated outside active episode")
                key = event.payload["hypothesis_id"]
                if key not in proposals:
                    raise ReplayValidationError("unknown hypothesis in evidence update")
                proposal = proposals[key]
                refs = set(event.payload["observation_refs"])
                if not refs <= observations.keys():
                    raise ReplayValidationError("unissued evidence in update")
                proposals[key] = proposal.model_copy(update={
                    "observation_refs": tuple(sorted(set(proposal.observation_refs) | refs)),
                    "missing_evidence": tuple(sorted(set(proposal.missing_evidence)
                                                     | set(event.payload["missing_evidence"]))),
                })
            elif event.event_type == EventType.DYNAMIC_VERIFICATION:
                if status != EpisodeStatus.ACTIVE or not discovery_completed or pending_request:
                    raise ReplayValidationError("dynamic verification is out of order")
                expected = provider.verification_reports(tuple(proposals.values()), tuple(observations.values()))
                verification_report = {k: v.model_dump(mode="json") for k, v in expected.items()}
                if event.payload.get("candidates") != verification_report:
                    raise ReplayValidationError("dynamic verification differs from recorded evidence")
            elif event.event_type == EventType.TOOL_REQUEST:
                if status != EpisodeStatus.ACTIVE or pending_request is not None:
                    raise ReplayValidationError("tool request is out of order")
                expected_ref = query_fingerprint(
                    str(event.payload["tool_name"]), event.payload["request"]
                )
                if event.payload["query_ref"] != expected_ref:
                    raise ReplayValidationError("query fingerprint does not match request")
                if event.payload["budget_before"] != budget_remaining:
                    raise ReplayValidationError(
                        "tool request budget does not match transcript state"
                    )
                pending_request = event.payload
                tool_calls += 1
            elif event.event_type == EventType.TOOL_RESPONSE:
                if pending_request is None or pending_response is not None:
                    raise ReplayValidationError("tool response has no matching request")
                pending_response = ToolResponse.model_validate(event.payload)
                if pending_response.query_ref != pending_request["query_ref"]:
                    raise ReplayValidationError("tool response query fingerprint differs")
                if pending_response.tool_name != pending_request["tool_name"]:
                    raise ReplayValidationError("tool response name differs from request")
            elif event.event_type == EventType.QUERY_COST:
                if pending_request is None or pending_response is None:
                    raise ReplayValidationError("query cost has no matching response")
                cost = pending_response.cost
                expected = {"query_ref": pending_response.query_ref, **cost.model_dump(mode="json")}
                if event.payload != expected:
                    raise ReplayValidationError("query cost event differs from tool response")
                if budget_remaining != cost.budget_before:
                    raise ReplayValidationError("query budget chain is broken")
                if cost.budget_after != cost.budget_before - cost.units:
                    raise ReplayValidationError("query cost arithmetic is invalid")
                budget_remaining = cost.budget_after
                for observation in pending_response.observations:
                    record = IssuedObservation(
                        observation, pending_response.query_ref,
                        not pending_response.truncated and pending_response.status.value == "success",
                    )
                    # Changed issued facts are rejected by the live runner, not admitted here.
                    observations.setdefault(observation.observation_id, record)
                pending_request = None
                pending_response = None
            elif event.event_type == EventType.TOOL_EXCEPTION:
                if pending_request is not None and pending_response is None:
                    pending_request = None
            elif event.event_type == EventType.ANSWER_FROZEN:
                if status != EpisodeStatus.ACTIVE or pending_request is not None:
                    raise ReplayValidationError("answer freeze is out of order")
                answer = AnswerArtifact.model_validate(event.payload)
                if not answer.verify_digest():
                    raise ReplayValidationError("answer artifact digest is invalid")
                if answer.transcript_digest != event.previous_digest:
                    raise ReplayValidationError("answer is not bound to the inference transcript")
                if answer.authoritative_decision.status == DecisionStatus.DISCOVERED_VERIFIED:
                    if verification_report is None or tuple(proposals.values()) != answer.policy_proposals:
                        raise ReplayValidationError("dynamic result lacks its verification record")
                    final_reports = provider.verification_reports(
                        answer.policy_proposals, tuple(observations.values()),
                    )
                    if verification_report != {k: v.model_dump(mode="json") for k, v in final_reports.items()}:
                        raise ReplayValidationError("dynamic verification record is stale")
                    expected = provider.decide(answer.policy_proposals, tuple(observations.values()),
                                               answer.termination_reason)
                    if expected != answer.authoritative_decision:
                        raise ReplayValidationError("dynamic root cause does not match Core replay")
                status = EpisodeStatus.ANSWER_FROZEN
            elif event.event_type == EventType.EVALUATION_STARTED:
                if status != EpisodeStatus.ANSWER_FROZEN or evaluation_started:
                    raise ReplayValidationError("evaluation start is out of order")
                evaluation_started = True
            elif event.event_type == EventType.EVALUATION_COMPLETED:
                if status != EpisodeStatus.ANSWER_FROZEN or not evaluation_started:
                    raise ReplayValidationError("evaluation completion is out of order")
                status = EpisodeStatus.EVALUATED
            elif status in {
                EpisodeStatus.ANSWER_FROZEN,
                EpisodeStatus.EVALUATED,
            } and event.event_type in {
                EventType.POLICY_ACTION,
                EventType.TOOL_REQUEST,
                EventType.TOOL_RESPONSE,
                EventType.QUERY_COST,
                EventType.HYPOTHESIS_SUBMITTED,
                EventType.CANDIDATE_EVIDENCE_UPDATED,
            }:
                raise ReplayValidationError("inference event appears after answer freeze")
        if pending_request is not None:
            raise ReplayValidationError("transcript ends with an incomplete tool call")
        if discovery_started and not discovery_completed:
            raise ReplayValidationError("transcript ends with incomplete discovery")
        if status is None:
            raise ReplayValidationError("transcript has no lifecycle state")
        return ReplayReport(
            valid=True,
            episode_id=episode_id,
            event_count=len(events),
            final_status=status,
            head_digest=events[-1].digest,
            tool_calls=tool_calls,
        )
