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
from rca_agent.runner.inquiry_outcomes import inquiry_outcomes
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
        completed_queries: dict[str, ToolResponse] = {}
        proposals = {}
        verification_report = None
        discovery_refs = ()
        discovery_queries = ()
        inquiry_handoff_ids: tuple[str, ...] = ()
        inquiry_rounds: list[dict] = []
        frozen_mechanisms = {}
        compiled_plans = {}
        extension_started = None
        extension_count = 0
        expected_extension_proposal = None
        planner_retry_seen = False
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
            if (expected_extension_proposal is not None
                    and event.event_type != EventType.HYPOTHESIS_SUBMITTED):
                raise ReplayValidationError("extended plan has no immediate candidate proposal")

            if expected_sequence == 1:
                if event.event_type != EventType.EPISODE_CREATED:
                    raise ReplayValidationError("first event must create the episode")
                status = EpisodeStatus.CREATED
                provider = CoreDecisionProvider(
                    frozenset(event.payload.get("excluded_catalog_candidate_ids", ()))
                )
                continue
            if event.event_type == EventType.ENVIRONMENT_RESET:
                if status != EpisodeStatus.CREATED:
                    raise ReplayValidationError("environment reset is out of order")
                status = EpisodeStatus.ACTIVE
                budget_remaining = int(event.payload["bootstrap"]["query_budget"]["remaining"])
            elif event.event_type == EventType.INQUIRY_ROUND_RECORDED:
                from rca_agent.langchain_acquisition import InquiryDraft

                if (status != EpisodeStatus.ACTIVE or discovery_started
                        or not isinstance(event.payload.get("round"), int)
                        or not 1 <= event.payload["round"] <= 5
                        or (inquiry_rounds
                            and event.payload["round"] <= inquiry_rounds[-1]["round"])
                        or len(inquiry_rounds) >= 5):
                    raise ReplayValidationError("inquiry round is out of order")
                InquiryDraft.model_validate(event.payload["draft"])
                inquiry_rounds.append(event.payload)
            elif event.event_type == EventType.DISCOVERY_STARTED:
                if status != EpisodeStatus.ACTIVE or discovery_started:
                    raise ReplayValidationError("discovery start is out of order")
                if event.payload.get("inquiry_handoff") is not None:
                    from rca_agent.langchain_acquisition import InquiryHandoff

                    handoff = InquiryHandoff.model_validate(event.payload["inquiry_handoff"])
                    if inquiry_rounds:
                        from rca_agent.langchain_acquisition import inquiry_candidate_records

                        if handoff.candidate_history != inquiry_candidate_records(inquiry_rounds):
                            raise ReplayValidationError("inquiry handoff omitted a recorded idea")
                    inquiry_handoff_ids = tuple(sorted(
                        item.hypothesis_id for item in handoff.hypotheses
                    ))
                    for link in handoff.query_links:
                        response = completed_queries.get(link.query_ref)
                        if (response is None or response.status.value != link.status
                                or set(link.observation_refs) != {
                                    item.observation_id for item in response.observations
                                }):
                            raise ReplayValidationError("inquiry handoff query result mismatch")
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
                mechanism_id = event.payload.get("mechanism_id")
                if mechanism_id is not None:
                    frozen = frozen_mechanisms.get(mechanism_id)
                    if frozen is None or event.payload.get("mechanism_digest") != frozen.digest:
                        raise ReplayValidationError("dynamic candidate changed frozen mechanism")
            elif event.event_type == EventType.MECHANISM_DRAFT_GENERATED:
                from rca_agent.core.dynamic import MechanismDraft

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism draft is out of order")
                MechanismDraft.model_validate(event.payload["draft"])
            elif event.event_type == EventType.MECHANISM_DRAFT_REJECTED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism rejection is out of order")
            elif event.event_type == EventType.MECHANISM_DRAFT_FROZEN:
                from rca_agent.core.dynamic import FrozenMechanism, MechanismDraft, freeze_mechanism

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism freeze is out of order")
                frozen = FrozenMechanism.model_validate(event.payload["mechanism"])
                draft = MechanismDraft.model_validate(
                    frozen.model_dump(exclude={"mechanism_id", "digest"})
                )
                if freeze_mechanism(draft).model_dump() != frozen.model_dump():
                    raise ReplayValidationError("frozen mechanism digest is invalid")
                if frozen.mechanism_id in frozen_mechanisms:
                    raise ReplayValidationError("mechanism frozen twice")
                frozen_mechanisms[frozen.mechanism_id] = frozen
            elif event.event_type == EventType.VERIFICATION_PLAN_RETRIED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("verification retry is out of order")
                if planner_retry_seen:
                    raise ReplayValidationError("verification planner retried more than once")
                if not set(event.payload["mechanism_ids"]) <= frozen_mechanisms.keys():
                    raise ReplayValidationError("retry references unknown frozen mechanism")
                planner_retry_seen = True
            elif event.event_type in {
                EventType.VERIFICATION_PLAN_GENERATED,
                EventType.VERIFICATION_PLAN_COMPILED,
            }:
                from rca_agent.core.dynamic import VerificationPlanDraft

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("verification plan is out of order")
                plan = VerificationPlanDraft.model_validate(event.payload["plan"])
                if plan.mechanism_id not in frozen_mechanisms:
                    raise ReplayValidationError("plan references unknown frozen mechanism")
                if event.payload["attempt"] not in {1, 2}:
                    raise ReplayValidationError("invalid verification-plan attempt")
                if event.payload["attempt"] == 2 and not planner_retry_seen:
                    raise ReplayValidationError("verification retry plan has no retry event")
                if event.event_type == EventType.VERIFICATION_PLAN_COMPILED:
                    frozen = frozen_mechanisms[plan.mechanism_id]
                    if event.payload["mechanism_digest"] != frozen.digest:
                        raise ReplayValidationError("compiled plan changed frozen mechanism")
                    compiled_plans[plan.mechanism_id] = plan
            elif event.event_type == EventType.VERIFICATION_PLAN_EXTENSION_STARTED:
                from rca_agent.core.dynamic import VerificationPlanDraft

                if (status != EpisodeStatus.ACTIVE or not discovery_completed
                        or extension_started is not None or extension_count >= 1):
                    raise ReplayValidationError("plan extension start is out of order")
                mechanism_id = event.payload["mechanism_id"]
                previous_plan = VerificationPlanDraft.model_validate(
                    event.payload["previous_plan"]
                )
                if (mechanism_id not in frozen_mechanisms
                        or previous_plan != compiled_plans.get(mechanism_id)
                        or provider.dynamic_compiled(event.payload["old_candidate_id"]) is None):
                    raise ReplayValidationError("plan extension has no frozen candidate")
                extension_started = (mechanism_id, event.payload["old_candidate_id"])
                extension_count += 1
            elif event.event_type == EventType.VERIFICATION_PLAN_EXTENSION_REJECTED:
                if (status != EpisodeStatus.ACTIVE or extension_started is None
                        or event.payload["mechanism_id"] != extension_started[0]):
                    raise ReplayValidationError("plan extension rejection is out of order")
                extension_started = None
            elif event.event_type == EventType.VERIFICATION_PLAN_EXTENDED:
                from rca_agent.core.dynamic import CompiledCandidate, VerificationPlanDraft
                from rca_agent.runner.plan_extension import added_predictions

                if (status != EpisodeStatus.ACTIVE or extension_started is None
                        or (event.payload["mechanism_id"], event.payload["old_candidate_id"])
                        != extension_started):
                    raise ReplayValidationError("plan extension is out of order")
                mechanism_id = event.payload["mechanism_id"]
                previous_plan = VerificationPlanDraft.model_validate(
                    event.payload["previous_plan"]
                )
                plan = VerificationPlanDraft.model_validate(event.payload["plan"])
                compiled = CompiledCandidate.model_validate(event.payload["compiled"])
                if (previous_plan != compiled_plans[mechanism_id]
                        or event.payload["mechanism_digest"] != frozen_mechanisms[mechanism_id].digest
                        or compiled.candidate.fault_type != frozen_mechanisms[mechanism_id].mechanism
                        or compiled.candidate.causal_entity
                        != frozen_mechanisms[mechanism_id].causal_entity_ref
                        or compiled.source_hypothesis_id != event.payload["new_hypothesis_id"]
                        or event.payload["new_hypothesis_id"]
                        != "h-discovered-" + compiled.version[:20]
                        or compiled.candidate.candidate_id != "dyn_" + compiled.version[:24]
                        or event.payload["old_hypothesis_id"] not in proposals
                        or proposals[event.payload["old_hypothesis_id"]].candidate_id
                        != event.payload["old_candidate_id"]):
                    raise ReplayValidationError("plan extension changed frozen identity")
                try:
                    added_predictions(previous_plan, plan)
                    old_compiled = provider.dynamic_compiled(event.payload["old_candidate_id"])
                    if old_compiled is None:
                        raise ValueError("old candidate is absent")
                    if (not {item.model_dump_json() for item in old_compiled.predictions}
                            <= {item.model_dump_json() for item in compiled.predictions}
                            or old_compiled.alternatives != compiled.alternatives
                            or not {item.model_dump_json() for item in old_compiled.rules}
                            <= {item.model_dump_json() for item in compiled.rules}
                            or not {item.model_dump_json() for item in old_compiled.signals}
                            <= {item.model_dump_json() for item in compiled.signals}
                            or old_compiled.discovery_query_refs != compiled.discovery_query_refs
                            or old_compiled.discovery_observation_refs
                            != compiled.discovery_observation_refs
                            or old_compiled.discovery_fact_digests
                            != compiled.discovery_fact_digests):
                        raise ValueError("compiled extension dropped old evidence contract")
                    old_queries = {
                        query_fingerprint(item.query.tool_name, item.query.request)
                        for item in old_compiled.signals
                    }
                    expected_new_queries = {
                        query_fingerprint(item.query.tool_name, item.query.request)
                        for item in compiled.signals
                    } - old_queries - completed_queries.keys()
                    if set(event.payload["new_query_refs"]) != expected_new_queries:
                        raise ValueError("extension query list differs from compiled plan")
                    provider.supersede_dynamic(event.payload["old_candidate_id"], compiled)
                except ValueError as exc:
                    raise ReplayValidationError("invalid additive plan extension") from exc
                new_query_refs = set(event.payload["new_query_refs"])
                if not new_query_refs or new_query_refs & completed_queries.keys():
                    raise ReplayValidationError("plan extension has no new query")
                compiled_plans[mechanism_id] = plan
                extension_started = None
                expected_extension_proposal = (
                    event.payload["new_hypothesis_id"],
                    event.payload["old_hypothesis_id"],
                    compiled.candidate.candidate_id,
                )
            elif event.event_type == EventType.VERIFICATION_PLAN_REJECTED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("verification rejection is out of order")
                if event.payload["mechanism_id"] not in frozen_mechanisms:
                    raise ReplayValidationError("rejected plan references unknown mechanism")
                if event.payload["attempt"] == 2 and not planner_retry_seen:
                    raise ReplayValidationError("retry rejection has no retry event")
            elif event.event_type == EventType.DISCOVERY_DRAFT_EVALUATED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("discovery draft evaluation is out of order")
                if event.payload.get("final_status") not in {"accepted", "rejected"}:
                    raise ReplayValidationError("invalid discovery draft evaluation status")
            elif event.event_type == EventType.DISCOVERY_PAYLOAD_REJECTED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("discovery payload rejection is out of order")
            elif event.event_type == EventType.DISCOVERY_COMPLETED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("discovery completion is out of order")
                if event.payload["registered"] != len(dynamic_ids):
                    raise ReplayValidationError("discovery registration count mismatch")
                if event.payload.get("status") == "completed" and inquiry_handoff_ids:
                    dispositions = event.payload.get("inquiry_dispositions", ())
                    if (tuple(sorted(item["hypothesis_id"] for item in dispositions))
                            != inquiry_handoff_ids
                            or any(item["disposition"] not in {
                                "kept", "revised", "discarded", "deferred", "unreported"
                            } for item in dispositions)):
                        raise ReplayValidationError("inquiry disposition mismatch")
                discovery_completed = True
            elif event.event_type == EventType.CATALOG_CANDIDATE_ALIGNED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("catalog alignment is out of order")
            elif event.event_type == EventType.HYPOTHESIS_SUBMITTED:
                if status != EpisodeStatus.ACTIVE:
                    raise ReplayValidationError("hypothesis submitted outside active episode")
                proposal = HypothesisProposal.model_validate(event.payload)
                if expected_extension_proposal is not None:
                    new_id, old_id, candidate_id = expected_extension_proposal
                    if (proposal.hypothesis_id != new_id
                            or proposal.candidate_id != candidate_id
                            or not set(proposals[old_id].observation_refs)
                            <= set(proposal.observation_refs)):
                        raise ReplayValidationError("extended plan proposal is not bound to old evidence")
                    expected_extension_proposal = None
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
                completed_queries[pending_response.query_ref] = pending_response
                pending_request = None
                pending_response = None
            elif event.event_type == EventType.TOOL_EXCEPTION:
                if pending_request is not None and pending_response is None:
                    pending_request = None
            elif event.event_type == EventType.ANSWER_FROZEN:
                if (status != EpisodeStatus.ACTIVE or pending_request is not None
                        or extension_started is not None
                        or expected_extension_proposal is not None):
                    raise ReplayValidationError("answer freeze is out of order")
                answer = AnswerArtifact.model_validate(event.payload)
                if not answer.verify_digest():
                    raise ReplayValidationError("answer artifact digest is invalid")
                if answer.transcript_digest != event.previous_digest:
                    raise ReplayValidationError("answer is not bound to the inference transcript")
                if "inquiry_candidates" in event.payload:
                    expected_outcomes = inquiry_outcomes(
                        events[:expected_sequence - 1],
                        provider.preview(tuple(proposals.values()), tuple(observations.values()))
                        .candidate_assessments,
                    )
                    if answer.inquiry_candidates != expected_outcomes:
                        raise ReplayValidationError("inquiry outcomes differ from transcript and Core")
                if answer.authoritative_decision.status in {
                    DecisionStatus.DISCOVERED_VERIFIED,
                    DecisionStatus.MULTIPLE_SUPPORTED,
                }:
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
