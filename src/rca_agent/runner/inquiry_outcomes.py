"""Project transcript-bound inquiry ideas into non-authoritative answer context."""
from __future__ import annotations

from rca_agent.models import CandidateAssessmentPreview, InquiryCandidateOutcome
from rca_agent.runner.transcript import EventType, TranscriptEvent


def inquiry_outcomes(
    events: tuple[TranscriptEvent, ...] | list[TranscriptEvent],
    assessments: tuple[CandidateAssessmentPreview, ...],
) -> tuple[InquiryCandidateOutcome, ...]:
    handoff = None
    inquiry_rounds = []
    dispositions = {}
    frozen: dict[str, str] = {}
    compiled: dict[str, str] = {}
    for event in events:
        if event.event_type == EventType.INQUIRY_ROUND_RECORDED:
            inquiry_rounds.append(event.payload)
        elif event.event_type == EventType.DISCOVERY_STARTED and event.payload.get("inquiry_handoff"):
            from rca_agent.langchain_acquisition import InquiryHandoff

            handoff = InquiryHandoff.model_validate(event.payload["inquiry_handoff"])
        elif event.event_type == EventType.DISCOVERY_COMPLETED:
            dispositions = {
                item["hypothesis_id"]: item
                for item in event.payload.get("inquiry_dispositions", ())
            }
        elif event.event_type == EventType.MECHANISM_DRAFT_FROZEN:
            mechanism = event.payload["mechanism"]
            frozen[mechanism["mechanism_id"]] = mechanism["mechanism"]
        elif (event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED
              and event.payload.get("mechanism_id")):
            compiled[event.payload["mechanism_id"]] = event.payload["compiled"][
                "candidate"
            ]["candidate_id"]
        elif (event.event_type == EventType.CATALOG_CANDIDATE_ALIGNED
              and event.payload.get("mechanism_id")):
            compiled[event.payload["mechanism_id"]] = event.payload["candidate_id"]
        elif event.event_type == EventType.VERIFICATION_PLAN_EXTENDED:
            compiled[event.payload["mechanism_id"]] = event.payload["compiled"][
                "candidate"
            ]["candidate_id"]
    if handoff is None:
        if not inquiry_rounds:
            return ()
        from rca_agent.langchain_acquisition import InquiryHandoff, inquiry_candidate_records

        history = inquiry_candidate_records(inquiry_rounds)
        handoff = InquiryHandoff(
            revision=inquiry_rounds[-1]["round"],
            hypotheses=tuple(item.hypothesis for item in history),
            needs=(), query_links=(), candidate_history=history,
        )
    assessment_by_id = {item.candidate_id: item for item in assessments}
    history = {item.hypothesis.hypothesis_id: item for item in handoff.candidate_history}
    outcomes = []
    for hypothesis in handoff.hypotheses:
        item = dispositions.get(hypothesis.hypothesis_id, {})
        named_mechanism = item.get("formal_mechanism")
        exact = named_mechanism or hypothesis.mechanism
        linked = tuple(sorted(
            mechanism_id for mechanism_id, mechanism in frozen.items()
            if mechanism.strip().casefold() == exact.strip().casefold()
        ))
        candidate_ids = tuple(sorted({
            compiled[mechanism_id] for mechanism_id in linked if mechanism_id in compiled
        }))
        record = history.get(hypothesis.hypothesis_id)
        links = tuple(link for link in handoff.query_links
                      if link.hypothesis_id == hypothesis.hypothesis_id)
        outcomes.append(InquiryCandidateOutcome(
            hypothesis_id=hypothesis.hypothesis_id,
            mechanism=hypothesis.mechanism,
            alternative=hypothesis.alternative,
            first_round=record.first_round if record else handoff.revision,
            last_round=record.last_round if record else handoff.revision,
            mention_count=record.mention_count if record else 1,
            query_refs=tuple(dict.fromkeys(link.query_ref for link in links)),
            observation_refs=tuple(sorted({ref for link in links
                                           for ref in link.observation_refs})),
            discovery_disposition=item.get("disposition", "unreported"),
            discovery_reason=item.get("reason", "Discovery did not report this idea."),
            formal_mechanism_ids=linked,
            compiled_candidate_ids=candidate_ids,
            core_evidence_statuses=tuple(
                assessment_by_id[candidate_id].evidence_status.value
                for candidate_id in candidate_ids if candidate_id in assessment_by_id
            ),
        ))
    return tuple(outcomes)
