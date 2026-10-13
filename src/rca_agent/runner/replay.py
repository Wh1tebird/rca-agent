from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.investigation import (
    InvestigationWorkspaceReducer,
    MechanismCoveragePolicy,
    MechanismCoverageRecord,
    build_mechanism_coverage_input,
    project_formal_coverage_candidates,
)
from rca_agent.models import (
    AnswerArtifact,
    DecisionStatus,
    EpisodeStatus,
    HypothesisProposal,
    ResetResponse,
    ToolResponse,
)
from rca_agent.runner.inquiry_outcomes import inquiry_outcomes
from rca_agent.runner.transcript import (
    GENESIS_DIGEST,
    EventType,
    TranscriptEvent,
    load_jsonl_events,
)
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest


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
        workspace_reducer = InvestigationWorkspaceReducer(episode_id)
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
        generated_mechanisms = []
        engineering_revision = 1
        inquiry_contract = None
        inquiry_contract_recorded = False
        handoff_contract_version = 1
        semantic_handoff = None
        handoff_initial = None
        handoff_validation = None
        handoff_expected_drafts = None
        handoff_generated = []
        bound_records = {}
        frozen_mechanisms = {}
        compiled_plans = {}
        generated_plans = {}
        extension_started = None
        extension_count = 0
        expected_extension_proposal = None
        planner_retry_seen = False
        mechanism_completion: dict | None = None
        mechanism_completion_count = 0
        completion_expected_drafts: set[str] | None = None
        completion_generated_drafts: set[str] = set()
        formalization_started = False
        formalization_contract_version = 1
        formalization_tracks: set[str] = set()
        formalization_retry_payload = None
        discovery_format_repair_context = None
        discovery_format_repair_count = 0
        discovery_format_repair_drafts: tuple[str, ...] = ()
        discovery_format_repair_seen = 0
        mechanism_draft_generated_count = 0
        bootstrap: ResetResponse | None = None
        coverage_mode = "off"
        coverage_recorded = False
        coverage_requires_answer = False
        active_offers_by_round: dict[int, tuple[str, ...]] = {}
        latest_query_registry = None
        expected_active_query: str | None = None
        for expected_sequence, event in enumerate(events, start=1):
            if event.episode_id != episode_id:
                raise ReplayValidationError("episode ID changed inside transcript")
            if event.sequence != expected_sequence:
                raise ReplayValidationError("event sequence is not contiguous")
            if event.previous_digest != previous:
                raise ReplayValidationError("previous-event digest chain is broken")
            if not event.verify_digest():
                raise ReplayValidationError("event payload digest is invalid")
            try:
                workspace_reducer.apply(event)
            except (ValueError, TypeError) as exc:
                raise ReplayValidationError("investigation workspace event is invalid") from exc
            previous = event.digest
            if (expected_extension_proposal is not None
                    and event.event_type != EventType.HYPOTHESIS_SUBMITTED):
                raise ReplayValidationError("extended plan has no immediate candidate proposal")
            if coverage_requires_answer and event.event_type != EventType.ANSWER_FROZEN:
                raise ReplayValidationError("coverage shadow record must immediately precede answer freeze")

            if expected_sequence == 1:
                if event.event_type != EventType.EPISODE_CREATED:
                    raise ReplayValidationError("first event must create the episode")
                status = EpisodeStatus.CREATED
                from rca_agent.investigation.contract_enforcement import validate_revision
                engineering_revision = validate_revision(event.payload.get("engineering_contract_revision", 1))
                from rca_agent.investigation.contract_enforcement import inquiry_host_contract
                inquiry_contract = inquiry_host_contract(event.payload, historical=True)
                inquiry_contract_recorded = "inquiry_contract_version" in event.payload
                provider = CoreDecisionProvider(
                    frozenset(event.payload.get("excluded_catalog_candidate_ids", ()))
                )
                coverage_mode = event.payload.get("mechanism_coverage_mode", "off")
                if coverage_mode not in {"off", "shadow"}:
                    raise ReplayValidationError("mechanism coverage mode is invalid")
                continue
            if event.event_type == EventType.ENVIRONMENT_RESET:
                if status != EpisodeStatus.CREATED:
                    raise ReplayValidationError("environment reset is out of order")
                status = EpisodeStatus.ACTIVE
                bootstrap = ResetResponse.model_validate(event.payload["bootstrap"])
                budget_remaining = bootstrap.query_budget.remaining
            elif event.event_type == EventType.STARTUP_ACTION_SELECTED:
                if status != EpisodeStatus.ACTIVE or pending_request is not None:
                    raise ReplayValidationError("startup host decision is out of order")
            elif event.event_type == EventType.INQUIRY_ATTEMPT_RECORDED:
                from rca_agent.inquiry_policy import validate_inquiry_attempt

                if status != EpisodeStatus.ACTIVE or discovery_started:
                    raise ReplayValidationError("inquiry attempt is out of order")
                try:
                    validate_inquiry_attempt(event.payload)
                except ValueError as exc:
                    raise ReplayValidationError("invalid inquiry attempt diagnostic") from exc
            elif event.event_type == EventType.INQUIRY_ROUND_RECORDED:
                from rca_agent.langchain_acquisition import parse_inquiry_draft

                if (status != EpisodeStatus.ACTIVE or discovery_started
                        or not isinstance(event.payload.get("round"), int)
                        or not 1 <= event.payload["round"] <= 5
                        or (inquiry_rounds
                            and event.payload["round"] <= inquiry_rounds[-1]["round"])
                        or len(inquiry_rounds) >= 5):
                    raise ReplayValidationError("inquiry round is out of order")
                try:
                    draft = parse_inquiry_draft(event.payload["draft"], live_version=inquiry_contract,
                        strict=engineering_revision == 2, recorded=inquiry_contract is not None,
                        historical=engineering_revision == 1 and not inquiry_contract_recorded)
                    if inquiry_contract is None:
                        inquiry_contract = 2 if draft.contract_version == 2 else 1
                except ValueError as exc:
                    raise ReplayValidationError("unsupported or invalid Inquiry contract") from exc
                if draft.contract_version == 2:
                    from rca_agent.investigation.inquiry_contract import validate_bindings
                    from rca_agent.investigation.query_strategy import issued_entities
                    from rca_agent.langchain_acquisition import query_options
                    facts = tuple(r.observation for r in observations.values())
                    menu = query_options({"bootstrap": bootstrap.model_dump(mode="json")},
                                         issued_entities(bootstrap, facts))
                    try:
                        validate_bindings(draft, event.payload.get("selected_actions"),
                                          event.payload["round"], menu,
                                          max_bindings=(16 if workspace_reducer.workspace._inquiry_loop_state
                                                        is not None else 2))
                    except ValueError as exc:
                        raise ReplayValidationError("invalid Inquiry contract bindings") from exc
                for selected in event.payload.get("selected_actions", ()):
                    from rca_agent.models import CallToolAction

                    action = CallToolAction.model_validate(selected["action"])
                    if selected.get("query_ref") != "aq-" + sha256_digest(
                        action.model_dump(mode="json")
                    )[:20]:
                        raise ReplayValidationError("active selected action binding is invalid")
                if event.payload.get("selected_actions"):
                    active_offers_by_round[event.payload["round"]] = tuple(sorted(
                        query_fingerprint(
                            CallToolAction.model_validate(item["action"]).tool_name,
                            CallToolAction.model_validate(item["action"]).request,
                        )
                        for item in event.payload["selected_actions"]
                    ))
                inquiry_rounds.append({**event.payload, "draft": draft.model_dump(mode="json")})
            elif event.event_type == EventType.DISCOVERY_STARTED:
                if status != EpisodeStatus.ACTIVE or discovery_started:
                    raise ReplayValidationError("discovery start is out of order")
                if event.payload.get("inquiry_handoff") is not None:
                    from rca_agent.langchain_acquisition import InquiryHandoff

                    handoff = InquiryHandoff.model_validate(event.payload["inquiry_handoff"])
                    semantic_handoff = handoff.model_dump(mode="json")
                    from rca_agent.investigation.contract_enforcement import (
                        execution_binding_responses,
                    )
                    from rca_agent.investigation.inquiry_contract import validate_handoff_links
                    try:
                        validate_handoff_links(handoff, inquiry_rounds, completed_queries,
                            binding_responses=(execution_binding_responses(events[:event.sequence - 1])
                                               if engineering_revision == 2 else None))
                    except ValueError as exc:
                        raise ReplayValidationError("invalid handoff measurement binding") from exc
                    if workspace_reducer.workspace._inquiry_loop_state is not None:
                        from rca_agent.investigation.inquiry_loop import loop_handoff

                        expected_handoff = loop_handoff(
                            workspace_reducer.workspace._inquiry_loop_state,
                            inquiry_rounds, events[:event.sequence - 1],
                        )
                        if semantic_handoff != expected_handoff:
                            raise ReplayValidationError("Inquiry loop handoff differs from reviewed state")
                    elif inquiry_rounds:
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
                handoff_contract_version = event.payload.get("handoff_contract_version", 1)
                if handoff_contract_version not in {1, 2, 3, 4, 5, 6, 7} or (
                    handoff_contract_version in {2, 3, 4, 5, 6, 7} and not inquiry_handoff_ids
                ):
                    raise ReplayValidationError("invalid semantic handoff version")
                discovery_refs = tuple(sorted(observations))
                discovery_queries = tuple(sorted({r.query_ref for r in observations.values()}))
            elif event.event_type == EventType.BOUND_CANDIDATE_FORMALIZATION_RECORDED:
                from rca_agent.investigation.bound_formalization import (
                    bind_candidate,
                    candidate_binding,
                )
                from rca_agent.runner.episode import _find_entity_refs

                if (handoff_contract_version not in {5, 6, 7} or not discovery_started or discovery_completed
                        or handoff_validation is not None or mechanism_draft_generated_count):
                    raise ReplayValidationError("bound candidate formalization is out of order")
                binding = event.payload["binding"]
                identifier = binding["hypothesis_id"]
                candidates = semantic_handoff["hypotheses"]
                candidate = next((c for c in candidates if c["hypothesis_id"] == identifier), None)
                if candidate is None or binding != candidate_binding(candidate, min(proposals), version=handoff_contract_version, handoff=semantic_handoff, revision=engineering_revision):
                    raise ReplayValidationError("bound candidate identity changed")
                previous_record = bound_records.get(identifier)
                attempt = event.payload["attempt"]
                if previous_record is not None:
                    if (attempt != 2 or previous_record["attempt"] != 1
                            or previous_record["response"] is not None):
                        raise ReplayValidationError("bound candidate retry is invalid")
                else:
                    if (len(bound_records) >= len(candidates)
                            or candidates[len(bound_records)]["hypothesis_id"] != identifier
                            or attempt != (0 if len(bound_records) >= 4 else 1)):
                        raise ReplayValidationError("bound candidate order/capacity is invalid")
                    if bound_records:
                        last = next(reversed(bound_records.values()))
                        if last["response"] is None and last["attempt"] == 1:
                            raise ReplayValidationError("bound candidate omitted retry")
                response = event.payload.get("response")
                if response is not None:
                    entities = {bootstrap.alert_entity.entity_ref} | set().union(*(
                        _find_entity_refs(r.model_dump(mode="json")) for r in completed_queries.values()
                    ))
                    bind_candidate(binding, response, entities, check_class=handoff_contract_version >= 6)
                    if event.payload["reason_code"] != response["reason_code"] or attempt == 0:
                        raise ReplayValidationError("bound candidate response status differs")
                elif attempt == 0 and event.payload["reason_code"] != "FORMALIZATION_CAPACITY":
                    raise ReplayValidationError("invalid bound capacity record")
                bound_records[identifier] = event.payload
            elif event.event_type == EventType.DISCOVERY_HANDOFF_REPAIR_REQUESTED:
                from rca_agent.core.dynamic import MechanismDiscoveryPayload
                from rca_agent.investigation.discovery_handoff import (
                    diagnostic_errors,
                    pending_formalizations,
                    validate_handoff,
                )

                if (not discovery_started or discovery_completed or handoff_contract_version not in {2, 3, 4}
                        or handoff_initial is not None or handoff_validation is not None
                        or mechanism_draft_generated_count or event.payload.get("attempt") != 1
                        or event.payload.get("version") != handoff_contract_version):
                    raise ReplayValidationError("semantic handoff retry is out of order")
                handoff_initial = MechanismDiscoveryPayload.model_validate(event.payload["initial_payload"])
                errors = validate_handoff(handoff_initial, semantic_handoff, [
                    r.observation.model_dump(mode="json") for r in observations.values()
                ])
                if not errors or event.payload["errors"] != diagnostic_errors(errors):
                    raise ReplayValidationError("semantic handoff retry has invalid trigger")
                if (handoff_contract_version == 4 and event.payload.get("pending_formalizations")
                        != pending_formalizations(handoff_initial, semantic_handoff)):
                    raise ReplayValidationError("semantic handoff missing-draft diagnostic differs")
            elif event.event_type == EventType.DISCOVERY_HANDOFF_VALIDATED:
                from rca_agent.core.dynamic import MechanismDiscoveryPayload
                from rca_agent.investigation.discovery_handoff import (
                    diagnostic_errors,
                    merge_partial_handoff,
                    pending_formalizations,
                    quarantine_handoff,
                    repair_errors,
                    validate_handoff,
                )

                if handoff_contract_version in {5, 6, 7}:
                    from rca_agent.investigation.bound_formalization import (
                        admit_novel,
                        bound_payload,
                    )
                    from rca_agent.runner.episode import _find_entity_refs

                    if (not discovery_started or discovery_completed or handoff_validation is not None
                            or event.payload.get("version") != handoff_contract_version or mechanism_draft_generated_count
                            or set(bound_records) != set(inquiry_handoff_ids)
                            or any(r["response"] is None and r["attempt"] == 1
                                   for r in bound_records.values())):
                        raise ReplayValidationError("incomplete bound formalization")
                    entities = {bootstrap.alert_entity.entity_ref} | set().union(*(
                        _find_entity_refs(r.model_dump(mode="json")) for r in completed_queries.values()
                    ))
                    base = bound_payload(semantic_handoff, bound_records, entities,
                                         check_class=handoff_contract_version >= 6)
                    novel_raw = event.payload.get("new_discovery_payload")
                    novel = MechanismDiscoveryPayload.model_validate(novel_raw) if novel_raw is not None else None
                    facts = [r.observation.model_dump(mode="json") for r in observations.values()]
                    expected = admit_novel(base, novel, semantic_handoff, facts, set(proposals), entities)
                    if (event.payload["initial_payload"] != base.model_dump(mode="json")
                            or event.payload["payload"] != expected.model_dump(mode="json")
                            or event.payload["outcome"] != "bound_candidates"
                            or event.payload["pending_formalizations"] != pending_formalizations(expected, semantic_handoff)
                            or validate_handoff(expected, semantic_handoff, facts)):
                        raise ReplayValidationError("bound formalization aggregate differs")
                    handoff_validation = expected
                    handoff_expected_drafts = [m.model_dump_json() for m in expected.mechanisms]
                    continue

                if (not discovery_started or discovery_completed or handoff_contract_version not in {2, 3, 4}
                        or handoff_validation is not None or mechanism_draft_generated_count
                        or event.payload.get("version") != handoff_contract_version):
                    raise ReplayValidationError("semantic handoff validation is out of order")
                initial = MechanismDiscoveryPayload.model_validate(event.payload["initial_payload"])
                facts = [r.observation.model_dump(mode="json") for r in observations.values()]
                errors = validate_handoff(initial, semantic_handoff, facts)
                if (event.payload["initial_errors"] != diagnostic_errors(errors)
                        or (errors and initial != handoff_initial)
                        or (not errors and handoff_initial is not None)):
                    raise ReplayValidationError("semantic handoff initial diagnostic differs")
                retry_raw = event.payload.get("retry_payload")
                retry = MechanismDiscoveryPayload.model_validate(retry_raw) if retry_raw is not None else None
                retry_issues = repair_errors(
                    initial, retry, semantic_handoff, facts,
                    recover_exact=handoff_contract_version in {3, 4},
                ) if retry else ()
                if event.payload["retry_errors"] != diagnostic_errors(retry_issues):
                    raise ReplayValidationError("semantic handoff retry diagnostic differs")
                if not errors:
                    if retry is not None or event.payload.get("retry_failure") is not None:
                        raise ReplayValidationError("unnecessary semantic handoff retry")
                    expected, outcome = initial, "accepted"
                elif retry is not None and not retry_issues:
                    if event.payload.get("retry_failure") is not None:
                        raise ReplayValidationError("successful handoff retry marked failed")
                    expected, outcome = retry, "repaired"
                else:
                    expected = quarantine_handoff(
                        initial, semantic_handoff, facts, recover_exact=handoff_contract_version in {3, 4},
                    )
                    outcome = "quarantined"
                    if handoff_contract_version == 4:
                        merged = merge_partial_handoff(initial, retry, semantic_handoff, facts)
                        outcome = "partially_repaired" if merged != expected else "quarantined"
                        expected = merged
                if event.payload["payload"] != expected.model_dump(mode="json") or (
                    event.payload["outcome"] != outcome or validate_handoff(expected, semantic_handoff, facts)
                ):
                    raise ReplayValidationError("semantic handoff accepted payload differs")
                if (handoff_contract_version == 4 and event.payload.get("pending_formalizations")
                        != pending_formalizations(expected, semantic_handoff)):
                    raise ReplayValidationError("semantic handoff final missing-draft diagnostic differs")
                handoff_validation = expected
                handoff_expected_drafts = [m.model_dump_json() for m in expected.mechanisms]
                if completion_expected_drafts is not None:
                    completion_expected_drafts = set(handoff_expected_drafts)
                if discovery_format_repair_context is not None or discovery_format_repair_drafts:
                    discovery_format_repair_drafts = list(handoff_expected_drafts)
            elif event.event_type == EventType.DISCOVERY_FORMAT_REPAIR_REQUESTED:
                from rca_agent.investigation.discovery_format import (
                    DiscoveryFormatRepairContext,
                    repair_context_from_raw,
                )

                if (
                    status != EpisodeStatus.ACTIVE or not discovery_started
                    or discovery_completed or discovery_format_repair_count
                    or mechanism_draft_generated_count
                    or frozen_mechanisms or event.payload.get("attempt") != 1
                ):
                    raise ReplayValidationError("Discovery format repair is out of order")
                context = DiscoveryFormatRepairContext.model_validate(event.payload["context"])
                rebuilt = repair_context_from_raw(
                    context.raw_payload,
                    proposal_ids=set(proposals),
                    issued_entities={item.causal_entity_ref for item in context.anchors},
                )
                if (
                    rebuilt != context
                    or event.payload.get("context_digest") != sha256_digest(
                        context.model_dump(mode="json")
                    )
                ):
                    raise ReplayValidationError("Discovery repair source is invalid")
                discovery_format_repair_context = context
                discovery_format_repair_count = 1
            elif event.event_type == EventType.DISCOVERY_FORMAT_REPAIR_COMPLETED:
                from rca_agent.core.dynamic import MechanismDiscoveryPayload
                from rca_agent.investigation.discovery_format import (
                    validate_discovery_format_repair,
                )

                if discovery_format_repair_context is None or discovery_completed:
                    raise ReplayValidationError("Discovery repair completion is out of order")
                payload = MechanismDiscoveryPayload.model_validate(event.payload["payload"])
                try:
                    validate_discovery_format_repair(payload, discovery_format_repair_context)
                except ValueError as exc:
                    raise ReplayValidationError("Discovery repair changed causal semantics") from exc
                if event.payload.get("payload_digest") != sha256_digest(
                    payload.model_dump(mode="json")
                ):
                    raise ReplayValidationError("Discovery repair payload digest is invalid")
                discovery_format_repair_drafts = tuple(
                    item.model_dump_json() for item in payload.mechanisms
                )
                discovery_format_repair_context = None
            elif event.event_type == EventType.DISCOVERY_FORMAT_REPAIR_REJECTED:
                if (
                    discovery_format_repair_context is None or discovery_completed
                    or not isinstance(event.payload.get("reason_code"), str)
                ):
                    raise ReplayValidationError("Discovery repair rejection is out of order")
                discovery_format_repair_context = None
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
            elif event.event_type == EventType.MECHANISM_FORMALIZATION_STARTED:
                if (
                    status != EpisodeStatus.ACTIVE
                    or not discovery_started
                    or discovery_completed
                    or formalization_started
                ):
                    raise ReplayValidationError("mechanism formalization start is out of order")
                tracks = event.payload.get("candidate_tracks", ())
                digests = event.payload.get("semantic_digests", ())
                candidates = {
                    item.candidate_track_id: item
                    for item in workspace_reducer.workspace.candidates
                }
                if (
                    not tracks
                    or len(tracks) != len(digests)
                    or any(
                        track not in candidates
                        or candidates[track].semantic_digest != digest
                        for track, digest in zip(tracks, digests, strict=True)
                    )
                ):
                    raise ReplayValidationError("formalization source candidates are invalid")
                formalization_started = True
                formalization_tracks = set(tracks)
                formalization_contract_version = event.payload.get("contract_version", 1)
                if formalization_contract_version not in (1, 2):
                    raise ReplayValidationError("unknown formalization contract version")
                formalization_retry_payload = None
            elif event.event_type == EventType.MECHANISM_FORMALIZATION_RETRIED:
                from rca_agent.investigation import MechanismFormalizationPayload

                if (
                    not formalization_started
                    or discovery_completed
                    or formalization_contract_version != 2
                    or formalization_retry_payload is not None
                    or event.payload.get("attempt") != 2
                ):
                    raise ReplayValidationError("formalization retry is out of order")
                initial = MechanismFormalizationPayload.model_validate(
                    event.payload["initial_payload"]
                )
                initial_tracks = {item.candidate_track_id for item in initial.mechanisms}
                initial_tracks.update(
                    item.candidate_track_id for item in initial.skipped_candidates
                )
                if not initial_tracks.issubset(formalization_tracks):
                    raise ReplayValidationError("formalization retry used an unissued candidate")
                incomplete = initial_tracks != formalization_tracks
                reason = (
                    "INCOMPLETE_CANDIDATE_ACCOUNTING" if incomplete
                    else "EMPTY_MECHANISM_SET"
                )
                if (
                    event.payload.get("reason_code") != reason
                    or (not incomplete and initial.mechanisms)
                ):
                    raise ReplayValidationError("formalization retry trigger is invalid")
                for item in initial.mechanisms:
                    candidate = workspace_reducer.workspace.candidate(item.candidate_track_id)
                    if (
                        candidate is None
                        or candidate.semantic_digest != item.source_semantic_digest
                        or " ".join(candidate.mechanism_summary.casefold().split())
                        != " ".join(item.mechanism.mechanism.casefold().split())
                        or item.mechanism.source_hypothesis_id not in proposals
                    ):
                        raise ReplayValidationError("formalization retry source is invalid")
                formalization_retry_payload = initial
            elif event.event_type == EventType.MECHANISM_FORMALIZATION_COMPLETED:
                from rca_agent.core.dynamic import MechanismDiscoveryPayload
                from rca_agent.investigation import MechanismFormalizationPayload

                if not formalization_started or discovery_completed:
                    raise ReplayValidationError("formalization completion is out of order")
                payload = MechanismFormalizationPayload.model_validate(event.payload["payload"])
                if formalization_contract_version == 2:
                    accounted = {item.candidate_track_id for item in payload.mechanisms}
                    accounted.update(item.candidate_track_id for item in payload.skipped_candidates)
                    if accounted != formalization_tracks:
                        raise ReplayValidationError("formalization candidate accounting is incomplete")
                    if not payload.mechanisms and formalization_retry_payload is None:
                        raise ReplayValidationError("empty formalization skipped bounded retry")
                    if formalization_retry_payload is not None:
                        first = {
                            item.candidate_track_id: item.mechanism.model_dump(mode="json")
                            for item in formalization_retry_payload.mechanisms
                        }
                        last = {
                            item.candidate_track_id: item.mechanism.model_dump(mode="json")
                            for item in payload.mechanisms
                        }
                        if any(last.get(track) != mechanism for track, mechanism in first.items()):
                            raise ReplayValidationError("formalization retry lost a mechanism")
                candidates = {
                    item.candidate_track_id: item
                    for item in workspace_reducer.workspace.candidates
                }
                for item in payload.mechanisms:
                    candidate = candidates.get(item.candidate_track_id)
                    if (
                        candidate is None
                        or candidate.semantic_digest != item.source_semantic_digest
                        or " ".join(candidate.mechanism_summary.casefold().split())
                        != " ".join(item.mechanism.mechanism.casefold().split())
                        or item.mechanism.source_hypothesis_id not in proposals
                    ):
                        raise ReplayValidationError("formalizer changed candidate semantics")
                if any(item.candidate_track_id not in formalization_tracks
                       for item in payload.skipped_candidates):
                    raise ReplayValidationError("formalizer skipped an unissued candidate")
                rebindings = event.payload.get("source_id_rebindings", 0)
                if (
                    type(rebindings) is not int
                    or not 0 <= rebindings <= len(payload.mechanisms)
                    or (rebindings and len(proposals) != 1)
                ):
                    raise ReplayValidationError("formalizer source ID binding is invalid")
                mechanism_payload = MechanismDiscoveryPayload(
                    mechanisms=tuple(item.mechanism for item in payload.mechanisms)
                )
                if event.payload["mechanism_payload_digest"] != sha256_digest(
                    mechanism_payload.model_dump(mode="json")
                ):
                    raise ReplayValidationError("formalization payload digest is invalid")
                formalization_started = False
            elif event.event_type == EventType.MECHANISM_FORMALIZATION_REJECTED:
                if not formalization_started or discovery_completed:
                    raise ReplayValidationError("formalization rejection is out of order")
                formalization_started = False
            elif event.event_type == EventType.MECHANISM_COMPLETION_REQUESTED:
                from rca_agent.core.dynamic import (
                    IdentifiedMechanismHint,
                    MechanismDiscoveryPayload,
                )

                if (
                    status != EpisodeStatus.ACTIVE
                    or not discovery_started
                    or discovery_completed
                    or mechanism_completion is not None
                    or mechanism_completion_count >= 1
                ):
                    raise ReplayValidationError("mechanism completion request is out of order")
                initial = MechanismDiscoveryPayload.model_validate(
                    event.payload["initial_payload"]
                )
                if (
                    event.payload.get("attempt") != 1
                    or event.payload.get("initial_payload_digest")
                    != sha256_digest(initial.model_dump(mode="json"))
                ):
                    raise ReplayValidationError("mechanism completion request is invalid")
                hints = tuple(
                    IdentifiedMechanismHint.model_validate(item)
                    for item in event.payload.get("allowed_hints", ())
                )
                initial_hints = {
                    item.model_dump_json()
                    for item in initial.identified_but_unsubmitted_mechanisms
                }
                submitted = {
                    (item.source_hypothesis_id, item.mechanism.strip().casefold())
                    for item in initial.mechanisms
                }
                dispositions = {
                    item.hypothesis_id: item for item in initial.inquiry_dispositions
                }
                keys = [
                    (item.source_hypothesis_id, item.mechanism.strip().casefold())
                    for item in hints
                ]
                if (
                    not hints
                    or len(keys) != len(set(keys))
                    or len(initial.mechanisms) + len(hints) > 4
                    or any(item.model_dump_json() not in initial_hints for item in hints)
                    or any(item.source_hypothesis_id not in proposals for item in hints)
                    or any(
                        inquiry_handoff_ids
                        and item.inquiry_hypothesis_id not in inquiry_handoff_ids
                        for item in hints
                    )
                    or any(key in submitted for key in keys)
                    or any(
                        item.inquiry_hypothesis_id not in dispositions
                        or item.mechanism.strip().casefold()
                        not in dispositions[
                            item.inquiry_hypothesis_id
                        ].reason.strip().casefold()
                        for item in hints
                    )
                ):
                    raise ReplayValidationError("mechanism completion changed discovery semantics")
                mechanism_completion = {
                    "allowed": set(keys),
                    "initial": submitted,
                    "initial_drafts": {
                        item.model_dump_json() for item in initial.mechanisms
                    },
                }
                mechanism_completion_count += 1
            elif event.event_type == EventType.MECHANISM_COMPLETION_COMPLETED:
                from rca_agent.core.dynamic import MechanismDiscoveryPayload, MechanismDraft

                if (
                    status != EpisodeStatus.ACTIVE
                    or discovery_completed
                    or mechanism_completion is None
                    or event.payload.get("attempt") != 1
                ):
                    raise ReplayValidationError("mechanism completion result is out of order")
                completion = MechanismDiscoveryPayload.model_validate(event.payload["payload"])
                accepted = tuple(
                    MechanismDraft.model_validate(item)
                    for item in event.payload.get("accepted_mechanisms", ())
                )
                returned = {item.model_dump_json() for item in completion.mechanisms}
                accepted_keys = [
                    (item.source_hypothesis_id, item.mechanism.strip().casefold())
                    for item in accepted
                ]
                if (
                    len(accepted_keys) != len(set(accepted_keys))
                    or any(item.model_dump_json() not in returned for item in accepted)
                    or not set(accepted_keys) <= mechanism_completion["allowed"]
                    or set(accepted_keys) & mechanism_completion["initial"]
                ):
                    raise ReplayValidationError("mechanism completion introduced a new meaning")
                completion_expected_drafts = {
                    *mechanism_completion["initial_drafts"],
                    *(item.model_dump_json() for item in accepted),
                }
                mechanism_completion = None
            elif event.event_type == EventType.MECHANISM_COMPLETION_REJECTED:
                if (
                    status != EpisodeStatus.ACTIVE
                    or discovery_completed
                    or mechanism_completion is None
                    or event.payload.get("attempt") != 1
                ):
                    raise ReplayValidationError("mechanism completion rejection is out of order")
                completion_expected_drafts = set(mechanism_completion["initial_drafts"])
                mechanism_completion = None
            elif event.event_type == EventType.MECHANISM_DRAFT_GENERATED:
                from rca_agent.core.dynamic import MechanismDraft

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism draft is out of order")
                generated = MechanismDraft.model_validate(event.payload["draft"])
                if engineering_revision == 2 and frozen_mechanisms:
                    raise ReplayValidationError("mechanism generated after freeze projection")
                generated_mechanisms.append((event.payload["draft_index"], generated))
                if handoff_contract_version in {2, 3, 4, 5, 6, 7}:
                    if handoff_validation is None or (
                        event.payload["draft_index"] != len(handoff_generated)
                        or len(handoff_generated) >= len(handoff_expected_drafts)
                        or generated.model_dump_json() != handoff_expected_drafts[len(handoff_generated)]
                    ):
                        raise ReplayValidationError("mechanism bypassed semantic handoff validation")
                    handoff_generated.append(generated.model_dump_json())
                mechanism_draft_generated_count += 1
                serialized = generated.model_dump_json()
                if discovery_format_repair_seen < len(discovery_format_repair_drafts):
                    if serialized != discovery_format_repair_drafts[discovery_format_repair_seen]:
                        raise ReplayValidationError("Discovery repair draft changed after acceptance")
                    discovery_format_repair_seen += 1
                if (
                    completion_expected_drafts is not None
                    and serialized not in completion_expected_drafts
                ):
                    raise ReplayValidationError("mechanism completion draft was not authorized")
                completion_generated_drafts.add(serialized)
            elif event.event_type == EventType.MECHANISM_DRAFT_REJECTED:
                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism rejection is out of order")
            elif event.event_type == EventType.MECHANISM_DRAFT_FROZEN:
                from rca_agent.core.dynamic import FrozenMechanism, MechanismDraft, freeze_mechanism

                if status != EpisodeStatus.ACTIVE or not discovery_started or discovery_completed:
                    raise ReplayValidationError("mechanism freeze is out of order")
                frozen = FrozenMechanism.model_validate(event.payload["mechanism"])
                if engineering_revision == 2:
                    from rca_agent.investigation.contract_enforcement import (
                        draft_rejection,
                        freeze_input,
                    )
                    from rca_agent.runner.episode import _find_entity_refs
                    entities = {bootstrap.alert_entity.entity_ref} | set().union(*(
                        _find_entity_refs(r.model_dump(mode="json")) for r in completed_queries.values()))
                    seen = set()
                    valid = [(index, item) for index, item in generated_mechanisms
                             if draft_rejection(item, proposals, entities, seen,
                                 reject_self=events[0].payload.get("bound_handoff_version", 6) == 7) is None]
                    original = next((item for index, item in valid
                                     if index == event.payload["draft_index"]), None)
                    if (original is None or freeze_mechanism(freeze_input(original, valid)) != frozen):
                        raise ReplayValidationError("frozen mechanism differs from authorized draft and peers")
                if events[0].payload.get("bound_handoff_version", 6) == 7:
                    from rca_agent.investigation.inquiry_contract import normalized
                    if any(normalized(a.mechanism) == normalized(frozen.mechanism)
                           for a in frozen.alternatives):
                        raise ReplayValidationError("SELF_ALTERNATIVE")
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
                else:
                    generated_plans[(plan.mechanism_id, event.payload["attempt"])] = plan
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
                if event.payload.get("diagnostics") is not None:
                    from rca_agent.core.dynamic import CapabilityHandle, VerificationPlanDraft
                    from rca_agent.runner.planner_feedback import plan_feedback

                    plan = VerificationPlanDraft.model_validate(event.payload["plan"])
                    handles = tuple(CapabilityHandle.model_validate(h)
                                    for h in event.payload["capabilities"])
                    if plan != generated_plans.get((plan.mechanism_id, event.payload["attempt"])):
                        raise ReplayValidationError("planner feedback changed the generated plan")
                    if event.payload["diagnostics"] != plan_feedback(
                        frozen_mechanisms[plan.mechanism_id], plan, handles,
                        event.payload["error_code"],
                    ):
                        raise ReplayValidationError("planner feedback differs from contract diagnostics")
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
                if mechanism_completion is not None:
                    raise ReplayValidationError("discovery completed during mechanism completion")
                if (
                    discovery_format_repair_context is not None
                    or discovery_format_repair_seen != len(discovery_format_repair_drafts)
                ):
                    raise ReplayValidationError("Discovery format repair is incomplete")
                if (
                    completion_expected_drafts is not None
                    and completion_generated_drafts != completion_expected_drafts
                ):
                    raise ReplayValidationError("mechanism completion drafts are incomplete")
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
                if (handoff_contract_version in {2, 3, 4, 5, 6, 7} and event.payload.get("status") == "completed"
                        and (handoff_validation is None or handoff_generated != handoff_expected_drafts
                            or event.payload.get("inquiry_dispositions") != [
                                d.model_dump(mode="json") for d in sorted(
                                    handoff_validation.inquiry_dispositions,
                                    key=lambda d: d.hypothesis_id,
                                )
                            ])):
                    raise ReplayValidationError("semantic handoff completion differs")
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
            elif event.event_type == EventType.MECHANISM_COVERAGE_SHADOW_RECORDED:
                if (
                    status != EpisodeStatus.ACTIVE
                    or coverage_mode != "shadow"
                    or coverage_recorded
                    or bootstrap is None
                    or pending_request is not None
                ):
                    raise ReplayValidationError("coverage shadow record is out of order")
                try:
                    record = MechanismCoverageRecord.model_validate(event.payload)
                    workspace = workspace_reducer.workspace.snapshot(
                        transcript_head_sequence=event.sequence - 1,
                        transcript_head_digest=event.previous_digest,
                    )
                    expected_input = build_mechanism_coverage_input(
                        workspace,
                        observations.values(),
                        bootstrap,
                        policy_version=record.input.policy_version,
                        formal_candidates=project_formal_coverage_candidates(
                            events[:event.sequence - 1]
                        ),
                    )
                    expected_assessment = MechanismCoveragePolicy().evaluate(expected_input)
                except (TypeError, ValueError) as exc:
                    raise ReplayValidationError("coverage shadow record is invalid") from exc
                if (
                    record.input != expected_input
                    or record.assessment != expected_assessment
                    or not record.assessment.verify_digest()
                ):
                    raise ReplayValidationError(
                        "coverage shadow record differs from replayed episode state"
                    )
                coverage_recorded = True
                coverage_requires_answer = True
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
                if expected_active_query is not None:
                    if event.payload["query_ref"] != expected_active_query:
                        raise ReplayValidationError("active query execution differs from selection")
                    expected_active_query = None
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
            elif event.event_type == EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED:
                from rca_agent.investigation import DiscriminationActiveSelection

                selection = DiscriminationActiveSelection.model_validate(event.payload)
                if (
                    status != EpisodeStatus.ACTIVE
                    or selection.offered_query_fingerprints != active_offers_by_round.get(
                        selection.source_round
                    )
                ):
                    raise ReplayValidationError(
                        "active Matrix selection differs from model-offered queries"
                    )
                if selection.strategy_version == 2:
                    if expected_active_query is not None:
                        raise ReplayValidationError("active selection has no executed predecessor")
                    expected_active_query = selection.selected_query_fingerprint
            elif event.event_type == EventType.CAPABILITY_REGISTRY_RESOLVED:
                from rca_agent.investigation import CapabilityRegistrySnapshot
                from rca_agent.investigation.query_strategy import query_capabilities

                latest_query_registry = CapabilityRegistrySnapshot.model_validate(event.payload)
                query_caps = tuple(c for c in latest_query_registry.capabilities
                                   if c.query_action is not None)
                if query_caps:
                    if bootstrap is None or not inquiry_rounds:
                        raise ReplayValidationError("query catalog has no episode context")
                    evidence_caps = [c for c in latest_query_registry.capabilities
                                     if c.query_action is None]
                    expected = query_capabilities(
                        bootstrap, tuple(r.observation for r in observations.values()),
                        SimpleNamespace(capability_handles=lambda caps=evidence_caps: caps),
                        inquiry_rounds[-1]["selected_actions"],
                    )
                    if query_caps != expected:
                        raise ReplayValidationError("query catalog differs from published legal actions")
            elif event.event_type == EventType.DISCRIMINATION_TASK_CREATED:
                from rca_agent.investigation import DiscriminationTask
                from rca_agent.investigation.query_strategy import scheduling_metadata

                task = DiscriminationTask.model_validate(event.payload)
                if task.scheduling is not None:
                    record = next(r for r in inquiry_rounds if r["round"] == task.source_round)
                    need = next(n for n in record["draft"]["needs"]
                                if n["need_id"] == task.source_need_id)
                    selected = next(a for a in record["selected_actions"]
                                    if a["need_id"] == task.source_need_id
                                    and query_fingerprint(
                                        CallToolAction.model_validate(a["action"]).tool_name,
                                        CallToolAction.model_validate(a["action"]).request,
                                    ) == task.legacy_query_ref)
                    action = CallToolAction.model_validate(selected["action"])
                    fingerprint = query_fingerprint(action.tool_name, action.request)
                    if (engineering_revision == 2 and record["draft"].get("contract_version") == 2
                            and task.source_binding_id != selected["binding_id"]):
                        raise ReplayValidationError("task omitted its declared binding identity")
                    if task.source_binding_id is not None:
                        from rca_agent.investigation.inquiry_contract import binding_id
                        if (task.source_binding_id != binding_id(task.source_round,
                                task.source_need_id, selected["query_ref"])
                                or task.observable_kind != selected["signal"]
                                or task.legacy_choice_ref != selected["query_ref"]
                                or task.question != need["question"]
                                or task.candidate_prediction != need["expected_if_true"]
                                or task.comparator_prediction != need["expected_if_alternative"]):
                            raise ReplayValidationError("task differs from accepted binding")
                    if (task.legacy_query_ref != fingerprint or not task.capability_options
                            or any(c.query_action != action.model_dump(mode="json")
                                   for c in task.capability_options)
                            or task.scheduling != scheduling_metadata(
                                need, task.capability_options[0], bootstrap,
                                tuple(r.observation for r in observations.values()),
                                SimpleNamespace(signals=latest_query_registry.capabilities),
                                binding=selected if record["draft"].get("contract_version") == 2 else None,
                            )):
                        raise ReplayValidationError("query need alignment differs from recorded input")
            elif event.event_type == EventType.INQUIRY_QUERY_CANCELLED:
                task = workspace_reducer.workspace.task(event.payload["task_id"])
                reason = event.payload["reason_code"]
                if (status != EpisodeStatus.ACTIVE
                        or reason == "DUPLICATE_EXECUTED_QUERY"
                        and event.payload["query_fingerprint"] not in completed_queries
                        or reason == "QUERY_BUDGET_UNAVAILABLE"
                        and task.capability_options[0].query_cost <= budget_remaining):
                    raise ReplayValidationError("query cancellation has no valid basis")
            elif event.event_type == EventType.DISCRIMINATION_TASK_COMPLETED and engineering_revision == 2:
                from rca_agent.investigation.contract_enforcement import execution_binding_responses
                task = workspace_reducer.workspace.task(event.payload["task_id"])
                if task.source_binding_id is not None:
                    response = execution_binding_responses(events[:event.sequence - 1]).get(task.source_binding_id)
                    if response is None or set(event.payload["observation_refs"]) != {
                            item["observation_id"] for item in response["observations"]}:
                        raise ReplayValidationError("task response predates its execution binding")
            elif event.event_type in {
                EventType.INVESTIGATION_CANDIDATE_ADDED,
                EventType.CANDIDATE_TRANSITIONED,
                EventType.CANDIDATE_SUPERSEDED,
                EventType.CAPABILITY_REGISTRY_RESOLVED,
                EventType.DISCRIMINATION_TASK_CREATED,
                EventType.DISCRIMINATION_TASK_SELECTED,
                EventType.DISCRIMINATION_TASK_COMPLETED,
                EventType.DISCRIMINATION_TASK_UNVERIFIABLE,
                EventType.DISCRIMINATION_SHADOW_SELECTION_RECORDED,
                EventType.MECHANISM_FORMALIZATION_STARTED,
                EventType.MECHANISM_FORMALIZATION_RETRIED,
                EventType.MECHANISM_FORMALIZATION_COMPLETED,
                EventType.MECHANISM_FORMALIZATION_REJECTED,
                EventType.DISCOVERY_FORMAT_REPAIR_REQUESTED,
                EventType.DISCOVERY_FORMAT_REPAIR_COMPLETED,
                EventType.DISCOVERY_FORMAT_REPAIR_REJECTED,
            }:
                if status != EpisodeStatus.ACTIVE:
                    if status in {
                        EpisodeStatus.ANSWER_FROZEN,
                        EpisodeStatus.EVALUATED,
                    }:
                        raise ReplayValidationError(
                            "inference event appears after answer freeze"
                        )
                    raise ReplayValidationError(
                        "investigation workspace event is outside the active episode"
                    )
            elif event.event_type == EventType.ANSWER_FROZEN:
                if (status != EpisodeStatus.ACTIVE or pending_request is not None
                        or extension_started is not None
                        or expected_extension_proposal is not None):
                    raise ReplayValidationError("answer freeze is out of order")
                if coverage_mode == "shadow" and not coverage_recorded:
                    raise ReplayValidationError(
                        "coverage shadow mode froze an answer without a coverage record"
                    )
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
                if event.payload.get("investigation_workspace") is not None:
                    assert answer.investigation_workspace is not None
                    expected_workspace = workspace_reducer.workspace.snapshot(
                        transcript_head_sequence=expected_sequence - 1,
                        transcript_head_digest=event.previous_digest,
                        registry_digest=answer.investigation_workspace.registry_digest,
                    )
                    workspace_matches = answer.investigation_workspace == expected_workspace
                    raw_workspace = event.payload["investigation_workspace"]
                    if not workspace_matches and "active_selections" not in raw_workspace:
                        actual_previous = answer.investigation_workspace.model_dump(
                            mode="json", exclude={"snapshot_digest", "active_selections"}
                        )
                        expected_previous = expected_workspace.model_dump(
                            mode="json", exclude={"snapshot_digest", "active_selections"}
                        )
                        workspace_matches = actual_previous == expected_previous
                    if not workspace_matches and "discrimination_matrix" not in raw_workspace:
                        actual_legacy = answer.investigation_workspace.model_dump(
                            mode="json",
                            exclude={
                                "snapshot_digest",
                                "discrimination_matrix",
                                "shadow_selections",
                                "active_selections",
                            },
                        )
                        expected_legacy = expected_workspace.model_dump(
                            mode="json",
                            exclude={
                                "snapshot_digest",
                                "discrimination_matrix",
                                "shadow_selections",
                                "active_selections",
                            },
                        )
                        workspace_matches = actual_legacy == expected_legacy
                    if (
                        not answer.investigation_workspace.verify_digest()
                        or not workspace_matches
                    ):
                        raise ReplayValidationError(
                            "workspace projection differs from transcript reduction"
                        )
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
                coverage_requires_answer = False
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
                EventType.INVESTIGATION_CANDIDATE_ADDED,
                EventType.CANDIDATE_TRANSITIONED,
                EventType.CANDIDATE_SUPERSEDED,
                EventType.CAPABILITY_REGISTRY_RESOLVED,
                EventType.DISCRIMINATION_TASK_CREATED,
                EventType.DISCRIMINATION_TASK_SELECTED,
                EventType.DISCRIMINATION_TASK_COMPLETED,
                EventType.DISCRIMINATION_TASK_UNVERIFIABLE,
                EventType.DISCRIMINATION_SHADOW_SELECTION_RECORDED,
                EventType.DISCRIMINATION_ACTIVE_SELECTION_RECORDED,
                EventType.MECHANISM_COVERAGE_SHADOW_RECORDED,
                EventType.MECHANISM_FORMALIZATION_STARTED,
                EventType.MECHANISM_FORMALIZATION_RETRIED,
                EventType.MECHANISM_FORMALIZATION_COMPLETED,
                EventType.MECHANISM_FORMALIZATION_REJECTED,
            }:
                raise ReplayValidationError("inference event appears after answer freeze")
        if pending_request is not None:
            raise ReplayValidationError("transcript ends with an incomplete tool call")
        if coverage_requires_answer:
            raise ReplayValidationError("transcript ends after coverage shadow without answer freeze")
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
