from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from rca_agent.core.dynamic import (
    AlternativeDraft,
    DiscoveryPayload,
    EvidenceRegistry,
    FrozenMechanism,
    MechanismDiscoveryPayload,
    VerificationPlanDraft,
    VerificationPlanExtension,
    VerificationPlanPayload,
    freeze_mechanism,
    observation_fact_digest,
    repair_mechanical_contract,
)
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
from rca_agent.runner.inquiry_outcomes import inquiry_outcomes
from rca_agent.runner.plan_extension import added_predictions
from rca_agent.runner.transcript import EventType, JsonlTranscriptStore, Transcript
from rca_agent.tools.fingerprint import query_fingerprint, sha256_digest

ENTITY_REF_RE = re.compile(r"^ent-[a-z0-9-]{3,40}$")


def _contract_error_code(message: str) -> str:
    mappings = (
        ("unknown capability reference", "UNKNOWN_CAPABILITY_REF"),
        ("unknown alternative reference", "UNKNOWN_ALTERNATIVE_REF"),
        ("unknown frozen mechanism", "UNKNOWN_FROZEN_MECHANISM"),
        ("refutation prediction", "MISSING_REFUTATION"),
        ("dedicated refutation prediction", "ALTERNATIVES_NOT_REFUTED"),
        ("at most one alternative", "REFUTATION_NOT_DEDICATED"),
        ("support prediction cannot distinguish", "SUPPORT_CANNOT_REFUTE"),
        ("cannot refute the alternative mechanism class", "REFUTATION_SCOPE_MISMATCH"),
        ("not eligible as root-cause support", "SUPPORT_RULE_NOT_ELIGIBLE"),
        ("independent support groups", "INSUFFICIENT_SUPPORT_GROUPS"),
        ("generic GC observations", "FULL_GC_SIGNAL_UNAVAILABLE"),
        ("duplicate predictions", "DUPLICATE_PREDICTIONS"),
        ("not executable", "CAPABILITY_NOT_EXECUTABLE"),
        ("removed or changed a frozen prediction", "NON_ADDITIVE_PLAN"),
        ("requires one or two new predictions", "NO_NEW_PREDICTION"),
    )
    return next((code for fragment, code in mappings if fragment in message), "CONTRACT_REJECTED")


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
        verification_planner: Any | None = None,
        evidence_registry: EvidenceRegistry | Callable[..., EvidenceRegistry] | None = None,
        discovery_timeout_seconds: float = 30,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if discovery_timeout_seconds <= 0 or discovery_timeout_seconds > 120:
            raise ValueError("discovery timeout must be in (0, 120]")
        if discovery is not None and evidence_registry is None:
            raise ValueError("discovery requires host-owned signal and rule contracts")
        if verification_planner is not None and discovery is None:
            raise ValueError("verification planner requires mechanism discovery")
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
        self.__verification_planner = verification_planner
        self.__registry = evidence_registry
        self.__discovery_timeout = discovery_timeout_seconds
        self.__discovery_attempted = False
        self.__dynamic_options: list[dict[str, Any]] = []
        self.__verification_queries: list[tuple[str, CallToolAction]] = []
        self.__resolved_registry: EvidenceRegistry | None = None
        self.__frozen_mechanisms: dict[str, FrozenMechanism] = {}
        self.__compiled_plans: dict[str, VerificationPlanDraft] = {}
        self.__candidate_for_mechanism: dict[str, str] = {}
        self.__feedback_planned = False
        self.__inquiry_rounds: list[dict[str, Any]] = []
        self.__bootstrap: ResetResponse | None = None
        self.__budget_remaining = 0
        self.__proposals: list[HypothesisProposal] = []
        self.__observation_refs: set[str] = set()
        self.__observation_ledger: dict[str, IssuedObservation] = {}
        self.__issued_entities: set[str] = set()
        self.__answer: AnswerArtifact | None = None
        self.transcript.append(
            EventType.EPISODE_CREATED,
            {
                "status": self.status,
                "max_rounds": max_rounds,
                "excluded_catalog_candidate_ids": sorted(
                    self.excluded_catalog_candidate_ids
                ),
            },
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

    @property
    def excluded_catalog_candidate_ids(self) -> frozenset[str]:
        provider = self.__decision_provider
        return (
            provider.excluded_catalog_candidate_ids
            if isinstance(provider, CoreDecisionProvider)
            else frozenset()
        )

    def record_inquiry_round(self, round_record: dict[str, Any]) -> None:
        """Bind each model-authored exploration draft before its selected query executes."""
        from rca_agent.langchain_acquisition import InquiryDraft

        self._require_status(EpisodeStatus.ACTIVE)
        if self.__discovery_attempted or len(self.__inquiry_rounds) >= 5:
            raise ValueError("inquiry round is out of bounds")
        round_number = round_record["round"]
        if (not 1 <= round_number <= 5
                or self.__inquiry_rounds and round_number <= self.__inquiry_rounds[-1]["round"]):
            raise ValueError("inquiry rounds must increase")
        draft = InquiryDraft.model_validate(round_record["draft"])
        payload = {"round": round_number, "draft": draft.model_dump(mode="json")}
        self.__inquiry_rounds.append(payload)
        self.transcript.append(EventType.INQUIRY_ROUND_RECORDED, payload)

    def should_discover(self, *, stopping: bool = False) -> bool:
        if (self.status != EpisodeStatus.ACTIVE or self.__discovery is None
                or self.__discovery_attempted or self.__rounds >= self.__max_rounds
                or self.__budget_remaining <= 0 or not self.__observation_ledger
                or not isinstance(self.__decision_provider, CoreDecisionProvider)):
            return False
        preview = self.preview_core_assessment()
        if preview.overall_decision.status in {
            DecisionStatus.CONFIRMED_ROOT_CAUSE, DecisionStatus.DISCOVERED_VERIFIED,
        }:
            return False
        resolved = {h for c in preview.candidate_assessments for h in c.hypothesis_ids}
        if any(p.hypothesis_id not in resolved for p in self.__proposals):
            return True
        return bool(self.__proposals) and (stopping or all(
            c.evidence_status.value == "contradicted" for c in preview.candidate_assessments
        ))

    def discover_candidates(self, *, inquiry_handoff: dict[str, Any] | None = None) -> dict[str, Any]:
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
        self.__resolved_registry = registry
        proposals = {p.hypothesis_id: p for p in self.__proposals}
        context = {
            "alert_summary": self.__bootstrap.alert_summary,
            "hypotheses": [p.model_dump(mode="json") for p in proposals.values()],
            "signals": [{"signal": s.signal, "entity_ref": s.entity_ref,
                         "data_type": s.data_type, "unit": s.unit, "description": s.description,
                         "available": self._signal_executable(s)} for s in registry.signals],
            "rules": [{"rule_id": r.rule_id, "signal": r.signal,
                       "phenomenon": r.description, "aggregation": r.aggregation,
                       "support_eligible": r.support_eligible}
                      for r in registry.rules],
            "entity_graph": self._discovery_entity_graph(registry),
            "observations": [r.observation.model_dump(mode="json")
                             for r in self.__observation_ledger.values()
                             if any(s.accepts(r.observation) for s in registry.signals)],
            "existing_candidates": [
                {"candidate_id": c.candidate_id, "mechanism": c.fault_type}
                for c in self.__decision_provider.catalog(self.__bootstrap.alert_entity.entity_ref)
            ],
        }
        if inquiry_handoff is not None:
            from rca_agent.langchain_acquisition import InquiryHandoff

            handoff = InquiryHandoff.model_validate(inquiry_handoff)
            if self.__inquiry_rounds:
                from rca_agent.langchain_acquisition import inquiry_candidate_records

                if handoff.candidate_history != inquiry_candidate_records(self.__inquiry_rounds):
                    raise ValueError("inquiry handoff omitted a recorded candidate")
            completed = {
                event.payload["query_ref"]: event.payload
                for event in self.transcript.events
                if event.event_type == EventType.TOOL_RESPONSE
            }
            for link in handoff.query_links:
                response = completed.get(link.query_ref)
                if (response is None or response["status"] != link.status
                        or set(link.observation_refs) != {
                            item["observation_id"] for item in response["observations"]
                        }):
                    raise ValueError("inquiry handoff references an unissued query result")
            context["inquiry_handoff"] = handoff.model_dump(mode="json")
        default_dispositions = [
            {"hypothesis_id": item["hypothesis_id"], "disposition": "unreported",
             "reason": "This discovery interface does not report inquiry dispositions."}
            for item in context.get("inquiry_handoff", {}).get("hypotheses", ())
        ]
        self.transcript.append(EventType.DISCOVERY_STARTED, {
            "context_digest": sha256_digest(context),
            "inquiry_handoff": context.get("inquiry_handoff"),
        })
        if self.__verification_planner is not None:
            return self._discover_two_stage(context, registry, proposals)
        mailbox: Queue = Queue(maxsize=1)
        generator = self.__discovery

        def invoke():
            try:
                mailbox.put((True, generator.discover(deepcopy(context))))
            except Exception as exc:  # noqa: BLE001 - failure is sanitized and bounded below
                failure = {"error_type": type(exc).__name__}
                raw_payload = getattr(exc, "raw_payload", None)
                if isinstance(raw_payload, (dict, list)):
                    failure["raw_payload"] = raw_payload
                mailbox.put((False, failure))

        Thread(target=invoke, daemon=True).start()
        registered = 0
        aligned = 0
        rejections = []
        try:
            ok, raw = mailbox.get(timeout=self.__discovery_timeout)
            if not ok:
                self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, raw)
                result = {
                    "status": "failed", "registered": 0,
                    "error_type": raw["error_type"],
                }
                self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
                return result
            payload = DiscoveryPayload.model_validate(raw)
            seen = set()
            for draft_index, draft in enumerate(payload.hypotheses):
                evaluated_draft = draft
                initial_error = None
                repair_actions: tuple[str, ...] = ()
                try:
                    original = proposals.get(draft.source_hypothesis_id)
                    discovery_key = (
                        draft.source_hypothesis_id,
                        draft.mechanism.strip().casefold(),
                    )
                    if original is None or discovery_key in seen:
                        raise ValueError("unknown or duplicate source hypothesis")
                    seen.add(discovery_key)
                    referenced_entities = {
                        draft.causal_entity_ref or original.causal_entity_ref,
                        *(prediction.target_entity_ref
                          for prediction in draft.predictions
                          if prediction.target_entity_ref is not None),
                    }
                    if not referenced_entities <= self.__issued_entities:
                        raise ValueError("dynamic hypothesis references an unissued entity")
                    compile_kwargs = {
                        "discovery_query_refs": tuple(sorted({
                            r.query_ref for r in self.__observation_ledger.values()
                        })),
                        "discovery_observation_refs": tuple(sorted(self.__observation_ledger)),
                        "discovery_fact_digests": tuple(sorted({
                            observation_fact_digest(r.observation)
                            for r in self.__observation_ledger.values()
                        })),
                    }
                    try:
                        compiled = registry.compile(
                            evaluated_draft, original.causal_entity_ref, **compile_kwargs
                        )
                    except ValueError as exc:
                        initial_error = str(exc)
                        evaluated_draft, repair_actions = repair_mechanical_contract(draft)
                        if not repair_actions:
                            raise
                        compiled = registry.compile(
                            evaluated_draft, original.causal_entity_ref, **compile_kwargs
                        )
                    if not all(self._signal_executable(s) for s in compiled.signals):
                        raise ValueError("query recipe is not executable in this episode")
                    existing = next((c for c in self.__decision_provider.catalog(
                                     compiled.candidate.causal_entity)
                                     if evaluated_draft.mechanism.casefold() in {
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
                            mechanism=existing.fault_type,
                            causal_entity_ref=compiled.candidate.causal_entity,
                            rationale=evaluated_draft.causal_explanation,
                        ))
                        queries = {query_fingerprint(s.query.tool_name, s.query.request): s.query
                                   for s in compiled.signals}
                        self.__verification_queries.extend((hypothesis_id, q) for q in queries.values())
                        aligned += 1
                        self._record_discovery_draft(
                            draft_index, draft, evaluated_draft, initial_error,
                            repair_actions, "accepted", None, "catalog_aligned",
                        )
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
                        mechanism=evaluated_draft.mechanism,
                        causal_entity_ref=compiled.candidate.causal_entity,
                        rationale=evaluated_draft.causal_explanation,
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
                    self._record_discovery_draft(
                        draft_index, draft, evaluated_draft, initial_error,
                        repair_actions, "accepted", None, "dynamic_registered",
                    )
                except ValueError as exc:
                    final_error = (str(exc) if type(exc) is ValueError
                                   else "invalid compiled candidate contract")
                    rejections.append(final_error)
                    self._record_discovery_draft(
                        draft_index, draft, evaluated_draft, initial_error,
                        repair_actions, "rejected", final_error, "none",
                    )
            result = {"status": "completed", "registered": registered,
                      "aligned": aligned, "rejections": rejections,
                      "inquiry_dispositions": default_dispositions}
        except Empty:
            result = {"status": "timeout", "registered": 0}
        except (ValueError, TypeError):
            result = {"status": "failed", "registered": 0}
        self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
        return result

    def _discover_two_stage(
        self,
        context: dict[str, Any],
        registry: EvidenceRegistry,
        proposals: dict[str, HypothesisProposal],
    ) -> dict[str, Any]:
        mechanism_context = {
            key: deepcopy(value)
            for key, value in context.items()
            if key not in {"signals", "rules"}
        }
        for node in mechanism_context["entity_graph"]["nodes"]:
            node.pop("available_rule_ids", None)
        ok, raw = self._invoke_discovery_component(
            lambda: self.__discovery.discover(mechanism_context)
        )
        if not ok:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "mechanism_discovery", **raw,
            })
            result = {"status": "failed", "registered": 0, "error_type": raw["error_type"]}
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result
        try:
            payload = MechanismDiscoveryPayload.model_validate(raw)
        except (ValueError, TypeError) as exc:
            failure = {"stage": "mechanism_discovery", "error_type": type(exc).__name__}
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, failure)
            result = {"status": "failed", "registered": 0, "error_type": type(exc).__name__}
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result

        inquiry_ids = {
            item["hypothesis_id"]
            for item in context.get("inquiry_handoff", {}).get("hypotheses", ())
        }
        model_dispositions = {
            item.hypothesis_id: item for item in payload.inquiry_dispositions
            if item.hypothesis_id in inquiry_ids
        }
        inquiry_dispositions = [
            (model_dispositions[item].model_dump(mode="json")
             if item in model_dispositions else {
                 "hypothesis_id": item, "disposition": "unreported",
                 "reason": "The discovery model did not provide a disposition.",
             })
            for item in sorted(inquiry_ids)
        ]

        frozen: dict[str, FrozenMechanism] = {}
        valid_drafts = []
        seen = set()
        rejection_codes: list[str] = []
        for draft_index, draft in enumerate(payload.mechanisms):
            self.transcript.append(EventType.MECHANISM_DRAFT_GENERATED, {
                "draft_index": draft_index, "draft": draft.model_dump(mode="json"),
            })
            key = (draft.source_hypothesis_id, draft.mechanism.strip().casefold())
            error_code = None
            if draft.source_hypothesis_id not in proposals:
                error_code = "UNKNOWN_SOURCE_HYPOTHESIS"
            elif key in seen:
                error_code = "DUPLICATE_MECHANISM"
            elif draft.causal_entity_ref not in self.__issued_entities:
                error_code = "UNISSUED_CAUSAL_ENTITY"
            if error_code is not None:
                rejection_codes.append(error_code)
                self.transcript.append(EventType.MECHANISM_DRAFT_REJECTED, {
                    "draft_index": draft_index, "error_code": error_code,
                })
                continue
            seen.add(key)
            valid_drafts.append((draft_index, draft))

        for draft_index, draft in valid_drafts:
            existing_alternatives = {
                item.mechanism.strip().casefold() for item in draft.alternatives
            }
            peers = tuple(
                AlternativeDraft(
                    alternative_id="alt-peer-" + sha256_digest({
                        "source": other.source_hypothesis_id,
                        "mechanism": other.mechanism,
                    })[:16],
                    mechanism=other.mechanism,
                    mechanism_class=other.mechanism_class,
                )
                for _, other in valid_drafts
                if other.source_hypothesis_id == draft.source_hypothesis_id
                and other.mechanism.strip().casefold() != draft.mechanism.strip().casefold()
                and other.mechanism.strip().casefold() not in existing_alternatives
            )
            frozen_draft = draft.model_copy(update={
                "alternatives": (*draft.alternatives, *peers),
            })
            item = freeze_mechanism(frozen_draft)
            frozen[item.mechanism_id] = item
            self.__frozen_mechanisms[item.mechanism_id] = item
            self.transcript.append(EventType.MECHANISM_DRAFT_FROZEN, {
                "draft_index": draft_index, "mechanism": item.model_dump(mode="json"),
            })

        if not frozen:
            result = {
                "status": "completed", "registered": 0, "aligned": 0,
                "rejections": rejection_codes,
                "inquiry_dispositions": inquiry_dispositions,
                "planner_attempts": 0,
            }
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result

        handles = registry.capability_handles()
        planner_base = {
            "frozen_mechanisms": [item.model_dump(mode="json") for item in frozen.values()],
            "capabilities": [item.model_dump(mode="json") for item in handles],
            "observations": context["observations"],
            "unavailable_signals": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unavailable_signals.items()
            },
            "unsupported_derivations": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unsupported_derivations.items()
            },
        }
        compile_kwargs = {
            "discovery_query_refs": tuple(sorted({
                record.query_ref for record in self.__observation_ledger.values()
            })),
            "discovery_observation_refs": tuple(sorted(self.__observation_ledger)),
            "discovery_fact_digests": tuple(sorted({
                observation_fact_digest(record.observation)
                for record in self.__observation_ledger.values()
            })),
        }
        registered = 0
        aligned = 0
        accepted: set[str] = set()

        first_payload, first_failure = self._invoke_planner({
            **planner_base, "attempt": 1, "errors": [],
        })
        if first_payload is None:
            code = first_failure or "PLANNER_FAILED"
            rejection_codes.append(code)
            initial_failures = {
                mechanism_id: {"mechanism_id": mechanism_id, "error_code": code}
                for mechanism_id in frozen
            }
            for mechanism_id in frozen:
                self._reject_plan(mechanism_id, 1, code)
            self.transcript.append(EventType.VERIFICATION_PLAN_RETRIED, {
                "mechanism_ids": sorted(frozen),
                "errors": [initial_failures[value] for value in sorted(frozen)],
            })
            retry_payload, retry_failure = self._invoke_planner({
                **planner_base,
                "attempt": 2,
                "errors": [initial_failures[value] for value in sorted(frozen)],
                "previous_plans": [],
            })
            if retry_payload is not None:
                retry_failed, counts = self._compile_plans(
                    retry_payload,
                    frozen,
                    registry,
                    proposals,
                    compile_kwargs,
                    accepted,
                    attempt=2,
                )
                registered += counts[0]
                aligned += counts[1]
                rejection_codes.extend(
                    item["error_code"] for item in retry_failed.values()
                )
            else:
                rejection_codes.append(retry_failure or "PLANNER_RETRY_FAILED")
            result = {
                "status": "completed", "registered": registered, "aligned": aligned,
                "rejections": rejection_codes, "planner_attempts": 2,
                "inquiry_dispositions": inquiry_dispositions,
            }
            self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
            return result
        failed, counts = self._compile_plans(
            first_payload, frozen, registry, proposals, compile_kwargs, accepted, attempt=1,
        )
        registered += counts[0]
        aligned += counts[1]
        rejection_codes.extend(item["error_code"] for item in failed.values())

        planner_attempts = 1
        if failed:
            planner_attempts = 2
            retry_ids = sorted(failed)
            self.transcript.append(EventType.VERIFICATION_PLAN_RETRIED, {
                "mechanism_ids": retry_ids,
                "errors": [failed[value] for value in retry_ids],
            })
            retry_context = {
                **planner_base,
                "frozen_mechanisms": [
                    frozen[value].model_dump(mode="json") for value in retry_ids
                ],
                "attempt": 2,
                "errors": [failed[value] for value in retry_ids],
                "previous_plans": [
                    failed[value]["plan"] for value in retry_ids if failed[value].get("plan")
                ],
            }
            retry_payload, retry_failure = self._invoke_planner(retry_context)
            if retry_payload is None:
                rejection_codes.append(retry_failure or "PLANNER_RETRY_FAILED")
            else:
                retry_failed, counts = self._compile_plans(
                    retry_payload,
                    {value: frozen[value] for value in retry_ids},
                    registry,
                    proposals,
                    compile_kwargs,
                    accepted,
                    attempt=2,
                )
                registered += counts[0]
                aligned += counts[1]
                rejection_codes.extend(
                    item["error_code"] for item in retry_failed.values()
                )
        result = {
            "status": "completed", "registered": registered, "aligned": aligned,
            "rejections": rejection_codes, "planner_attempts": planner_attempts,
            "inquiry_dispositions": inquiry_dispositions,
        }
        self.transcript.append(EventType.DISCOVERY_COMPLETED, result)
        return result

    def _invoke_discovery_component(self, invoke) -> tuple[bool, Any]:
        from queue import Empty, Queue
        from threading import Thread

        mailbox: Queue = Queue(maxsize=1)

        def target():
            try:
                mailbox.put((True, invoke()))
            except Exception as exc:  # noqa: BLE001 - bounded structured output only
                failure = {"error_type": type(exc).__name__}
                raw_payload = getattr(exc, "raw_payload", None)
                if isinstance(raw_payload, (dict, list)):
                    failure["raw_payload"] = raw_payload
                mailbox.put((False, failure))

        Thread(target=target, daemon=True).start()
        try:
            return mailbox.get(timeout=self.__discovery_timeout)
        except Empty:
            return False, {"error_type": "TimeoutError"}

    def _invoke_planner(
        self, context: dict[str, Any]
    ) -> tuple[VerificationPlanPayload | None, str | None]:
        ok, raw = self._invoke_discovery_component(
            lambda: self.__verification_planner.plan(deepcopy(context))
        )
        if not ok:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "verification_planner", **raw,
            })
            return None, "PLANNER_" + raw["error_type"].upper()
        try:
            return VerificationPlanPayload.model_validate(raw), None
        except (ValueError, TypeError) as exc:
            self.transcript.append(EventType.DISCOVERY_PAYLOAD_REJECTED, {
                "stage": "verification_planner", "error_type": type(exc).__name__,
            })
            return None, "PLANNER_INVALID_PAYLOAD"

    def _compile_plans(
        self,
        payload: VerificationPlanPayload,
        mechanisms: dict[str, FrozenMechanism],
        registry: EvidenceRegistry,
        proposals: dict[str, HypothesisProposal],
        compile_kwargs: dict[str, Any],
        accepted: set[str],
        *,
        attempt: int,
    ) -> tuple[dict[str, dict[str, Any]], tuple[int, int]]:
        failed: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        registered = 0
        aligned = 0
        for plan in payload.plans:
            self.transcript.append(EventType.VERIFICATION_PLAN_GENERATED, {
                "attempt": attempt, "plan": plan.model_dump(mode="json"),
            })
            mechanism = mechanisms.get(plan.mechanism_id)
            if mechanism is None or plan.mechanism_id in seen or plan.mechanism_id in accepted:
                code = "UNKNOWN_OR_DUPLICATE_MECHANISM"
                self._reject_plan(plan.mechanism_id, attempt, code)
                continue
            seen.add(plan.mechanism_id)
            try:
                normalized_plan, plan_repairs = registry.normalize_plan(mechanism, plan)
                compiled = registry.compile_plan(mechanism, normalized_plan, **compile_kwargs)
                if not all(self._signal_executable(signal) for signal in compiled.signals):
                    raise ValueError("query recipe is not executable in this episode")
                registration = self._register_two_stage_candidate(
                    compiled, mechanism, proposals[mechanism.source_hypothesis_id]
                )
                self.__compiled_plans[plan.mechanism_id] = normalized_plan
                self.__candidate_for_mechanism[plan.mechanism_id] = compiled.candidate.candidate_id
                accepted.add(plan.mechanism_id)
                registered += registration == "dynamic_registered"
                aligned += registration == "catalog_aligned"
                self.transcript.append(EventType.VERIFICATION_PLAN_COMPILED, {
                    "attempt": attempt,
                    "mechanism_id": plan.mechanism_id,
                    "mechanism_digest": mechanism.digest,
                    "plan": normalized_plan.model_dump(mode="json"),
                    "plan_repairs": plan_repairs,
                    "registration": registration,
                })
            except ValueError as exc:
                code = _contract_error_code(str(exc))
                failed[plan.mechanism_id] = {
                    "mechanism_id": plan.mechanism_id, "error_code": code,
                    "plan": plan.model_dump(mode="json"),
                }
                self._reject_plan(plan.mechanism_id, attempt, code)
        for mechanism_id in mechanisms.keys() - seen - accepted:
            code = "MISSING_VERIFICATION_PLAN"
            failed[mechanism_id] = {
                "mechanism_id": mechanism_id, "error_code": code,
            }
            self._reject_plan(mechanism_id, attempt, code)
        return failed, (registered, aligned)

    def _reject_plan(self, mechanism_id: str, attempt: int, error_code: str) -> None:
        self.transcript.append(EventType.VERIFICATION_PLAN_REJECTED, {
            "attempt": attempt, "mechanism_id": mechanism_id, "error_code": error_code,
        })

    def _register_two_stage_candidate(
        self,
        compiled,
        mechanism: FrozenMechanism,
        original: HypothesisProposal,
    ) -> str:
        existing = next((candidate for candidate in self.__decision_provider.catalog(
            compiled.candidate.causal_entity
        ) if mechanism.mechanism.casefold() in {
            candidate.candidate_id.casefold(), candidate.fault_type.casefold()
        } and candidate.family == mechanism.family), None)
        if existing is not None:
            required = {condition.signal for condition in existing.conditions}
            if not required <= {signal.signal for signal in compiled.signals}:
                raise ValueError("catalog alignment lacks executable signal recipes")
            hypothesis_id = "h-aligned-" + compiled.version[:20]
            self.transcript.append(EventType.CATALOG_CANDIDATE_ALIGNED, {
                "candidate_id": existing.candidate_id,
                "hypothesis_id": hypothesis_id,
                "source_hypothesis_id": original.hypothesis_id,
                "mechanism_id": mechanism.mechanism_id,
            })
            candidate_id = existing.candidate_id
            registration = "catalog_aligned"
        else:
            hypothesis_id = "h-discovered-" + compiled.version[:20]
            rebound = compiled.model_copy(update={"source_hypothesis_id": hypothesis_id})
            self.__decision_provider.register_dynamic(rebound)
            self.transcript.append(EventType.DYNAMIC_CANDIDATE_REGISTERED, {
                "compiled": rebound.model_dump(mode="json"),
                "source_hypothesis_id": original.hypothesis_id,
                "mechanism_id": mechanism.mechanism_id,
                "mechanism_digest": mechanism.digest,
            })
            candidate_id = compiled.candidate.candidate_id
            candidate = compiled.candidate
            self.__dynamic_options.append({
                "candidate_id": candidate.candidate_id,
                "fault_type": candidate.fault_type,
                "family": candidate.family.value,
                "required_signals": tuple(signal.signal for signal in compiled.signals),
                "required_capabilities": tuple(candidate.required_capabilities),
            })
            registration = "dynamic_registered"
        self._submit_hypothesis(HypothesisProposal(
            hypothesis_id=hypothesis_id,
            candidate_id=candidate_id,
            mechanism=mechanism.mechanism if existing is None else existing.fault_type,
            causal_entity_ref=compiled.candidate.causal_entity,
            rationale=mechanism.causal_explanation,
        ))
        queries = {
            query_fingerprint(signal.query.tool_name, signal.query.request): signal.query
            for signal in compiled.signals
        }
        self.__verification_queries.extend((hypothesis_id, query) for query in queries.values())
        return registration

    def _record_discovery_draft(
        self,
        draft_index: int,
        raw_draft,
        final_draft,
        initial_error: str | None,
        repair_actions: tuple[str, ...],
        final_status: str,
        final_error: str | None,
        registration: str,
    ) -> None:
        self.transcript.append(EventType.DISCOVERY_DRAFT_EVALUATED, {
            "draft_index": draft_index,
            "raw_draft": raw_draft.model_dump(mode="json"),
            "final_draft": final_draft.model_dump(mode="json"),
            "initial_status": "rejected" if initial_error is not None else final_status,
            "initial_error": initial_error,
            "repair_actions": repair_actions,
            "final_status": final_status,
            "final_error": final_error,
            "registration": registration,
        })

    def _discovery_entity_graph(self, registry: EvidenceRegistry) -> dict[str, Any]:
        assert self.__bootstrap is not None
        alert = self.__bootstrap.alert_entity
        nodes: dict[str, dict[str, Any]] = {
            alert.entity_ref: alert.model_dump(mode="json")
        }
        edges: set[tuple[str, str, str, str]] = set()
        for record in self.__observation_ledger.values():
            observation = record.observation
            target = None
            relation = observation.details.get("relation")
            direction = observation.details.get("direction", "outgoing")
            if observation.signal in {"related_entity", "trace_participant"} and isinstance(
                observation.value, dict
            ):
                target = observation.value
            elif observation.signal == "topology_edge":
                target = observation.details.get("neighbor")
            resolved = observation.details.get("resolved_entity")
            if isinstance(resolved, dict):
                nodes[resolved["entity_ref"]] = resolved
                edges.add((observation.entity_ref, resolved["entity_ref"],
                           "trace_resolves_to", "outgoing"))
            resolved_service = observation.details.get("resolved_service")
            if isinstance(resolved_service, dict):
                nodes[resolved_service["entity_ref"]] = resolved_service
                source = resolved["entity_ref"] if isinstance(resolved, dict) else observation.entity_ref
                edges.add((source, resolved_service["entity_ref"], "operation_of", "parent"))
            if isinstance(target, dict) and "entity_ref" in target:
                nodes[target["entity_ref"]] = target
                edges.add((observation.entity_ref, target["entity_ref"],
                           str(relation or observation.value), str(direction)))
        rules_by_signal: dict[str, list[str]] = {}
        for rule in registry.rules:
            rules_by_signal.setdefault(rule.signal, []).append(rule.rule_id)
        available_by_entity: dict[str, set[str]] = {}
        for signal in registry.signals:
            available_by_entity.setdefault(signal.entity_ref, set()).update(
                rules_by_signal.get(signal.signal, ())
            )
        return {
            "nodes": [
                {**node, "available_rule_ids": sorted(available_by_entity.get(ref, set()))}
                for ref, node in sorted(nodes.items())
            ],
            "edges": [
                {"source_entity_ref": source, "target_entity_ref": target,
                 "relation": relation, "direction": direction}
                for source, target, relation, direction in sorted(edges)
            ],
        }

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
            r.observation.details.get("trace_ref") == trace
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

    def feedback_plan_available(self) -> bool:
        if (self.status != EpisodeStatus.ACTIVE or self.__feedback_planned
                or self.__verification_planner is None or self.__resolved_registry is None
                or self.__verification_queries or self.__budget_remaining <= 0
                or self.__rounds + 2 >= self.__max_rounds):
            return False
        return self._feedback_plan_target() is not None

    def _feedback_plan_target(self):
        assert self.__resolved_registry is not None
        attempted = {event.payload["query_ref"] for event in self.transcript.events
                     if event.event_type == EventType.TOOL_REQUEST}
        registry = self.__resolved_registry
        reports = self.__decision_provider.verification_reports(
            tuple(self.__proposals), tuple(self.__observation_ledger.values())
        )
        for assessment in self.preview_core_assessment().candidate_assessments:
            report = reports.get(assessment.candidate_id)
            if (not assessment.verification_pending or report is None
                    or report.state.value == "failed"):
                continue
            for mechanism_id, candidate_id in self.__candidate_for_mechanism.items():
                if candidate_id != assessment.candidate_id:
                    continue
                used = {item.capability_ref for item in
                        self.__compiled_plans[mechanism_id].predictions}
                for handle in registry.capability_handles():
                    if handle.capability_ref in used:
                        continue
                    signal = next((item for item in registry.signals
                                   if item.entity_ref == handle.entity_ref
                                   and item.signal == handle.signal), None)
                    if signal is not None and self._signal_executable(signal) and (
                        query_fingerprint(signal.query.tool_name, signal.query.request)
                        not in attempted
                    ):
                        return mechanism_id, assessment
        return None

    def extend_verification_plan(self) -> dict[str, Any]:
        """Request one additive plan revision from the existing Planner."""
        if not self.feedback_plan_available():
            return {"status": "unavailable"}
        assert self.__resolved_registry is not None and self.__bootstrap is not None
        self.__feedback_planned = True
        self.__rounds += 1
        registry = self.__resolved_registry
        target = self._feedback_plan_target()
        if target is None:
            return {"status": "unavailable"}
        mechanism_id, assessment = target
        frozen = self.__frozen_mechanisms[mechanism_id]
        previous = self.__compiled_plans[mechanism_id]
        old_candidate_id = assessment.candidate_id
        old_compiled = self.__decision_provider.dynamic_compiled(old_candidate_id)
        assert old_compiled is not None
        reports = self.__decision_provider.verification_reports(
            tuple(self.__proposals), tuple(self.__observation_ledger.values())
        )
        blockers = reports[old_candidate_id].blockers if old_candidate_id in reports else ()
        feedback = {
            "candidate_id": old_candidate_id,
            "missing_signals": assessment.missing_signals,
            "unavailable_signals": assessment.unavailable_signals,
            "blockers": blockers,
        }
        self.transcript.append(EventType.VERIFICATION_PLAN_EXTENSION_STARTED, {
            "mechanism_id": mechanism_id,
            "old_candidate_id": old_candidate_id,
            "previous_plan": previous.model_dump(mode="json"),
            "core_feedback": feedback,
        })
        context = {
            "frozen_mechanisms": [frozen.model_dump(mode="json")],
            "capabilities": [item.model_dump(mode="json")
                             for item in registry.capability_handles()],
            "observations": [record.observation.model_dump(mode="json")
                             for record in self.__observation_ledger.values()],
            "existing_plan": previous.model_dump(mode="json"),
            "core_feedback": feedback,
            "unavailable_signals": {
                signal: reason for capability in self.__bootstrap.capabilities
                for signal, reason in capability.unavailable_signals.items()
            },
        }
        structured_extension = getattr(
            self.__verification_planner, "supports_plan_extension", False
        )
        ok, raw = self._invoke_discovery_component(
            (lambda: self.__verification_planner.plan_extension(deepcopy(context)))
            if structured_extension else
            (lambda: self.__verification_planner.plan(deepcopy(context)))
        )
        try:
            if not ok:
                raise ValueError("PLANNER_" + raw["error_type"].upper())
            if structured_extension:
                addition = VerificationPlanExtension.model_validate(raw)
                if addition.mechanism_id != mechanism_id:
                    raise ValueError("UNKNOWN_FROZEN_MECHANISM")
                proposed = VerificationPlanDraft(
                    mechanism_id=mechanism_id,
                    predictions=(*previous.predictions, *addition.added_predictions),
                )
            else:
                payload = VerificationPlanPayload.model_validate(raw)
                if len(payload.plans) != 1:
                    raise ValueError("MISSING_ADDITIVE_PLAN")
                proposed = payload.plans[0]
            plan, repairs = registry.normalize_plan(frozen, proposed)
            new_refs = added_predictions(previous, plan)
            compiled = registry.compile_plan(
                frozen, plan,
                discovery_query_refs=old_compiled.discovery_query_refs,
                discovery_observation_refs=old_compiled.discovery_observation_refs,
                discovery_fact_digests=old_compiled.discovery_fact_digests,
            )
            if not all(self._signal_executable(item) for item in compiled.signals):
                raise ValueError("CAPABILITY_NOT_EXECUTABLE")
            attempted = {event.payload["query_ref"] for event in self.transcript.events
                         if event.event_type == EventType.TOOL_REQUEST}
            old_queries = {query_fingerprint(item.query.tool_name, item.query.request)
                           for item in old_compiled.signals}
            new_queries = {
                query_fingerprint(item.query.tool_name, item.query.request): item.query
                for item in compiled.signals
                if query_fingerprint(item.query.tool_name, item.query.request)
                not in old_queries | attempted
            }
            query_cost = sum(
                next(capability.query_cost for capability in self.__bootstrap.capabilities
                     if capability.tool_name == action.tool_name)
                for action in new_queries.values()
            )
            if (not new_queries or self.__budget_remaining < query_cost
                    or self.__rounds + 2 * len(new_queries) > self.__max_rounds):
                raise ValueError("NO_NEW_EXECUTABLE_QUERY")
            if not set(new_refs) <= {item.capability_ref
                                    for item in registry.capability_handles()}:
                raise ValueError("UNKNOWN_CAPABILITY_REF")
        except (ValueError, TypeError) as exc:
            code = str(exc) if type(exc) is ValueError and str(exc).isupper() else (
                _contract_error_code(str(exc))
            )
            self.transcript.append(EventType.VERIFICATION_PLAN_EXTENSION_REJECTED, {
                "mechanism_id": mechanism_id, "error_code": code,
            })
            return {"status": "rejected", "reason": code}

        new_hypothesis_id = "h-discovered-" + compiled.version[:20]
        rebound = compiled.model_copy(update={"source_hypothesis_id": new_hypothesis_id})
        self.__decision_provider.supersede_dynamic(old_candidate_id, rebound)
        previous_proposal = next(item for item in self.__proposals
                                 if item.candidate_id == old_candidate_id)
        self.transcript.append(EventType.VERIFICATION_PLAN_EXTENDED, {
            "mechanism_id": mechanism_id,
            "mechanism_digest": frozen.digest,
            "old_candidate_id": old_candidate_id,
            "old_hypothesis_id": previous_proposal.hypothesis_id,
            "new_hypothesis_id": new_hypothesis_id,
            "previous_plan": previous.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "plan_repairs": repairs,
            "compiled": rebound.model_dump(mode="json"),
            "new_query_refs": tuple(new_queries),
        })
        self._submit_hypothesis(HypothesisProposal(
            hypothesis_id=new_hypothesis_id,
            candidate_id=rebound.candidate.candidate_id,
            mechanism=frozen.mechanism,
            causal_entity_ref=frozen.causal_entity_ref,
            rationale=frozen.causal_explanation,
            observation_refs=previous_proposal.observation_refs,
            missing_evidence=previous_proposal.missing_evidence,
        ))
        self.__verification_queries.extend((new_hypothesis_id, action)
                                           for action in new_queries.values())
        self.__compiled_plans[mechanism_id] = plan
        self.__candidate_for_mechanism[mechanism_id] = rebound.candidate.candidate_id
        self.__dynamic_options = [item for item in self.__dynamic_options
                                  if item["candidate_id"] != old_candidate_id]
        self.__dynamic_options.append({
            "candidate_id": rebound.candidate.candidate_id,
            "fault_type": rebound.candidate.fault_type,
            "family": rebound.candidate.family.value,
            "required_signals": tuple(item.signal for item in rebound.signals),
            "required_capabilities": tuple(rebound.candidate.required_capabilities),
        })
        return {"status": "extended", "mechanism_id": mechanism_id,
                "new_queries": len(new_queries)}

    def feedback_query_options(self) -> tuple[tuple[str, str, CallToolAction], ...]:
        """Unattempted frozen-plan queries for Core-reported evidence gaps."""
        available_slots = min(self.__budget_remaining,
                              (self.__max_rounds - self.__rounds) // 2)
        if (self.status != EpisodeStatus.ACTIVE or self.__verification_queries
                or available_slots <= 0
                or not isinstance(self.__decision_provider, CoreDecisionProvider)):
            return ()
        attempted = {
            event.payload["query_ref"] for event in self.transcript.events
            if event.event_type == EventType.TOOL_REQUEST
        }
        result = []
        for assessment in self.preview_core_assessment().candidate_assessments:
            compiled = self.__decision_provider.dynamic_compiled(assessment.candidate_id)
            if compiled is None or not assessment.verification_pending:
                continue
            gaps = {*assessment.missing_signals, *assessment.unavailable_signals}
            for signal in compiled.signals:
                if signal.signal not in gaps:
                    continue
                fingerprint = query_fingerprint(signal.query.tool_name, signal.query.request)
                if fingerprint in attempted:
                    continue
                for hypothesis_id in assessment.hypothesis_ids:
                    result.append((hypothesis_id, signal.signal, signal.query))
        return tuple(result[:available_slots])

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
        if isinstance(self.__decision_provider, CoreDecisionProvider):
            reports = self.__decision_provider.verification_reports(
                proposals, tuple(self.__observation_ledger[item] for item in used_observations),
            )
            if reports:
                self.transcript.append(EventType.DYNAMIC_VERIFICATION, {
                    "candidates": {key: report.model_dump(mode="json") for key, report in reports.items()},
                })
        frozen_at = datetime.now(UTC)
        candidate_assessments = self.preview_core_assessment().candidate_assessments
        fields = {
            "episode_id": self.episode_id,
            "termination_reason": termination_reason,
            "policy_proposals": proposals,
            "authoritative_decision": decision,
            "observation_refs": used_observations,
            "missing_evidence": missing,
            "inquiry_candidates": inquiry_outcomes(self.transcript.events, candidate_assessments),
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
