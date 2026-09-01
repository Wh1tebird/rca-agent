from __future__ import annotations

from rca_agent.core.evidence_control import (
    DecisionStatus,
    EvidenceAssertion,
    EvidenceController,
    EvidencePolarity,
    FamilyRoute,
    RouteStatus,
)
from rca_agent.core.v3.capabilities import compile_typed_plans, resolve_conditions
from rca_agent.core.v3.execution import ObservationProvider
from rca_agent.core.v3.models import (
    Candidate,
    Comparison,
    ConditionPolarity,
    EvidenceAssessment,
    Incident,
    InvestigationRound,
    InvestigationStatus,
    Observation,
    ObservationCapability,
    OpenRcaResult,
)
from rca_agent.core.v3.proposal import ProposalGenerator, RuleProposalGenerator


class OpenRcaEngine:
    """Rule-first v3 loop; proposals cannot bypass the frozen v2 safety kernel."""

    def __init__(
        self,
        capabilities: list[ObservationCapability],
        proposal_generator: ProposalGenerator | None = None,
        max_rounds: int = 2,
    ) -> None:
        self._capabilities = capabilities
        self._proposal_generator = proposal_generator or RuleProposalGenerator()
        self._max_rounds = max_rounds

    def investigate(
        self, incident: Incident, provider: ObservationProvider
    ) -> OpenRcaResult:
        try:
            scout = provider.scout()
        except Exception as exc:  # noqa: BLE001 - query failure is a modeled terminal state
            return OpenRcaResult(
                incident=incident,
                capabilities=self._capabilities,
                proposal=self._empty_proposal("scout_query_failed"),
                condition_availability=[],
                rounds=[],
                status=InvestigationStatus.QUERY_FAILED,
                winner=None,
                supported_hypotheses=[],
                missing_evidence=[f"scout query failed: {type(exc).__name__}"],
                stop_reason="observation scout failed before candidate generation",
            )
        proposal = self._proposal_generator.propose(
            incident, self._capabilities, scout
        )
        availability = resolve_conditions(proposal.candidates, self._capabilities)
        executable = {item.condition_id for item in availability if item.executable}
        all_plans = compile_typed_plans(
            proposal.candidates,
            self._capabilities,
            incident.window_start,
            incident.window_end,
        )
        modality_by_capability = {
            item.capability_id: item.modality for item in self._capabilities
        }
        plan_groups = [
            [plan for plan in all_plans if modality_by_capability[plan.capability_id] == "metric"],
            [plan for plan in all_plans if modality_by_capability[plan.capability_id] != "metric"],
        ]
        assertions: list[EvidenceAssertion] = []
        rounds = []
        missing = [
            f"{item.condition_id}: {item.reason}" for item in availability if not item.executable
        ]
        final_decision = EvidenceController().decide(
            [item.candidate_id for item in proposal.candidates],
            self._route(proposal.candidates, assertions),
            assertions,
        ) if proposal.candidates else None
        for round_index, plans in enumerate(plan_groups[: self._max_rounds], start=1):
            if not plans:
                continue
            try:
                observations, unavailable = provider.execute(plans)
            except Exception as exc:  # noqa: BLE001 - query failure is a modeled terminal state
                failure = EvidenceController().decide(
                    [item.candidate_id for item in proposal.candidates],
                    FamilyRoute(status=RouteStatus.QUERY_FAILED),
                    assertions,
                )
                rounds.append(
                    InvestigationRound(
                        round_index=round_index,
                        plans=plans,
                        assessments=[],
                        decision=failure,
                    )
                )
                final_decision = failure
                missing.append(f"round query failed: {type(exc).__name__}")
                break
            missing.extend(unavailable)
            assessments = self._assess(proposal.candidates, observations, executable)
            assertions.extend(
                item.assertion for item in assessments if item.assertion is not None
            )
            final_decision = EvidenceController().decide(
                [item.candidate_id for item in proposal.candidates],
                self._route(proposal.candidates, assertions),
                assertions,
            )
            rounds.append(
                InvestigationRound(
                    round_index=round_index,
                    plans=plans,
                    assessments=assessments,
                    decision=final_decision,
                )
            )
            if final_decision.status == DecisionStatus.CONFIRMED:
                break
        if final_decision is None:
            status = InvestigationStatus.OUT_OF_SCOPE
            winner = None
            reason = "candidate generator produced no capability-grounded candidates"
            supported = []
        else:
            status = self._status(
                final_decision.status,
                bool(rounds),
                bool(missing),
                any(item.support_strength > 0 for item in final_decision.candidate_states),
            )
            winner = final_decision.winner
            reason = final_decision.reason
            supported = [
                item.candidate
                for item in final_decision.candidate_states
                if item.support_strength > 0
            ]
        return OpenRcaResult(
            incident=incident,
            capabilities=self._capabilities,
            proposal=proposal,
            condition_availability=availability,
            rounds=rounds,
            status=status,
            winner=winner,
            supported_hypotheses=supported,
            missing_evidence=sorted(set(missing)),
            stop_reason=reason,
        )

    @staticmethod
    def _route(
        candidates: list[Candidate], assertions: list[EvidenceAssertion]
    ) -> FamilyRoute:
        if not candidates:
            return FamilyRoute(status=RouteStatus.OUT_OF_SCOPE)
        support_ids = {
            item.evidence_id.removeprefix("v3-")
            for item in assertions
            if item.polarity == EvidencePolarity.SUPPORT
        }
        eligible = [
            item.candidate_id
            for item in candidates
            if set(item.positive_evidence_requirements) <= support_ids
        ]
        if not eligible:
            return FamilyRoute(
                status=RouteStatus.UNKNOWN,
                supported_families=sorted({item.family for item in candidates}),
                reasons=["no candidate has completed all required positive evidence"],
            )
        return FamilyRoute(
            status=RouteStatus.ELIGIBLE,
            eligible_candidates=eligible,
            supported_families=sorted({item.family for item in candidates}),
            reasons=["candidate-owned required positive evidence is complete"],
        )

    @staticmethod
    def _assess(
        candidates: list[Candidate],
        observations: list[Observation],
        executable: set[str],
    ) -> list[EvidenceAssessment]:
        observations_by_signal = {item.signal: item for item in observations}
        assessments = []
        for candidate in candidates:
            for condition in candidate.conditions:
                if condition.condition_id not in executable:
                    continue
                observation = observations_by_signal.get(condition.signal)
                if observation is None:
                    assessments.append(
                        EvidenceAssessment(
                            condition_id=condition.condition_id,
                            candidate_id=candidate.candidate_id,
                            matched=False,
                            assertion=None,
                            observation=None,
                            unavailable_reason="typed query returned no observation",
                        )
                    )
                    continue
                matched = _matches(condition.comparison, observation.value, condition.threshold)
                assertion = None
                if matched:
                    assertion = EvidenceAssertion(
                        evidence_id=f"v3-{condition.condition_id}",
                        candidate=candidate.candidate_id,
                        polarity=(
                            EvidencePolarity.SUPPORT
                            if condition.polarity == ConditionPolarity.SUPPORT
                            else EvidencePolarity.CONTRADICT
                        ),
                        strength=condition.strength,
                        independent_group=condition.independent_group,
                        explanation=condition.reason,
                        query_ref=observation.query_ref,
                    )
                assessments.append(
                    EvidenceAssessment(
                        condition_id=condition.condition_id,
                        candidate_id=candidate.candidate_id,
                        matched=matched,
                        assertion=assertion,
                        observation=observation,
                    )
                )
        return assessments

    @staticmethod
    def _status(
        decision: DecisionStatus,
        had_round: bool,
        has_missing: bool,
        has_support: bool,
    ) -> InvestigationStatus:
        if decision == DecisionStatus.CONFIRMED:
            return InvestigationStatus.CONFIRMED_ROOT_CAUSE
        if decision == DecisionStatus.OUT_OF_SCOPE:
            return InvestigationStatus.OUT_OF_SCOPE
        if decision == DecisionStatus.QUERY_FAILED:
            return InvestigationStatus.QUERY_FAILED
        if had_round and has_missing and has_support:
            return InvestigationStatus.SUPPORTED_HYPOTHESIS
        if had_round:
            return InvestigationStatus.INCONCLUSIVE
        return InvestigationStatus.CONTINUE_INVESTIGATION

    @staticmethod
    def _empty_proposal(generator: str):
        from rca_agent.core.v3.models import CandidateProposal

        return CandidateProposal(generator=generator, candidates=[])


def _matches(comparison: Comparison, value: float | str, threshold: float | str) -> bool:
    if comparison == Comparison.GTE:
        return (
            isinstance(value, (int, float))
            and isinstance(threshold, (int, float))
            and value >= threshold
        )
    if comparison == Comparison.LTE:
        return isinstance(value, (int, float)) and isinstance(threshold, (int, float)) and value <= threshold
    if comparison == Comparison.EQ:
        return str(value) == str(threshold)
    if comparison == Comparison.CONTAINS:
        return str(threshold).casefold() in str(value).casefold()
    raise ValueError(f"unsupported comparison: {comparison}")
