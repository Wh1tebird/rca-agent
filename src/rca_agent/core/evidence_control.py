from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidencePolarity(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"


class RouteStatus(StrEnum):
    ELIGIBLE = "eligible"
    UNKNOWN = "unknown"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_FAILED = "query_failed"


class DecisionStatus(StrEnum):
    CONFIRMED = "confirmed"
    INCONCLUSIVE = "inconclusive"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_FAILED = "query_failed"


class EvidenceAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    candidate: str
    polarity: EvidencePolarity
    strength: float = Field(gt=0.0, le=10.0)
    independent_group: str
    explanation: str
    query_ref: str


class FamilyRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RouteStatus
    eligible_candidates: list[str] = Field(default_factory=list)
    supported_families: list[str] = Field(default_factory=list)
    competing_families: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def eligible_route_requires_candidates(self) -> FamilyRoute:
        if self.status == RouteStatus.ELIGIBLE and not self.eligible_candidates:
            raise ValueError("eligible route requires candidates")
        return self


class CandidateEvidenceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: str
    support_strength: float
    contradict_strength: float
    net_score: float
    positive_groups: list[str]
    supporting_evidence: list[str]
    contradicting_evidence: list[str]


class EvidenceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DecisionStatus
    winner: str | None
    reason: str
    candidate_states: list[CandidateEvidenceState]
    winner_margin: float = Field(ge=0.0)


class EvidenceControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_positive_strength: float = Field(default=2.0, gt=0.0)
    minimum_positive_groups: int = Field(default=1, ge=1)
    minimum_winner_margin: float = Field(default=2.0, ge=0.0)


class EvidenceController:
    """Safety controller: ranking cannot substitute for candidate-owned proof."""

    def __init__(self, config: EvidenceControllerConfig | None = None) -> None:
        self._config = config or EvidenceControllerConfig()

    def decide(
        self,
        candidates: list[str],
        route: FamilyRoute,
        assertions: list[EvidenceAssertion],
    ) -> EvidenceDecision:
        self._validate(candidates, assertions)
        states = self._states(candidates, assertions)
        if route.status == RouteStatus.QUERY_FAILED:
            return self._terminal(DecisionStatus.QUERY_FAILED, "preflight query failed", states)
        if route.status == RouteStatus.OUT_OF_SCOPE:
            return self._terminal(
                DecisionStatus.OUT_OF_SCOPE,
                "preflight found a competing mechanism family outside the candidate set",
                states,
            )
        if route.status == RouteStatus.UNKNOWN:
            return self._terminal(
                DecisionStatus.INCONCLUSIVE,
                "preflight could not establish candidate-set applicability",
                states,
            )
        eligible = [state for state in states if state.candidate in route.eligible_candidates]
        qualified = [
            state
            for state in eligible
            if state.support_strength >= self._config.minimum_positive_strength
            and len(state.positive_groups) >= self._config.minimum_positive_groups
        ]
        if not qualified:
            return self._terminal(
                DecisionStatus.INCONCLUSIVE,
                "no candidate met its own minimum positive-evidence requirement",
                states,
            )
        ordered = sorted(qualified, key=lambda item: (-item.net_score, item.candidate))
        winner = ordered[0]
        runner_up = max(
            (item.net_score for item in eligible if item.candidate != winner.candidate),
            default=0.0,
        )
        margin = max(0.0, winner.net_score - runner_up)
        if winner.net_score <= 0 or margin < self._config.minimum_winner_margin:
            return EvidenceDecision(
                status=DecisionStatus.INCONCLUSIVE,
                winner=None,
                reason="positive evidence exists but conflict or margin prevents confirmation",
                candidate_states=states,
                winner_margin=margin,
            )
        return EvidenceDecision(
            status=DecisionStatus.CONFIRMED,
            winner=winner.candidate,
            reason="winner has candidate-owned positive evidence and passed the safety margin",
            candidate_states=states,
            winner_margin=margin,
        )

    @staticmethod
    def _validate(candidates: list[str], assertions: list[EvidenceAssertion]) -> None:
        if len(candidates) != len(set(candidates)) or not candidates:
            raise ValueError("candidates must be non-empty and unique")
        unknown = sorted({item.candidate for item in assertions} - set(candidates))
        if unknown:
            raise ValueError(f"evidence refers to unknown candidates: {unknown}")
        ids = [item.evidence_id for item in assertions]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")

    @staticmethod
    def _states(
        candidates: list[str], assertions: list[EvidenceAssertion]
    ) -> list[CandidateEvidenceState]:
        result = []
        for candidate in candidates:
            support = [
                item
                for item in assertions
                if item.candidate == candidate and item.polarity == EvidencePolarity.SUPPORT
            ]
            contradict = [
                item
                for item in assertions
                if item.candidate == candidate and item.polarity == EvidencePolarity.CONTRADICT
            ]
            support_strength = sum(item.strength for item in support)
            contradict_strength = sum(item.strength for item in contradict)
            result.append(
                CandidateEvidenceState(
                    candidate=candidate,
                    support_strength=support_strength,
                    contradict_strength=contradict_strength,
                    net_score=support_strength - contradict_strength,
                    positive_groups=sorted({item.independent_group for item in support}),
                    supporting_evidence=[item.evidence_id for item in support],
                    contradicting_evidence=[item.evidence_id for item in contradict],
                )
            )
        return sorted(result, key=lambda item: (-item.net_score, item.candidate))

    @staticmethod
    def _terminal(
        status: DecisionStatus,
        reason: str,
        states: list[CandidateEvidenceState],
    ) -> EvidenceDecision:
        return EvidenceDecision(
            status=status,
            winner=None,
            reason=reason,
            candidate_states=states,
            winner_margin=0.0,
        )
