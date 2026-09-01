import pytest

from rca_agent.core.evidence_control import (
    DecisionStatus,
    EvidenceAssertion,
    EvidenceController,
    EvidencePolarity,
    FamilyRoute,
    RouteStatus,
)

CANDIDATES = ["A", "B"]


def _route(status: RouteStatus = RouteStatus.ELIGIBLE) -> FamilyRoute:
    return FamilyRoute(
        status=status,
        eligible_candidates=CANDIDATES if status == RouteStatus.ELIGIBLE else [],
    )


def _evidence(
    evidence_id: str,
    candidate: str,
    polarity: EvidencePolarity,
    strength: float = 2.0,
) -> EvidenceAssertion:
    return EvidenceAssertion(
        evidence_id=evidence_id,
        candidate=candidate,
        polarity=polarity,
        strength=strength,
        independent_group=evidence_id,
        explanation="test evidence",
        query_ref="query:test",
    )


def test_contradicting_b_does_not_support_a():
    decision = EvidenceController().decide(
        CANDIDATES,
        _route(),
        [_evidence("against-b", "B", EvidencePolarity.CONTRADICT)],
    )

    assert decision.status == DecisionStatus.INCONCLUSIVE
    assert decision.winner is None
    state_a = next(item for item in decision.candidate_states if item.candidate == "A")
    assert state_a.support_strength == 0


def test_candidate_can_win_only_with_its_own_positive_evidence():
    decision = EvidenceController().decide(
        CANDIDATES,
        _route(),
        [_evidence("for-a", "A", EvidencePolarity.SUPPORT)],
    )

    assert decision.status == DecisionStatus.CONFIRMED
    assert decision.winner == "A"


@pytest.mark.parametrize(
    ("route_status", "decision_status"),
    [
        (RouteStatus.UNKNOWN, DecisionStatus.INCONCLUSIVE),
        (RouteStatus.OUT_OF_SCOPE, DecisionStatus.OUT_OF_SCOPE),
        (RouteStatus.QUERY_FAILED, DecisionStatus.QUERY_FAILED),
    ],
)
def test_preflight_terminal_states_block_even_strong_positive_evidence(
    route_status: RouteStatus, decision_status: DecisionStatus
):
    decision = EvidenceController().decide(
        CANDIDATES,
        _route(route_status),
        [_evidence("for-a", "A", EvidencePolarity.SUPPORT, strength=10.0)],
    )

    assert decision.status == decision_status
    assert decision.winner is None


def test_conflicting_positive_evidence_is_inconclusive():
    decision = EvidenceController().decide(
        CANDIDATES,
        _route(),
        [
            _evidence("for-a", "A", EvidencePolarity.SUPPORT),
            _evidence("for-b", "B", EvidencePolarity.SUPPORT),
        ],
    )

    assert decision.status == DecisionStatus.INCONCLUSIVE
    assert decision.winner is None


def test_negative_net_candidate_cannot_win_despite_positive_evidence():
    decision = EvidenceController().decide(
        CANDIDATES,
        _route(),
        [
            _evidence("for-a", "A", EvidencePolarity.SUPPORT),
            _evidence("against-a", "A", EvidencePolarity.CONTRADICT, strength=4.0),
        ],
    )

    assert decision.status == DecisionStatus.INCONCLUSIVE
    assert decision.winner is None


def test_unknown_candidate_evidence_is_rejected():
    with pytest.raises(ValueError, match="unknown candidates"):
        EvidenceController().decide(
            CANDIDATES,
            _route(),
            [_evidence("for-c", "C", EvidencePolarity.SUPPORT)],
        )
