from evaluation.unknown_benchmark_batch import mechanism_matches, score_case, summarize
from rca_agent.models import DecisionStatus


def result(*, status="supported_hypothesis", mechanism="SQL execution dominates latency"):
    return {
        "case_id": "t-test",
        "holdout_candidate": "slowSQL",
        "status": status,
        "dynamic_mechanisms": [mechanism],
        "winner": {"mechanism": mechanism} if status == "discovered_verified" else None,
        "discovery": {"status": "completed"},
        "ground_truth": {"root_cause_types": ["slowSQL"]},
        "replay_valid": True,
    }


def test_post_freeze_mechanism_scorer_is_conservative_about_connection_pool():
    assert DecisionStatus.QUERY_FAILED.value == "query_failed"
    assert mechanism_matches("slowSQL", "SQL execution dominates operation latency")
    assert mechanism_matches("dbNetworkLatency", "Network latency between app and database")
    assert not mechanism_matches("dbNetworkLatency", "Database connection pool exhaustion")


def test_batch_score_separates_proposal_recall_from_wrong_verification():
    proposed = score_case(result())
    assert proposed["gt_mechanism_proposed"]
    assert not proposed["discovered_verified"]

    wrong = score_case(result(
        status="discovered_verified",
        mechanism="Database connection pool exhaustion",
    ))
    assert wrong["wrong_verified"]
    assert not wrong["correct_verified"]
    aggregate = summarize([proposed, wrong], [])
    assert aggregate["gt_mechanism_proposed"] == 1
    assert aggregate["wrong_verified"] == 1
