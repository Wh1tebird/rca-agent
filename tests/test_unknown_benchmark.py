from pathlib import Path

from evaluation.capability_audit import build_matrix, render_markdown
from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy
from evaluation.unknown_benchmark_batch import mechanism_matches, score_case, summarize
from rca_agent.core.dynamic import DiscoveryPayload
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.environment import Rca100Environment
from rca_agent.langchain_policy import LangChainPolicyAdapter
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import DecisionStatus
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, RecordedReplay


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
    assert mechanism_matches("threadExhaustion", "Application thread pool saturation")
    assert mechanism_matches("memoryPressure", "High JVM heap memory pressure")
    assert mechanism_matches("fullGC", "Long garbage collection pause")


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


def test_capability_audit_keeps_missing_network_and_pool_signals_explicit(
    cases_root: Path,
):
    matrix = build_matrix(cases_root.parent / "answer_key", cases_root)
    mechanisms = {item["mechanism"]: item for item in matrix["mechanisms"]}

    assert mechanisms["slowSQL"]["queryability"] == "full"
    assert mechanisms["dbNetworkLatency"]["queryability"] == "partial"
    assert "db_network_rtt_ms" in mechanisms["dbNetworkLatency"]["missing"]
    assert mechanisms["connectionPoolExhaustion"]["queryability"] == "partial"
    assert "connection_pool_max" in mechanisms["connectionPoolExhaustion"]["missing"]
    assert mechanisms["threadExhaustion"]["queryability"] == "partial"
    tools = {item["tool_name"]: item for item in matrix["tools"]}
    assert tools["get_service_trace_summary"]["supported_entity_types"] == [
        "apm.operation", "apm.service",
    ]
    signals = {
        (item["entity_type"], item["signal"]): item for item in matrix["signals"]
    }
    assert signals[("k8s.node", "cpu_usage_rate")]["runtime_registered"]
    assert signals[("apm.service", "memory_usage")]["raw_data_case_count"] > 0
    assert signals[("apm.operation", "sql_shape")]["raw_data_case_count"] > 0
    assert signals[("apm.service", "thread_pool_active")]["runtime_registered"]
    assert signals[("apm.service", "jvm_gc_pause_seconds_delta")][
        "raw_data_case_count"
    ] == matrix["dataset_case_count"]
    rendered = render_markdown(matrix)
    assert "not network RTT" in rendered
    assert "not database-server" in rendered


def test_trace_resolved_service_advertises_jvm_rules_to_discovery(cases_root: Path):
    class CaptureDiscovery:
        def discover(self, context):
            self.context = context
            return DiscoveryPayload()

    discovery = CaptureDiscovery()
    runner = EpisodeRunner(
        Rca100Environment.from_case(cases_root, "t021", query_budget=20),
        max_rounds=30,
        decision_provider=CoreDecisionProvider(frozenset({"slowSQL"})),
        discovery=discovery,
        evidence_registry=rca100_evidence_registry,
    )
    report = LangGraphInvestigation(
        runner, LangChainPolicyAdapter(HoldoutAcquisitionPolicy())
    ).run(recursion_limit=100)

    service_nodes = [
        node for node in discovery.context["entity_graph"]["nodes"]
        if node["entity_type"] == "apm.service"
    ]
    assert service_nodes
    advertised = set().union(*(node["available_rule_ids"] for node in service_nodes))
    assert {
        "thread_pool_active", "thread_pool_queue_depth", "thread_pool_rejected_tasks",
        "jvm_runnable_threads", "jvm_gc_collection_delta", "jvm_gc_pause_seconds_delta",
    } <= advertised
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert report.answer is not None
