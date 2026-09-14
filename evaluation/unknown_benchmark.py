"""Run one post-freeze RCA100 Catalog-holdout experiment with real discovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from evaluation.ground_truth import FlatGroundTruthRepository
from rca_agent.core.runtime_registry import rca100_evidence_registry
from rca_agent.deepseek_policy import create_deepseek_discovery
from rca_agent.environment import Rca100Environment
from rca_agent.langchain_policy import LangChainPolicyAdapter, PolicyEnvelope
from rca_agent.langgraph_orchestrator import LangGraphInvestigation
from rca_agent.models import (
    CallToolAction,
    GetServiceTraceSummaryRequest,
    HypothesisProposal,
    RequestStopAction,
    SubmitHypothesisAction,
    ToolName,
)
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner, EventType, RecordedReplay


class HoldoutAcquisitionPolicy(Runnable[dict[str, Any], PolicyEnvelope]):
    """Acquire one trace reference, then force a genuinely unmatched hypothesis."""

    def __init__(self) -> None:
        self._step = 0

    def invoke(self, input: dict[str, Any], config: Any = None, **kwargs: Any) -> PolicyEnvelope:
        del config, kwargs
        context = input["context"]
        if self._step == 0:
            self._step += 1
            bootstrap = context["bootstrap"]
            return PolicyEnvelope(action=CallToolAction(
                tool_name=ToolName.GET_SERVICE_TRACE_SUMMARY,
                request=GetServiceTraceSummaryRequest(
                    entity_ref=bootstrap["alert_entity"]["entity_ref"],
                    window=bootstrap["observation_horizon"],
                    min_duration_ms=100,
                    max_traces=1,
                ),
            ))
        if self._step == 1:
            self._step += 1
            bootstrap = context["bootstrap"]
            return PolicyEnvelope(action=SubmitHypothesisAction(
                proposal=HypothesisProposal(
                    hypothesis_id="h-catalog-holdout",
                    candidate_id="catalogHoldoutUnknown",
                    mechanism="unexplained operation latency",
                    causal_entity_ref=bootstrap["alert_entity"]["entity_ref"],
                    rationale=(
                        "The bounded slow-operation alert is not assigned to an available Catalog "
                        "mechanism. Discover causal mechanisms testable by the issued trace."
                    ),
                )
            ))
        return PolicyEnvelope(action=RequestStopAction(
            summary="The bounded holdout investigation has completed available verification."
        ))


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    environment = Rca100Environment.from_case(
        args.cases_root, args.case_id, query_budget=args.query_budget,
    )
    runner = EpisodeRunner(
        environment,
        max_rounds=args.max_rounds,
        decision_provider=CoreDecisionProvider(frozenset({args.holdout_candidate})),
        discovery=create_deepseek_discovery(
            model=args.model,
            timeout_seconds=args.model_timeout_seconds,
            thinking_enabled=not args.disable_thinking,
        ),
        evidence_registry=rca100_evidence_registry,
        discovery_timeout_seconds=args.discovery_timeout_seconds,
    )
    policy = LangChainPolicyAdapter(HoldoutAcquisitionPolicy())
    report = LangGraphInvestigation(runner, policy).run(recursion_limit=args.recursion_limit)
    replay = RecordedReplay.verify(runner.transcript.events)

    # Ground truth is loaded only after AnswerArtifact freeze and replay verification.
    truth = FlatGroundTruthRepository(args.answer_key_root).load(args.case_id)
    decision = report.answer.authoritative_decision
    winner = next(
        (p for p in report.answer.policy_proposals
         if p.hypothesis_id == decision.winning_hypothesis_id),
        None,
    )
    discovery = [event.payload for event in runner.transcript.events
                 if event.event_type == EventType.DISCOVERY_COMPLETED]
    registered = [event.payload for event in runner.transcript.events
                  if event.event_type == EventType.DYNAMIC_CANDIDATE_REGISTERED]
    verification = [event.payload for event in runner.transcript.events
                    if event.event_type == EventType.DYNAMIC_VERIFICATION]
    return {
        "case_id": args.case_id,
        "holdout_candidate": args.holdout_candidate,
        "model": args.model,
        "thinking_enabled": not args.disable_thinking,
        "catalog_candidate_visible": args.holdout_candidate not in runner.excluded_catalog_candidate_ids,
        "status": decision.status.value,
        "winner": winner.model_dump(mode="json") if winner else None,
        "dynamic_mechanisms": [item["compiled"]["candidate"]["fault_type"] for item in registered],
        "dynamic_plans": [{
            "mechanism": item["compiled"]["candidate"]["fault_type"],
            "alternatives": item["compiled"]["alternatives"],
            "predictions": item["compiled"]["predictions"],
        } for item in registered],
        "verification": verification[-1] if verification else None,
        "discovery": discovery[-1] if discovery else None,
        "ground_truth": {
            "root_cause_types": truth.root_cause_types,
            "root_cause_entities": truth.root_cause_entities,
            "expected_fault_id": truth.expected_fault_id,
        },
        "graph_steps": report.graph_steps,
        "tool_calls": replay.tool_calls,
        "replay_valid": replay.valid,
    }


def parser() -> argparse.ArgumentParser:
    default_dataset = Path.home() / "study/AgenticOpsEval/RCA100"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case-id", default="t025")
    value.add_argument("--holdout-candidate", default="dbNetworkLatency")
    value.add_argument("--cases-root", type=Path,
                       default=Path(os.getenv("RCA100_CASES_ROOT", default_dataset / "cases")))
    value.add_argument("--answer-key-root", type=Path,
                       default=Path(os.getenv("RCA100_ANSWER_KEY_ROOT",
                                               default_dataset / "answer_key")))
    value.add_argument("--model", default="deepseek-v4-pro")
    value.add_argument("--disable-thinking", action="store_true")
    value.add_argument("--query-budget", type=int, default=20)
    value.add_argument("--max-rounds", type=int, default=30)
    value.add_argument("--recursion-limit", type=int, default=100)
    value.add_argument("--model-timeout-seconds", type=float, default=45)
    value.add_argument("--discovery-timeout-seconds", type=float, default=50)
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run_case(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
