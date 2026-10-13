"""Run only the bounded inquiry stage for Catalog-holdout diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.unknown_benchmark import HoldoutAcquisitionPolicy
from evaluation.unknown_benchmark import parser as case_parser
from evaluation.unknown_benchmark_batch import mechanism_matches
from rca_agent.deepseek_policy import create_deepseek_chat_model
from rca_agent.environment import Rca100Environment
from rca_agent.langchain_acquisition import LangChainInquiryAdapter
from rca_agent.models import CallToolAction, RequestStopAction, SubmitHypothesisAction, ToolResponse
from rca_agent.runner import CoreDecisionProvider, EpisodeRunner


def _score(root_cause: str, drafts: list[dict[str, Any]]) -> dict[str, Any]:
    all_mechanisms = [
        hypothesis["mechanism"]
        for draft in drafts
        for hypothesis in draft["draft"]["hypotheses"]
    ]
    final_mechanisms = [
        hypothesis["mechanism"]
        for hypothesis in drafts[-1]["draft"]["hypotheses"]
    ] if drafts else []
    return {
        "any_inquiry_gt_direction": any(
            mechanism_matches(root_cause, mechanism) for mechanism in all_mechanisms
        ),
        "final_inquiry_gt_direction": any(
            mechanism_matches(root_cause, mechanism) for mechanism in final_mechanisms
        ),
        "matched_any_mechanisms": [
            mechanism for mechanism in all_mechanisms
            if mechanism_matches(root_cause, mechanism)
        ],
        "matched_final_mechanisms": [
            mechanism for mechanism in final_mechanisms
            if mechanism_matches(root_cause, mechanism)
        ],
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    defaults = case_parser().parse_args([])
    environment = Rca100Environment.from_case(
        defaults.cases_root, args.case_id, query_budget=defaults.query_budget,
    )
    runner = EpisodeRunner(
        environment,
        max_rounds=defaults.max_rounds,
        decision_provider=CoreDecisionProvider(frozenset({args.holdout_candidate})),
    )
    bootstrap = runner.start().model_dump(mode="json")
    inquiry = LangChainInquiryAdapter.from_chat_model(
        create_deepseek_chat_model(
            model=args.model,
            timeout_seconds=args.model_timeout_seconds,
            max_tokens=args.model_max_tokens,
            max_retries=0,
            thinking_enabled=args.thinking_enabled,
        ),
        structured_method="json_mode",
    )
    acquisition = HoldoutAcquisitionPolicy(inquiry=inquiry)
    history: list[dict[str, Any]] = []
    stopped_before_discovery = False
    error = None
    try:
        for _ in range(defaults.max_rounds):
            action = acquisition.invoke({
                "context": {"bootstrap": bootstrap, "history": tuple(history)}
            }).action
            if isinstance(action, SubmitHypothesisAction | RequestStopAction):
                stopped_before_discovery = True
                break
            if not isinstance(action, CallToolAction):
                error = type(action).__name__
                break
            response = runner.handle(action)
            serialized = (
                {"kind": "tool_response", "value": response.model_dump(mode="json")}
                if isinstance(response, ToolResponse)
                else {"kind": type(response).__name__ if response is not None else "none",
                      "value": None}
            )
            history.append({"action": action.model_dump(mode="json"), "result": serialized})
    except Exception as exc:  # noqa: BLE001 - diagnostic captures failure class only.
        error = type(exc).__name__
    return {
        "case_id": args.case_id,
        "holdout_candidate": args.holdout_candidate,
        "role": args.role,
        "provider": "deepseek",
        "model": args.model,
        "thinking_enabled": args.thinking_enabled,
        "model_timeout_seconds": args.model_timeout_seconds,
        "model_max_tokens": args.model_max_tokens,
        "stopped_before_discovery": stopped_before_discovery,
        "error": error,
        "inquiry_rounds": len(acquisition.inquiry_drafts),
        "selection_calls": acquisition.selection_calls,
        "selection_errors": acquisition.selection_errors,
        "selected_queries": len(acquisition.selected_actions),
        **_score(args.holdout_candidate, acquisition.inquiry_drafts),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--case-id", required=True)
    value.add_argument("--holdout-candidate", required=True)
    value.add_argument("--role")
    value.add_argument("--model", default="deepseek-flash")
    value.add_argument("--thinking-enabled", action="store_true")
    value.add_argument("--model-timeout-seconds", type=float, default=300)
    value.add_argument("--model-max-tokens", type=int, default=65536)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    result = run_case(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
