from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evaluation.benchmark_metrics import (
    authoritative_allowed,
    benchmark_config_digest,
    classify_setup_exception,
    invalid_action_diagnostics,
    reasoning_mode,
    unavailable_usage,
)


def test_missing_case_metadata_is_case_setup_error():
    assert classify_setup_exception(ValueError("Invalid isoformat string: ''"))
    assert not classify_setup_exception(RuntimeError("provider unavailable"))


def test_unavailable_usage_is_explicit_and_never_estimated():
    assert unavailable_usage() == {
        "input_tokens": "unavailable",
        "output_tokens": "unavailable",
        "total_tokens": "unavailable",
        "model_call_durations_seconds": "unavailable",
    }


def test_reasoning_mode_records_supported_value_only():
    assert reasoning_mode(thinking_enabled=False) == "disabled"
    assert reasoning_mode(thinking_enabled=True) == "high"


@pytest.mark.parametrize(
    ("replay_valid", "answer_frozen", "allowed"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_authoritative_conclusion_requires_freeze_and_valid_replay(
    replay_valid: bool, answer_frozen: bool, allowed: bool
):
    assert authoritative_allowed(
        replay_valid=replay_valid, answer_frozen=answer_frozen
    ) is allowed


@pytest.mark.parametrize("status", ["OUT_OF_SCOPE", "QUERY_FAILED", "INCONCLUSIVE"])
def test_terminal_statuses_are_distinct(status):
    payload = {"run_status": "completed", "final_status": status, "failure_taxonomy": []}
    assert payload["final_status"] == status


def test_case_setup_error_output_keeps_case_and_raw_error(monkeypatch, capsys, tmp_path: Path):
    from evaluation import unknown_benchmark

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    output = tmp_path / "t001.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unknown_benchmark",
            "--case-id",
            "t001",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-flash",
            "--output",
            str(output),
        ],
    )
    unknown_benchmark.main()
    result = json.loads(output.read_text())
    assert result["case_id"] == "t001"
    assert result["run_status"] == "CASE_SETUP_ERROR"
    assert result["failure_taxonomy"] == ["CASE_SETUP_ERROR"]
    assert result["error_type"] == "ValueError"
    assert "isoformat" in result["error_message"]
    assert result["authoritative_conclusion"] is None
    assert result["total_tokens"] == "unavailable"


class _Event:
    def __init__(self, event_type: str, payload: dict):
        self.event_type = event_type
        self.payload = payload


def test_invalid_action_diagnostics_classify_schema_and_admission_safely():
    digest = benchmark_config_digest(
        provider="deepseek", model="deepseek-flash", profile="fixed",
        temperature=0, reasoning_mode="disabled", query_budget=20,
        max_rounds=20, inquiry_limit=10, retry_count=0,
    )
    result = invalid_action_diagnostics(
        events=[
            _Event("policy_exception", {"error_type": "ValidationError", "payload": {"secret": "x"}}),
            _Event("action_rejected", {"error_type": "unissued_reference", "payload_digest": "abc"}),
        ], provider="deepseek", profile="fixed", config_digest=digest,
    )
    assert result[0]["classification"] == "PROVIDER_ACTION_SCHEMA_ERROR"
    assert result[1]["classification"] == "ADMISSION_ERROR"
    assert all("secret" not in str(item) for item in result)
    assert all(item["config_digest"] == digest for item in result)


def test_outer_policy_exception_is_not_misclassified_as_admission():
    result = invalid_action_diagnostics(
        events=[
            _Event("action_rejected", {"error_type": "RuntimeError"}),
            _Event("policy_exception", {"error_type": "RuntimeError"}),
        ], provider="deepseek", profile="fixed", config_digest="d" * 64,
    )
    assert [item["classification"] for item in result] == [
        "UNATTRIBUTED", "UNATTRIBUTED"
    ]


@pytest.mark.parametrize("error_type", ["ValueError", "TypeError", None])
def test_invalid_action_diagnostics_fallback_is_unattributed(error_type):
    result = invalid_action_diagnostics(
        events=[_Event("policy_exception", {"error_type": error_type})],
        provider="deepseek", profile="fixed", config_digest="d" * 64,
    )
    assert result[0]["classification"] == "UNATTRIBUTED"
    assert result[0]["action_type"] == "unavailable"
