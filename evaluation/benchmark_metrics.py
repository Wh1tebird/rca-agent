"""Pure helpers for benchmark-only result metadata.

These helpers intentionally live outside ``src/rca_agent``.  They do not alter
episode semantics and keep unavailable provider metrics explicit.
"""
from __future__ import annotations

import json
from typing import Any

from rca_agent.tools.fingerprint import sha256_digest


def reasoning_mode(*, thinking_enabled: bool) -> str:
    return "high" if thinking_enabled else "disabled"


def classify_setup_exception(exc: Exception) -> bool:
    """Return whether an initialization exception is a structured setup error."""
    message = str(exc)
    return isinstance(exc, (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError)) and (
        "isoformat" in message
        or "case" in message.casefold()
        or "directory" in message.casefold()
    )


def unavailable_usage() -> dict[str, Any]:
    return {
        "input_tokens": "unavailable",
        "output_tokens": "unavailable",
        "total_tokens": "unavailable",
        "model_call_durations_seconds": "unavailable",
    }


def authoritative_allowed(*, replay_valid: bool, answer_frozen: bool) -> bool:
    """Gate conclusions on both immutable answer freeze and valid replay."""
    return replay_valid and answer_frozen


def benchmark_config_digest(*, provider: str, model: str, profile: str,
                            temperature: int, reasoning_mode: str,
                            query_budget: int, max_rounds: int,
                            inquiry_limit: int, retry_count: int) -> str:
    """Digest only public benchmark configuration; never include secrets or paths."""
    return sha256_digest({
        "provider": provider,
        "model": model,
        "profile": profile,
        "temperature": temperature,
        "reasoning_mode": reasoning_mode,
        "query_budget": query_budget,
        "max_rounds": max_rounds,
        "inquiry_limit": inquiry_limit,
        "retry_count": retry_count,
    })


def invalid_action_diagnostics(*, events: list[Any], provider: str,
                              profile: str, config_digest: str) -> list[dict[str, Any]]:
    """Project safe invalid-action diagnostics from immutable transcript events.

    Raw actions are intentionally absent. Unknown details remain ``UNATTRIBUTED``.
    """
    diagnostics: list[dict[str, Any]] = []
    for event in events:
        event_type = getattr(event, "event_type", None)
        event_type = getattr(event_type, "value", event_type)
        if event_type not in {"action_rejected", "policy_exception"}:
            continue
        payload = getattr(event, "payload", {})
        error_type = payload.get("error_type") if isinstance(payload, dict) else None
        category = "UNATTRIBUTED"
        if event_type == "policy_exception" and error_type == "ValidationError":
            category = "PROVIDER_ACTION_SCHEMA_ERROR"
        elif event_type == "action_rejected" and error_type in {
            "unissued_reference", "duplicate_hypothesis_id", "unknown_hypothesis_id",
            "candidate_evidence_admission_error", "profile_mapping_error",
        }:
            category = "ADMISSION_ERROR"
        diagnostics.append({
            "classification": category,
            "action_type": "unavailable",
            "action_id": "unavailable",
            "proposal_kind": "unavailable",
            "schema_validation_error": error_type if category == "PROVIDER_ACTION_SCHEMA_ERROR" else "unavailable",
            "conversion_error": error_type if category == "UNATTRIBUTED" and event_type == "policy_exception" else "unavailable",
            "admission_error": error_type if category == "ADMISSION_ERROR" else "unavailable",
            "invalid_field_names": [],
            "stop_reason": "policy_invalid_action",
            "phase": "admission" if event_type == "action_rejected" else "policy",
            "provider": provider,
            "profile": profile,
            "config_digest": config_digest,
        })
    return diagnostics

