from __future__ import annotations

import pytest

from evaluation.unknown_benchmark import parser as case_parser
from evaluation.unknown_benchmark_batch import parser as batch_parser
from rca_agent.local_model import DEFAULT_LOCAL_MODEL, create_local_chat_model
from rca_agent.qwen_policy import create_qwen_discovery_pipeline, create_qwen_policy


def test_local_model_requires_environment_credential(monkeypatch):
    monkeypatch.delenv("RCA_LOCAL_API_KEY", raising=False)
    monkeypatch.setattr("rca_agent.local_model.sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError, match="RCA_LOCAL_API_KEY is not set"):
        create_local_chat_model()


def test_local_model_uses_openai_compatible_base_without_leaking_key(monkeypatch):
    monkeypatch.setenv("RCA_LOCAL_API_KEY", "test-only-local-secret")
    model = create_local_chat_model()

    assert model.model_name == DEFAULT_LOCAL_MODEL
    assert str(model.openai_api_base) == "http://127.0.0.1:11025/v1"
    assert model.temperature == 0
    assert model.max_tokens == 8192
    assert model.max_retries == 0
    assert "test-only-local-secret" not in repr(model)


def test_local_model_prompts_on_terminal_without_echoing_credential(monkeypatch):
    monkeypatch.delenv("RCA_LOCAL_API_KEY", raising=False)
    monkeypatch.setattr("rca_agent.local_model.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("rca_agent.local_model.getpass", lambda _prompt: "typed-test-secret")

    model = create_local_chat_model()

    assert model.openai_api_key.get_secret_value() == "typed-test-secret"
    assert "typed-test-secret" not in repr(model)


def test_benchmark_defaults_to_local_provider_without_key_access(monkeypatch):
    monkeypatch.delenv("RCA_LOCAL_API_KEY", raising=False)
    case_args = case_parser().parse_args([])
    batch_args = batch_parser().parse_args([])
    assert case_args.provider == "qwen-local"
    assert batch_args.provider == "qwen-local"
    assert case_args.model_timeout_seconds == 300
    assert case_args.model_max_tokens == 65536
    assert case_args.discovery_timeout_seconds == 300
    assert batch_args.model_timeout_seconds == 300
    assert batch_args.model_max_tokens == 65536
    assert batch_args.discovery_timeout_seconds == 300


def test_qwen_policy_and_discovery_bind_without_network(monkeypatch):
    monkeypatch.setenv("RCA_LOCAL_API_KEY", "test-only-local-secret")

    policy = create_qwen_policy()
    discovery, planner = create_qwen_discovery_pipeline()

    assert policy is not None
    assert discovery is not None
    assert planner is not None


def test_local_model_rejects_public_http_before_reading_credential(monkeypatch):
    monkeypatch.delenv("RCA_LOCAL_API_KEY", raising=False)
    monkeypatch.setattr("rca_agent.local_model.sys.stdin.isatty", lambda: False)

    with pytest.raises(ValueError, match="HTTPS or a loopback-only tunnel"):
        create_local_chat_model(base_url="https://provider.example/v1")


def test_public_http_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("RCA_LOCAL_API_KEY", "test-only-local-secret")
    model = create_local_chat_model(
        base_url="https://provider.example/v1", allow_insecure_http=True,
    )
    assert str(model.openai_api_base) == "https://provider.example/v1"
