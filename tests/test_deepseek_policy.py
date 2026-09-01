from pydantic import SecretStr

from rca_agent.deepseek_policy import (
    DEFAULT_DEEPSEEK_MODEL,
    create_deepseek_chat_model,
    create_deepseek_policy,
)
from rca_agent.langchain_policy import LangChainPolicyAdapter


def test_deepseek_model_uses_accuracy_first_agent_configuration():
    model = create_deepseek_chat_model(api_key=SecretStr("test-only-secret"))

    assert model.model_name == "deepseek-v4-pro"
    assert model.model_name == DEFAULT_DEEPSEEK_MODEL
    assert model.temperature == 0
    assert model.max_tokens == 8192
    assert model.max_retries == 2
    assert model.reasoning_effort == "high"
    assert model.extra_body == {"thinking": {"type": "enabled"}}
    assert "test-only-secret" not in repr(model)


def test_deepseek_policy_binds_to_closed_structured_output_without_network_access():
    policy = create_deepseek_policy(api_key=SecretStr("test-only-secret"))

    assert isinstance(policy, LangChainPolicyAdapter)


def test_deepseek_model_reads_default_environment_credential(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-environment-secret")

    model = create_deepseek_chat_model()

    assert model.api_key.get_secret_value() == "test-environment-secret"
    assert "test-environment-secret" not in repr(model)
