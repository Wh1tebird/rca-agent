"""Host-only configuration. These paths are never returned from the API."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rca_agent.local_model import DEFAULT_LOCAL_BASE_URL, DEFAULT_LOCAL_MODEL


@dataclass(frozen=True, slots=True)
class Settings:
    cases_root: Path
    runs_root: Path
    live_workers: int = 1
    qwen_model: str = DEFAULT_LOCAL_MODEL
    qwen_base_url: str = DEFAULT_LOCAL_BASE_URL
    qwen_allow_insecure_http: bool = False
    qwen_timeout_seconds: float = 300
    qwen_discovery_timeout_seconds: float = 300
    qwen_query_budget: int = 20
    qwen_max_rounds: int = 20
    qwen_recursion_limit: int = 100

    @classmethod
    def from_environment(cls) -> Settings:
        default_dataset = Path.home() / "study" / "AgenticOpsEval" / "RCA100" / "cases"
        default_runs = Path(__file__).resolve().parents[1] / "var" / "runs"
        workers = int(os.getenv("RCA_AGENT_LIVE_WORKERS", "1"))
        if not 1 <= workers <= 4:
            raise ValueError("RCA_AGENT_LIVE_WORKERS must be between one and four")
        settings = cls(
            cases_root=Path(os.getenv("RCA100_CASES_ROOT", default_dataset)).resolve(),
            runs_root=Path(os.getenv("RCA_AGENT_RUNS_ROOT", default_runs)).resolve(),
            live_workers=workers,
            qwen_model=os.getenv("RCA_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
            qwen_base_url=os.getenv("RCA_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL),
            qwen_allow_insecure_http=_boolean_env("RCA_LOCAL_ALLOW_INSECURE_HTTP", False),
            qwen_timeout_seconds=float(os.getenv("RCA_LOCAL_TIMEOUT_SECONDS", "300")),
            qwen_discovery_timeout_seconds=float(
                os.getenv("RCA_AGENT_DISCOVERY_TIMEOUT_SECONDS", "300")
            ),
            qwen_query_budget=int(os.getenv("RCA_AGENT_QUERY_BUDGET", "20")),
            qwen_max_rounds=int(os.getenv("RCA_AGENT_MAX_ROUNDS", "20")),
            qwen_recursion_limit=int(os.getenv("RCA_AGENT_RECURSION_LIMIT", "100")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 1 <= self.live_workers <= 4:
            raise ValueError("live_workers must be between one and four")
        if not 1 <= self.qwen_query_budget <= 100:
            raise ValueError("qwen_query_budget must be between one and one hundred")
        if not 1 <= self.qwen_max_rounds <= 100:
            raise ValueError("qwen_max_rounds must be between one and one hundred")
        if not 1 <= self.qwen_recursion_limit <= 500:
            raise ValueError("qwen_recursion_limit must be between one and five hundred")
        if not 1 <= self.qwen_timeout_seconds <= 600:
            raise ValueError("qwen_timeout_seconds must be between one and six hundred")
        if not 1 <= self.qwen_discovery_timeout_seconds <= 600:
            raise ValueError(
                "qwen_discovery_timeout_seconds must be between one and six hundred"
            )


def _boolean_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")
