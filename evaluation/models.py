from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.tools.fingerprint import sha256_digest


class EvaluationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(pattern=r"^ep-[a-f0-9]{32}$")
    answer_artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluator_version: str
    evaluated_at: datetime
    evaluation_status: str
    evaluator_record_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        return (
            sha256_digest(self.model_dump(mode="json", exclude={"artifact_digest"}))
            == self.artifact_digest
        )
