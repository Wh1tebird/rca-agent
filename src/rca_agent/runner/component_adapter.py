"""JSON-only component adapter specs and worker-backed adapter shims.

The shims preserve the existing adapter interfaces used by ``EpisodeRunner``.
Only the spec and typed call context cross the process boundary; validation and
all episode mutations remain in the parent.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rca_agent.core.dynamic import (
    DiscoveryPayload,
    MechanismDiscoveryPayload,
    VerificationPlanExtension,
    VerificationPlanPayload,
)
from rca_agent.investigation.discovery_format import DiscoveryFormatRepairContext
from rca_agent.investigation.formalizer import (
    MechanismFormalizationContext,
    MechanismFormalizationPayload,
)
from rca_agent.runner.component_worker import (
    ComponentCallContext,
    ComponentWorkerHost,
    ComponentWorkerSpec,
    WorkerCrashedError,
    WorkerExecutionError,
    WorkerTimeoutError,
)

ComponentName = Literal["discovery", "formalizer", "planner"]


class ComponentAdapterSpec(BaseModel):
    """Serializable child-side factory selection; no Runnable or model fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: ComponentName
    factory_id: str = Field(min_length=1, max_length=120)
    config: dict[str, Any] = Field(default_factory=dict)
    protocol_version: int = Field(default=1, ge=1)

    @field_validator("config")
    @classmethod
    def _json_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        import json

        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter config must be JSON serializable") from exc
        return value

    def worker_spec(self) -> ComponentWorkerSpec:
        return ComponentWorkerSpec(
            worker_type="component",
            factory_id=self.factory_id,
            config=self.config,
            protocol_version=self.protocol_version,
        )


class ComponentAdapterCallError(RuntimeError):
    """Failure raised by a worker-backed adapter before Runner admission."""

    def __init__(self, error_type: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.details = details or {}


class ComponentAdapterHost:
    """Run exactly one typed component call in a fresh, reaped worker."""

    def __init__(
        self,
        spec: ComponentAdapterSpec,
        *,
        timeout_seconds: float = 1.0,
        terminate_grace_seconds: float = 0.2,
    ) -> None:
        self.spec = spec
        self.timeout_seconds = timeout_seconds
        self.terminate_grace_seconds = terminate_grace_seconds

    def call(self, operation: str, context: dict[str, Any]) -> dict[str, Any]:
        call = ComponentCallContext(
            component=self.spec.component,
            operation=operation,
            context=context,
        )
        worker = ComponentWorkerHost(
            self.spec.worker_spec(),
            ready_timeout_seconds=self.timeout_seconds,
            request_timeout_seconds=self.timeout_seconds,
            terminate_grace_seconds=self.terminate_grace_seconds,
        )
        try:
            worker.start()
            return worker.request({"call": call.model_dump(mode="json")})
        except WorkerTimeoutError as exc:
            raise ComponentAdapterCallError("TimeoutError") from exc
        except WorkerExecutionError as exc:
            raise ComponentAdapterCallError(
                str(exc.details.get("error_type", "RuntimeError")),
                exc.details,
            ) from exc
        except WorkerCrashedError as exc:
            raise ComponentAdapterCallError("WorkerCrashedError") from exc
        finally:
            if worker.is_alive:
                worker.terminate()


class WorkerBackedPlannerAdapter:
    """Planner interface backed by a JSON-only component worker."""

    worker_backed = True

    def __init__(self, spec: ComponentAdapterSpec, *, timeout_seconds: float = 1.0) -> None:
        if spec.component != "planner":
            raise ValueError("planner adapter requires planner spec")
        self._host = ComponentAdapterHost(spec, timeout_seconds=timeout_seconds)
        self.supports_plan_extension = bool(spec.config.get("supports_plan_extension", False))

    def plan(self, context: dict[str, Any]) -> VerificationPlanPayload:
        try:
            return VerificationPlanPayload.model_validate(self._host.call("plan", context))
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc

    def plan_extension(self, context: dict[str, Any]) -> VerificationPlanExtension:
        try:
            return VerificationPlanExtension.model_validate(
                self._host.call("plan_extension", context)
            )
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc


class WorkerBackedDiscoveryAdapter:
    """Discovery interface backed by a JSON-only component worker."""

    worker_backed = True

    def __init__(self, spec: ComponentAdapterSpec, *, timeout_seconds: float = 1.0) -> None:
        if spec.component != "discovery":
            raise ValueError("discovery adapter requires discovery spec")
        self._host = ComponentAdapterHost(spec, timeout_seconds=timeout_seconds)
        self.supports_bound_candidates = bool(spec.config.get("supports_bound_candidates", False))

    def discover(self, context: dict[str, Any]) -> DiscoveryPayload | MechanismDiscoveryPayload:
        try:
            response = self._host.call("discover", context)
            try:
                return DiscoveryPayload.model_validate(response)
            except ValueError:
                return MechanismDiscoveryPayload.model_validate(response)
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc

    def formalize_candidate(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._host.call("formalize_candidate", context)
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc


class WorkerBackedMechanismFormalizerAdapter:
    """Mechanism formalizer interface backed by a JSON-only worker."""

    supports_candidate_accounting = True
    worker_backed = True

    def __init__(self, spec: ComponentAdapterSpec, *, timeout_seconds: float = 1.0) -> None:
        if spec.component != "formalizer":
            raise ValueError("formalizer adapter requires formalizer spec")
        self._host = ComponentAdapterHost(spec, timeout_seconds=timeout_seconds)

    def formalize(self, context: MechanismFormalizationContext) -> MechanismFormalizationPayload:
        try:
            return MechanismFormalizationPayload.model_validate(
                self._host.call("formalize", context.model_dump(mode="json"))
            )
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc

    def retry_formalize(
        self,
        context: MechanismFormalizationContext,
        previous: MechanismFormalizationPayload,
    ) -> MechanismFormalizationPayload:
        try:
            return MechanismFormalizationPayload.model_validate(
                self._host.call(
                    "retry_formalize",
                    {
                        "context": context.model_dump(mode="json"),
                        "previous": previous.model_dump(mode="json"),
                    },
                )
            )
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc


class WorkerBackedDiscoveryFormatFormalizerAdapter:
    worker_backed = True

    def __init__(self, spec: ComponentAdapterSpec, *, timeout_seconds: float = 1.0) -> None:
        if spec.component != "formalizer":
            raise ValueError("format formalizer requires formalizer spec")
        self._host = ComponentAdapterHost(spec, timeout_seconds=timeout_seconds)

    def repair(self, context: DiscoveryFormatRepairContext) -> MechanismDiscoveryPayload:
        try:
            return MechanismDiscoveryPayload.model_validate(
                self._host.call("repair", context.model_dump(mode="json"))
            )
        except ComponentAdapterCallError as exc:
            raise _runner_error(exc) from exc


def _runner_error(error: ComponentAdapterCallError) -> RuntimeError:
    if error.error_type == "TimeoutError":
        return TimeoutError("component worker request timed out")
    return RuntimeError(error.error_type)
