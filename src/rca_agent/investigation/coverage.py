"""Non-authoritative mechanism-family coverage diagnostics.

The policy is deliberately narrower than candidate generation: it only reports
that a relevant investigation domain has no live candidate for one or more
general mechanism families.  It cannot create candidates, queries, mechanisms,
or Core evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rca_agent.investigation.models import (
    CandidateDisposition,
    CandidateStage,
    InvestigationWorkspaceSnapshot,
)
from rca_agent.models import DataStatus, ResetResponse
from rca_agent.tools.fingerprint import sha256_digest

MECHANISM_COVERAGE_POLICY_VERSION = "mechanism-coverage-v3"


class CoverageDomain(StrEnum):
    DATABASE = "database"
    JVM = "jvm"


class CoverageFamily(StrEnum):
    DATABASE_EXECUTION = "database_execution"
    DATABASE_TRANSPORT = "database_transport"
    DATABASE_CONNECTION_ACQUISITION = "database_connection_acquisition"
    JVM_THREAD_QUEUE_PRESSURE = "jvm_thread_queue_pressure"
    JVM_GC_PRESSURE = "jvm_gc_pressure"
    JVM_MEMORY_PRESSURE = "jvm_memory_pressure"
    JVM_CPU_LOAD_PRESSURE = "jvm_cpu_load_pressure"


class CoverageCapabilityState(StrEnum):
    ADVERTISED = "advertised"
    PARTIAL = "partial"
    NOT_ADVERTISED = "not_advertised"


class CoverageCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_id: str = Field(pattern=r"^track-[a-f0-9]{20}$")
    mechanism_summary: str | None = Field(default=None, max_length=2000)
    mechanism_class: str | None = Field(default=None, max_length=80)
    system_layer: str | None = Field(default=None, max_length=100)
    stage: CandidateStage
    disposition: CandidateDisposition


class CoverageFormalCandidateInput(BaseModel):
    """A typed, compiled Candidate with a recorded Core verification state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=3, max_length=120)
    mechanism_id: str = Field(pattern=r"^mech-[a-f0-9]{20}$")
    mechanism_class: str = Field(min_length=1, max_length=80)
    mechanism_summary: str = Field(min_length=3, max_length=500)
    family: str = Field(min_length=1, max_length=80)
    state: Literal["verified", "unresolved", "failed"]


class CoverageObservationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_ref: str = Field(pattern=r"^obs-[a-f0-9]{16}$")
    query_ref: str = Field(min_length=1, max_length=120)
    signal: str = Field(min_length=1, max_length=120)
    entity_ref: str = Field(pattern=r"^ent-[a-z0-9-]{3,40}$")
    data_status: DataStatus


class CoverageCapabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1, max_length=80)
    available: bool
    advertised_signals: tuple[str, ...] = ()
    unavailable_signals: tuple[str, ...] = ()
    unsupported_derivations: tuple[str, ...] = ()


class MechanismCoverageInput(BaseModel):
    """Bounded Agent-visible input persisted for deterministic Replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = MECHANISM_COVERAGE_POLICY_VERSION
    workspace_snapshot_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    alert_entity_type: str = Field(min_length=1, max_length=120)
    candidates: tuple[CoverageCandidateInput, ...] = ()
    formal_candidates: tuple[CoverageFormalCandidateInput, ...] = ()
    observations: tuple[CoverageObservationInput, ...] = ()
    capabilities: tuple[CoverageCapabilityInput, ...] = ()
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        excluded = {"input_digest"}
        payload = self.model_dump(mode="json", exclude=excluded)
        if self.policy_version in {"mechanism-coverage-v1", "mechanism-coverage-v2"}:
            payload.pop("formal_candidates", None)
        if self.policy_version == "mechanism-coverage-v1":
            for candidate in payload["candidates"]:
                candidate.pop("mechanism_summary", None)
        return sha256_digest(payload) == self.input_digest


class CoverageTriggerBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_track_ids: tuple[str, ...] = ()
    observation_refs: tuple[str, ...] = ()
    observed_signals: tuple[str, ...] = ()
    advertised_capability_signals: tuple[str, ...] = ()


class CoverageDiscriminatorNeed(BaseModel):
    """Abstract observability need; never an executable query or Core rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    need_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")
    missing_family: CoverageFamily
    competing_families: tuple[CoverageFamily, ...] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=500)
    signal_semantics: tuple[str, ...] = Field(min_length=1, max_length=8)
    capability_state: CoverageCapabilityState


class MechanismCoverageGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: CoverageDomain
    missing_families: tuple[CoverageFamily, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=500)
    discriminator_needs: tuple[CoverageDiscriminatorNeed, ...] = Field(min_length=1)
    trigger_basis: CoverageTriggerBasis


class MechanismCoverageAssessment(BaseModel):
    """Shadow-only diagnostic output with no candidate or verification authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = MECHANISM_COVERAGE_POLICY_VERSION
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    relevant_domains: tuple[CoverageDomain, ...] = ()
    covered_families: tuple[CoverageFamily, ...] = ()
    gaps: tuple[MechanismCoverageGap, ...] = ()
    assessment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")

    def verify_digest(self) -> bool:
        return sha256_digest(
            self.model_dump(mode="json", exclude={"assessment_digest"})
        ) == self.assessment_digest


class MechanismCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["shadow"] = "shadow"
    input: MechanismCoverageInput
    assessment: MechanismCoverageAssessment


_DB_FAMILIES = (
    CoverageFamily.DATABASE_EXECUTION,
    CoverageFamily.DATABASE_TRANSPORT,
    CoverageFamily.DATABASE_CONNECTION_ACQUISITION,
)
_JVM_FAMILIES = (
    CoverageFamily.JVM_THREAD_QUEUE_PRESSURE,
    CoverageFamily.JVM_GC_PRESSURE,
    CoverageFamily.JVM_MEMORY_PRESSURE,
    CoverageFamily.JVM_CPU_LOAD_PRESSURE,
)

_MECHANISM_FAMILY = {
    "database_execution": CoverageFamily.DATABASE_EXECUTION,
    "database_connection_pool": CoverageFamily.DATABASE_CONNECTION_ACQUISITION,
    "thread_pool_saturation": CoverageFamily.JVM_THREAD_QUEUE_PRESSURE,
    "gc_pause": CoverageFamily.JVM_GC_PRESSURE,
    "memory_pressure": CoverageFamily.JVM_MEMORY_PRESSURE,
    "cpu_saturation": CoverageFamily.JVM_CPU_LOAD_PRESSURE,
}

# Only explicit causal phrases count as coverage. Ambiguous prose remains untyped;
# this is a non-authoritative coverage audit, never a mechanism classifier for Core.
_UNTYPED_FAMILY_PATTERNS = (
    (CoverageFamily.DATABASE_EXECUTION, re.compile(
        r"\bslow[ -]?sql\b|\bsql (?:query )?execution\b|\b(?:db|database) query execution\b"
    )),
    (CoverageFamily.DATABASE_TRANSPORT, re.compile(
        r"\b(?:db|database) (?:network|transport) (?:latency|delay)\b"
    )),
    (CoverageFamily.DATABASE_CONNECTION_ACQUISITION, re.compile(
        r"\b(?:db|database) connection pool (?:exhaustion|saturation)\b"
        r"|\bconnection acquisition (?:wait|latency)\b"
    )),
    (CoverageFamily.JVM_THREAD_QUEUE_PRESSURE, re.compile(
        r"\bthread pool (?:exhaustion|saturation)\b"
    )),
    (CoverageFamily.JVM_GC_PRESSURE, re.compile(r"\b(?:gc|garbage collection) pressure\b")),
    (CoverageFamily.JVM_MEMORY_PRESSURE, re.compile(r"\bmemory pressure\b")),
    (CoverageFamily.JVM_CPU_LOAD_PRESSURE, re.compile(r"\b(?:cpu saturation|cpu overload)\b")),
)

_DATABASE_LAYER_NAMES = frozenset({"database", "db", "data_store", "datastore"})

_DB_SIGNALS = frozenset({
    "connection_acquisition_ms",
    "connection_duration_ms",
    "db_client_span_duration_ms",
    "sql_duration_ms",
    "sql_duration_share",
    "sql_shape",
})
_JVM_SIGNALS = frozenset({
    "thread_pool_active",
    "thread_pool_active_baseline_ratio",
    "thread_pool_queue_depth",
    "thread_pool_queue_depth_baseline_ratio",
    "thread_pool_rejected_tasks",
    "jvm_runnable_threads",
    "jvm_waiting_threads",
    "jvm_timed_waiting_threads",
    "jvm_thread_count",
    "jvm_thread_cpu_time_ns",
    "jvm_heap_used_bytes",
    "jvm_heap_used_bytes_baseline_ratio",
    "jvm_heap_max_bytes",
    "jvm_heap_committed_bytes",
    "jvm_gc_collection_delta",
    "jvm_gc_collection_delta_baseline_ratio",
    "jvm_gc_pause_seconds_delta",
    "jvm_gc_pause_seconds_delta_baseline_ratio",
    "jvm_gc_count_total",
    "jvm_gc_seconds_total",
    "cpu_usage_rate",
    "cpu_usage_vs_limits",
    "cpu_usage_vs_requests",
    "node_cpu_peak",
    "service_cpu_ratio",
})

_NEED_DEFINITIONS: dict[
    CoverageFamily, tuple[str, str, tuple[str, ...]]
] = {
    CoverageFamily.DATABASE_EXECUTION: (
        "DISTINGUISH_DATABASE_EXECUTION",
        "Can database-side execution be separated from acquisition and transport time?",
        ("database server execution time", "bounded SQL shape"),
    ),
    CoverageFamily.DATABASE_TRANSPORT: (
        "DISTINGUISH_DATABASE_TRANSPORT",
        "Can transport delay be separated from database execution and connection acquisition?",
        ("independent network round-trip time", "database server execution time"),
    ),
    CoverageFamily.DATABASE_CONNECTION_ACQUISITION: (
        "DISTINGUISH_DATABASE_CONNECTION_ACQUISITION",
        "Can connection acquisition or pool wait be separated from execution and transport?",
        ("connection acquisition time", "pool occupancy or wait semantics"),
    ),
    CoverageFamily.JVM_THREAD_QUEUE_PRESSURE: (
        "DISTINGUISH_JVM_THREAD_QUEUE",
        "Can runnable, queued, or rejected work be separated from CPU load, GC, and dependency wait?",
        ("queue depth or wait time", "active and rejected work", "thread state semantics"),
    ),
    CoverageFamily.JVM_GC_PRESSURE: (
        "DISTINGUISH_JVM_GC",
        "Can GC activity be separated from queue pressure, memory growth, and CPU load?",
        ("GC pause and collection activity", "heap recovery after collection"),
    ),
    CoverageFamily.JVM_MEMORY_PRESSURE: (
        "DISTINGUISH_JVM_MEMORY",
        "Can sustained memory pressure be separated from transient GC activity and load?",
        ("heap usage trend", "old-generation or allocation semantics"),
    ),
    CoverageFamily.JVM_CPU_LOAD_PRESSURE: (
        "DISTINGUISH_JVM_CPU_LOAD",
        "Can application CPU or host load be separated from queueing, GC, and wait time?",
        ("process and host CPU load", "CPU limit or throttling semantics"),
    ),
}

_NEED_SIGNALS: dict[CoverageFamily, frozenset[str]] = {
    CoverageFamily.DATABASE_EXECUTION: frozenset({
        "db_server_execution_ms",
        "db_client_span_duration_ms",
        "sql_duration_share",
        "sql_shape",
    }),
    CoverageFamily.DATABASE_TRANSPORT: frozenset({
        "db_network_rtt_ms",
        "db_server_execution_ms",
        "network_stage_ms",
    }),
    CoverageFamily.DATABASE_CONNECTION_ACQUISITION: frozenset({
        "connection_acquisition_ms",
        "connection_pool_active",
        "connection_pool_max",
        "connection_pool_wait_ms",
        "connection_duration_ms",
    }),
    CoverageFamily.JVM_THREAD_QUEUE_PRESSURE: frozenset({
        "jvm_runnable_threads",
        "thread_pool_queue_wait_ms",
        "thread_pool_active",
        "thread_pool_queue_depth",
        "thread_pool_rejected_tasks",
    }),
    CoverageFamily.JVM_GC_PRESSURE: frozenset({
        "heap_after_gc",
        "full_gc_count",
        "full_gc_pause_duration",
        "jvm_gc_collection_delta",
        "jvm_gc_pause_seconds_delta",
    }),
    CoverageFamily.JVM_MEMORY_PRESSURE: frozenset({
        "allocation_rate",
        "heap_after_gc",
        "jvm_heap_max_bytes",
        "jvm_heap_used_bytes",
        "old_gen_usage",
    }),
    CoverageFamily.JVM_CPU_LOAD_PRESSURE: frozenset({
        "cpu_throttling",
        "cpu_usage_rate",
        "cpu_usage_vs_limits",
        "node_cpu_peak",
    }),
}


def _candidate_family(
    candidate: CoverageCandidateInput,
    *,
    database_relevant: bool,
    policy_version: str,
) -> CoverageFamily | None:
    if candidate.mechanism_class == "network_latency":
        layer = (candidate.system_layer or "").casefold().replace("-", "_")
        if database_relevant or layer in _DATABASE_LAYER_NAMES:
            return CoverageFamily.DATABASE_TRANSPORT
        return None
    typed_family = _MECHANISM_FAMILY.get(candidate.mechanism_class or "")
    if typed_family is not None or policy_version == "mechanism-coverage-v1":
        return typed_family
    if candidate.mechanism_class is not None or not candidate.mechanism_summary:
        return None
    matches = {
        family for family, pattern in _UNTYPED_FAMILY_PATTERNS
        if pattern.search(candidate.mechanism_summary.casefold())
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _formal_family(candidate: CoverageFormalCandidateInput) -> CoverageFamily | None:
    if candidate.mechanism_class == "network_latency":
        if (
            candidate.family == "database_dependency"
            or re.search(r"\b(?:db|database)\b", candidate.mechanism_summary.casefold())
        ):
            return CoverageFamily.DATABASE_TRANSPORT
        return None
    return _MECHANISM_FAMILY.get(candidate.mechanism_class)


def project_formal_coverage_candidates(events: Iterable[Any]) -> tuple[CoverageFormalCandidateInput, ...]:
    """Rebuild typed Candidate viability solely from pre-freeze Transcript events."""

    mechanisms: dict[str, dict[str, Any]] = {}
    compiled_mechanisms: dict[str, str] = {}
    verification: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event_type == "mechanism_draft_frozen":
            mechanism = event.payload["mechanism"]
            mechanisms[mechanism["mechanism_id"]] = mechanism
        elif event.event_type in {
            "dynamic_candidate_registered", "verification_plan_extended",
        }:
            compiled = event.payload["compiled"]
            compiled_mechanisms[compiled["candidate"]["candidate_id"]] = (
                event.payload["mechanism_id"]
            )
        elif event.event_type == "dynamic_verification":
            verification = event.payload["candidates"]
    projected = []
    for candidate_id, mechanism_id in sorted(compiled_mechanisms.items()):
        report = verification.get(candidate_id)
        mechanism = mechanisms.get(mechanism_id)
        if report is None or mechanism is None:
            continue
        projected.append(CoverageFormalCandidateInput(
            candidate_id=candidate_id,
            mechanism_id=mechanism_id,
            mechanism_class=mechanism["mechanism_class"],
            mechanism_summary=mechanism["mechanism"],
            family=mechanism["family"],
            state=report["state"],
        ))
    return tuple(projected)


def _input_digest(payload: dict[str, Any]) -> str:
    return sha256_digest(payload)


def build_mechanism_coverage_input(
    workspace: InvestigationWorkspaceSnapshot,
    issued_observations: Iterable[Any],
    bootstrap: ResetResponse,
    *,
    policy_version: str = MECHANISM_COVERAGE_POLICY_VERSION,
    formal_candidates: Iterable[CoverageFormalCandidateInput] = (),
) -> MechanismCoverageInput:
    """Project only legal episode state into the CoveragePolicy contract."""

    candidates = tuple(sorted((
        CoverageCandidateInput(
            candidate_track_id=item.candidate_track_id,
            mechanism_summary=(item.mechanism_summary
                               if policy_version != "mechanism-coverage-v1" else None),
            mechanism_class=item.mechanism_class,
            system_layer=item.system_layer,
            stage=item.stage,
            disposition=item.disposition,
        )
        for item in workspace.candidates
    ), key=lambda item: item.candidate_track_id))
    observations = tuple(sorted((
        CoverageObservationInput(
            observation_ref=item.observation.observation_id,
            query_ref=item.query_ref,
            signal=item.observation.signal,
            entity_ref=item.observation.entity_ref,
            data_status=item.observation.data_status,
        )
        for item in issued_observations
    ), key=lambda item: item.observation_ref))
    capabilities = tuple(sorted((
        CoverageCapabilityInput(
            tool_name=item.tool_name,
            available=item.available,
            advertised_signals=tuple(sorted({metric.signal for metric in item.metrics})),
            unavailable_signals=tuple(sorted(item.unavailable_signals)),
            unsupported_derivations=tuple(sorted(item.unsupported_derivations)),
        )
        for item in bootstrap.capabilities
    ), key=lambda item: item.tool_name))
    raw = {
        "policy_version": policy_version,
        "workspace_snapshot_digest": workspace.snapshot_digest,
        "alert_entity_type": bootstrap.alert_entity.entity_type,
        "candidates": [item.model_dump(
            mode="json", exclude={"mechanism_summary"}
            if policy_version == "mechanism-coverage-v1" else None
        ) for item in candidates],
        "observations": [item.model_dump(mode="json") for item in observations],
        "capabilities": [item.model_dump(mode="json") for item in capabilities],
    }
    if policy_version == "mechanism-coverage-v3":
        raw["formal_candidates"] = [item.model_dump(mode="json") for item in sorted(
            formal_candidates, key=lambda item: item.candidate_id
        )]
    return MechanismCoverageInput(**raw, input_digest=_input_digest(raw))


class MechanismCoveragePolicy:
    """Deterministic DB/JVM coverage audit for shadow recording only."""

    def evaluate(self, coverage_input: MechanismCoverageInput) -> MechanismCoverageAssessment:
        if not coverage_input.verify_digest():
            raise ValueError("mechanism coverage input digest is invalid")
        if coverage_input.policy_version not in {
            "mechanism-coverage-v1", "mechanism-coverage-v2",
            MECHANISM_COVERAGE_POLICY_VERSION,
        }:
            raise ValueError("unsupported mechanism coverage policy version")

        live_candidates = tuple(
            item for item in coverage_input.candidates
            if item.stage != CandidateStage.CLOSED
            and item.disposition == CandidateDisposition.OPEN
        )
        trusted_observations = tuple(
            item for item in coverage_input.observations
            if item.data_status in {DataStatus.OBSERVED, DataStatus.DERIVED}
        )
        observed_signals = {item.signal for item in trusted_observations}
        initial_families = {
            family for item in live_candidates
            if (family := _candidate_family(
                item, database_relevant=False,
                policy_version=coverage_input.policy_version,
            )) is not None
        }
        viable_formal = tuple(
            item for item in coverage_input.formal_candidates
            if item.state in {"verified", "unresolved"}
        ) if coverage_input.policy_version == "mechanism-coverage-v3" else ()
        initial_families.update(
            family for item in viable_formal
            if (family := _formal_family(item)) is not None
        )
        db_relevant = bool(initial_families.intersection(_DB_FAMILIES)) or bool(
            observed_signals.intersection(_DB_SIGNALS)
        )
        candidate_families = {
            family for item in live_candidates
            if (family := _candidate_family(
                item, database_relevant=db_relevant,
                policy_version=coverage_input.policy_version,
            )) is not None
        }
        candidate_families.update(
            family for item in viable_formal
            if (family := _formal_family(item)) is not None
        )
        jvm_relevant = bool(candidate_families.intersection(_JVM_FAMILIES)) or bool(
            observed_signals.intersection(_JVM_SIGNALS)
        )
        relevant_domains = tuple(
            domain for domain, relevant in (
                (CoverageDomain.DATABASE, db_relevant),
                (CoverageDomain.JVM, jvm_relevant),
            ) if relevant
        )
        covered_families = tuple(sorted(candidate_families, key=str))
        advertised = {
            signal for item in coverage_input.capabilities if item.available
            for signal in item.advertised_signals
        }
        unavailable = {
            signal for item in coverage_input.capabilities
            for signal in (*item.unavailable_signals, *item.unsupported_derivations)
        }

        gaps: list[MechanismCoverageGap] = []
        if db_relevant:
            gap = self._gap(
                CoverageDomain.DATABASE,
                _DB_FAMILIES,
                candidate_families,
                live_candidates,
                trusted_observations,
                advertised,
                unavailable,
                coverage_input.policy_version,
            )
            if gap is not None:
                gaps.append(gap)
        if jvm_relevant:
            gap = self._gap(
                CoverageDomain.JVM,
                _JVM_FAMILIES,
                candidate_families,
                live_candidates,
                trusted_observations,
                advertised,
                unavailable,
                coverage_input.policy_version,
            )
            if gap is not None:
                gaps.append(gap)

        raw = {
            "policy_version": coverage_input.policy_version,
            "input_digest": coverage_input.input_digest,
            "relevant_domains": relevant_domains,
            "covered_families": covered_families,
            "gaps": tuple(gaps),
        }
        assessment = MechanismCoverageAssessment(
            **raw,
            assessment_digest=sha256_digest({
                key: [item.model_dump(mode="json") for item in value]
                if key == "gaps" else value
                for key, value in raw.items()
            }),
        )
        if not assessment.verify_digest():  # pragma: no cover - construction invariant
            raise AssertionError("mechanism coverage assessment digest construction failed")
        return assessment

    def _gap(
        self,
        domain: CoverageDomain,
        domain_families: tuple[CoverageFamily, ...],
        covered: set[CoverageFamily],
        candidates: tuple[CoverageCandidateInput, ...],
        observations: tuple[CoverageObservationInput, ...],
        advertised: set[str],
        unavailable: set[str],
        policy_version: str,
    ) -> MechanismCoverageGap | None:
        missing = tuple(family for family in domain_families if family not in covered)
        if not missing:
            return None
        relevant_signals = _DB_SIGNALS if domain == CoverageDomain.DATABASE else _JVM_SIGNALS
        basis_observations = tuple(
            item for item in observations if item.signal in relevant_signals
        )
        basis_candidates = tuple(
            item for item in candidates
            if _candidate_family(
                item, database_relevant=domain == CoverageDomain.DATABASE,
                policy_version=policy_version,
            ) in domain_families
        )
        needs = tuple(
            self._need(family, domain_families, advertised, unavailable)
            for family in missing
        )
        return MechanismCoverageGap(
            domain=domain,
            missing_families=missing,
            rationale=(
                f"{domain.value} is relevant to the current investigation, but one or more "
                "general competing mechanism families have no live open candidate."
            ),
            discriminator_needs=needs,
            trigger_basis=CoverageTriggerBasis(
                candidate_track_ids=tuple(sorted(
                    item.candidate_track_id for item in basis_candidates
                )),
                observation_refs=tuple(sorted(
                    item.observation_ref for item in basis_observations
                )),
                observed_signals=tuple(sorted({
                    item.signal for item in basis_observations
                })),
                advertised_capability_signals=tuple(sorted(
                    advertised.intersection(relevant_signals)
                )),
            ),
        )

    def _need(
        self,
        family: CoverageFamily,
        domain_families: tuple[CoverageFamily, ...],
        advertised: set[str],
        unavailable: set[str],
    ) -> CoverageDiscriminatorNeed:
        code, question, semantics = _NEED_DEFINITIONS[family]
        needed_signals = _NEED_SIGNALS[family]
        advertised_matches = needed_signals.intersection(advertised)
        unavailable_matches = needed_signals.intersection(unavailable)
        if advertised_matches and len(advertised_matches) == len(needed_signals):
            capability_state = CoverageCapabilityState.ADVERTISED
        elif advertised_matches or unavailable_matches:
            capability_state = CoverageCapabilityState.PARTIAL
        else:
            capability_state = CoverageCapabilityState.NOT_ADVERTISED
        return CoverageDiscriminatorNeed(
            need_code=code,
            missing_family=family,
            competing_families=tuple(item for item in domain_families if item != family),
            question=question,
            signal_semantics=semantics,
            capability_state=capability_state,
        )
