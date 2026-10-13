"""Bounded model selection of host-built queries; all execution stays with Runner."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rca_agent.models import (
    CallToolAction,
    CompareMetricWindowsRequest,
    GetMetricSeriesRequest,
    ListRelatedEntitiesRequest,
    TimeWindow,
)
from rca_agent.tools.fingerprint import sha256_digest


class AcquisitionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_refs: tuple[str, ...] = Field(max_length=5)


class InquiryNeed(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    need_id: str = Field(min_length=1, max_length=40)
    hypothesis_id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=2000)
    expected_if_true: str = Field(min_length=1, max_length=2000)
    expected_if_alternative: str = Field(min_length=1, max_length=2000)
    observable_signals: tuple[str, ...] | None = Field(
        default=None, max_length=12, exclude_if=lambda value: value is None,
    )
    entity_ref: str | None = Field(
        default=None, max_length=120, exclude_if=lambda value: value is None,
    )
    measurement_kind: Literal["series", "window_comparison", "entity_exploration"] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    windows: dict[str, TimeWindow] | None = Field(default=None, exclude_if=lambda value: value is None)


class InquiryHypothesis(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    hypothesis_id: str = Field(min_length=1, max_length=40)
    mechanism: str = Field(min_length=1, max_length=2000)
    alternative: str = Field(min_length=1, max_length=2000)
    alternative_id: str | None = Field(default=None, exclude_if=lambda v: v is None)


class MeasurementTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    entity_ref: str
    signal: str
    measurement_kind: Literal["series", "window_comparison", "entity_exploration"]
    windows: dict[str, TimeWindow]


class InquiryChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    need_id: str
    query_ref: str
    mode: Literal["direct", "exploratory"] | None = Field(default=None, exclude_if=lambda v: v is None)
    purpose: str | None = Field(default=None, min_length=1, max_length=1000, exclude_if=lambda v: v is None)
    measurement_target: MeasurementTarget | None = Field(default=None, exclude_if=lambda v: v is None)


class InquiryDraft(BaseModel):
    """Revisable exploration notes, never a Core proposal or verification plan."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    contract_version: int | None = Field(default=None, exclude_if=lambda value: value is None)
    hypotheses: tuple[InquiryHypothesis, ...] = Field(min_length=1, max_length=4)
    needs: tuple[InquiryNeed, ...] = Field(min_length=1, max_length=12)
    choices: tuple[InquiryChoice, ...] = Field(min_length=1, max_length=8)


class InquiryChoiceV2(InquiryChoice):
    """New live output requires binding declarations; old records retain optional fields."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_ref: str = Field(
        description=(
            "Menu query ref copied from the current menu (aq-... or qN alias). "
            "For batch_assessments use the completed result fingerprint instead."
        )
    )
    mode: Literal["direct", "exploratory"]
    purpose: str = Field(min_length=1, max_length=1000)
    measurement_target: MeasurementTarget


class InquiryAdmissionChoice(InquiryChoice):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InquiryAdmissionDraft(InquiryDraft):
    choices: tuple[InquiryAdmissionChoice, ...] = Field(min_length=1, max_length=8)


def parse_inquiry_draft(response, *, live_version=2, strict=False, recorded=False,
                       historical=False):
    from rca_agent.investigation.contract_enforcement import resolve_inquiry_version
    raw = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
    version = resolve_inquiry_version(raw.get("contract_version"), host_version=live_version,
                                      recorded=recorded, historical=historical)
    raw = {**raw, "contract_version": 2 if version == 2 else None}
    schema = InquiryAdmissionDraft if strict or (version == 2 and not historical) else InquiryDraft
    draft = schema.model_validate(raw)
    if version == 2:
        from rca_agent.investigation.inquiry_contract import fail, validate_draft
        validate_draft(draft)
        if not historical:
            for choice in draft.choices:
                if choice.mode is None:
                    fail("BINDING_MODE_REQUIRED", "choices.mode")
                if not choice.purpose or not choice.purpose.strip():
                    fail("BINDING_PURPOSE_REQUIRED", "choices.purpose")
                if choice.measurement_target is None:
                    fail("MEASUREMENT_TARGET_MISMATCH", "choices.measurement_target")
            draft = InquiryDraftV2.model_validate(draft.model_dump(mode="json"))
    return draft


class InquiryDraftV2(InquiryDraft):
    choices: tuple[InquiryChoiceV2, ...] = Field(min_length=1, max_length=8)


class InquiryQueryLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    source_round: int | None = Field(default=None, exclude_if=lambda v: v is None)
    source_need_id: str | None = Field(default=None, exclude_if=lambda v: v is None)
    mode: Literal["direct", "exploratory"] | None = Field(default=None, exclude_if=lambda v: v is None)
    measurement_target: MeasurementTarget | None = Field(default=None, exclude_if=lambda v: v is None)
    question: str = Field(min_length=1, max_length=2000)
    hypothesis_id: str | None = Field(default=None, max_length=40)
    signal: str = Field(min_length=1, max_length=100)
    entity_ref: str = Field(min_length=1, max_length=120)
    query_ref: str = Field(min_length=1, max_length=120)
    status: str = Field(min_length=1, max_length=40)
    observation_refs: tuple[str, ...] = Field(max_length=100)


class InquiryCandidateRecord(BaseModel):
    """Stable, non-authoritative record of an idea across inquiry rounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    hypothesis: InquiryHypothesis
    first_round: int = Field(ge=1, le=5)
    last_round: int = Field(ge=1, le=5)
    mention_count: int = Field(ge=1, le=20)


def inquiry_candidate_records(drafts: list[dict[str, Any]]) -> tuple[InquiryCandidateRecord, ...]:
    """Merge only exact normalized mechanism and alternative text, never inferred synonyms."""
    records: dict[tuple[str, str], InquiryCandidateRecord] = {}
    used_ids: set[str] = set()
    for item in drafts:
        round_number = int(item["round"])
        for raw in item["draft"]["hypotheses"]:
            hypothesis = InquiryHypothesis.model_validate(raw)
            key = (" ".join(hypothesis.mechanism.casefold().split()),
                   " ".join(hypothesis.alternative.casefold().split()))
            previous = records.get(key)
            if previous is not None:
                records[key] = previous.model_copy(update={
                    "last_round": round_number,
                    "mention_count": previous.mention_count + 1,
                })
                continue
            identifier = hypothesis.hypothesis_id
            if identifier in used_ids:
                identifier = identifier[:30] + "-" + sha256_digest(key)[:8]
            used_ids.add(identifier)
            records[key] = InquiryCandidateRecord(
                hypothesis=hypothesis.model_copy(update={"hypothesis_id": identifier}),
                first_round=round_number, last_round=round_number, mention_count=1,
            )
    return tuple(records.values())


class InquiryHandoff(BaseModel):
    """Tentative context for Discovery, without evidence or decision authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = Field(ge=1, le=5)
    hypotheses: tuple[InquiryHypothesis, ...] = Field(max_length=20)
    needs: tuple[InquiryNeed, ...] = Field(max_length=12)
    query_links: tuple[InquiryQueryLink, ...] = Field(max_length=80)
    candidate_history: tuple[InquiryCandidateRecord, ...] = Field(default=(), max_length=20)
    loop_contract_version: Literal[1] | None = Field(default=None, exclude_if=lambda v: v is None)
    loop_state_revision: int | None = Field(default=None, ge=1, exclude_if=lambda v: v is None)
    loop_state_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$",
                                        exclude_if=lambda v: v is None)
    candidate_priorities: tuple[dict[str, Any], ...] = Field(default=(), max_length=20,
        exclude_if=lambda v: not v)
    all_needs: tuple[dict[str, Any], ...] = Field(default=(), max_length=60,
        exclude_if=lambda v: not v)
    need_assessments: tuple[dict[str, Any], ...] = Field(default=(), max_length=80,
        exclude_if=lambda v: not v)
    pending_outcomes: tuple[dict[str, Any], ...] = Field(default=(), max_length=40,
        exclude_if=lambda v: not v)

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.loop_contract_version is None and (
            len(self.query_links) > 10 or self.loop_state_revision is not None
            or self.loop_state_digest is not None or self.candidate_priorities
            or self.all_needs or self.need_assessments or self.pending_outcomes
        ):
            raise ValueError("legacy handoff cannot use batched fields or extra bindings")
        if self.loop_contract_version == 1 and (
            self.loop_state_revision is None or self.loop_state_digest is None
        ):
            raise ValueError("batched handoff requires its recorded state revision")
        ids = [item.hypothesis_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate inquiry candidate identifiers")
        if self.candidate_history and tuple(
            item.hypothesis for item in self.candidate_history
        ) != self.hypotheses:
            raise ValueError("inquiry candidate history does not match handoff")
        if any(link.hypothesis_id is not None and link.hypothesis_id not in ids
               for link in self.query_links):
            raise ValueError("inquiry query link references an unknown candidate")
        return self


class InquiryOutputError(ValueError):
    """Bounded diagnostic for a model response that cannot become an inquiry draft."""

    def __init__(self, code: str, raw_text: str = "", fields: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.raw_digest = sha256_digest(raw_text) if raw_text else None
        excerpt = raw_text[:2000]
        excerpt = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted-key]", excerpt)
        excerpt = re.sub(r"/Users/[^\s\"']+", "[redacted-path]", excerpt)
        self.raw_excerpt = excerpt
        self.fields = fields[:8]

    def diagnostic(self) -> dict[str, Any]:
        return {"code": self.code, "raw_digest": self.raw_digest,
                "raw_excerpt": self.raw_excerpt, "fields": list(self.fields)}


def query_options(context: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, dict]:
    bootstrap = context["bootstrap"]
    horizon = TimeWindow.model_validate(bootstrap["observation_horizon"])
    incident = TimeWindow(start=horizon.start + (horizon.end - horizon.start) / 2, end=horizon.end)
    capabilities = {c["tool_name"]: c for c in bootstrap["capabilities"] if c["available"]}
    options = {}
    for entity in entities:
        for tool in ("get_metric_series", "compare_metric_windows"):
            for metric in capabilities.get(tool, {}).get("metrics", ()):
                if metric["entity_type"] != entity["entity_type"]:
                    continue
                request = (
                    GetMetricSeriesRequest(entity_ref=entity["entity_ref"],
                                           signal=metric["signal"], window=horizon,
                                           max_points=min(100, capabilities[tool]["max_results"]))
                    if tool == "get_metric_series" else
                    CompareMetricWindowsRequest(entity_ref=entity["entity_ref"],
                                                signal=metric["signal"], incident_window=incident)
                )
                action = CallToolAction(tool_name=tool, request=request)
                ref = "aq-" + sha256_digest(action.model_dump(mode="json"))[:20]
                options[ref] = {"action": action, "entity": entity,
                                "description": metric["description"], "unit": metric.get("unit"),
                                "capability_max_results": capabilities[tool]["max_results"],
                                "limitations": capabilities[tool].get("unsupported_derivations", {})}
        if "list_related_entities" in capabilities:
            action = CallToolAction(tool_name="list_related_entities",
                                   request=ListRelatedEntitiesRequest(
                                       entity_ref=entity["entity_ref"], max_results=50))
            ref = "aq-" + sha256_digest(action.model_dump(mode="json"))[:20]
            options[ref] = {"action": action, "entity": entity,
                            "capability_max_results": capabilities["list_related_entities"]["max_results"],
                            "description": "Explore related instances, services or infrastructure."}
    return options


class LangChainAcquisitionSelector:
    def __init__(self, runnable: Any):
        self.runnable = runnable

    @classmethod
    def from_chat_model(cls, model, *, structured_method: str = "json_mode"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", ("Select up to five unique query_refs from the offered inventory to "
             "investigate the incident. Prefer measurements that distinguish plausible causes, "
             "baseline comparisons for volume, and entity exploration when needed. Use the "
             "observations and metric descriptions; never infer a missing measurement as zero. "
             "Do not invent query references. Return the requested JSON schema. You select "
             "queries only and cannot assert a final root cause.")),
            ("human", "Schema: {schema}\nContext: {context}"),
        ])
        return cls(prompt | model.with_structured_output(
            AcquisitionSelection, method=structured_method,
        ))

    def select(self, context, options, limit=5) -> tuple[CallToolAction, ...]:
        public = {
            ref: {**item, "action": item["action"].model_dump(mode="json")}
            for ref, item in options.items()
        }
        selection = AcquisitionSelection.model_validate(self.runnable.invoke({
            "schema": json.dumps(AcquisitionSelection.model_json_schema()),
            "context": json.dumps({"incident": context, "queries": public}, default=str),
        }))
        if len(selection.query_refs) > limit or len(set(selection.query_refs)) != len(
            selection.query_refs
        ) or any(ref not in options for ref in selection.query_refs):
            raise ValueError("invalid acquisition selection")
        return tuple(options[ref]["action"] for ref in selection.query_refs)


class LangChainInquiryAdapter:
    """Plans a short evidence-seeking round without obtaining decision authority."""

    def __init__(self, runnable: Any, *, contract_version: Literal[1, 2] = 2):
        from rca_agent.investigation.contract_enforcement import validate_inquiry_host
        validate_inquiry_host(contract_version)
        self.runnable = runnable
        self.contract_version = contract_version

    @classmethod
    def from_chat_model(cls, model, *, structured_method: str = "json_mode", contract_version: Literal[1, 2] = 2):
        from rca_agent.investigation.contract_enforcement import validate_inquiry_host
        validate_inquiry_host(contract_version)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are making REVISABLE investigation notes, not a final root-cause decision. "
                "State up to three plausible causal mechanisms and an alternative for each. "
                "State evidence needs with a question and contrasting predictions. "
                "Choose one or two distinct offered query_refs, each bound to a need_id. "
                "Link each chosen query to the evidence need it answers. "
                "Every choice must declare mode (direct or exploratory), purpose and an exact "
                "measurement_target copied from the menu's entity_ref, signal, measurement_kind "
                "and windows. Direct must match an explicit Need entity and observable signal; "
                "Exploratory records indirect exploration without direct coverage. Never use "
                "the cause itself as its alternative. "
                "In each need, name observable_signals and entity_ref from the offered menu "
                "when known. A request-count query cannot measure SQL duration. If no direct "
                "measurement exists, describe an exploratory question instead of claiming it "
                "tests the unavailable signal. "
                "Use compare_metric_windows for a baseline when useful; explore related entities "
                "when the causal entity is uncertain. Prefer measurements that separate causes "
                "over generic latency or error symptoms. Do not invent signals, entities, or "
                "query references. Do not treat missing data as zero. Return the JSON schema."
            )),
            ("human", "Schema: {schema}\nInvestigation: {context}"),
        ])
        return cls(prompt | model.with_structured_output(
            InquiryDraftV2 if contract_version == 2 else InquiryDraft,
            method=structured_method, include_raw=True,
        ), contract_version=contract_version)

    def plan(self, context: dict[str, Any], options: dict[str, dict]) -> tuple[InquiryDraft,
                                                                               tuple[CallToolAction, ...]]:
        from rca_agent.investigation.inquiry_contract import (
            public_query,
            validate_choice,
            validate_draft,
        )
        aliases = {f"q{index}": (ref, item)
                   for index, (ref, item) in enumerate(options.items(), start=1)}
        public = {
            alias: public_query(item)
            for alias, (_, item) in aliases.items()
        }
        response = self.runnable.invoke({
            "schema": json.dumps((InquiryDraftV2 if self.contract_version == 2 else InquiryDraft).model_json_schema()),
            "context": json.dumps({"incident": context, "queries": public}, default=str),
        })
        if isinstance(response, dict) and "parsed" in response and "raw" in response:
            parsed = response.get("parsed")
            if parsed is None:
                raw = response.get("raw")
                metadata = getattr(raw, "response_metadata", {}) or {}
                if metadata.get("finish_reason") == "length":
                    raise InquiryOutputError("output_truncated")
                content = getattr(raw, "content", "")
                raw_text = content if isinstance(content, str) else ""
                calls = getattr(raw, "tool_calls", ()) or ()
                if len(calls) == 1 and calls[0].get("name") in {"InquiryDraft", "InquiryDraftV2"}:
                    parsed = calls[0].get("args")
                stripped = raw_text.strip()
                if stripped.startswith("```json") and stripped.endswith("```"):
                    stripped = stripped[7:-3].strip()
                if parsed is None:
                    if not stripped:
                        raise InquiryOutputError("empty_response")
                    try:
                        parsed = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise InquiryOutputError("invalid_json", raw_text) from exc
            response = parsed
        try:
            draft = parse_inquiry_draft(response, live_version=self.contract_version)
        except ValidationError as exc:
            fields = tuple((".".join(map(str, error["loc"])) or "response") + ":" + error["type"]
                           for error in exc.errors(include_input=False))
            raise InquiryOutputError("schema_mismatch", json.dumps(response, default=str),
                                     fields) from exc
        except InquiryOutputError:
            raise
        except ValueError as exc:
            raise InquiryOutputError("UNSUPPORTED_INQUIRY_VERSION", fields=("contract_version",)) from exc
        if self.contract_version == 2:
            validate_draft(draft)
            from rca_agent.investigation.inquiry_contract import original_alternative_id
            draft = draft.model_copy(update={"hypotheses": tuple(
                h.model_copy(update={"alternative_id": original_alternative_id(h.model_dump())})
                for h in draft.hypotheses)})
        hypotheses = {item.hypothesis_id for item in draft.hypotheses}
        needs = {item.need_id: item for item in draft.needs}
        if len(hypotheses) != len(draft.hypotheses) or len(needs) != len(draft.needs):
            raise InquiryOutputError("duplicate inquiry identifiers", fields=("hypotheses", "needs"))
        if any(item.hypothesis_id not in hypotheses for item in draft.needs):
            raise InquiryOutputError("inquiry need has no hypothesis", fields=("needs.hypothesis_id",))
        selected_choices = draft.choices[:2]
        if not selected_choices:
            raise InquiryOutputError("invalid inquiry choices", fields=("choices.query_ref",))
        actions = []
        normalized_choices = []
        seen_queries = set()
        for choice in selected_choices:
            need = needs.get(choice.need_id)
            offered = aliases.get(choice.query_ref)
            if need is None or offered is None:
                fields = (("choices.need_id",) if need is None else ("choices.query_ref",))
                raise InquiryOutputError("inquiry choice is not offered", fields=fields)
            ref, option = offered
            if self.contract_version == 2:
                validate_choice(need, choice, option["action"])
            action = option["action"]
            if self.contract_version == 1 and ref in seen_queries:
                continue
            if ref not in seen_queries:
                actions.append(action)
                seen_queries.add(ref)
            normalized = choice.model_copy(update={"query_ref": ref})
            if any(c.need_id == normalized.need_id and c.query_ref == ref and c != normalized
                   for c in normalized_choices):
                raise InquiryOutputError("CONFLICTING_BINDING", fields=("choices",))
            if normalized not in normalized_choices:
                normalized_choices.append(normalized)
        return draft.model_copy(update={"choices": tuple(normalized_choices)}), tuple(actions)
