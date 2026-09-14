# RCA Agent handoff

## Latest increment — 2026-09-14

Bounded dynamic discovery is integrated into LangGraph/Runner. Configure a separate discovery
adapter and an EvidenceRegistry (or builder). Existing Catalog candidates remain available.
The compiler checks signal types, units, threshold sources, reference data, independent groups
and query recipes. Runner owns registration, verification, evidence binding and transcript events.
Dynamic predicates default to support/inconclusive; confirmation requires a host-reviewed
sufficiency contract and Core evidence checks.

See `analysis/dynamic_discovery.md` for setup and limits. The initial RCA100 registry uses existing
issued trace/resource signals; thread-pool tests use synthetic data. No live inference was performed.
Next integration work is backend-specific inventories and a separately assessed live model run.
Validation for this increment: Ruff passed; 117 tests passed (23 new dynamic-discovery tests).
The older sections below describe the baseline before this increment.

Updated: 2026-09-01

## Current status

Phases 0, 1, and 2 of the RCA v4 Agent Environment are complete. The LangChain structured-policy
boundary and LangGraph Fake/Scripted orchestration stage are also complete. The real-provider model
factory now targets `deepseek-v4-pro`; a live investigation has not yet been run. Deterministic
diagnosis strategy and accuracy experiments have not started. A one-call provider smoke test has
validated credential loading and schema-constrained `REQUEST_STOP`; no RCA100 live investigation
has run yet. Ruff and all 94 tests pass.

The package no longer depends on the frozen sibling `rca-core`. The RCA runtime domain is now
owned under `src/rca_agent/core`, including the v3 candidate, condition, capability, proposal,
discovery, assessment, candidate-state, and safety-decision modules. Research code remains outside
Agent Runtime; evaluator-only ground-truth loading stays under top-level `evaluation/`.

The Episode Runner now invokes the internal domain core by default. It records issued Observations
with query provenance, applies a trust gate to proposal-referenced facts, resolves optional
`candidate_id` values against the trusted Catalog, and freezes the Core's authoritative decision.
Real RCA100 smoke runs confirm t025 as `dbNetworkLatency` and t030 as `slowSQL` through this chain.

The Runner now also exposes a read-only, non-freezing Core assessment preview. LangGraph invokes
Policy, sends every action through the Runner, previews the resulting Core state, loops while the
result is inconclusive, and requests Runner-controlled freeze after confirmation. Invalid model
output fails before any tool execution. Budget, max rounds, Transcript, Replay, and answer-freeze
tests remain enforced without an API key.

The preview is now genuinely multi-candidate. Observations remain episode-global facts, while
`UPDATE_CANDIDATE_EVIDENCE` binds issued references to one hypothesis. Core evaluates each Candidate
only within that scope and returns safe per-candidate status plus discriminating signal gaps. Graph
rebuilds active/supported/contradicted/inconclusive/confirmed Candidate frontiers from the preview;
it cannot author those states. A deterministic `slowSQL` versus `dbNetworkLatency` E2E confirms
that evidence does not leak across candidates and that one differential Trace query can contradict
slowSQL while Core confirms dbNetworkLatency.

## Implemented

- independent Git repository and Python package skeleton;
- phase boundaries and trust-state documentation;
- minimal `RcaEnvironment` protocol;
- private RCA100 host adapter using the existing observation directory;
- anonymous entity and trace references;
- Metric series and equal-window comparison tools;
- related-entity and topology tools;
- slow-operation trace summary and dependent stage-breakdown tools;
- explicit unavailable Log capability;
- bounded windows, entities, return sizes, query costs, and episode budget;
- candidate-neutral observations and stable replay fingerprints.

Acceptance evidence is recorded in `analysis/phase1_acceptance.md`.

Phase 2 additionally implements:

- closed typed Policy Actions;
- `CREATED -> ACTIVE -> ANSWER_FROZEN -> EVALUATED` lifecycle enforcement;
- append-only hash-chained JSONL Transcripts;
- query fingerprint, cost, budget, lifecycle, and answer-binding Recorded Replay;
- single-use typed Answer Artifacts with proposal/decision separation;
- episode-scoped issued-Observation ledger and query provenance;
- trust-aware Observation normalization and default `CoreDecisionProvider`;
- evaluator-only post-freeze loading and separate Evaluation Artifacts;
- sanitized Scripted/Fake Policy child-process messaging;
- explicit invalid-action, crash, timeout, budget, max-round, and tool-failure termination.

Acceptance evidence is recorded in `analysis/phase2_acceptance.md`.

## Decisions to preserve

- Do not modify the frozen sibling `rca-core` repository or reintroduce it as a package dependency.
- Evolve the extracted runtime slice independently inside `rca-agent` and expand it only for a
  concrete Agent need.
- Do not expose raw paths, case selectors, trace IDs, core query references, or evaluator data.
- Do not reinterpret missing observations as negative evidence.
- Keep threshold, weight, sufficiency, and stopping authority in the internal Agent domain core.
- Keep Ground Truth, evaluation, Oracle, Benchmark, holdout, audit, and historical experiments out
  of `src/rca_agent`.
- Keep t025/t030 as exposed development regressions only.

## Stable interfaces for the next stage

Phase 3 may depend on:

1. `RcaEnvironment` and all phase 1 request/response contracts;
2. `PolicyAction`, `CallToolAction`, `SubmitHypothesisAction`, and `RequestStopAction`;
3. `EpisodeRunner`, lifecycle errors, and the Transcript event contract;
4. `RecordedReplay` and stable query fingerprints;
5. `AnswerArtifact` proposal/decision separation and the post-freeze evaluator gate;
6. Scripted child-process messaging as the process-isolation reference implementation.
7. `CoreDecisionProvider`, the issued-Observation ledger, and trusted `candidate_id` resolution.
8. `EpisodeRunner.preview_core_assessment()` as the sole read-only graph-routing view of Core state.
9. `LangChainPolicyAdapter`, `PolicyEnvelope`, and `LangGraphInvestigation` as non-authoritative
   orchestration interfaces.
10. `CoreAssessmentPreview`, Candidate-scoped evidence updates, safe discriminating-signal views,
    and the Core-derived Candidate Frontier.

The next objective is to provide `DEEPSEEK_API_KEY` to the host process and run one controlled live
investigation through `create_deepseek_policy()`, then proceed to RCA100 blind investigations.
Before accuracy claims, add model-facing prompt regressions and verify that candidate-option
disclosure remains within the intended blind evaluation protocol. A non-LLM deterministic
investigation-policy baseline is still useful as a comparison arm. No Core-interface blocker is
currently known.

Backend Verification Replay, signed transcript anchoring, concurrent transcript storage, evaluator
scoring, and a general untrusted-code sandbox remain explicit follow-up work.
