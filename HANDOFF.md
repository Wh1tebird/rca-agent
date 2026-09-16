# RCA Agent handoff

## Latest increment — two-stage discovery and planning

Dynamic unknown-root-cause work now uses two adapters over one DeepSeek model binding. The first
adapter proposes only mechanisms, causal entities, explanations and alternatives. Runner validates
and freezes each mechanism. The second adapter receives immutable mechanisms and opaque capability
handles, and writes verification plans. It may retry once after a bounded planner/compile failure,
but cannot change mechanism semantics. Transcript and Recorded Replay validate the complete draft,
freeze, retry, plan and compilation chain. The legacy one-stage adapter remains supported for
existing callers.

The final clean 12-case run completed and replayed 12/12, compiled at least one mechanism in 12/12,
and converted all 5/5 stated GT-like mechanisms through the mechanism contract. The one-stage
baseline compiled any mechanism in 7/12 and converted 4/6 GT-like raw ideas. Final accuracy did not
increase: both runs correctly verified 2/12 and wrongly verified 0. The new correct pair is t055
thread exhaustion and t009 memory pressure. Seven planner retries recovered five compilable plans,
but no additional GT-like plan. Current evidence therefore says the split fixes much of the
"thought of it but could not say it in the contract" loss; raw semantic recall and evidence
discrimination remain the larger bottlenecks.

The first naive split produced four wrong verifications and was not retained as the final design.
Safety hardening now treats sibling proposals as mandatory alternatives, requires contradiction
rather than support to distinguish alternatives, groups correlated thread-pool/GC signals, and
blocks a candidate when a same-origin competitor still has two independent matched support groups.
A deterministic regression covers this condition. Final results and interpretation are in
`analysis/jvm_two_stage_final*`.

Next work should repeat the fixed benchmark enough times to quantify model variance, and separately
stratify semantic misses, capability gaps and evidence-closure failures. Do not add a semantic
auto-rewriter or weaken `DISCOVERED_VERIFIED`.

## Latest increment — discovery A/B attribution

Discovery diagnostics now retain raw model drafts before compilation, per-draft initial and final
contract outcomes, rejection reasons, repair provenance, and bounded invalid structured JSON. These
events are hash-chained and order-checked by Recorded Replay. The only automatic repair is exact
case/whitespace canonicalization for references to an already-declared alternative; Signal, Entity,
Rule and mechanism semantics remain model-owned.

The repeated 12-case `deepseek-chat` run completed 12/12 with valid replay and no query failures.
Raw semantic recall was 6/12; four of those six correct directions initially compiled; two were
correctly `DISCOVERED_VERIFIED` (t055 thread exhaustion and t021 slow SQL); and no wrong mechanism
verified. t024 and t025 show "thought of it but could not ground it" failures. t007 and t009 show
correct compiled ideas that did not close refutation/alternatives. Six cases did not state the GT
mechanism specifically enough. The deterministic repair fired zero times, so the current bottleneck
is not cosmetic JSON cleanup. Results are in `analysis/jvm_discovery_funnel*`.

## Latest increment — 2026-09-15

The JVM/thread/GC follow-up is complete. Trace breakdown now issues bounded neutral participant
services even without a DB stage bundle; downstream-depth ranking plus a ten-query JVM screen locates
inventory in both t007 and t055. Metric comparison emits a typed baseline-ratio fact. Runtime rules
now use equal-window changes for active threads, queues, runnable/total threads, grouped thread CPU,
heap, GC count and GC pause; only rejected tasks retains a meaningful absolute positive threshold.
The ambiguous active-thread aggregate is no longer treated as a portable capacity ratio.

Generic GC observations cannot compile into a specific Full GC claim. The evaluation-only profiler
scanned 103 cases / 824 records, and the 12-case DeepSeek diagnostic completed 12/12 with valid
replay, 6/6 positive target services issued, 0 query-failed cases and 0 wrong verifications. t055
recovered thread exhaustion and t009 recovered heap/GC pressure, both safely remaining supported
because alternative/refutation closure was incomplete. Ruff and all 147 tests pass. Results live in
`analysis/jvm_signal_profile.*`, `analysis/jvm_diagnostic_probe_summary.json`, and
`analysis/jvm_diagnostic_probe/`.

The main remaining bottleneck is Discovery contract adherence and discriminating evidence, not JVM
entity reachability. Eight cases had compiler rejections, especially non-exact alternative strings
and entity/rule mismatch. Before the formal 20-30 case benchmark, improve structured alternative
references (prefer IDs over repeated free text) and add capability-specific refutation signals.

## Previous capability increment — 2026-09-15

The first capability increment delivered the 1-4 implementation slice, but not every acceptance
condition in the full seven-step plan. The audit and entity graph are now implemented; database
differentiation remains partial because transport, pool-capacity and server-side timing are absent;
the fixed seven-case rerun is complete. The evaluation-only generator writes
`analysis/capability_matrix.{md,json}`. The runtime trace registry exposes truthful Hikari/JDBC
aliases, keys recipes by target entity, and reserves an unattempted issued trace for verification
when available.

The audit has since been upgraded from a static list to an executable inventory: it scans tool and
entity contracts, query bounds, mapped Signal/Rule metadata, raw Case/sample coverage, unmapped
metrics and Environment initialization failures. This revealed aggregate JVM and thread-pool data
across all 103 raw metric sets. The Environment now maps active/max threads, queue depth, rejected
tasks, runnable/total JVM threads, heap used/max and generic GC count/pause deltas for issued
services. Core advertises bounded reusable rules without requiring those metrics to be pre-queried.
Trace resolution issues both the downstream operation and its owning service, closing the entity
bridge needed for Discovery. These capabilities remain PARTIAL: the aggregate data lacks queue-wait,
old-gen/allocation/heap-after-GC and Full-GC-specific semantics, and thresholds still need holdout
validation. The generic verification standard was not weakened.

Acquisition can traverse issued operation/service relationships and request bounded slow traces for
a discovered service. Trace breakdown preserves exact operation matches and otherwise resolves a
complete Hikari/JDBC bundle to its downstream database-owning operation. Runner exposes this as an
issued entity graph; Discovery may choose only advertised entity/rule pairs. Caller trace references
remain legal recipes for the resolved downstream target, while returned evidence must still match
that target.

The fixed seven-case live batch was rerun with `deepseek-chat`: 7/7 completed, 7/7 replayed, 4/7
compiled at least one draft, 4/7 proposed the GT mechanism, 1/7 reached a correct
`DISCOVERED_VERIFIED` (t021/slowSQL), and 0 wrong mechanisms verified. Final `QUERY_FAILED` fell from
3 to 0. The proposed DB-network mechanism remained `SUPPORTED_HYPOTHESIS`, correctly reflecting the
lack of RTT and server-execution telemetry. Ruff passes, as do all 143 tests.

The older sections below describe the baseline before this increment.

## Previous dynamic-discovery increment — 2026-09-14

Bounded dynamic discovery is integrated into LangGraph/Runner. Configure a separate discovery
adapter and an EvidenceRegistry (or builder). Existing Catalog candidates remain available.
The compiler checks signal types, units, threshold sources, reference data, independent groups
and query recipes. Runner owns registration, verification, evidence binding and transcript events.
Dynamic mechanisms can now return `DISCOVERED_VERIFIED` as this investigation's root-cause result.
The prior per-mechanism ConfirmationContract was removed. All dynamic mechanisms use the same
generic standard: independent support, executed refutation and declared-alternative checks, new
verification evidence and Core arbitration. Incomplete/contradicted plans do not pass. This result
does not promote the mechanism to Catalog or claim exhaustive exclusion of all possible alternatives.
Transcript records the plan's discovery evidence snapshot and verification outcomes; replay
recomputes the generic check and validates the dynamic root-cause result from issued evidence.

See `analysis/dynamic_discovery.md` for setup and limits. The initial RCA100 registry uses existing
issued trace/resource signals; thread-pool tests use synthetic data. No live inference was performed.
Next integration work is backend-specific inventories and a separately assessed live model run.
Validation for the generic-verification revision: Ruff passed; 131 tests passed.
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

`evaluation/unknown_benchmark.py` now runs a post-freeze Catalog-holdout probe with real discovery.
Catalog exclusions are enforced in Policy options, Discovery context, Core resolution and Replay.
Live `deepseek-chat` probes on t025 (`dbNetworkLatency`) and t030 (`slowSQL`) both produced compiled
dynamic mechanisms and valid replay. The t030 model proposal semantically recovered slow SQL and its
two supporting predicates matched, but neither run reached `DISCOVERED_VERIFIED`: the model left
`Prediction.distinguishes` empty, so Core retained `SUPPORTED_HYPOTHESIS`. This is the desired safe
failure. The full record and next benchmark design are in `analysis/unknown_rca_benchmark.md`.

The next objective is a preregistered 20-30 case holdout manifest with post-freeze semantic scoring,
failure-stage attribution and separate queryable/non-queryable strata. Do not weaken the generic
verification standard to improve the headline recovery rate. Improve discovery contract adherence
and expand host capability coverage, then measure Ground-Truth recall and wrong-verification rate.
A non-LLM deterministic investigation-policy baseline remains useful as a comparison arm.

The first preregistered expansion has now run all 7 available `dbNetworkLatency`/`slowSQL` cases with
`deepseek-chat` and thinking disabled. Results: 7 valid replays, 6 discovery completions, 6 cases with
compiled drafts, 4 GT-like proposals, 1 correct `DISCOVERED_VERIFIED`, 0 wrong verifications,
3 `QUERY_FAILED`, and 1 `OUT_OF_SCOPE`. t030 is the verified success. DB-network recall remains 0/2;
both misses were safer connection-pool guesses that Core declined to verify. Before expanding to
20-30 cross-mechanism cases, add blind acquisition and at least two independent executable evidence
groups for those mechanism families; otherwise the larger number mostly measures missing capability.

Backend Verification Replay, signed transcript anchoring, concurrent transcript storage, evaluator
scoring, and a general untrusted-code sandbox remain explicit follow-up work.
