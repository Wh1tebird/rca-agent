# RCA Agent handoff

## Latest increment — 2026-09-18 inquiry retention and additive feedback

Completed the bounded candidate-retention and feedback implementation without changing Core's
generic verification standard. Every valid inquiry round is transcripted before query execution.
Exact-text candidate records survive model ID reuse and later omissions; the handoff and Recorded
Replay check them against those rounds. Answer Artifacts return tentative candidates with their
query/Observation references, Discovery disposition, formal mechanism links and Core evidence
status. Ideas remain visible even if Discovery does not run or fails.

The existing Verification Planner has a separate additive structured response for Core feedback:
the model supplies only 1–2 new capability predictions, while Runner keeps old predictions,
recompiles the same frozen mechanism with the original discovery snapshot, versions the candidate,
executes new bounded queries and asks Core to reassess. Transcript and Replay validate the revision
and old evidence binding. A fixed Graph test reached `DISCOVERED_VERIFIED` through this route;
failed candidates are not reopened. The inquiry benchmark now accepts `--inquiry-query-limit` to
reserve budget for verification and feedback. Ruff and all 197 tests passed.

Three distinct Qwen thinking/json_schema Case probes completed with valid Replay; no 12-case
batch was run. t007 retained 3/3 inquiry ideas and compiled two, but Core rejected the GT-like
thread-pool plan's evidence. t055 retained 3/3, while Discovery timed out. t021 retained 3/3 and
recorded both inquiry rounds in Transcript; Discovery discarded the ideas and compiled none. No
real Qwen probe offered an eligible nonfailed compiled candidate for additive feedback, so live
feedback success remains unobserved. Results and exact limits are in
`analysis/inquiry_retention_feedback_acceptance.md` and `analysis/qwen_chain_smoke_20260918/`.
Do not report improved model recall or end-to-end verification from these three probes. A future
evaluation may select a naturally unresolved, compilable candidate to observe real feedback, then
run the separately approved 12-case comparison.

## Latest increment — 2026-09-17 local Qwen model adapter

Added `src/rca_agent/qwen_policy.py` and a secret-free OpenAI-compatible model factory. The
single-case and batch holdout benchmarks select Qwen3-30B-A3B-Thinking-2507 by default and retain
`--provider deepseek` for comparison. The API key comes only from `RCA_LOCAL_API_KEY` or a hidden
interactive terminal prompt. The model factory rejects credentials embedded in URLs and
`/chat/completions` used as the base URL. Public HTTP requires explicit
`--allow-insecure-http`; the user approved this exception after HTTPS on the advertised port
failed and the private address proved unreachable. The user placed the key into the macOS login
keychain under `rca-agent-qwen-local`; the test process reads it into an environment variable at
launch, while the key is absent from tool arguments, source and reports. The clipboard was cleared.
The complete Qwen 12-case diagnostic finished with 1 GT-like formal mechanism, 7 compiled cases,
0 correct and 0 wrong verified. DeepSeek's historical 12-case run had 1, 11 and 0/0 respectively.
Qwen had 16 acquisition format/choice errors versus DeepSeek's 1, so model-intelligence claims
are not justified. A JSON-schema compatibility probe succeeded, but its follow-up batch was
stopped at the user's request; do not include partial results in accuracy comparisons. See
`analysis/qwen_model_comparison.md`. No further provider calls should be made without a new request.

## Latest increment — 2026-09-17 inquiry handoff and feedback

The inquiry arm now records bounded field-level structured-output diagnostics and tolerates longer
mechanism explanations or more than two suggested queries (only the first two execute). A typed
handoff carries tentative hypotheses, evidence questions, and issued query results into formal
Discovery. Runner validates the issued references; Transcript and Recorded Replay bind the handoff.
Discovery can state why it kept, revised or discarded each inquiry hypothesis. An optional one-pass
Core feedback route can execute up to two still-unattempted queries from a frozen verification plan
and bind their results through Runner. It does not change the frozen mechanism.

The complete handoff-only 12-case live run finished and replayed all 12, selected 120 acquisition
queries, and recovered 3 model-output errors with bounded retry. Final inquiry drafts had six
GT-like lexical directions; formal Discovery retained two. Correct/wrong verification stayed 0/0.
No Core feedback query was eligible after existing verification. A second 12-case run exercised
the disposition schema. After the user recharged, only the three cases interrupted by HTTP 402
(t003, t004, t098) were rerun: 12/12 Discovery calls completed, 12/12 Replays passed, and 36/36
dispositions were reported. It compiled 11 cases, retained one GT-like formal mechanism, and had
0/0 correct/wrong verification, with zero eligible Core feedback queries. Ruff and all 185 tests
pass. Detailed limitations and per-run results: `analysis/inquiry_handoff_acceptance.md`.

The planned three complete repetitions per arm remain outstanding; no further paid runs were made
to conserve credits. Do not treat the earlier provider interruption as a reasoning failure. The next functional extension,
if pursued, is a separately versioned verification plan for a genuinely new distinguishing
prediction; rerunning already executed frozen queries cannot resolve missing upstream telemetry.

## Latest increment — 2026-09-17 evidence-need-guided acquisition

Added optional `--acquisition inquiry` for the 12-Case Catalog-holdout benchmark. One structured
DeepSeek binding proposes revisable preliminary hypotheses and evidence questions, then chooses
one or two host-built queries per round. The host maps short `qN` menu references back to opaque
typed Actions, records evidence-need bindings, and Runner executes under existing budgets.
At most five rounds / ten queries and one output retry per round; old fixed and AI modes remain.
Core verification and Replay are unchanged. Ruff and all 181 tests pass.

After two interface revisions and a final complete rerun: 12/12 cases completed with valid Replay;
86 acquisition queries selected over 47 model selection calls; 11 parser errors, no remaining
reference/entity-binding errors. Formal Discovery's raw GT-like recall was 4/12 versus 2/12 in the
prior AI-query arm, but eligible evidence was 79/162 versus 73/139 and alternative resolution was
14/144 versus 11/110. Both arms had zero correct and zero wrong verified results. This is a
diagnostic single-run difference, not proven improvement. t040 and t004 selected no exploratory
queries because two consecutive structured outputs failed to parse. Preliminary ideas are not
currently handed explicitly to formal Discovery. See `analysis/inquiry_acquisition_acceptance.md`.

Next work: make structured inquiry output more reliable, add a bounded and auditable handoff from
revisable exploration to official Discovery without granting decision authority, and repeat the
same 12 cases multiple times before claiming gains. Do not lower Core's alternative/refutation
requirements to improve verified counts.

## Final completion — capability plan

All five plan items are implemented and audited. Added the remaining 17 source mappings, making
35 new / 65 total public entity-signal pairs; the raw-field unmapped inventory is empty. This does
not fix the 18 pre-existing initialization gaps or create absent transport/Full-GC telemetry.
Counter/error-ratio semantics were audited over all 103 cases; unsupported derivations are now
explicit contract entries with measured reasons in `analysis/metric_semantics_audit.md`.

Added a bounded AI query selector with opaque references; the fixed acquisition screen remains
default and `--acquisition ai` enables the alternative. Completed both 12-case arms: fixed compiled
12, AI compiled 11, 24/24 replays, zero correct and zero wrong verified. AI selected 120 queries,
including new Waiting/Timed Waiting signals, with no selector failures. Its query success and
evidence coverage improved, but raw GT recall and alternative resolution did not. No accuracy
improvement is claimed. Detailed per-candidate coverage inputs and post-freeze semantic review are
saved; the review is by the coding agent, not an independent expert annotation.

Final acceptance: `analysis/capability_plan_acceptance.md`, `capability_final_acceptance.md`,
`capability_final_coverage.json`, and `capability_semantic_review.md`. Prior expansion delivery
reports remain historical snapshots. Future work is acquisition quality and new upstream
discriminator telemetry; it is not unfinished implementation of this plan.

## Latest increment — 2026-09-16 capability expansion

Added 18 public metric mappings, typed public metric descriptors and 15 explicit missing-source
signals. Registered neutral recipes for issued instance/deployment/Pod entities in addition to
existing service/operation/node capabilities. Added request-volume, latency and error-ratio
equal-window recipes with conservative support eligibility; request-volume averages have no
generic traffic-refutation authority. Seven new aggregate thread/pool statistics remain context
only because scale and counter/reset semantics are not portable.

Updated the executable capability audit and completed 172 tests plus Ruff. The same 12-case
DeepSeek diagnostic completed with 12 compiled, 12 valid replays, 3 raw GT-like mechanisms,
10 supported, 2 inconclusive, 0 correct verified and 0 wrong verified. Eight cases selected new
request-volume rules; 12 retained unresolved alternatives (previously 11). This is not demonstrated
accuracy improvement. The acquisition screen and budgets are unchanged for comparability.

Reports: `analysis/capability_expansion_comparison.md`, `analysis/capability_expansion_delivery.md`,
`analysis/capability_expansion_summary.json`, and per-case JSON in `analysis/capability_expansion_run/`.
General Policy now sees neutral metric inventory, but this fixed benchmark uses scripted acquisition.
Follow-up: evaluate bounded capability-aware acquisition separately; validate upstream aggregate
semantics before adding task rates/pool utilization, and obtain genuinely missing discriminator
telemetry before expecting verification closure. Existing unmapped raw metrics remain listed in
the capability matrix; this increment does not claim all raw fields are exposed.

## Latest increment — failed competitors and refutation semantics

Dynamic verification now has explicit `verified`, `unresolved` and `failed` states. Observed support
failure or an observed refutation condition is decisive failure, so that candidate no longer blocks
a same-origin verified mechanism. Missing data remains unresolved. Two strongly supported viable
mechanisms return `MULTIPLE_SUPPORTED` with all supported hypothesis IDs and no winner. Deterministic
fixed-input tests cover both the multiple-candidate result and the failed-competitor release path;
Recorded Replay recomputes them.

Refutation is now scope checked. Discovery classifies mechanisms and alternatives with a small
generic `MechanismClass`; host capabilities advertise which classes they can refute. A contradiction
may discharge one alternative only when the scopes match. Invalid exclusion claims and
support/refutation conflicts are removed as transcripted plan repairs, leaving the Candidate
supported or inconclusive instead of rejecting it or falsely verifying it. One independent support
group is enough to retain a Candidate, but never enough to verify it.

The final live 12-case run completed and replayed all cases, compiled mechanisms in all cases, and
returned 10 supported plus 2 inconclusive results. Correct verified and wrong verified were both 0.
Across 23 mechanisms, 18 unvalidated alternative exclusions were removed; 11/12 cases retained an
unresolved alternative. The previous two verified results therefore relied on refutation claims the
host could not semantically authorize. Results are in `analysis/jvm_refutation_final*`; findings
are in `analysis/competition_refinement.md` and `analysis/refutation_semantic_audit.md`.

Next work should add host capabilities for request volume, database transport versus server
execution, lock/wait state and application/deployment errors. Do not weaken the alternative check to
restore the old verified count.

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
