# RCA Agent project context

Last consolidated: 2026-09-15 after JVM/thread/GC diagnostic hardening

## 1. Purpose

`rca-agent` turns RCA100 observations into a controlled, interactive, multi-turn investigation
environment. It prevents an investigation policy from reading a complete case in one step while
preserving a path from neutral observations into internal evidence validation and safe stopping.

The intended full pipeline is:

```text
RCA100 observations
  -> private RCA100 adapter
  -> Agent-visible RCA Environment Contract
  -> Episode Runner
  -> LangGraph investigation orchestration
  -> LangChain structured investigation policy
  -> Episode Runner actions and read-only Core assessment preview
  -> internal Agent domain-core evidence assessment and safe stopping
  -> frozen answer artifact
  -> evaluator-only scoring
```

## 2. Goals and non-goals

Implemented goals now include a strict environment boundary, bounded acquisition, neutral
observations, budget enforcement, a typed episode lifecycle, append-only transcript records,
Recorded Replay, answer freezing, post-freeze evaluator loading, and Scripted/Fake Policy process
isolation. The run chain also maintains an issued-Observation ledger and invokes the internal
evidence-controlled domain core when an answer is frozen.

Current non-goals are production security, arbitrary query languages, broad Log semantics, new
evidence thresholds, and changes to the frozen sibling `rca-core` repository. The LangChain
interface and LangGraph control loop are implemented against a Fake/Scripted model. An
accuracy-first `deepseek-v4-pro` model factory is available for provider-backed runs, but live and
blind diagnosis quality is not yet claimed.

## 3. Domain-core boundary

- The sibling `rca-core` repository is frozen, read-only, and no longer a package dependency.
- `src/rca_agent/core` owns the RCA Agent runtime domain: blind RCA100 case loading, bounded
  window/trace/topology queries, trace-stage extraction, candidate and evidence contracts,
  capability compilation, proposal and discovery, evidence assessment, candidate-state
  aggregation, and safe decisions.
- The copied implementations use the validated `rca-core` behavior as their starting baseline but
  now evolve independently inside `rca-agent`.
- Research stays outside runtime: Ground Truth, evaluator scoring, Oracle, Benchmark, holdout,
  audit, dataset evaluation, historical automatic experiments, CLI, and local research config are
  not present under `src/rca_agent`.
- Dataset-specific v3 providers are not duplicated in the domain core; `RcaEnvironment` remains the
  Agent-owned observation boundary.
- Evaluator-only answer-key loading is copied under top-level `evaluation/`, outside the inference
  package. Ground truth remains unavailable before answer freeze.
- Environment tools return neutral observations only. Root-cause confirmation runs later over
  observations explicitly referenced by a submitted Policy proposal.

## 4. Environment contract

`reset()` returns only event time, alert summary, one anonymous alert-entity reference, a bounded
observation horizon, capability descriptors, and the query budget. It omits case selectors, paths,
labels, evaluator notes, full facts, and query internals.

Available tools:

- `get_metric_series`;
- `compare_metric_windows`;
- `list_related_entities`;
- `get_service_trace_summary`;
- `get_trace_stage_breakdown`;
- `inspect_topology`.

`search_log_markers` is advertised as unavailable because current Log semantics do not justify a
portable bounded implementation.

Entities and traces use opaque stable references. The environment accepts only references already
issued in the current episode. All query responses use stable query fingerprints that contain no
core query reference or dataset identifier.

## 5. Data trust model

The trust states are `OBSERVED`, `DERIVED`, `IMPUTED`, `MODEL_INFERRED`, `MISSING`, and `UNKNOWN`.
Raw bounded samples and durations are `OBSERVED`. Equal-window summaries, ratios, structural trace
shape, and synthetic operation-to-service linkage are replayable `DERIVED` values. Missing source
coverage is never converted into a numeric value. Phase 1 emits no imputed or model-inferred value.

Only trusted observed facts and replayable deterministic derivations may later become confirmation
inputs. An Agent cannot set evidence strength or convert an observation into a final decision.

## 6. Isolation rule

The host chooses a case and observation directory when constructing the private adapter. The
Agent receives only the `RcaEnvironment` interface. Evaluator code lives outside the inference
package under top-level `evaluation/` and may load its private record only after the Episode Runner
freezes an answer. Evaluator configuration remains in the host and is never serialized to Policy.

Phase 2 places the Scripted/Fake Policy behind a serialized child-process boundary. The child gets
only the reset payload, typed Action script, and serialized host acknowledgements. Its working
directory is temporary and its environment variables are allowlisted. This validates process and
configuration separation for fixed policy code; it is not an OS sandbox for arbitrary code.

## 7. Episode and replay contracts

`EpisodeRunner` owns `CREATED -> ACTIVE -> ANSWER_FROZEN -> EVALUATED`. It records every action,
tool request and response, cost, budget transition, proposal, stop, freeze, evaluation transition,
and explicit failure. Reverse transitions, duplicate freeze/evaluation, and tools after freeze are
invalid.

Policy Actions are the closed union `CALL_TOOL | SUBMIT_HYPOTHESIS |
UPDATE_CANDIDATE_EVIDENCE | REQUEST_STOP`. A policy can register a mechanism and explicitly attach
episode-issued Observation references to it, but cannot supply a decision status, threshold,
evidence weight, SQL, path, or confirmation authority.

Transcript events form an append-only SHA-256 chain with normalized payloads and contiguous
sequence numbers. Recorded Replay is backend-free and evaluator-free. It validates the chain,
query fingerprints, request/response/cost grouping, budget arithmetic, lifecycle, and answer
binding. Backend Verification Replay remains deferred.

The Answer Artifact separately records policy proposals and the authoritative decision. The
default `CoreDecisionProvider` resolves trusted Catalog candidates, admits only `OBSERVED` and
deterministic `DERIVED` facts, evaluates candidate-owned conditions, and maps the safety-controller
result into the authoritative Agent status. Evaluator code remains outside the inference package
and can load its private record only after the answer is frozen.

`EpisodeRunner.preview_core_assessment()` exposes the same Core assessment as a read-only view while
the episode is active. It does not append transcript events, mutate proposal or observation state,
or freeze an answer. LangGraph uses the preview only for routing; finalization is still a
`RequestStopAction` handled by the Runner, which recomputes the authoritative decision.

Observations are episode-global immutable facts, while evidence admission is Candidate-scoped.
Each Candidate is evaluated only against Observation references explicitly attached to its own
hypothesis. `CoreAssessmentPreview` returns a safe per-candidate projection and discriminating
signal view without condition rules, thresholds, weights, strengths, or margin values.

## 8. LangChain and LangGraph boundary

`LangChainPolicyAdapter` owns prompt construction and strict `PolicyEnvelope` validation. Policy
can select one advertised tool action, submit one proposal, or request stop. The adapter does not
expose Core conditions, thresholds, weights, sufficiency logic, or decision authority.

`create_deepseek_policy()` supplies the current real-model binding: `deepseek-v4-pro`, thinking
enabled, high reasoning effort, and deterministic sampling for the schema-constrained action. The
provider reads `DEEPSEEK_API_KEY` from the host environment; runtime code never persists or logs it.

`LangGraphInvestigation` owns only the investigation loop: Policy -> Runner action -> Core preview ->
Policy, with a confirmed preview routed to Runner-controlled freeze. It must not call the
environment, Observation ledger, evidence controller, or evaluator directly. Budget, max rounds,
Transcript, Replay, and answer freeze therefore retain their existing semantics.

After every Core preview, the Graph rebuilds a Candidate Frontier containing active, supported,
contradicted, inconclusive, confirmed, and unsubmitted Candidate IDs. This frontier is a projection,
not a second authority: Graph and Policy cannot independently change Candidate status. A
deterministic `slowSQL` versus `dbNetworkLatency` E2E proves differential Trace planning,
Candidate-scoped evidence updates, safe confirmation, and replay integrity.

## 9. Roadmap and acceptance

- Phase 0: repository, architecture boundary, data isolation, trust model — complete.
- Phase 1: Environment Contract, RCA100 adapter, bounded tools — complete; acceptance recorded in
  `analysis/phase1_acceptance.md`.
- Phase 2: Episode Runner, Recorded Replay, answer freeze, evaluator-only loader and process
  boundary — complete; acceptance recorded in `analysis/phase2_acceptance.md`.
- Phase 3: deterministic v3 investigation-policy baseline using the phase 1 tools.
- Phase 4: LangChain/LangGraph Agent without decision authority — framework and Fake/Scripted model
  loop complete; real provider adapter and blind-model tests remain.
- Phase 5: development regression, preregistered evaluation, and comparative report.

Phase 3 must preserve the Environment, Action, Transcript, answer-freeze, and evaluator boundaries.
Its deterministic policy may choose actions and propose hypotheses, but authoritative confirmation
must still be delegated to the internal Agent domain core.

## 10. Claim boundary

The optional dynamic discovery increment is documented in `analysis/dynamic_discovery.md`.
The main LangGraph/Runner chain accepts a separate discovery adapter and a host-owned typed signal
and rule registry. Static Catalog candidates remain available. Core compiles selected predicates;
Runner registers episode-local candidates, executes bounded queries and explicitly binds evidence.
Dynamic mechanisms passing the mechanism-independent `generic-root-cause-v1` standard and Core
arbitration return `DISCOVERED_VERIFIED` as this investigation's root-cause result. No per-mechanism
human sufficiency contract is required. The standard requires completed independent support,
refutation and declared-alternative checks plus new verification evidence. This does not promote
the mechanism to Catalog or exhaust every theoretical alternative. Discovery is bounded by attempts, model waiting,
candidate count, rounds and the existing query budget. Preview remains read-only and definitions
are transcript-bound. The initial registry exposes only existing issued resource/trace capabilities.

t025 and t030 are exposed development regressions. Their controlled Trace difference validates the
environment interface, not Agent accuracy, blind generalization, production readiness, arbitrary
mechanism discovery, or multi-root handling.

The optional Unknown RCA Benchmark now supports a real Catalog holdout across Policy disclosure,
Discovery context, Core resolution and Recorded Replay. Initial live `deepseek-chat` probes on t025
and t030 proposed plausible Catalog-external mechanisms, including the correct slow-SQL mechanism
for t030, but Core correctly stopped at `SUPPORTED_HYPOTHESIS` because the model did not bind its
predictions to declared alternatives. See `analysis/unknown_rca_benchmark.md`; these two probes are
diagnostic observations, not accuracy estimates.

A preregistered seven-case batch now covers all available `dbNetworkLatency` and `slowSQL` cases.
The capability-aware rerun completed all 7 with valid replay: 4 proposed a GT-like mechanism, 1
(t021/slowSQL) reached a correct `DISCOVERED_VERIFIED`, and 0 wrong mechanisms were verified. Bounded
service/operation exploration and downstream DB-stage resolution removed all final `QUERY_FAILED`
outcomes. The proposed DB-network mechanism safely remained supported because
true RTT and database-server timing are unavailable. Treat these as diagnostic counts, not general
RCA accuracy; the deterministic post-freeze lexical scorer still requires semantic review. The
evaluation-only matrix in `analysis/capability_matrix.md` records full, partial and unsupported
mechanism families without exposing Ground Truth to inference.

The capability audit now inventories tools, supported entity types, bounds, executable Signal/Rule
contracts, raw Case/sample coverage, unmapped metrics and Environment initialization gaps. RCA100
contains aggregate JVM and application-thread-pool metrics in all 103 raw Case metric files. They
are now exposed as neutral service Signals and host-owned rules for issued `apm.service` entities.
Trace resolution also issues the owning service, so Discovery can select its JVM rules without
inventing a service name. These JVM/thread contracts remain PARTIAL pending holdout validation and
because queue-wait, old-gen/allocation/heap-after-GC and Full-GC-specific semantics are absent.
The compiler now distinguishes causal-support predicates from symptoms/exclusion predicates;
operation latency, JDBC client duration and low SQL share cannot be used to manufacture a second
root-cause support group.

The JVM/thread/GC increment now derives neutral trace participants even when a Trace lacks a complete
database stage bundle. Participants are ranked by downstream depth, and the diagnostic acquisition
policy screens no more than two services and ten JVM/thread Metric queries before Discovery. t007
and t055 both locate and screen the inventory service within this bound. Equal-window comparison now
also emits a typed scalar baseline ratio, so host rules can verify changes without fixed absolute
counts. Missing and zero baselines stay unavailable.

An evaluation-only 103-case profile found that the source active-thread aggregate has inconsistent
scale across cases. Runtime therefore does not treat it as a portable active/max capacity ratio; it
uses only change relative to the same signal's own baseline. Grouped thread CPU time is now mapped.
Generic GC count/pause can establish only generic GC pressure, and the compiler rejects a specific
Full GC claim unless Full-GC-specific support telemetry is present.

The frozen 12-case diagnostic probe completed 12/12 with valid replay, 0 `QUERY_FAILED`, and 0 wrong
`DISCOVERED_VERIFIED`. All six JVM/thread/GC positives issued their target service. DeepSeek proposed
the GT-like mechanism in t055 and t009, but both remained `SUPPORTED_HYPOTHESIS` because refutation
or declared-alternative checks were incomplete. This is a safe diagnostic result, not an accuracy
claim. See `analysis/jvm_signal_profile.md` and `analysis/jvm_diagnostic_probe.md`.

The follow-up discovery funnel now preserves every schema-valid raw draft and its initial/final
compile outcome in the replayable Transcript. Invalid structured provider JSON is also retained in
bounded form for post-freeze diagnosis. A deterministic repair may only canonicalize case and
whitespace in references to an already-declared alternative; it cannot change a mechanism, family,
rule, entity, polarity or causal explanation. On the rerun, 6/12 cases had a raw GT-like mechanism,
4/6 of those passed the initial contract, and 2/4 reached a correct `DISCOVERED_VERIFIED`; no wrong
mechanism verified. The safe mechanical repair did not fire because the observed failures required
entity/rule grounding or evidence capability rather than harmless text normalization. See
`analysis/jvm_discovery_funnel.md` and `analysis/jvm_discovery_funnel_interpretation.md`.

The next increment splits dynamic discovery into two structured calls over the same DeepSeek model
binding. Mechanism Discovery sees incident evidence and the issued entity graph, but no Signal/Rule
inventory; it proposes mechanism, entity, explanation and alternatives. Runner freezes accepted
mechanisms. Verification Planner sees those immutable mechanisms plus opaque capability handles and
selects evidence checks. It gets at most one bounded retry for invocation or compilation failure and
cannot rewrite causal semantics. Transcript and Replay bind drafts, frozen digests, plans, retries
and compiled candidates.

The clean 12-case rerun compiled at least one mechanism in 12/12 cases versus 7/12 in the one-stage
baseline. All five GT-like raw mechanisms passed the mechanism contract, versus four of six before.
Correct `DISCOVERED_VERIFIED` remained 2/12 (t055 thread exhaustion and t009 memory pressure), raw
GT recall was 5/12, and wrong verification remained 0. Planner retry recovered a compilable plan in
5/7 attempted cases, but rescued no additional GT-like result. The split therefore improves
contract reliability, not demonstrated semantic recall or final accuracy.

An initial naive split exposed four wrong verifications because sibling explanations reused generic
correlated evidence. The production revision makes siblings mandatory peer alternatives, permits
only contradiction checks to discharge alternatives, groups correlated thread-pool/GC evidence,
and blocks promotion while another same-origin mechanism retains two matched independent support
groups. See `analysis/jvm_two_stage_final.md` and
`analysis/jvm_two_stage_final_interpretation.md`.

Dynamic competition and refutation semantics are now refined. Verification reports explicitly
classify mechanisms as `verified`, `unresolved` or `failed`. A decisively failed mechanism no longer
blocks a same-origin candidate; missing evidence remains unresolved. When two strongly supported
mechanisms remain viable, the authoritative result is `MULTIPLE_SUPPORTED` with all supported
hypothesis IDs and no invented winner. Replay recomputes both this result and its candidate states.

Mechanisms and alternatives now carry a generic `MechanismClass`, while host capability handles
advertise a narrow `refutes` set. A contradiction can discharge exactly one alternative only when
those scopes match. Invalid exclusion claims and support/refutation conflicts are removed as
auditable safe plan repairs. Candidates with incomplete refutation or only one independent support
group are retained as supported/inconclusive but cannot reach `DISCOVERED_VERIFIED`.

The refutation-scoped 12-case run completed and replayed 12/12, compiled mechanisms in 12/12, and
returned 10 supported plus 2 inconclusive results, with 0 correct and 0 wrong verified. Eighteen
unvalidated refutation claims were removed across 23 compiled mechanisms. Eleven cases retained an
unresolved alternative. This shows the present bottleneck is missing discriminating capability,
especially traffic, network transport, locks and application defects. See
`analysis/competition_refinement.md` and `analysis/refutation_semantic_audit.md`.
