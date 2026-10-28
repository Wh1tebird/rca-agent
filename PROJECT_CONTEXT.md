# RCA Agent project context

## RCA Agent 1.0 current baseline — 2026-10-28

P1-1 through P2-5 are complete at commit `721a28221e76801bd07077b6495d097b7f6de2e5`, and
`main` is synchronized with `origin/main`. This section is the canonical current architecture;
the dated sections below are historical implementation notes and must not override it.

The stable host-owned layer is `EpisodeRunner`, `EvidenceProvider`, `CoreDecisionProvider`,
`Transcript`, `Replay`, `EpisodeRecovery`, `AuditRecord`, `OperationalMetrics` and `Answer Freeze`.
`EpisodeRunner` owns admission, Capability checks, Query execution, Observation issuance,
Candidate-scoped Evidence admission, Core decision, budget, stop, Handoff, projection writes and
Answer Freeze. Workspace and LangGraph are projections or compatibility caches only.

The replaceable model-facing layer is `ReasoningAdapter` plus the Inquiry, Discovery, Formalizer,
Planner and LangGraph compatibility wrappers. These components can propose a Candidate, Query,
Hypothesis, plan, next action, stop or Handoff, but cannot write Observation, Evidence, Transcript,
Workspace, Answer, budget, permission or stop authority. Replay does not invoke them.

Worker calls use JSON-only typed payloads and bounded framed messages. The parent remains the sole
state writer; timeout, cancel, exception, crash and malformed frame lead to stream cleanup and
terminate/kill/wait/reap. A late result cannot enter Transcript, Workspace, Observation or a frozen
Answer. `EpisodeRecovery` returns `RESUMABLE`, `RETRYABLE`, `TERMINAL`, `FROZEN` or `CORRUPTED`.
`RETRYABLE` is a host action plan for an explicit pending operation, not proof that recovery has
already happened. Cross-process restart cannot currently reconstruct live Environment or adapters.

Runtime never imports evaluator-only loaders. Evaluation loads Ground Truth/RCA100 and scoring data
only after Answer artifact Freeze. Current tests use deterministic fixtures and offline stubs; no
real provider smoke test is part of this baseline. RCA100, evaluator data, `rca-core` and frontend
remain outside this runtime change boundary.

## Historical P1-1 cancellation boundary assessment — 2026-10-26

The following assessment predates the delivered worker protocol and is retained for audit history.

P1-1 is **not implemented**. The existing Discovery and shared Formalizer/Planner calls still
use daemon threads with a waiting deadline; expiry does not cancel the underlying invocation.
Component protocols currently accept live Python adapter objects, and internal invocations use
closures. There is no serialized component-construction contract equivalent to ScriptedPolicySpec.
The scripted Policy subprocess is a lifecycle reference, not a drop-in executor for those adapters.

A fork-based draft was withdrawn before commit. Forking a multithreaded host can inherit locked
provider/native state; polling a Pipe before recv does not bound a partial-frame read; inherited
Transcript paths and handles are not isolated by copy-on-write; copying arbitrary adapter state
back is not a safe episode-admission contract. Passing deterministic tests did not establish the
requested guarantees. A safe replacement needs child-side adapter construction, bounded framed
messages and explicit worker cleanup, with Runner remaining the sole state writer. Keep this
execution-boundary work separate from any P1-2 Runtime reducer change.

## Deterministic batched Inquiry menu projection — 2026-10-15

The batched Inquiry host now keeps its complete typed menu private while projecting a bounded
v4 model menu. `project_inquiry_menu()` uses only issued entity/signal/measurement/window fields,
existing query fingerprints, capability availability and remaining budget. Focused entries satisfy
all non-empty fields of an open Need; when there are too few or no matches, the view exposes a
small explicitly labeled exploration layer. No Action is created or repriced. v1/v3 menu behavior,
Runner admission, pending transitions, Observation visibility and Replay semantics remain unchanged.

v4 records stable `qN -> aq-... -> act-...` mappings for every exposed entry, a projection digest
bound to the LoopState digest, and the existing attempt digest binds the full host view. Runner and
Workspace Replay reject a draft choice outside the current projected references before state or
budget mutation. Deterministic projection, alias, hidden-action, budget, digest and replay tests
pass; no model, Benchmark, Ground Truth or evaluator was used.

## Legacy query menu compatibility repair — 2026-10-09

Status: **全量确定性测试失败已修复，待独立复验**. The independent recheck's
808/1 and five isolated failures remain historical evidence. The unchanged legacy v1
single-round/two-query regression now passes three protected fresh-process repeats;
three protected pre-fix repeats failed. The current unconditional 64-item menu prefix
discarded a legacy caller's query reference, causing KeyError and retry. The prefix
projection now applies only to the newer host Inquiry contract; v1 retains the full
typed menu. No execution grant, tool capability, budget, Core or Replay rule changed.
Without a trustworthy earlier source snapshot, do not attribute the menu defect to
the preceding startup admission repair.

This run: 809 full, 326 query/Inquiry/Workspace and 100 admission/loop/worker tests pass.
Synthetic 6/6 and historical 23/23, total 54/54 replay; original four batch plus two
startup illegal records still reject. Exact 129-file protection plus four startup
evidence hashes remain unchanged in this repair. Source/tests Ruff and diff checks
pass; repository Ruff has 71 evidence-script findings (60 prior, 11 copied here).
Early diagnostic tests missed the guard entry point; the report discloses that gap.
Subsequent protected baseline and formal validation are distinct from those diagnostics.
Main/worker canaries pass 6/6 before/after; one double-wrapped canary launcher was
blocked, so do not claim zero total formal-log blocks. Keep prior independent verdicts.
See [repair evidence and limits](analysis/batched_inquiry_loop_legacy_query_repair_20261009.md).
No independent acceptance, model experiment or next phase is authorized.

## Bootstrap admission repair — 2026-10-08

Current status: **启动取证准入修复完成，待独立复验**. Preserve both second-batch
independent “不通过” reports. The last recheck closed the original four admission
defects but found startup authorization bypass and reported 758 passed / 1 worker
failure; the earlier implementation's 759 result below does not supersede that review.

New batched hosts lock loop envelope **3** (LoopState/LoopStep remain 1; Inquiry and
engineering enforcement remain 2). `select_startup_query()` accepts no Action and
reconstructs the real bounded acquisition policy from issued history, with model
selection disabled. Graph calls it before Policy. Runner retains a private selection
bound to its exact decision-event digest; a caller's event/source/Action cannot mint
a grant. Startup execution requires that current unspent selection and the complete
deterministically selected Action. Other state events expire it; ToolRequest consumes
it. Replay validates the decision basis/order and rebuilds the same selection. Saved
envelopes 1/2 need no added fields but their startup requests must match the reconstructed
host policy. Genuine legacy remains supported; missing/unknown batched versions cannot
downgrade. Roll back only new runs with explicit legacy + original Policy, retaining
this Replay reader and immutable existing artifacts.

Worker startup has a separate bounded ready deadline (default 5s); per-message task
deadline begins after ready/ack and is unchanged. Byte reads keep partial frames within
the deadline; all exits/cancellation reap and close the worker. The original .3s crash
test/expectations are unchanged. Its pre-repair rerun here passed, unlike the saved
review log; protected controlled startup delay independently demonstrates the shared
deadline misclassification. Exact old machine timing/guard overhead is not established.

Final protected validation: **809 full / 419 targeted / 10 isolated worker tests passed**,
no skips. Original four assertions pass; original four plus two startup illegal frozen
pairs are rejected by Replay. Saved synthetic 6/6, historical batch 23/23, total 54/54
Replay and freeze bindings pass. The 129-item protection list is 109 historical files
(54 pairs + one unpaired transcript), 12 synthetic files, 8 original illegal files;
all repair-before/final hashes match. Four additional original startup evidence files
were captured mid-repair and remain unchanged during subsequent verification.
Main/actual-worker GT/derived-score/network canaries block 6/6 before and after; formal
guard has no unexpected blocks. Source/tests and new helper Ruff pass; repository-wide
Ruff attempt reports preserved evidence-script findings, detailed in the report.

See [repair and limitations](analysis/batched_inquiry_loop_bootstrap_admission_repair_20261008.md).
No model, Benchmark, GT/score artifacts, sibling Core changes or Git commits; no
second-batch acceptance, real experiment or next phase is authorized by this status.

## Batched execution admission repair — 2026-09-30

Earlier implementation status: **执行准入缺陷已修复，待独立复验**. The original second-batch
independent report remains “不通过”. New batched episodes record loop envelope
version 2. Runner checks an accepted, current, unspent batch member against the
whole typed Action before recording PolicyAction, ToolRequest, rounds or cost;
planning, deferred, cancelled, consumed and old batch requests are rejected.
Host acquisition and Planner verification use separate Runner entry points,
with constrained Action shapes or compiled plan membership. Replay reconstructs
the same eligibility from prior events and checks the recorded execution source.
Saved version-1 batched artifacts retain their original event format while receiving
batch membership checks; legacy episodes retain their existing tool path.

Evidence and limits: [admission repair report](analysis/batched_inquiry_loop_admission_repair_20260930.md).
The four unchanged reviewer negative assertions fail before the repair and pass
after it; the four pre-repair frozen illegal transcripts are rejected by Replay.
Synthetic result-driven paths and six saved pairs still Replay. Guarded full suite:
759 passed; targeted 365 passed. Historical 23/23 and 54/54 Replay; the explicit 109 protected
files have identical repair-before and final SHA-256. No real model or Benchmark
was run, and no improvement in real investigation quality is claimed.

## Batched result-driven Inquiry loop — 2026-09-30

Earlier implementation snapshot: **实现完成，待独立验收**; the later independent
review found the admission defect now addressed above. The first engineering-contract
batch received an independent **通过** in `analysis/engineering_contract_final_recheck_20260929.md`;
earlier “待独立复验” sections below are historical implementation snapshots.

Second-batch hosts default to `inquiry_loop_mode="legacy"`. Explicit `batched` requires
strict host Inquiry v2 / enforcement revision 2, matching batched Policy, and Workspace
shadow or active. EpisodeCreated records loop contract 1 and immutable 1–2 physical
query batch / ≤10 Inquiry query limits. The host-owned Workspace reducer records stable
candidates, original alternatives, Needs, pending dispositions, exact batch members,
ToolResponse-backed assessments, planning/review counts and revision/digest. Model
steps are atomically admitted through the real typed menu and first-batch binding rules.
The same transition is rederived during Replay. Results can drive retain/defer/cancel
and subsequent query choice; no unselected pending action can execute. Query costs
are checked for the entire accepted batch before execution; Runner still owns actual
costs, hard stops, Core evidence and Answer freezing.

InquiryStateView exposes full typed visible observations within a deterministic 128 KiB
projection, with omitted references, and separately exposes the actual menu. A model's
Need assessment and candidate priority are tentative investigation state only. Direct
`answered` requires a successful, complete, visible matching value; it never means
Core verification or Task resolution. Discovery receives priority-ordered current
candidates plus all Needs, assessments, pending outcomes and execution bindings;
review state revision is separate from planning round. Batched Handoff version fields
are explicit; historical serialization/Replay remain unchanged. New Replay checks
recorded host input, step transitions, actual responses and final Handoff. Source:
`analysis/batched_inquiry_loop_implementation_20260930.md`.

GT/network-guarded synthetic full suite: 749 passed. Six synthetic Transcript/Answer
episodes Replay valid, including result-dependent branching, two-query, shared-binding,
nonfatal empty and fatal partial batch. Fixed historical whitelist: 23/23 paired batch,
54/54 total frozen Replay; 109 protected files unchanged during this implementation.
Ruff/diff checks pass. This is mechanism verification only; no real model, Benchmark,
GT scoring or independent second-batch acceptance has occurred.

## Inquiry host-version lock repair — 2026-09-29

Current status: **版本降级问题已修复，待独立复验**. The independent recheck's
"不通过" conclusion remains unchanged. Scope is only the missing/null Inquiry downgrade.
Host configuration now fixes Inquiry v2 by default; missing/null live markers receive full v2
schema and binding admission before canonical serialization. Unknown/mismatched markers reject.
EpisodeCreated records `inquiry_contract_version`; Replay locks that host contract, or the
first-round format of genuine old records, and rejects inconsistent/midstream downgrade.
Explicit legacy remains supported; complete rollback uses Adapter v1, Runner Inquiry 1,
Discovery/Runner bound v6 and enforcement revision 1. Keep the updated Replay reader.

Evidence: `analysis/engineering_contract_version_fix_20260929.md`. Independent 63, new version
regressions 291, complete deterministic suite 717 pass; Ruff/diff checks pass. Historical batch
23/23 and all 54/54 Replay/Answer bindings pass; only the explicit 109 Transcript/Answer whitelist
was hashed, unchanged during this repair. GT/network hooks cover actual Python main/worker
processes: no unexpected blocks, six deliberate canaries blocked. Native-library limitations
remain; no arbitrary rewrite authenticity or real-model improvement claim. No second batch.

## Engineering contract repair enforcement revision 2 — 2026-09-29

Earlier implementation status: **六类缺陷已修复，待独立复验**. The original independent review remains
"不通过"; the earlier completion section below is a historical implementation snapshot.

New hosts keep Inquiry v2 and bound handoff v7 and record
`EpisodeCreated.engineering_contract_revision=2` (Runner constructor default). v7 candidate
bindings also record `validation_revision=2`. Replay treats absent enforcement revision as 1;
it does not apply the new freeze/order/strict-extra rules to saved old records. Unknown revisions
and Inquiry versions are rejected rather than falling through legacy parsing. The Adapter rejects
unsupported raw versions before stamping the host version, including raw/tool-call fallback.
An unversioned model response to the live v2 Adapter is validated and stamped v2; historical
unversioned recorded drafts remain distinct from live model admission. Explicit adapter v1
continues to serialize the old unversioned contract.

Revision 2 reconstructs each freeze from accepted generated drafts, deterministic admission and
the unchanged same-source peer projection. Bound drafts are independently reconstructed from
original Handoff bindings and formalization events. Alternative text, ID and provenance must agree.
Hash-chain self-consistency is not an external authenticity anchor: an attacker able to rewrite
all original events and host revision metadata can fabricate a different internally consistent
record. Arbitrary tamper resistance or historical authenticity is not claimed.

External v7 bindings retain every original before checking normalized self alternatives and
duplicate IDs. Conflicting IDs cause explicit deferred formalization, without deleting explanations;
independent candidates remain isolated. Typed response IDs and Inquiry choice keys are checked
before dictionary construction. Same-class explanations concerning different entities remain legal.

Execution links and task completion must derive from a binding declaration preceding its actual
typed request and matching response. Policy stores the host history position at declaration;
callers lacking positions cannot claim a response for a later repeated offer. Runner and Replay
validate against Transcript sequences, so a cancelled later Need cannot borrow an old response.
This does not reinterpret historical evidence, resolve tasks semantically or replan queries.

Live v2 Choice extra fields are forbidden by both published schema and local runtime admission;
Runner/Replay use the same strict choice parser under revision 2. Missing required binding fields
retain the existing precise contract diagnostics. Menus expose actual request `max_points`/
`max_results`, public capability `max_results`, and partial-result/truncation semantics; actions,
windows, capacities and budgets are unchanged.

Explicit rollback for new runs: adapter `contract_version=1`, matching Discovery and Runner
`bound_handoff_version=6`, and Runner `engineering_contract_revision=1`. Retain all Replay readers;
never rewrite frozen artifacts. Unknown versions are rejected even under rollback. Workspace mode
is independent of these contract controls. See `analysis/engineering_contract_repair_20260929.md`
for actual validation, isolation coverage limits and this-run file protection evidence.

## Versioned engineering contracts — 2026-09-29

New live Inquiry uses contract v2 with mandatory binding mode, purpose and exact typed
measurement target. Host-built menus disclose measurement kind, real timezone-aware windows,
public units, limitations and return shape without changing menu order. Pure contract checks
are shared by Adapter, Runner and Replay; invalid Direct declarations receive the existing
bounded retry and are never automatically converted to Exploratory.

Bindings and physical executions are separate: one Need can own several query tasks, and
several predeclared Needs can share one physical response and budget charge. Tasks retain a
deterministic source binding ID; Handoff retains source/mode/target and exact response links.
Both shadow and active use the same new admission checks. Executed tasks remain ambiguous
or unavailable, with no semantic resolution or Core evidence authority. Exploratory scheduling
does not claim direct coverage; utility coefficients and query/call budgets are unchanged.

Bound formalization v7 has separate cause and original-alternative typing slots. The host
preserves original causal text and competing explanation ID/text/provenance, including multiple
alternatives of the same cause across rounds. Strict normalized self alternatives are rejected
before freeze. Per-candidate isolation and the existing four-original/one-retry limits remain.
New Alternative provenance is representation metadata only; Core evidence semantics are unchanged.
Legacy omitted fields stay omitted in serialization, and historical handoff v1–v6 retain old Replay.
Host rollback chooses Inquiry v1 plus bound schema/Runner v6; Workspace routing is a separate gate.

Acceptance: analysis/engineering_contract_acceptance_20260929.md. 374 deterministic tests,
Ruff and diff checks pass; all 23 paired-batch frozen results and 31 other saved results replay
with unchanged files. No provider or Benchmark ran. An existing capability-audit test initially
read real answer keys during full-suite verification; its label dependency was replaced by an
empty synthetic directory, retaining capability assertions. Final checks use no real GT; the
earlier reads were not used for implementation or tuning. This establishes engineering
contracts only; per-query semantic feedback and dynamic replanning remain a second batch.

## Host-bound Discovery formalization v6 — 2026-09-28

Retained Inquiry candidates are no longer regenerated in a single free-form Discovery call.
The host fixes their identity and causal text; the same Qwen Discovery adapter fills typed
entity/class/explanation/alternatives independently, or explicitly defers. A failed candidate
cannot discard other candidates. Four original attempts and one retry per item bound model
cost; remaining originals stay visible as capacity-deferred. A separate new-discovery call
may add provenanced mechanisms in remaining slots. Typed classification and causal explanation
remain model-authored and fallible; identity preservation is not evidence verification.
Planner and Core still own executable verification and sufficiency respectively.
Transcript/Replay v5/v6 verify the host binding and aggregate without changing old replay rules.
v6 adds conservative, Case-independent explicit-terminology class compatibility rejection
and feedback; it does not infer evidence or assign a class. Multiple matched layers remain
ambiguous, and unmatched terminology is not rejected. Evidence and the competing alternative
are excluded from the typing prompt to prevent re-diagnosis; the issued entity graph remains.

## Per-candidate partial repair continuation (handoff v4) — 2026-09-28

Version 4 admits independently valid groups from a partially invalid Discovery
retry instead of reverting wholly to the first output. Initial safe formal drafts
remain byte-for-byte unchanged; valid explicit discards remain protected. Previously
deferred candidates may receive a valid draft. Candidate groups are admitted in
handoff order only after full schema/contract checks; duplicate/conflicting names
are skipped, and new discoveries need explicit valid provenance. No causal draft
or evidence is synthesized by the host. v2/v3 Replay retain their old projection.

Missing original drafts are listed with exact candidate ID/text and the model's
reported reason, both in the existing single retry and final diagnostics. Replay
recomputes those lists and the v4 partial merge. Frozen previous t030 projection
retains its retry GC draft while h1/h2 stay deferred. Synthetic Runner tests prove
partial success can reach Planner, with omission/retry errors still explicit.
314 tests pass; Ruff/diff checks pass; 15 selected historical Replays remain valid.
One same-config Qwen t030 check is recorded under
`analysis/discovery_handoff_v4_qwen_20260928/`; consult its report for real outcome.

## Exact original-draft recovery (handoff v3) — 2026-09-28

Handoff v3 preserves a uniquely matching original formal draft when only model
disposition bookkeeping is inconsistent. It restores an exact text link, never
rewrites draft contents, guesses synonyms, overrides an explicit revision/discard
or grants evidence authority. Duplicate/cross-linked provenance remains rejected.
Repair receives a safe projection to preserve; if repair still fails, valid
original drafts can reach Planner independently. Version 2 replay keeps the
previous fail-closed projection, so old frozen artifacts remain unchanged.

One authorized t030 run stopped at Inquiry duplicate selections before handoff;
safe diagnostic records identified repeated query refs on both attempts.
The adapter now deduplicates the two bounded choices after validating each need
and offered ref, retaining all hypotheses/needs and executing the first valid
binding once. No invented query or evidence assignment is permitted. This last
dedup change has local validation only, not another live call. See
`analysis/discovery_handoff_v3_qwen_20260928/report.md`.

## Inquiry attempt diagnostics — 2026-09-28

Application Inquiry no longer silently enters Discovery after two invalid model
responses. The same single retry receives structured code/field feedback; if it
fails, the episode freezes via the existing policy-failure boundary. Explicit
no-offered-query exhaustion is distinguished from provider failure. Each attempt
is recorded through Runner and checked by Replay; recording artifacts expose only
safe codes/locations, never raw model reasoning or provider exceptions. Empty,
truncated, schema and query/need-binding failures are distinguished. Older records
remain compatible. Core and budgets are unchanged. Local regression: 302 tests.

## Discovery handoff contract v2 — 2026-09-28

Two-stage Discovery with a nonempty Inquiry handoff now validates candidate
accounting and causal-text preservation before Planner. `kept` must preserve
the original causal text (case/whitespace normalization only); revisions need
issued observation references and explicit retained/deferred handling of the
original. Discarded candidates cannot simultaneously become formal mechanisms.
Additional discoveries require explicit source and observation provenance.
The same Discovery component gets at most one semantic-contract repair attempt;
failed repair quarantines invalid drafts and preserves original Inquiry ideas.
Valid independent drafts cannot be dropped by repair. This is not an additional
Agent and does not change Core sufficiency or establish causal truth from refs.

Transcript and Replay recompute the v2 contract, repair outcome and exact draft
bindings. Historical transcripts remain version 1. Standalone Formalizer mode
and empty handoffs retain their existing paths. Workspace/Answer link original
Inquiry ideas to their own mechanism, not a revised cause's Core verdict.
Full local regression: 297 passed, Ruff/diff checks passed; 11 historical
Recorded Replays passed. Frozen t030 output exposes seven contract violations;
this is an offline audit, not a new model run or VERIFIED result. No live model,
GT scoring, frontend changes or Benchmark in this increment. Acceptance:
`analysis/discovery_handoff_contract_acceptance_20260928.md`.

## Active query utility and Planner feedback — 2026-09-28

Authorized real Qwen two-case checks are complete, both Replay-valid and
INCONCLUSIVE. t007 executes all 10 offered Inquiry queries; t030 records no
Inquiry rounds and cannot validate active scheduling. Each compiles one
non-target mechanism; thread-pool/slow-SQL Planner success is not established.
See `analysis/query_strategy_qwen_20260928/report.md` for safe diagnostics.

Active now retains legal runners-up across rounds and deduplicates executed,
not merely proposed, queries. Version-2 tasks use a deterministic utility score
for declared comparator coverage, new groups, location gain, cost, duplication,
symptoms, alignment and missing-data risk. Query-only capabilities come from
exact published menu actions and cannot grant evidence authority. Planner retry
receives indexed role errors, normalized prediction counts and safe eligible
support handles. Replay checks v2 scheduling/action bindings and recomputes
Planner diagnostics; legacy serialization and scoring remain compatible.
No Agent, Playbook, Core rule or budget change. Full tests: 268 passed; Ruff
and diff check pass. See `analysis/query_strategy_fix_20260928.md`.
Real Qwen t007/t030 checks were separately authorized and are recorded under
`analysis/query_strategy_qwen_20260928/`; do not infer model improvement from
the deterministic tests alone.

## Conditional Discovery format formalization — 2026-09-24

An opt-in `discovery_format_formalizer` now runs after the original semantic
Discovery and before Planner only when Discovery's structured payload is
invalid but its bounded raw JSON still anchors every mechanism's exact text,
source proposal and issued causal entity. A valid Discovery payload passes
through with zero extra model calls. The Qwen format adapter may fill malformed
typed fields once, but host validation rejects missing/added/rewritten causes,
changed source/entity, loss of valid classifications or existing alternative
texts. Unsafe raw output is not repaired. Transcript and Recorded Replay bind
the repair source, output and generated drafts; legacy runs remain compatible.
Existing `mechanism_formalizer` and frontend routes are unchanged; Core,
Registry and GT boundaries are unchanged.

One real Qwen t007/t030 run each used the new opt-in combination. Both Discovery
responses were valid and bypassed the format AI. t007 compiled one thread-pool
mechanism and ended `SUPPORTED_HYPOTHESIS`; t030 independently proposed a
complex-SQL mechanism but Planner twice rejected its verification plan, so no
Candidate compiled and the final state was `OUT_OF_SCOPE`. Both Replays passed.
See `analysis/investigation_discovery_formalizer_20260924/report.md`.

## Active Inquiry with legacy Discovery diagnostic — 2026-09-24

The existing standalone GT-free Qwen recorder now has an opt-in
`--mechanism-stage discovery` mode. It holds active Inquiry selection, model,
schema, budget, Planner, Registry and Core constant while replacing only the
Formalizer stage with the existing semantic Discovery adapter; the default
Formalizer mode and frontend route are unchanged. One t007/t030 run each
completed with valid Replay. t007 froze three mechanisms, compiled one and
ended `SUPPORTED_HYPOTHESIS`. t030's Discovery independently recovered a
complex-SQL database-execution mechanism despite Inquiry not retaining it,
but Planner produced no plan for that mechanism and no candidate compiled;
the final state was `OUT_OF_SCOPE`. This isolates a live Discovery opportunity,
not a verified diagnosis or accuracy gain. See
`analysis/investigation_active_discovery_20260924/report.md`.

## Formalizer candidate-accounting contract — 2026-09-24

The opt-in Qwen Mechanism Formalizer now accounts for every retained candidate
with either a semantics-preserving typed mechanism or a typed skip reason and
short explanation. Missing verification evidence is not a valid reason to
avoid formalization; Core still decides evidence sufficiency later. An empty or
incompletely accounted first response receives one bounded, transcripted
semantics-preserving retry. A final response with unaccounted candidates is
rejected instead of silently completing with zero mechanisms. Retry cannot drop
or rewrite a mechanism accepted from the first response. Replay checks the
versioned accounting, retry trigger, candidate identities and final payload;
older formalization transcripts retain their v1 semantics. This changes no
Core verification rule, Capability, query budget, or GT boundary.
Two real Qwen t007/t030 runs with the existing GT-free script each formalized
and compiled three candidates with valid Replay; neither needed the retry.
Their final states were `SUPPORTED_HYPOTHESIS` and `INCONCLUSIVE`, respectively.
This validates the live formalization chain, not the live retry branch or a
diagnosis-quality improvement. See
`analysis/investigation_formalizer_accounting_20260924/report.md`.

## Active Qwen two-Case rerun after query routing fix — 2026-09-24

t007 and t030 were each rerun once through the standalone active Inquiry plus
Qwen Formalizer entrypoint. Both frozen Answers and Recorded Replays are valid.
Each Transcript proves later-round active queries actually executed: t007 4/10
offered Inquiry actions and t030 5/10, versus 1/10 each in the first pass.
Both Formalizer calls were rejected with recorded `ValueError`; neither Run
froze a formal mechanism or compiled a Candidate, so end-to-end formalization
remains unverified. Core stayed unchanged and returned `OUT_OF_SCOPE` in both.
See `analysis/investigation_active_qwen_rerun_20260924/report.md`.

## Active Inquiry + Qwen Formalizer two-Case first pass — 2026-09-24

A standalone GT-free script now uses Qwen `json_schema` with
`investigation_workspace_mode="active"` and the opt-in dedicated
`LangChainMechanismFormalizerAdapter`; the frontend keeps its legacy route.
Only t007/t030 were run. Both frozen Answers replay. Each Transcript proves one
active query selection and execution. t007 Formalizer completed with zero
mechanisms; t030 Formalizer was rejected with `ValueError`. Neither produced a
compiled Candidate, so formalization-to-Core is not yet live-validated. No Core,
Catalog, Capability, Case, budget or existing Prompt change was made. See
`analysis/investigation_active_qwen_20260924/report.md` for paired old-chain
counts, artifacts, fixed command and next review questions. Stop before a batch.

## Coverage v3 formal Candidate projection — 2026-09-24

Coverage now projects typed frozen mechanisms, compiled dynamic Candidates and
their recorded Core verification states from pre-freeze Transcript events.
Only `unresolved`/`verified` formal Candidates count as covered families;
`failed` and unassessed drafts do not. The projection is included in versioned
`mechanism-coverage-v3` input and recomputed by Recorded Replay. Historical v1/v2
coverage records keep their original digest and evaluation semantics. Frozen
t030/t007 v2 Runs still replay; deterministic v3 recomputation covers t030's
previously missed database-execution and database-transport formal Candidates,
while t007 is unchanged. No Qwen call, Core rule, query or Answer changed.
Coverage remains `off` by default and shadow-only. Details:
`analysis/mechanism_coverage_qwen_20260924/report.md`.

## Coverage v2 and two script-driven Qwen shadow runs — 2026-09-24

The Phase 5 Inquiry false gap is partially corrected: `mechanism-coverage-v2`
conservatively recognizes an explicit single-family causal phrase in an untyped
open Workspace candidate. Ambiguous text remains unclassified, and v1 coverage
input digests remain readable by Replay. The new `evaluation/record_coverage_qwen.py`
reuses the existing Qwen live pipeline without the frontend and records per-Case
Manifest/Transcript/frozen Answer with Replay validation. A two-Case t030/t007
Qwen run completed and replayed 2/2; both Core results were supported, neither
verified. GT was loaded only in the separate post-freeze offline evaluator.

The live t030 run exposed a remaining Coverage projection defect: a slow-SQL
Discovery mechanism compiled and remained Core-unresolved, but its Inquiry
Workspace track was absent, so Coverage still reported database execution
missing. Do not enable active Coverage. The next repair must include nonfailed
formal/compiled Candidates in replayable Coverage input, then use the saved
Transcript for deterministic regression before any more paid model calls. See
`analysis/mechanism_coverage_qwen_20260924/report.md`.

Last consolidated: 2026-09-23 after MechanismCoveragePolicy shadow

## MechanismCoveragePolicy shadow — 2026-09-23

Phase 4 now has a separate `mechanism_coverage_mode="off"|"shadow"` gate; `off` is the safe
default and no active mode exists. The deterministic policy runs once after Core decision and
Workspace finalization, immediately before Answer freeze. It audits only two general domains:
database execution/transport/connection acquisition and JVM thread-queue/GC/memory/CPU-load.
Workload/availability and DiagnosticPlaybook remain design-only.

Input is restricted to the read-only InvestigationWorkspace Snapshot, live typed Candidate family
and layer, legally issued Observation identity/signal/status, Episode-published Capability signal
summaries and the safe incident entity type. Capability presence cannot trigger a domain by itself.
Output is a non-authoritative coverage diagnostic: covered/missing families, generic discriminator
needs, advertised observability state and replayable trigger basis. It contains no Candidate seed,
query, ordering, budget, threshold, evidence weight or Core status and is ignored by the Workspace
reducer, Planner, Core and Answer fields.

Transcript records the versioned input and assessment with independent digests. Recorded Replay
rebuilds the pre-event Workspace, reconstructs the legal input, reruns the policy and requires exact
equality; the coverage event must immediately precede Answer freeze. Old Transcripts default to
coverage `off`. Focused tests prove `off` and `shadow` have identical Tool Requests, budget,
Candidates, formalization records and Core decision, while rehashed diagnostic tampering is
rejected. Ruff, all 234 tests and `git diff --check` pass. Details:
`analysis/mechanism_coverage_shadow_acceptance.md`.

This is ready for offline shadow measurement, not active Candidate creation. Do not add a
DiagnosticPlaybook or activate coverage until fixed-input analysis shows stable incremental recall,
acceptable noise and no confusion with Planner, Capability or Core closure failures.

## Active discrimination and Mechanism Formalizer — 2026-09-23

The approved third batch is implemented without changing Core verification. Workspace mode now
supports opt-in `active` routing while retaining `shadow` as the default and `off` as rollback. An
active Inquiry round persists and validates its structured Need-to-Action bindings, resolves the
Episode Registry before execution, creates Tasks, computes the derived Matrix and executes only the
highest-ranked exact capability binding among actions the model already offered. Other queued
actions from that round are suppressed rather than replaced or added, so query budget never grows.

Active selection is recorded before the tool request with the Matrix digest, selected Task and
Capability, offered and suppressed query fingerprints. The later Task completion is bound to the
exact ToolResponse and Observation refs. A successful query remains `AMBIGUOUS`; active routing has
no causal adjudication or verification authority. Unresolved Needs become task-owned
`UNVERIFIABLE`. Replay checks both the Inquiry-offer binding and the exact query result.

An opt-in `MechanismFormalizer` protocol now accepts retained open InvestigationCandidates plus the
safe entity graph and observations, but no Registry rules, Core thresholds or GT. Each typed output
must preserve its track ID, source semantic digest and normalized causal mechanism. Semantic drift
is rejected before freeze. The existing immutable mechanism, Planner, compiler and Core path is
unchanged, preserving `candidate_track_id -> formal_mechanism_id -> compiled_core_candidate_id`.
Legacy `discovery=` behavior remains available; `DiscoveryMechanismFormalizerAdapter` provides a
strict migration wrapper and the new formalizer path is opt-in. Ruff and all 227 tests pass; no
provider or Benchmark was run. Details: `analysis/investigation_phase2_active_phase3_acceptance.md`.

## Discrimination tasks and Matrix shadow projection — 2026-09-23

The approved second batch is implemented as a retrospective, behavior-neutral shadow. Once the
host has resolved the Episode-specific `EvidenceRegistry`, Runner transcripts a digest-bound safe
capability projection containing opaque capability refs, entity/signal bindings, independent
evidence groups, query fingerprints, costs and executability. It contains no Core thresholds or
Ground Truth.

Every recorded Inquiry evidence need becomes a persistent `DiscriminationTask` against the
candidate's recorded alternative. An exact capability binding is admitted only when the completed
legacy query's fingerprint and entity match an executable Registry recipe. Natural-language needs
without that exact binding become task-owned `UNVERIFIABLE`; the Candidate itself remains open and
its overall reachability is derived. A Candidate with both resolved and unavailable primary tasks
is `PARTIALLY_ACTIONABLE`, while `DISCRIMINATION_COMPLETE` remains distinct from Core verification.

`DiscriminationMatrix` is recomputed from Transcript-reduced Tasks and is stored only inside the
read-only Workspace Snapshot. Transcript persists task creation/selection/completion, the Registry
snapshot and a small shadow-selection comparison containing the Matrix digest—not a second
authoritative Matrix state. The current ranking prefers explicit contrasting predictions with an
exact executable binding, then lower query cost, with deterministic IDs as tie-breakers. It does
not interpret observations as causal proof. Therefore an executed successful query remains
`AMBIGUOUS` unless a future active-stage adjudicator can justify a distinguishing outcome.

The shadow path runs after legacy Inquiry queries have completed. It does not execute, replace,
reorder or increase queries, and it cannot affect Discovery, Planner or Core. Replay verifies exact
query/Observation binding, task state transitions, Registry and Matrix digests, frozen Snapshot
equality, active-episode ordering and legacy Phase 1 Answer digests. The Workspace `off` mode
remains the rollback. Ruff and all 224 tests pass; no model or Benchmark was run. The next approved
batch is Phase 2 active plus Phase 3 Mechanism Formalizer, subject to review of this shadow contract.

## Investigation Workspace contracts and shadow projection — 2026-09-23

The first batch of the investigation-continuity plan is implemented without changing acquisition,
Discovery, Planner or Core behavior. `EpisodeRunner` now owns a non-authoritative
`InvestigationWorkspace` reconstructed deterministically from new Transcript events. Schema-valid
Inquiry ideas receive a separate `candidate_track_id`; that ID never substitutes for a formal
mechanism ID or a Catalog/Dynamic/Core Candidate ID. The frozen Answer carries a read-only
`InvestigationWorkspaceSnapshot`, and LangGraph carries the same projection separately from its
existing Core-derived `CandidateFrontier`.

The track identity hashes only the causal mechanism, optional mechanism class and causal entity.
Changing or adding a comparison alternative therefore does not split one causal idea into a new
track; alternatives are added by a replayable transition and retain their own deterministic IDs.

Candidate stage (`TENTATIVE|ACTIVE|DEFERRED|CLOSED`), non-authoritative disposition
(`OPEN|DISFAVORED`) and formal/Core links are separate. `CLARIFY` must preserve the semantic digest;
semantic change uses one atomic `CANDIDATE_SUPERSEDED` event. `REACTIVATE` changes only
`DEFERRED -> ACTIVE`, while `RECONSIDER` changes only `DISFAVORED -> OPEN`. The task/reachability
contracts for the next phase are defined, including task-owned `UNVERIFIABLE` and the derived
`DISCRIMINATION_COMPLETE` name, but no discrimination task is generated or used for query selection
yet.

Recorded Replay folds the Workspace events, validates their referenced prior events, queries and
Observations, and compares the recomputed Snapshot with the frozen Answer. Older Answer Artifacts
without a Workspace projection remain valid. `investigation_workspace_mode="off"` disables the
shadow events and projection as a rollback; `shadow` is the default. This Phase 0/1 baseline passed
222 tests and Ruff before the Phase 2 shadow increment.

## Evidence-role correction and versioned verification semantics — 2026-09-22

Verification-plan normalization now removes two auditable role errors: a class-specific positive
predicate cannot support an unrelated primary mechanism, and it cannot contradict the same primary
mechanism. Partial positive support is retained as unresolved rather than automatically treated as
decisive failure; a matched valid contradiction or complete observed support failure still fails a
Candidate. New Candidates record `generic-root-cause-v2`; historical compiled Candidates default
to v1 so immutable Transcript Replay is preserved.

A fixed-input recomputation of the saved t007 and t030 Runs used the same frozen mechanisms, plans
and issued Observations. t007 thread-pool saturation moved from failed to unresolved with one
support group; t030 complex SQL moved from failed to unresolved with both SQL support groups and
only unresolved alternatives. Neither became verified. Inquiry history also now skips the null
payload of candidate-evidence-update events, removing the t030 post-verification `AttributeError`.
Details: `analysis/t007_t030_frozen_replay.md`.

## Investigation console dark workspace — 2026-09-21

The Vue console now uses a Chinese-first, dark observability-workbench layout while preserving the
existing API/Mock data-source boundary and all recorded Run semantics. The left product navigation,
compact incident header, investigation tabs, three-column Candidate/Timeline/Evidence workspace and
bottom Core decision panel mirror the actual investigation chain rather than a chat interface.
Candidate proposal/formalization/compilation remain visually distinct, the Timeline remains the
primary surface, and Developer Mode still exposes only the existing safe projection. Local Vite
origins on localhost or 127.0.0.1 are accepted by the FastAPI host so a fallback development port
can connect without weakening non-local origin access.

## Extended provider generation budget — 2026-09-20

DeepSeek thinking calls now default to a 65,536-token generation ceiling and a 300-second
provider wait. Single-case and batch Unknown RCA entrypoints expose the token ceiling explicitly,
record all three limits in result artifacts, and invalidate cached batch results when the limits
differ. Runner permits a Discovery wait up to 600 seconds and the benchmark defaults to 300
seconds. These changes only prevent premature provider truncation/timeouts; they do not change
prompts, mechanisms, evidence rules, Core thresholds or Ground-Truth isolation.

## Semantics-preserving mechanism completion — 2026-09-20

Two-stage Discovery can explicitly report a mechanism that its disposition reason already names
but omitted from the formal mechanism array. Runner accepts only a typed hint whose mechanism text
is copied verbatim from that recorded reason and whose source hypothesis already exists. It makes
at most one completion call. The returned draft must preserve the hinted source ID and normalized
mechanism text; semantic drift is discarded before freezing or planning. The completion attempt,
initial payload, allowed hints, returned payload, accepted drafts and failures are transcripted.
Recorded Replay checks the one-attempt limit, initial-payload digest, prose-to-hint binding and
the exact set of drafts allowed into the normal freeze, Planner, compiler and Core path. Core
verification rules and thresholds are unchanged. One live Qwen t021 probe directly formalized and
compiled the slow-SQL direction without requiring completion; it replayed successfully and ended
`INCONCLUSIVE`. The completion branch itself is proven by deterministic Runner and Replay tests.

## Inquiry retention and additive verification feedback — 2026-09-18

The inquiry arm records each schema-valid exploration round in the episode Transcript before its
selected query executes. A stable candidate record merges only exact normalized mechanism and
alternative text; reused model IDs for different ideas remain distinct. Discovery receives every
recorded direction, and Runner plus Recorded Replay compare the handoff against the recorded rounds.
The Answer Artifact includes every tentative direction, its inquiry query/Observation references,
Discovery disposition, any exact formal mechanism/compiled candidate link, and Core-derived evidence
status. If Discovery never runs or fails, the recorded ideas remain visible as unformalized ideas;
they never gain Core decision authority by being retained.

The existing Verification Planner can answer a Core evidence gap with one or two new capability
predictions. The host appends these to the immutable previous plan, recompiles the same frozen
mechanism with the original discovery evidence snapshot, and registers a versioned replacement
candidate. Old predictions and prior evidence cannot be removed. Runner executes only new bounded
queries and binds their observations to the revised candidate; Core recomputes its unchanged generic
verification standard. Transcript and Recorded Replay validate the plan revision and evidence
binding. A failed candidate is not reopened for feedback. The inquiry benchmark exposes a bounded
query-limit setting so small live probes can reserve queries for verification and feedback.

## Optional local Qwen provider — 2026-09-17

`src/rca_agent/qwen_policy.py` binds the same Policy, Discovery and Verification Planner
interfaces to the Qwen3-30B-A3B-Thinking-2507 OpenAI-compatible model. The holdout benchmark
defaults to this local provider; `--provider deepseek` retains the old provider. The local binding
reads `RCA_LOCAL_API_KEY` from the process environment or prompts without echo on an interactive
terminal. It never writes the key into code or case reports. It accepts HTTPS or a loopback-only
HTTP tunnel by default. Public HTTP requires an explicit `--allow-insecure-http` opt-in; this
protects against accidental cleartext transmission while permitting the user's stated exception.
One complete 12-case Qwen diagnostic is recorded in `analysis/qwen_model_comparison.md`:
1 GT-like formal mechanism, 7 cases with compiled mechanisms, and no correct or wrong verified
answer. The result is constrained by structured-output failures and model-run variance.

## Inquiry handoff and Core feedback — 2026-09-17

The optional inquiry arm now gives formal Discovery a typed, transcript-bound summary of the
latest tentative hypotheses, evidence questions, and issued query results. The Runner validates
query/Observation references, and Recorded Replay rechecks the handoff against prior tool
responses. Mechanism Discovery can report a kept/revised/discarded reason for each inquiry
hypothesis; omitted reasons are explicit `unreported` diagnostics and carry no decision authority.
Core feedback may execute at most two previously unattempted queries from a frozen candidate plan
when Core reports corresponding missing evidence. Runner still owns execution and candidate
evidence binding. It does not rewrite mechanism semantics or predictions. The existing 12-case
dataset offered no such unattempted frozen-plan query after verification, so this path is covered
by deterministic tests but did not fire in live runs. See
`analysis/inquiry_handoff_acceptance.md` for both completed probes; the second required targeted
recovery of three provider-interrupted cases.

## Evidence-need-guided acquisition — 2026-09-17

The optional `--acquisition inquiry` benchmark arm adds a non-authoritative, revisable
pre-discovery inquiry. One structured DeepSeek call per round proposes up to three preliminary
mechanisms, competing explanations and evidence questions with contrasting predictions. It chooses
one or two short menu references from host-built, entity-scoped query options. The host resolves
those references to typed actions and records the need-to-query binding; Runner alone executes them.
After their observations arrive, the next round replans. At most five rounds / ten selected queries
are allowed, with one bounded output retry per round. A preliminary idea is not a Core proposal,
does not freeze verification predictions, and has no effect on sufficiency or stopping. Official
Discovery, Planner, Core and Recorded Replay remain unchanged. The old fixed and AI acquisition
arms remain available for comparison.

## Final capability acceptance — 2026-09-16

All raw metric entity/field pairs discovered across 103 source files now have a query mapping:
65 public entity/signal mappings in total, 35 added during this plan. Field coverage is distinct
from per-entity sample availability, initialization coverage (still 85/103) and causal sufficiency.
Host audit of aggregate counters finds frequent decreases; service errors sometimes exceed aligned
workload. Four unsupported derivations are explicitly disclosed rather than fabricated. Existing
source GC deltas remain usable without re-differentiation.

`LangChainAcquisitionSelector` selects only opaque host-built query references on issued entities.
The diagnostic AI screen makes at most two selection calls and ten queries; Runner executes every
action with normal cost, scope and replay checks. The fixed screen remains default. Planner also
receives missing-signal and unsupported-derivation explanations.

The final fixed/AI arms each ran the same 12 cases with matching model and query budgets; all 24
replayed. Fixed: 12 compiled, 10 supported, 2 inconclusive; AI: 11 compiled, 6 supported,
5 inconclusive, 1 out of scope. Neither arm verified a correct or wrong root cause. AI querying
increased successful queries from 62.9% to 81.0% and eligible evidence coverage from 42.8% to 52.5%,
but alternative resolution fell from 14.3% to 10.0%. These are diagnostic observations with model
variance, not causal improvement claims. Post-freeze coverage reuses Core checks with candidate
bindings, complete responses and query fingerprints. Semantic review distinguishes partial
memory/GC directions from exact GT recovery. See `analysis/capability_plan_acceptance.md`,
`analysis/capability_final_acceptance.md` and `analysis/capability_semantic_review.md`.

## Capability expansion — 2026-09-16

The Environment now exposes 18 additional entity/signal mappings: seven aggregate JVM/thread
statistics, five deployment resource/availability metrics, five instance metrics and Pod restart
count. Bootstrap metric descriptors disclose public signal names, entity types, units, semantics
and explicit unavailable signals, without private source names, paths or Core thresholds.
The runtime registry registers neutral query recipes for every issued supported entity type.
Request-volume baseline growth is one support group; latency/error-rate baseline changes are
symptom-only rules. Service data has error counts but no documented error ratio; ratio comparisons
are available for operation/instance entities. Missing and zero baselines retain existing semantics.

Aggregate pool sizes, waiting-thread statistics and task/restart counters have insufficient scale
or reset semantics for portable utilization/rate derivations. They remain queryable context and do
not refute lock contention, OOM or throttling. No new refutation authority is invented. The fixed
12-case run preserves the existing five-signal screen, model and query budget: all cases compiled
and replayed; raw lexical GT recall stayed 3/12; 10 supported and 2 inconclusive; correct and wrong
verified both 0. Eight cases selected new request-volume rules, but all retained unresolved
alternatives. This measures verification-capability expansion, not autonomous acquisition quality.
See `analysis/capability_expansion_comparison.md` and `analysis/capability_expansion_delivery.md`.

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

## Web console read-only integration — 2026-09-21

The repository now has an independent `backend/` FastAPI presentation host and the existing
Vue console can switch between its retained Mock source and this API source. The host dynamically
enumerates public RCA100 `task.json` metadata, records incomplete/invalid Case data explicitly,
and does not embed a benchmark manifest. It projects only a stored Run Manifest, append-only
Transcript and frozen AnswerArtifact into Case, Candidate, Timeline and Evidence UI DTOs.

Run storage lives under ignored `backend/var/runs/<run-id>/` and is intentionally distinct from
`analysis/`: benchmark result files can include evaluator/GT content and are never served. The
backend never imports evaluator code. Answer digests and Recorded Replay are rechecked before a
final decision is exposed; an invalid replay stays available only as non-authoritative forensic
timeline data. Core standards, Ground-Truth isolation and the Agent-visible Environment contract
are unchanged.

The first local generated `t021` demonstration Run used only the real Environment, EpisodeRunner,
bounded trace query, Transcript, AnswerArtifact and Replay. It did not call a model or Benchmark.
Live model execution, persistent jobs, concurrent storage and event streaming remain follow-up
work. API contract: `backend/docs/api-v1.md`.

## Scripted Live Run control path — 2026-09-21

The web host now accepts `POST /api/v1/cases/{case_id}/runs` for the single server-approved
`scripted-trace-demo` profile. The request cannot supply provider credentials, model URLs, raw
queries, budgets, Core controls or holdout selectors. A bounded background worker persists
`queued -> running -> succeeded|failed` transitions atomically, streams the Runner's append-only
Transcript to the existing Run directory, freezes the normal AnswerArtifact and requires Recorded
Replay before success. A host restart converts stale queued/running manifests into an explicit
`HostRestarted` failure rather than silently resuming or claiming success.

The Vue API data source can start this profile, poll its persistent status and load the final
projection after completion. A real endpoint smoke on t021 returned HTTP 202, ran two bounded Trace
queries, produced 15 Observations, froze an `INCONCLUSIVE` answer and passed Replay. This validates
the control and persistence path only; it is not an AI diagnosis or accuracy result. Model-backed
profiles, cross-process durable queues, cancellation and event streaming remain deferred.

## Qwen Live Run path — 2026-09-21

The web host now accepts a second server-approved `qwen-investigation` profile. Provider model,
endpoint, timeout, query budget, round limit and insecure-HTTP opt-in are host-only settings; the
API accepts none of them and rejects this profile with HTTP 503 when `RCA_LOCAL_API_KEY` is absent.
Credentials are not persisted in Manifest, Transcript, Answer or presentation responses.

The profile uses the existing Qwen `json_schema` adapters and the full GT-free path:
`Inquiry -> transcript-bound handoff -> Discovery -> Verification Planner -> EpisodeRunner ->
Core -> frozen Answer -> Recorded Replay`. `InquiryInvestigationPolicy` is a non-authoritative
runtime acquisition policy: it selects only host-offered typed queries, records revisable ideas,
and cannot set evidence status or a final decision. The scripted profile remains available.

A real t030 smoke run (`run-live-87c10830a6214f2f`) used Qwen3-30B-A3B-Thinking-2507, recorded five
Inquiry rounds, executed fourteen queries, froze two mechanisms, compiled two dynamic candidates,
finished `SUPPORTED_HYPOTHESIS`, and passed Recorded Replay. The console projection shows one
failed and one unresolved formal candidate plus the discarded tentative ideas. No Ground Truth or
evaluator was loaded; this proves integration and replayability, not diagnosis accuracy.

## Console UI polish — 2026-09-21

The Vue console now presents investigation records as a desktop Observability workbench rather than
a flat event log. The candidate rail separates formal Discovery mechanisms from Inquiry-only ideas
and displays their lifecycle plus Core-derived FAILED/UNRESOLVED state. The Timeline groups existing
safe Transcript projection into Incident, Inquiry, Discovery, Verification and Core Decision phases;
it does not infer new facts or alter any recorded state. The Evidence Inspector supports browsing
the frozen evidence set and developer-only Summary/References/Raw tabs. The final decision now
shows Replay status alongside overall Core status and the separate formal-candidate states.

The UI consumes the same API DTOs; no backend contract, Agent policy, Core rule, Prompt or GT
boundary changed. Lint, Vue type-check/build, Ruff and the full Python test suite pass. Browser
automation was unavailable for screenshot capture on this host, but the real replay-valid t030 API
payload (6 candidates, 43 Timeline events, 57 evidence records) and Mock mode were used for static
and integration-shape validation. The Vite chunk-size warning remains non-blocking.

## DeepSeek batched pending projection repair — 2026-10-10

Implemented the bounded batched pending-contract repair. New batched Episodes use loop envelope v4
with an actionable/history public projection; LoopStep remains v1 and nested Inquiry Draft remains
v2. The internal pending ledger and unavailable migration semantics are unchanged. Pending-set
admission now reports structured missing/unexpected/duplicate/invalid-state diagnostics, while
retry feedback is included in the actual Adapter input and attempt digest. Historical loop v3
 Replay remains supported. Deterministic repair validation passed; a separately authorized single
 DeepSeek t030 batched re-test later stopped after one accepted query and did not complete the
 dynamic loop. No Benchmark, GT or evaluator was run.
