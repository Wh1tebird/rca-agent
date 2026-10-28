# RCA Agent handoff

## Current handoff — RCA Agent 1.0 documentation hardening — 2026-10-28

P1-1 through P2-5 are complete and pushed at `721a28221e76801bd07077b6495d097b7f6de2e5`.
The documentation/comment hardening pass is complete without runtime, public API, event schema,
Replay digest, EvaluationReport, timeout, cancel, Recovery or Answer Freeze changes. The canonical
architecture is the stable host-owned layer (`EpisodeRunner`, `EvidenceProvider`,
`CoreDecisionProvider`, `Transcript`, `Replay`, `EpisodeRecovery`, `AuditRecord` and `Answer Freeze`)
plus the replaceable `ReasoningAdapter`/legacy compatibility layer. See
`docs/architecture/overview.md` and `docs/architecture/glossary.md`.

Future work is limited to final acceptance and documentation maintenance. Do not start another
structural phase. Cross-process restart still requires the host to reconstruct live Environment and
adapter dependencies for a `RETRYABLE` operation; it is not automatic recovery. Real provider
smoke tests remain outside the offline deterministic test boundary.

## Historical P1-1 stopped at execution-boundary blocker — 2026-10-26

This section predates the delivered worker protocol and is retained as historical audit context.

Status: **未完成；按本次任务 E 条停止编码，试验 runtime/test 改动已撤回**. The original
daemon-thread timeout defect remains. Do not interpret earlier passing tests or progress messages
as P1-1 acceptance. No incomplete runtime fix has been committed or pushed to GitHub.

Current chain: LangGraph routes a stop/unknown-candidate investigation to
EpisodeRunner.discover_candidates(). The one-stage Discovery uses its own Thread/Queue; two-stage
Discovery, bound typing, completion/repair, MechanismFormalizer, Planner retry and plan extension
use _invoke_discovery_component(). Returned payloads are admitted by Runner before compilation,
Workspace events and Answer freeze. Timeouts currently preserve stage-specific diagnostics and
bounded retries; Discovery timeout does not itself introduce a new Answer StopReason. Preserve
that behavior explicitly when replacing execution, rather than silently changing it to POLICY_TIMEOUT.

The attempted minimal fork executor was rejected during final review:

- The real host can be multithreaded; Python emitted a fork/deadlock warning in the fake suite.
  Existing adapters contain live LangChain/provider runnables, and closures can capture Runner;
  switching to spawn requires a reconstruction/serialization contract that does not exist yet.
- Pipe.poll followed by recv is not a deadline-safe partial-frame protocol. A complete response
  must also be size-bounded and received under the same monotonic deadline.
- Fork copies memory but inherits file descriptors and accessible Transcript paths. That alone
  cannot establish the requested sole-writer boundary. Generic adapter __dict__ synchronization
  also cannot be used as an accepted-state protocol merely to preserve fake call counters.
- Tests that only check a delayed marker remains absent do not prove PID reaping, partial-frame
  cancellation, SIGTERM escalation or every Formalizer/Planner integration path.

Minimum follow-up split, still P1-1 and independent of P1-2:

1. Define a host-only worker spec/factory for existing supported Discovery/Formalizer/Planner
   adapters and deterministic fixtures. Construct provider clients inside a fresh interpreter;
   send only typed call contexts, never Runner, Workspace, Environment, Transcript paths or handles.
   Define fake state/call accounting explicitly instead of reflecting arbitrary object attributes.
2. Add one controlled worker host with bounded startup/task deadlines, bounded framed responses,
   terminate/kill escalation, reaping and pipe closure on success, failure, timeout and cancellation.
   Keep parent-side schema/semantic admission, failure codes, retry counts and freeze semantics.
3. Route every existing component operation through it. Add handshake-driven PID/reaping tests,
   partial output, late output, exception/normal results, and actual Runner Formalizer/Planner
   timeout-to-freeze/Replay tests before claiming completion. No real model call is needed.

The withdrawn draft's full fake suite and repository Ruff passed, but are not evidence for a
delivered fix. Rerun required checks after rollback; their result is recorded in the chat. Do not
start P1-2 to compensate for this blocker. No RCA100, evaluator/ground-truth, sibling rca-core or
frontend change belongs to this work. A later user message authorized Git submission after
completion; that condition is not met, so the blocked assessment remains local.

## Batched Inquiry menu projection — 2026-10-15

Status: **确定性投影实现完成，待只读代码审查和单 Case 低预算复验授权**. The v4 host keeps
the complete menu for internal resolution and Replay, but the model receives only a deterministic
focused projection (8–20 target) or a small exploration layer when the open Need has insufficient
matches. Filtering uses host fields only; no embedding, LLM filtering, prefix truncation, or new
Action is involved. Stable alias records bind exposed `qN` aliases to `aq-...` menu refs and
`act-...` action fingerprints; completed results retain separate `qry-...` refs. Hidden actions fail
the projected-reference admission gate before state, budget or execution changes.

Full deterministic pytest passes, targeted source/test Ruff passes, and old v3 Replay plus current
v4 Transcript paths remain covered. Repository-wide Ruff still reports the pre-existing and newly
present evidence-script findings; no claim of a clean repository-wide Ruff run is made. No real
DeepSeek call, Benchmark, GT/evaluator read or Core/Runner admission relaxation was performed.

## Legacy query menu repair handoff — 2026-10-09

Status: **全量确定性测试失败已修复，待独立复验**. Review the unchanged
`test_scripted_inquiry_uses_two_queries_in_one_round_without_a_model` and
[new report](analysis/batched_inquiry_loop_legacy_query_repair_20261009.md).
Protected original repeats: 3 failed before, 3 passed after. Restore the complete typed
menu for explicit Inquiry host v1; retain the newer-contract 64-item projection.
The missing old query reference caused a KeyError retry, not two successful rounds.
Do not claim a proven chronology linking that defect to the preceding startup repair.

Actual full suite: 809 passed; directed sets 326 and 100 passed. Historical 54/54
(batch 23/23), synthetic 6/6 and original six illegal-record rejections hold;
129+4 exact protected hashes unchanged. Source/tests Ruff and diff pass; whole Ruff
has 71 evidence-script diagnoses, listed by file/rule/source in the evidence directory.
Early unguarded diagnosis and one incorrectly double-wrapped canary launcher are
explicit limitations; later formal validation used the protected entry and actual
worker probes 6/6. Host versions/rollback and all admission/Replay rules are unchanged.
Preserve prior independent “不通过” reports; arrange independent recheck, do not
close the second batch or begin real-model experiments/another phase automatically.

## Bootstrap admission handoff — 2026-10-08

Status: **启动取证准入修复完成，待独立复验**. Review only this repair using
[new report](analysis/batched_inquiry_loop_bootstrap_admission_repair_20261008.md);
keep all previous “不通过” reports. Do not close the second batch, run real experiments
or start a new phase. The independent recheck's 758/1 result remains historical evidence.

New batched envelope is 3, host-locked in EpisodeCreated. The Graph invokes
`Runner.select_startup_query()` before Policy; it takes no submitted Action. Runner
holds a private selected Action plus exact decision-event digest, verifies current
state and full Action at both execution boundaries, and consumes it at ToolRequest.
Caller-supplied authorization events cannot create/refresh that private grant. Shared
deterministic policy reconstruction validates selection and startup execution during
Replay. Saved envelopes 1/2 retain bytes/format but illegal startup requests fail;
legacy remains explicit and legal. Startup after handoff/stop/freeze, stale or consumed
selection, parameter changes and Inquiry disguised as startup all fail before cost.

Worker readiness has an independent 5s default deadline, and task deadlines still
begin after ready/ack. Partial output cannot bypass the read deadline; crash, timeout,
normal completion and host cancellation reap and close actual workers. Original .3s
test unchanged; its local pre-repair rerun passed, so do not claim exact reproduction
of the saved timing failure. Controlled protected .4s startup proves the old shared
deadline can classify a crash as timeout before readiness; isolated tests verify both
deadlines, cleanup and valid completion.

Validation: 809 full, 419 targeted, 10 isolated worker passed; original four assertions
remain closed. Four original illegal and two startup illegal saved pairs rejected;
synthetic 6/6 and historical 23/23, 54/54 Replay valid. Explicit 129-file hashes match
repair-before/final (109 historical, 12 synthetic, 8 illegal); separately captured four
startup evidence files unchanged after capture. Guarded main/worker canaries 6/6 before
and after, zero unexpected formal blocks. Source/tests/new helpers Ruff and diff checks
pass; broad Ruff has preserved evidence-script findings, not a clean repository claim.

Rollback: explicit legacy host + matching original Inquiry Policy for new episodes,
retain current Replay; never switch an active episode or rewrite historical files.
Isolation is Python audit/worker wrapping, not an OS sandbox. Replay proves internal
consistency only, not authenticity of completely coherently rewritten history. Exact
old machine startup overhead and real model quality remain unmeasured.

## Batched execution admission repair handoff — 2026-09-30

Earlier implementation status: **执行准入缺陷已修复，待独立复验**. Keep the second-batch independent
“不通过” report as the historical verdict; a separate review must check this
repair. See [repair report](analysis/batched_inquiry_loop_admission_repair_20260930.md)
and the unchanged four reviewer repro scripts copied into its evidence directory.
Do not start real experiments or the next batch from this handoff.

New batched runs record `inquiry_loop_version=2`. Direct `Runner.handle` checks
current scheduled member identity, full typed Action and unspent batch status
before any transcripted execution or budget change. The Graph routes deterministic
startup acquisition and compiled verification through constrained host calls.
Workspace Replay checks the episode's prior accepted batch, tool request source,
startup facts and plan capabilities; old version-1 and legacy frozen results remain
readable. Roll back new runs by selecting `inquiry_loop_mode="legacy"` with the
matching original Inquiry Policy; retain this Replay reader and never switch a
running episode's mode.

Guarded validation: 759 complete deterministic tests, 10 new admission tests,
four original repros now passing, six saved synthetic pairs and 54/54 historical
pairs Replay, 109 historical hashes unchanged. Main/worker GT and network
negative probes blocked 6/6 before and after. Python audit hooks do not guard
native calls that bypass the interpreter; arbitrary coherent rewriting of all
host facts has no external authenticity anchor. No real-model effect is known.

## Second-batch implementation handoff — 2026-09-30

Earlier status: **实现完成，待独立验收**; the later independent review found the
admission defect now addressed above. First-batch independent final recheck passed;
the earlier pending statuses below are historical. Next action is a separate
independent acceptance of the result-driven batched Inquiry loop, using
`analysis/batched_inquiry_loop_plan_20260930.md` and
`analysis/batched_inquiry_loop_implementation_20260930.md`. Do not start a real
model experiment, Benchmark or third batch from this handoff.

Use the guarded entry `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python
analysis/run_engineering_guarded.py pytest -o addopts='' -q tests` in the repo;
final implementation run: 749 passed. New tests are
`tests/test_batched_inquiry_state.py` and `tests/test_batched_inquiry_graph.py`
(32 targeted passed). Six generated synthetic Transcript/Answer pairs under
`analysis/batched_inquiry_loop_20260930/evidence/` are checked by the fixed-name
`audit_synthetic_evidence.py`; the explicit 109-file historical whitelist is
checked by `audit_whitelist.py` (23/23 batch, 54/54 total, no bytes changed).
Ruff and diff checks pass. Guarded main/worker canaries blocked 6/6 expected
attempts; no unexpected test access. Read the report for process/hook limits and
intermediate failures that were fixed before the final run.

Inspect live admission, pending transitions, state view and Handoff in
`src/rca_agent/investigation/inquiry_loop.py`; Runner and Workspace events in
`runner/episode.py` / `investigation/workspace.py`; Graph/Policy wiring in
`langgraph_orchestrator.py`, `langchain_policy.py` and `batched_inquiry_policy.py`;
recorded verification in `runner/replay.py`. Verify that actual Observation
content changes the next query, cancelled/deferred pending cannot execute,
two members finish before the next Inquiry call, shared bindings preserve one
physical cost, review/priority reaches Discovery, and legacy history remains
byte-identical. New Episode rollback is `inquiry_loop_mode="legacy"` with the
existing first-batch Policy and retained newer Replay reader. A running batched
Episode cannot switch contracts midstream. The model's semantic judgment and
real-world effect remain unmeasured; Core is unchanged.

## Missing/null Inquiry version repair — 2026-09-29

Status: **版本降级问题已修复，待独立复验**. Preserve both original independent
"不通过" reports; next action is independent re-review, not the second batch.
The shared host/version resolver governs Adapter, Policy, Runner and Replay. New live omission/null
executes full host-v2 admission, while unsupported/conflicting declarations reject before accepted
state or tools. EpisodeCreated stores the Inquiry host version. Recorded v2 omissions/null and
mixed versions reject even after chain/Answer reconstruction. True old formats infer once and lock;
explicit legacy remains usable. Default enforcement revision 2/bound v7 implies Inquiry 2.
Complete new-run rollback: Adapter 1, Runner `inquiry_contract_version=1`, matching Discovery/Runner
bound 6, Runner enforcement revision 1. Retain new Replay and immutable historical artifacts.

See `analysis/engineering_contract_version_fix_20260929.md` and its evidence directory:
independent 63/63, version regressions 291/291, full deterministic 717/717, no skips; Ruff/diff pass.
Historical 23/23 batch and 54/54 total valid; 109 explicitly listed Transcript/Answer files unchanged
between this repair's snapshots. Guarded main/actual Python worker validation has zero unexpected
blocks; six controlled GT/derived-file/network canaries block before access. Python audit hooks
are not an OS sandbox for native libraries. No real model, Benchmark, GT/score read, Core changes,
Git staging/commit/push or dynamic replanning. Whole-record consistent rewrites still lack external
authenticity guarantees; old interruptions, shadow t024, language semantics and actual model effect
remain out of scope. The earlier six-defect completion claim below is a historical snapshot.

## Independent-review repairs — 2026-09-29

Earlier status: **六类缺陷已修复，待独立复验**. The original independent review and its "不通过"
conclusion are preserved. This section supersedes the earlier completion claim for current status.

All six reported boundaries were reproduced before implementation: the exact 32-test appendix
produced 21 passed / 11 failed under GT/network guards. Those assertions are now repository
regressions. Repairs bind frozen Alternatives to accepted drafts and original formalization,
explicitly defer conflicting external IDs without deletion, reject unknown versions before execution,
enforce declaration/request/response order, forbid live Choice extras at actual admission, and
disclose typed result limits. Added focused tests cover raw fallback, direct Runner/Replay entry,
isolated workers, strict version types and explicit historical compatibility.

New hosts record enforcement revision 2 in EpisodeCreated and v7 bindings. Inquiry v2 and bound
handoff v7 schemas remain; absent enforcement revision means historical revision 1. Rollback for
new runs is Inquiry adapter v1 + matching Discovery/Runner bound v6 + Runner
`engineering_contract_revision=1`; keep new Replay support and leave old artifacts immutable.
Unknown versions never select legacy by default. Workspace mode remains a separate route control.

Validation and protection evidence: `analysis/engineering_contract_repair_20260929.md`, with
persistent logs/manifests under `analysis/engineering_contract_repair_20260929/`. The repeatable
guarded entry is `uv run python analysis/run_engineering_guarded.py pytest -o addopts='' -q`;
it guards actual pytest threads and isolated policy-worker subprocesses. It is a reviewed Python
audit harness, not an OS sandbox for arbitrary native programs. Initial full-suite harness errors
and the temporary legacy-task timing regression are disclosed in the report, alongside final results.

No real provider/model, Benchmark, real GT content, sibling Core modification, staging/commit/push,
Core rule/threshold/score/budget change, new Agent/Playbook or second-batch implementation.
Saved history remains old semantics; this-run digest comparison cannot prove first-implementation
immutability. Replay has no external trusted anchor and cannot establish arbitrary rewrite resistance.
Actual model effect, old four interruptions, shadow t024, and broad natural-language type/entity
semantics remain unresolved/out of scope. Next action is independent re-review of the six repairs.

## First engineering-contract batch completed — 2026-09-29

All six approved repairs are implemented: Need multi-query bindings throughout Policy/Runner/
Task/Handoff/Replay; typed complete menus; explicit Direct/Exploratory; pre-accepted deterministic
mismatch checks; host-owned original Alternative identity/text/provenance; strict self-alternative
rejection before freeze. Four baseline regressions first failed on old behavior. New fake-model
tests cover shared physical execution without double charge, mode and identity tampering,
per-candidate isolation and the full synthetic Inquiry-to-Answer/Replay chain.

Current contracts: live Inquiry v2, bound handoff v7. Rollback is host-selected Inquiry v1 and
matching Discovery schema/Runner bound_handoff_version=6, while keeping new Replay readers.
Workspace off/shadow/active is separate from schema version. Old omitted metadata remains
omitted; old self alternatives are not retroactively rejected. Core rules, coefficients, budgets,
capability inventory and sibling rca-core are unchanged. Alternate typing may be UNKNOWN or
deferred; no competing explanation is deleted to obtain verification.

Validation: 374 tests pass; Ruff and git diff --check pass. Read-only checks pass all 23 valid
paired-batch frozen Answers and 31 additional saved results (handoff v1–v6), verifying digests,
exact freeze bindings, Replay and unchanged source file bytes. No real model or Benchmark;
no Git staging/commit/push. An existing capability-audit test initially read real answer keys
during full-suite checks, contrary to this batch's restriction. This was disclosed, and that
test now uses an empty synthetic label directory with unchanged capability assertions. Final
checks pass all 374 tests with real answer-key file opens intercepted (0 attempts);
earlier reads were not used for code decisions. Existing user
modifications were preserved by local patches.
Delivery: analysis/engineering_contract_acceptance_20260929.md and
analysis/engineering_contract_replay_final_20260929.json; the audit script supports repeatable
read-only checking and optional exclusive new-report creation.

Remaining boundaries: actual model effect is unmeasured; exact causes of the four old accepted-to-
recorded interruptions and shadow t024 failure remain unknown. Natural-language cause/entity and
general type semantics cannot be fully verified mechanically. Second-batch per-query semantic
feedback, candidate state changes and dynamic replanning were not implemented.

## Active query paired batch — 2026-09-28 (completed)

All 24 attempts finished, with no extra model reruns. 23 frozen Answers/Recorded Replays
are valid; shadow t024 failed with ValidationError and has no scored Answer. Active t009,
t021, t041 and t098 accepted a final Inquiry attempt but rejected the following action,
leaving that round unrecorded and stopping before Discovery. Keep this distinct from model
recall failure; only safe ValueError class/digest is available, not the exact failing field.

Post-freeze manual-reviewed scores: both arms have 2 exact mechanism directions (t007,
t055) plus 2 GC-family-related directions (t040, t059), all 4 retained in final Answer.
Full GC was not explicitly diagnosed; t055 Pod identity was not scored. t021's thread-pool
cause with slow SQL as consequence is NOT a SQL-root-cause hit. Matched typed formalization
is 3 Cases per arm; matched mechanisms compile in 2 shadow / 3 active
Cases. No VERIFIED. Active recall in its four incomplete rounds remains unknown.
Raw trusted-data queries: shadow122/163, active106/150. Seven uninterrupted paired Inquiry
Cases: shadow78/105, active71/101; no demonstrated query effectiveness improvement.
Active selection runs and reranks, but engineering interruptions and uncertain Planner /
evidence closure prevent claiming end-to-end or accuracy improvement.
See analysis/active_query_batch_20260928/comparison.md and runs/offline_scores.json.
334 tests, Ruff and diff check pass. No inference changes, frontend/API changes or commit
in this batch. Next: diagnose accepted-to-recorded Inquiry rejection using deterministic
reproduction and safe reason codes; do NOT infer a Core fix from this batch.

User authorized a first batch focused on GT-like proposal/retention and effective queries.
Existing record_active_formalizer_qwen now supports --paired --workers 2 and shadow/active
query arms. All inference components are shared and fingerprinted; only Workspace routing
mode changes. The original 12 historical Case IDs were preregistered without loading GT.
All 12 initialize; global initialization remains 85/103 with 18 invalid Cases excluded.
Catalog uses the recent smoke default unchanged in both arms, NOT per-GT Catalog holdout.
Do not compare this score directly with historical holdout accuracy.

Artifacts: analysis/active_query_batch_20260928/runs; protocol in the parent directory.
No inference code or prompt changes during the running batch, no score-driven reruns.
Post-freeze scoring uses evaluation/active_query_comparison.py only after all inference ends.
It checks Answer/Transcript binding and Replay before each GT load. Shadow Transcript omits
selected_actions by design: raw recorder Inquiry query counters are therefore zero; offline
evaluation must derive exact executed refs from validated handoff query_links and offers from
normalized draft choices, not interpret zero as no investigation. No frozen outputs rewritten.

## Host-bound original formalization v6 — 2026-09-28

Production Discovery adapters now formalize each retained Inquiry candidate independently.
Host-owned immutable binding carries the original ID, exact causal text, source and digest;
the model schema contains only typed details or an explicit defer reason. Model output cannot
rename the original or substitute its alternative. At most four originals are attempted, each
with one retry; over-capacity originals remain explicitly deferred. Independent new Discovery
is retained as a separate provenance-checked channel and cannot replace original candidates.
This changes mechanism-stage model call allocation, not query budgets or Core standards.
Replay v5/v6 recomputes bindings, ordered attempts, host-built drafts and novel admission.
v6 also rejects typed classes incompatible with explicit original causal terminology; this
conservative taxonomy guard never selects a replacement class or root cause. Multi-layer text
allows all matched types; unmatched text has no added constraint. Formalization inputs exclude
competing text and evidence to avoid re-diagnosis while typing; issued entity graph remains.
Legacy adapters and historical v1-v5 Replay retain their prior behavior.

Initial checks: 323 tests, Ruff and diff check passed; 16 historical Replays valid. Initial Qwen
t030 smoke is under analysis/bound_discovery_qwen_20260928; typed-prompt t030/t007 under
analysis/bound_discovery_qwen_typed_20260928. Final v6 smoke and report under
analysis/bound_discovery_v6_qwen_20260928; consult its report before batch
execution. No GT-driven tuning, frontend/API changes, extra Agent or Git commit.

Final v6 smoke completed: t030 Inquiry 7/7 executed, 3 originals frozen, SQL correctly typed
database_execution and compiled (1/3 total); t007 Inquiry 10/10 executed, 3 originals frozen
and 3 compiled. Both SUPPORTED_HYPOTHESIS and Replay/Answer valid, not VERIFIED. Remaining
failures are per-candidate Planner gaps and evidence/alternative closure, not original loss.
326 tests pass. Existing entry now reports bound-formalization counts separately from legacy
Formalizer counters. No full batch or GT scoring performed; freeze this implementation before
a query-strategy-only paired comparison. See final report for exact run IDs and limitations.

Real v4 t030 outcome: Inquiry 5 successful rounds, 7/7 dynamic queries executed,
9 total queries. Discovery first/retry both submitted only the pool alternative
as kept for the SQL original; no independent valid formal draft was available.
Result OUT_OF_SCOPE, zero freezes/compiles, Planner not invoked, Answer/Replay
valid. No additional model calls. This differs from the frozen previous run where
retry GC can now be admitted. See analysis/discovery_handoff_v4_qwen_20260928/report.md.

## Partial repair continuation v4 — 2026-09-28

Approved per-candidate Discovery repair implemented in discovery_handoff.py.
Initial safe formal groups and valid discarded dispositions are protected;
independent safe retry groups and provenanced new discoveries may be added without
requiring every candidate to pass. No host-generated MechanismDraft, synonym
inference, evidence admission or Core sufficiency change. Deferred original ideas
can be completed; conflicting names/source claims are not automatically merged.
The existing one retry now includes pending original formalizations with exact
IDs/text and reported reasons. Transcript records final pending originals and
partially_repaired when incremental valid groups survive; Replay recomputes both.
Old handoff v2/v3 behavior is preserved for historical records.

314 full tests passed, Ruff/diff checks passed, 15/15 historical Replays valid.
Frozen latest t030 now admits its retry GC draft and keeps two other ideas deferred;
synthetic Runner test confirms initial/retry independent groups reach Planner.
Only one same-config real t030 check authorized this increment, stored in
analysis/discovery_handoff_v4_qwen_20260928; report determines remaining acceptance
gap. No automatic extra calls, no GT read, no frontend/backend API edits or commit.

## Handoff v3 and bounded duplicate choice normalization — 2026-09-28

Approved Discovery/Planner continuation increment implemented. v3 quarantine
recovers only unique exact original-to-existing-draft links for kept/deferred
bookkeeping, without changing a MechanismDraft or replacing causal meaning.
Explicit discarded/revised candidates, ambiguous originals, duplicate drafts and
cross-links are not recovered. Repair protects the recovered original draft and
receives its safe projection. Existing v2 Replay still uses the old projection.
Synthetic end-to-end test confirms the rescued draft reaches Planner and compiles
even if semantic repair repeats the bad link. Missing Planner plans are explicitly
rejected per mechanism on both attempts; no new Core rule or Agent.

Frozen t030 audit retains the exact original SQL draft, defers two other invalid
ideas, rejects the unlinked pool cause and passes contract validation. 13 selected
historical Replays valid. The single real t030 run in
`analysis/discovery_handoff_v3_qwen_20260928/` instead failed Inquiry twice on
duplicate query choices and froze INCONCLUSIVE, Replay valid, before Discovery.
Added bounded duplicate-choice dedup retaining all hypotheses and needs while
rejecting unoffered/unbound refs. This post-run change has no live revalidation;
no automatic extra model calls. Consult the report for remaining acceptance gap.

## Inquiry failure visibility and bounded retry — 2026-09-28

Inquiry adapter now distinguishes empty/truncated/schema/reference failures and
can validate a single standard InquiryDraft tool-call payload when structured
parsing supplies no parsed value. Binding failures carry safe field locations.
Application Inquiry retries once with specific code/fields, as before; exhausted
attempts raise InquiryUnavailableError rather than silently submit runtimeUnknown
and bypass Inquiry. No-offered-query exhaustion remains explicitly recorded and
does not call the model. Graph flushes bounded attempt diagnostics even on failure,
Runner/Replay validate them, and the existing recording entry includes them in
evidence/result artifacts. No raw reasoning, provider error text or keys enter
these diagnostic events. Legacy runs do not require the new event.

Local full regression passed (302 tests); Ruff and diff checks passed. One real
t030 run is separately authorized under analysis/inquiry_diagnostics_qwen_20260928;
consult its report for outcome, not the prior zero-Inquiry run. Past runs did not
persist failure details, so their exact Inquiry errors cannot be reconstructed.

## Discovery handoff contract repair — 2026-09-28

The approved complete repair is implemented, not only a prompt adjustment.
New `investigation/discovery_handoff.py` validates exact candidate accounting,
kept causal semantics, explicit original handling for revisions, issued factual
observation provenance and additive new-mechanism declarations. Runner allows
one repair by the existing Discovery component before Planner, protects valid
drafts from repair loss, and quarantines invalid output while preserving the
original Inquiry candidates. Workspace does not silently supersede originals;
Answer links never borrow a revised mechanism's Core status.

Contract v2 is enabled only for the two-stage Discovery path with a nonempty
Inquiry handoff, not standalone Formalizer. Recorded Replay recomputes checks
and accepted drafts; legacy v1 records remain valid. Frozen t030 auditing caught
the kept SQL-to-pool rewrite plus discarded-but-formalized thread/GC mechanisms
and missing provenance. These are protocol findings, not GT-based corrections.
Observation provenance does not prove causal reasoning; Core stays authoritative.

297 tests passed; Ruff and diff checks passed; 11/11 selected historical Replays
valid. No live model or Benchmark called, no Core rule/threshold, frontend or
budget change, no Git commit. Next optional step, requiring separate approval:
one t030 Qwen check using the same execution configuration and a fresh directory.
See `analysis/discovery_handoff_contract_acceptance_20260928.md`.

## Active query fixes and Planner feedback — 2026-09-28

Real Qwen checks completed: t007 Inquiry 10/10 executed (14 total requests),
2 mechanisms/1 compiled; t030 had no recorded Inquiry rounds (7 total requests),
2 mechanisms/1 compiled. Both INCONCLUSIVE, Answer digests/Replay valid. t007
thread-pool plan omitted on retry; t030 slow-SQL plan retained incorrect roles
despite specific feedback. Neither target mechanism compiled. See
`analysis/query_strategy_qwen_20260928/report.md`. No extra runs and no GT read.

Implemented approved three query fixes plus precise Planner retry diagnostics.
Pending legal actions survive selection and later rounds; executed actions drive
deduplication. Query utility v2 replaces cost-only ordering for new active tasks.
Separate query-only capabilities permit exploration without granting causal
evidence eligibility. Indexed Planner errors and retained counts survive into
retry and are checked by Replay. Old records keep legacy digests/scoring.
268 tests pass, Ruff/diff check pass. No new Agent/Playbook, no Core/budget changes.
Acceptance details: `analysis/query_strategy_fix_20260928.md`.
The user subsequently approved one real Qwen run each for t007/t030, using
active Inquiry + legacy Discovery + conditional format Formalizer; new output:
`analysis/query_strategy_qwen_20260928/runs/`. Summarize those records separately
before claiming any end-to-end or model-effect improvement. No Git commit.

## Conditional Discovery Formalizer — 2026-09-24

The approved chain is implemented as an opt-in `discovery_format_formalizer`
after legacy Discovery, with active Inquiry querying unchanged. Valid typed
Discovery output goes directly to Planner; malformed but safely anchored raw
mechanisms receive at most one format-only Qwen call. No Inquiry track ID is
required for a Discovery-new cause. The exact cause/source/entity and existing
valid classifications/alternative texts are host-checked, and Transcript /
Replay bind the repair and downstream draft. Unsafe or drifted results fail.
The prior standalone Formalizer mode and frontend profile are untouched.

Two real Qwen cases ran through the new mode and Replay-valid Answer freeze.
Neither needed format repair. t007 compiled a thread-pool mechanism and stayed
supported. t030 Discovery again proposed complex SQL, but Planner returned
`CONTRACT_REJECTED` twice, leaving zero compiled Candidates. Do not call this
a Core rejection or verified SQL root cause. The next separate task is to
diagnose Planner/Registry contract mismatch on the frozen t030 plans, without
changing Core standards or using GT during inference. Full report and artifacts:
`analysis/investigation_discovery_formalizer_20260924/`.

## Active query + legacy Discovery comparison — 2026-09-24

The standalone Qwen recorder now supports `--mechanism-stage discovery` with
active Inquiry querying and the original semantic Discovery adapter; default
Formalizer mode is unchanged. One t007/t030 run each completed and replayed.
t030 recovered a complex-SQL mechanism at Discovery but Planner omitted its
verification plan on both attempts, so no candidate compiled and Core never
adjudicated that SQL mechanism. Do not describe this as a Core contradiction or
a verified root cause. Report and saved artifacts:
`analysis/investigation_active_discovery_20260924/`. The next narrow audit is
why Planner did not submit a plan for the frozen SQL mechanism; keep GT, Core
standards and query budget unchanged.

## Formalizer empty-result repair — 2026-09-24

The Qwen Formalizer's formerly legal silent empty output is now a versioned
candidate-accounting contract. Each retained candidate must become a typed
mechanism or carry a typed skip reason. Empty/incomplete first output gets at
most one retry, recorded with its original structured payload. An incomplete
final result is explicitly rejected; a retry cannot delete a mechanism already
formalized. Recorded Replay validates the accounting and retry, while old v1
transcripts remain compatible. Deterministic tests cover an accepted explained
empty result, a rejected unexplained empty result and rehashed Transcript
tampering. Two bounded real Qwen Runs subsequently completed: t007 and t030
both returned three formal mechanisms, compiled three Candidates and passed
Recorded Replay, without needing the retry. t007 ended `SUPPORTED_HYPOTHESIS`;
t030 ended `INCONCLUSIVE` and did not propose slow SQL in this run. These are
chain-integration observations, not evidence of accuracy improvement or live
retry effectiveness. Details and artifacts:
`analysis/investigation_formalizer_accounting_20260924/report.md`.

## Active Qwen t007/t030 rerun — 2026-09-24

The corrected active routing was exercised once more on each Case. t007
executed 4/10 offered Inquiry queries, including new queries after round 3
had no selection; t030 executed 5/10. Both Answers and Replays are valid.
Both opt-in Formalizer calls were rejected with `ValueError`; no mechanism
froze or compiled, so formalization-to-Core remains unverified. Read the new
per-Case artifacts and `analysis/investigation_active_qwen_rerun_20260924/report.md`
before any further model call or batch. No Prompt, evidence or Core change was
made in this rerun.

## Active Qwen investigation first pass — 2026-09-24

`evaluation/record_active_formalizer_qwen.py` is the independent, reusable
script entrypoint for Qwen `json_schema`, active Workspace routing and the
opt-in `LangChainMechanismFormalizerAdapter`. Frontend remains on legacy
Inquiry → Discovery → Planner → Core. The script isolates per-Case config,
Transcript, Answer, Replay, evidence and safe failure result, then continues.
Only t007/t030 were run. Both replayed. Each had one executed active choice;
t007 Formalizer completed with zero mechanisms and t030 rejected with
`ValueError`, so the new formalization/compilation/Core path remains unverified.
Review `analysis/investigation_active_qwen_20260924/report.md` and the frozen
artifacts before approving more Cases. Do not claim accuracy improvement or
start the full batch from this two-Case diagnostic.

## Latest increment — 2026-09-24 Coverage v3 formal Candidate fix

The t030 false gap is fixed locally without another provider call. A pure
projection rebuilds frozen formal mechanisms, compiled Candidates and their
Core verification state from pre-freeze Transcript. Coverage v3 counts only
`unresolved`/`verified` formal Candidates and excludes `failed` or unassessed
drafts; Replay recomputes the same projection. The saved v2 t030/t007 Runs still
Replay, and frozen-input v3 recomputation now recognizes t030 database execution
and transport without changing t007. Historical v1/v2 digests retain their old
semantics. Keep default `off` and shadow-only; two samples do not establish a
safe active noise rate. See `analysis/mechanism_coverage_qwen_20260924/report.md`.

## Latest increment — 2026-09-24 script-driven Qwen Coverage shadow

Without the frontend, `evaluation/record_coverage_qwen.py` recorded new t030
and t007 Qwen thinking Runs under `analysis/mechanism_coverage_qwen_20260924/runs/`.
Both froze Answers and passed independent Recorded Replay. The existing
offline evaluator scored GT only after freeze and Replay. Both ended
`SUPPORTED_HYPOTHESIS`, with 0 wrong VERIFIED. The explicit untyped Inquiry
family false gap is fixed in Coverage v2 (conservative phrases, versioned
v1 compatibility), but t030 revealed a different false gap: its live formal
slow-SQL mechanism was compiled/Core-unresolved yet not represented in the
Inquiry Workspace Coverage input. Keep Coverage off by default and shadow-only;
fix formal Candidate projection using replayable Transcript evidence before
any active decision or another paid run. Details and artifacts:
`analysis/mechanism_coverage_qwen_20260924/report.md`.

## Latest increment — 2026-09-24 MechanismCoveragePolicy shadow Phase 5 offline assessment

The offline audit found 18 saved Runs across 8 Cases with valid frozen Answer digests and
Recorded Replay, but none has a Phase 4 Workspace Snapshot, so no historical shadow Run can be
reconstructed or scored for incremental recall/noise. The 18 legacy Runs are data-insufficient,
not Replay-incompatible; the separate 18 RCA100 Cases that cannot initialize are excluded from
all valid denominators. Fixed inputs repeat Coverage Gap/order/digest deterministically and
the prior off/shadow fixed Runner pair remains behavior-identical; there is no nonempty paired
historical Discovery/Planner sample. Two observation-only capability probes found 0/7 exact,
5/7 partial and 2/7 unresolvable abstract needs; these are not recall or noise samples.

A fixed Runner regression reproduces a design risk: an Inquiry Candidate that already names
slow SQL has no typed `mechanism_class`, so Coverage still reports database execution missing.
Phase 5 conclusion: **CoveragePolicy design needs modification and reevaluation before active**.
Keep the default `off`, do not enable Coverage active or DiagnosticPlaybook, and do not change
Core verification. The next evaluation needs saved Phase 4 shadow Runs or fixed paired full-chain
inputs with typed Candidate family, frozen Answer and valid Replay before any GT scoring.
Report, raw JSON, per-Case table and reproducible script are in
`analysis/mechanism_coverage_phase5_*` and `evaluation/mechanism_coverage_shadow.py`.

## Latest increment — 2026-09-23 MechanismCoveragePolicy shadow

Implemented the approved Phase 4 shadow only. `EpisodeRunner` has an independent
`mechanism_coverage_mode="off"|"shadow"` flag with `off` as the default; there is deliberately no
active value. The policy runs once after Core decision and final Workspace transitions, immediately
before Answer freeze, so it cannot affect the investigation that produced the result.

The pilot audits DB execution/transport/connection-acquisition and JVM thread-queue/GC/memory/
CPU-load family coverage. It sees only the final read-only Workspace, live typed Candidate classes,
legally issued Observation summaries, public Capability summaries and safe incident entity type.
Capability presence alone never triggers a domain. Output is limited to non-authoritative missing
families, abstract discriminator needs, observability state and exact trigger basis. It cannot add a
Candidate, MechanismSeed, query, budget, Formalizer call, Matrix selection, Core state or Answer
field. Workload/availability and all DiagnosticPlaybook content remain out of runtime.

The versioned input/output is stored in one `MECHANISM_COVERAGE_SHADOW_RECORDED` event. Replay
reconstructs the pre-event Workspace and legal inputs, recomputes the policy, requires exact
equality and rejects a changed gap even with a rebuilt hash chain. The event must immediately
precede Answer freeze. Missing flags in older Transcripts mean `off`.

Focused acceptance covers DB/JVM gaps, already-covered families, context-free suppression,
behavioral equivalence, no Workspace mutation, evaluation lifecycle replay, flag rollback and
tamper rejection. Ruff, all 234 tests and `git diff --check` pass. See
`analysis/mechanism_coverage_shadow_acceptance.md`.

Next step is offline shadow evaluation on fixed/replayed inputs. Do not activate Candidate creation
or add a DiagnosticPlaybook until the shadow demonstrates low-noise incremental mechanism recall.
Keep coverage misses separate from Planner, Capability and Core alternative-closure failures.

## Latest increment — 2026-09-23 Phase 2 active + Phase 3 Formalizer

Implemented the approved third batch. `EpisodeRunner` now accepts
`investigation_workspace_mode="active"`; `shadow` remains the default and `off` remains rollback.
Active mode validates the exact structured actions selected by Inquiry, resolves them against the
Episode Registry before execution, builds Tasks/Matrix, and executes only the top exact binding
among the model-offered actions. It suppresses the other queued actions and never adds a query or
increases budget. Successful observations remain `AMBIGUOUS`; no new component may mark a Task
resolved or a Candidate verified.

Transcript and Replay now bind active selections to the Inquiry round, Matrix digest, Capability,
query fingerprint and exact ToolResponse/Observation refs. The Answer freezes active Workspace
state. Phase 2 shadow and older Workspace Answer digests remain compatible.

Added the opt-in `MechanismFormalizer` protocol and strict legacy Discovery adapter. Formalizer
input contains retained candidates, safe observations and issued entity graph only. Output must
bind the same track and `semantic_digest`, preserve normalized causal mechanism text, and name an
existing source proposal. Drift is rejected before the existing freeze, Planner, compiler and Core
chain. The three-layer ID mapping remains projected by the Workspace finalizer. Existing
`discovery=` integrations are unchanged; formalizer and legacy Discovery are mutually exclusive.

Validation: `uv run ruff check .` and all 227 tests pass. No real model or Benchmark was run. See
`analysis/investigation_phase2_active_phase3_acceptance.md`.

Next approved batch is Phase 4 decision plus Phase 5 evaluation. First measure active routing and
formalization on fixed/replayed inputs: query reduction, exact capability-binding rate, candidate
retention, formalization yield, Replay validity and wrong-verification safety. Only then decide
between a simple `MechanismCoveragePolicy` and a small declarative `DiagnosticPlaybook`; do not add
Playbook rules from Case labels or weaken Core.

## Latest increment — 2026-09-23 Discrimination Phase 2 shadow

Implemented the approved second batch without changing executed queries. When Discovery resolves
the Episode Registry, Runner records a digest-bound capability projection and retrospectively
turns all recorded Inquiry needs into `DiscriminationTask` records. Exact entity plus query-
fingerprint matches bind executable capabilities; unresolved natural-language needs are task-owned
`UNVERIFIABLE` rather than guessed or promoted to Candidate failure.

Tasks now have replayed `OPEN -> SELECTED -> AMBIGUOUS|UNVERIFIABLE` shadow lifecycles and bind the
actual legacy query cost, capability, independent group and exact Observation refs. Shadow never
marks a task `RESOLVED`: successful observations remain `AMBIGUOUS` because this batch does not
introduce semantic outcome adjudication. Candidate reachability is derived, including the corrected
`PARTIALLY_ACTIONABLE` result when some primary tasks resolve and others are unavailable.

The `DiscriminationMatrix` is a deterministic Workspace projection, not persisted authority. The
Transcript stores only its digest and the diagnostic top selection versus the first legacy choice.
Ranking currently uses explicit contrasting predictions, exact executable binding and query cost;
it has no Core threshold, GT knowledge or causal-evidence promotion. Because it is generated after
legacy query completion, it cannot execute, replace, reorder or add a query.

Replay validates Registry/Matrix digests, exact task-query-Observation binding, lifecycle order,
Answer freeze and Phase 1 nested Workspace digest compatibility. Workspace `off` mode still removes
the complete Workspace/shadow path. Ruff, `git diff --check` and all 224 tests pass. No model or
Benchmark was invoked.

Next approved batch: Phase 2 active plus Phase 3. Before activating selection, move discriminator
creation ahead of execution and add an explicit structured observable/Capability-need binding; do
not infer a signal from prose or treat the retrospective shadow ranking as an active planner.
Mechanism Formalizer must preserve the selected InvestigationCandidate semantics and retain the
three-level ID mapping. Core and its thresholds remain unchanged.

## Latest increment — 2026-09-23 Investigation Workspace Phase 0/1

Implemented the approved first batch in behavior-neutral shadow mode. New contracts live under
`src/rca_agent/investigation/`: investigation candidates use stable `track-*` IDs, separate stage
and disposition axes, replayable transition bases, atomic semantic supersession, task-owned
unverifiability contracts, derived investigation frontiers and frozen workspace snapshots. Core
Candidate IDs and statuses remain separate and unchanged.

Runner records each valid Inquiry idea as a Workspace candidate event, retains exact normalized
ideas across rounds, disambiguates reused model hypothesis IDs using the existing handoff identity
rules, and projects later Discovery dispositions/formal mechanism links without influencing any
query or Core decision. Transcript remains the sole persisted fact source. Recorded Replay rebuilds
the Workspace and checks the Answer projection; legacy Answers remain readable. LangGraph carries
the read-only Snapshot beside the existing Core-derived Candidate Frontier.

Candidate identity excludes the current comparison alternative: the same causal mechanism keeps
one track when Inquiry changes its alternative, while each additional alternative is attached by a
replayable transition with a deterministic alternative ID.

The final design corrections are enforced: `CLARIFY` cannot change `semantic_digest`;
`SUPERSEDE` is a single atomic event; `REACTIVATE` and `RECONSIDER` affect different state axes;
and the derived terminal reachability label is `DISCRIMINATION_COMPLETE`, not `COMPLETE`.
`investigation_workspace_mode="off"` provides immediate rollback; default is `shadow`.

Validation: Ruff passes and all 222 tests pass. No model or Benchmark was invoked. Phase 2 types are
present but no Task is emitted or used. Next approved work is the second batch, Phase 2 shadow:
transcript the resolved Registry digest, derive `DiscriminationTask`/attempt opportunities and a
deterministic Matrix ranking, then compare it with legacy query choices without changing execution.

## Latest increment — 2026-09-22 t007/t030 frozen-evidence correction

Corrected evidence-role normalization without lowering thresholds or verification sufficiency.
Unambiguous class-scoped predicates are removed when used as support for another mechanism class
or as a contradiction of their own class. A partially supported dynamic Candidate now remains
unresolved; complete observed support failure and valid matched contradiction remain decisive.
The behavior is versioned as `generic-root-cause-v2`, while stored Candidates without the field
retain v1 semantics, preserving both saved Runs' Recorded Replay.

Frozen-data recomputation changed t007 thread-pool saturation and t030 complex SQL from failed to
unresolved/`SUPPORTED_HYPOTHESIS`; t030 retains `sql_duration_share` and `sql_shape` as matched
support and only lacks alternative closure. No Candidate became verified. The unrelated t030
`AttributeError` was fixed by ignoring null non-tool history results. Ruff and all 212 tests pass.
See `analysis/t007_t030_frozen_replay.md`. Next algorithm work may proceed to the separately planned
InvestigationWorkspace prototype; these correctness regressions no longer need to be carried into it.

## Latest increment — 2026-09-20 extended DeepSeek thinking budget

DeepSeek model calls now default to `max_tokens=65536` and a 300-second provider timeout. The
single-case, batch and inquiry-only experiment entrypoints expose and record those values. Runner's
Discovery timeout guard now permits up to 600 seconds, while experiment defaults use 300 seconds.
Batch cache reuse includes these limits so an old short-budget result is not mistaken for a new
long-budget run. This is an execution-budget change only; prompts, Core rules and evidence
semantics are unchanged.

## Latest increment — 2026-09-20 semantics-preserving mechanism completion

Two-stage Discovery now has one bounded completion attempt for the specific failure where its
recorded disposition reason explicitly names a new causal mechanism but the structured mechanisms
array omits it. The model must copy that exact phrase into a typed
`identified_but_unsubmitted_mechanisms` hint. Runner never extracts a cause from prose itself. It
allows one second call containing only those hints, accepts only unchanged source/mechanism pairs,
and sends accepted drafts through the unchanged freeze, Planner, compiler, queries and Core.
Semantic drift is recorded and discarded.

Three new Transcript event types and Recorded Replay checks bind the initial output, allowed hint,
completion result and generated drafts. Deterministic tests cover successful completion, semantic
drift rejection and transcript tampering. This is a contract-completion repair, not a new root-cause
inference step and not a relaxation of Core evidence sufficiency. A live t021 Qwen probe completed
with valid Replay and directly formalized and compiled a slow-SQL/database-execution mechanism, so
the retry was not needed in that run. Core returned `INCONCLUSIVE` under the unchanged standard.
The live run therefore confirms the formerly lost direction reached the normal pipeline, while the
completion retry itself remains demonstrated deterministically. See
`analysis/mechanism_completion_acceptance.md` and
`analysis/qwen_mechanism_completion_t021.json`.

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

## Latest increment — 2026-09-21 web console read-only integration

Implemented the approved first front/back-end integration slice without changing Core, Replay
rules, GT isolation, Benchmark code or invoking a model. `backend/` contains a FastAPI host,
dynamic RCA100 Case catalog, read-only file RunStore, Transcript/Answer projector, public OpenAPI
and `backend/docs/api-v1.md`. The server does not import `evaluation` and deliberately ignores
`analysis/` as a Run source because historical benchmark JSON can contain Ground Truth.

`frontend/` retains the Mock data source and now switches to API mode with
`VITE_RCA_DATA_MODE=api`. It lists Cases dynamically, selects saved Runs and displays Timeline,
Observation, Candidate and Final Decision data derived from recorded artifacts. Missing Case
timestamps are classified `invalid` rather than crashing the 103-Case listing. A local ignored
scripted t021 run was generated through the actual Environment/Runner/Transcript/Answer/Replay
path; it has valid Replay, four timeline events and four observations, with no provider call.

Verification: backend unit tests, full Python suite, Ruff, frontend TypeScript build and ESLint
all pass. Next optional slice is a GT-free live Run executor with server-approved profiles and
background job state; do not call the evaluator benchmark entrypoint from the web API.

## Latest increment — 2026-09-21 scripted Live Run path

Added the first GT-free Live Run control path. The API accepts only the server-owned
`scripted-trace-demo` profile and returns HTTP 202 with a persistent Run ID. A bounded local worker
writes atomic queued/running/terminal manifests, the real EpisodeRunner writes its append-only
Transcript, and success requires a frozen AnswerArtifact plus valid Recorded Replay. Safe failure
codes are persisted without exception text. Startup marks interrupted queued/running jobs failed
with `HostRestarted`.

The frontend now exposes a Case-aware `开始调查` action, polls Run status, refreshes history and
loads the completed Timeline/Evidence/decision projection. The real t021 API smoke followed
queued/running/succeeded, executed two Trace queries, returned 15 Observations and a replay-valid
`INCONCLUSIVE` result. No LLM, evaluator or Benchmark was called. Five focused web tests, Ruff,
the full Python suite, frontend build and ESLint pass. Next work may add a separately approved
model-backed profile; do not turn the current scripted smoke profile into an accuracy claim.

## Latest increment — 2026-09-21 Qwen Live Run path

Added the server-owned `qwen-investigation` profile without importing `evaluation` or exposing GT.
The API request contains only the profile ID. Qwen model, URL, timeouts, query budget, round limit
and the explicit public-HTTP exception remain backend environment settings; a missing host key is
a safe HTTP 503. The frontend API mode starts this profile by default and can select the scripted
profile through `VITE_RCA_RUN_PROFILE` for no-model demos.

The accepted path is Inquiry-driven rather than a free-running Policy loop. Five bounded Inquiry
rounds preserve tentative candidates and query links, then semantics flow through the existing
Handoff, Discovery, Planner, Core, Answer and Replay contracts. A first generic-Policy smoke was
replay-valid but spent nineteen queries without submitting a Candidate; it motivated using the
already validated Inquiry path, not a Prompt or Core change.

The final real t030 smoke (`run-live-87c10830a6214f2f`) completed with fourteen queries, five
recorded Inquiry rounds, two frozen mechanisms, two compiled candidates, valid Replay and
`SUPPORTED_HYPOTHESIS`. Presentation projection now derives each compiled candidate's FAILED or
UNRESOLVED state from recorded dynamic-verification events instead of copying the overall Answer
status. This run is integration evidence only; no Ground Truth scoring was performed.

Next optional web increment: cancellation, incremental progress/event streaming, and durable queue
recovery. Keep provider controls server-owned and continue treating interrupted/timeout states as
explicit failures rather than partial success.

The user chose UI polish as the next and likely final near-term increment. The approved planning
artifact is `frontend/UI_OPTIMIZATION_PLAN.md`. Its boundary is front-end presentation only: keep
Core, model prompts, API schemas, GT isolation and the Qwen execution chain unchanged. Use the
replay-valid t030 Qwen Run `run-live-87c10830a6214f2f` plus the scripted t021 Run and Mock mode as
the three visual acceptance fixtures.

## Latest increment — 2026-09-21 UI polish

The approved UI plan is implemented under `frontend/` only. CandidateSidebar groups formal
mechanisms and Inquiry-only ideas, surfaces Formal/Failed/Pending/Verified counts, and keeps Core
status visually distinct from the AI mention lifecycle. InvestigationTimeline groups the existing
safe Timeline projection into five investigation phases and retains per-event expandable details.
EvidenceInspector adds frozen-evidence paging and developer-only Summary/References/Raw tabs.
FinalDecision now distinguishes overall decision, Replay state and formal candidate states.

The frontend does not infer evidence roles, change Backend DTOs, read Run files directly, expose
GT/secrets/hidden reasoning, or alter any Core/Prompt/Agent behavior. `uv run ruff check .`,
`uv run pytest -q`, `npm run lint`, and `npm run build` pass. The production build retains the
pre-existing Vite large-chunk warning. Local browser automation was blocked by host policy, so
visual screenshot verification remains optional follow-up rather than an unreported claim.
## Latest increment — 2026-09-21 Chinese investigation workspace

The frontend was restyled into a dark Chinese-first observability investigation workspace based on
the approved concept mockup. Existing Case/Run selection, API mode, Mock mode, live-run trigger,
Candidate/Timeline/Evidence interactions, raw drawer and Core decision data are retained. Static
workflow language is simplified for non-authors, while identifiers and raw technical payloads
remain in Developer Mode. FastAPI now permits browser requests from local Vite ports through a
localhost-only CORS origin regex. The standalone concept and rendered comparison image live under
analysis/ui; they are design references and are not runtime dependencies.

## Current handoff — 2026-10-10 pending projection repair

The DeepSeek batched pending repair is implemented and deterministic validation is complete. New
batched Episodes use loop envelope v4; LoopStep remains v1 and nested Inquiry Draft remains v2.
The public view separates actionable `pending/deferred/unavailable` entries from read-only
`scheduled/executed/cancelled` history without changing the internal ledger. Structured pending
diagnostics distinguish missing, unexpected, duplicate and invalid-state submissions. Retry
feedback is rendered into the real Adapter prompt and included in v4 attempt binding. Historical
v3 Replay remains supported.

Validation: full `uv run pytest -q` passed; targeted Ruff for the changed files passed. A separately
authorized DeepSeek t030 batched re-test executed one dynamic query and stopped after three model
calls because the second step was not accepted; Replay remained valid. No Benchmark, GT or
evaluator was called. Further real re-test requires a new authorization after the smoke-entry
diagnostic gap is repaired.
