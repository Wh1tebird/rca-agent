# RCA Agent instructions

Read `PROJECT_CONTEXT.md` and `HANDOFF.md` completely before planning or changing this repository.

## Repository boundary

- `rca-agent` must not depend on the sibling `rca-core` package at runtime.
- `rca-core` is frozen and read-only. Runtime capabilities already extracted into
  `src/rca_agent/core` are owned by `rca-agent` and evolve independently.
- Do not synchronize later Agent changes back into `rca-core`, or expand the extracted slice without
  a concrete Agent runtime need and regression coverage.
- Keep Research out of `src/rca_agent`: Ground Truth, evaluation, Oracle, Benchmark, scoring,
  holdout, audit, historical experiments, and dataset-evaluation harnesses are not runtime domain.
- Do not copy RCA100 data. Configure the existing observation directory at host construction time.

## Inference and evaluation isolation

- The Agent-visible contract must never expose a dataset case selector, filesystem path, answer
  label, evaluator annotation, raw query implementation, or complete precomputed case facts.
- Evaluation data may be loaded only by a future evaluator after an answer artifact is frozen.
- Inference modules under `src/rca_agent` must not import evaluator-only loaders.
- Keep raw trace identifiers and core query references private; issue opaque stable references.

## Tool and evidence safety

- Tools are typed, read-only, entity-scoped, time-bounded, result-bounded, costed, and budgeted.
- Tools accept neither SQL nor filesystem paths.
- Tool output is a neutral `Observation`, never candidate support, contradiction, or a root-cause
  conclusion.
- Missing or unavailable data stays `MISSING` or `UNKNOWN`; do not interpolate it.
- `IMPUTED` and `MODEL_INFERRED` values may assist future exploration but cannot become critical
  positive evidence.
- Candidate thresholds, evidence weights, sufficient-evidence checks, and stopping authority belong
  to the internal Agent domain core, never to Policy output.
- Policy may name a trusted Catalog `candidate_id`; only observations issued in the same episode and
  explicitly referenced by that proposal may enter the default Core decision provider.
- Observations are immutable episode-global facts. Evidence admission is Candidate-scoped: an
  Observation attached to one hypothesis must not enter another Candidate's assessment unless the
  same issued reference is explicitly attached there too.
- Per-candidate status and discriminating-signal views come only from `CoreAssessmentPreview`. They
  may expose signal/capability gaps but never condition operators, thresholds, weights, strengths,
  winner-margin values, or sufficiency rules.
- LangChain/LangGraph are non-authoritative orchestration layers. They must send every action through
  `EpisodeRunner`; they must not call `RcaEnvironment`, the Observation ledger, evidence controller,
  evaluator, or answer-freeze internals directly.
- `EpisodeRunner.preview_core_assessment()` is read-only routing information. A graph must request
  stop through the Runner and may not turn a preview into an answer artifact itself.
- Candidate Frontier fields in Graph State must be derived from the latest Core preview. LangGraph
  and Policy must not independently confirm, contradict, exclude, or reactivate a Candidate.

## Working practice

- Do not run `git add`, `git commit`, `git push`, create tags, or configure remotes unless a future
  user request explicitly replaces this local-workspace rule.
- Preserve unrelated user changes.
- Use `apply_patch` for manual edits.
- Run `uv run ruff check .` and `uv run pytest -q` after changes.
- Put stable architectural facts in `PROJECT_CONTEXT.md` and current next steps in `HANDOFF.md`.
