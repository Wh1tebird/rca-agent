# RCA v4 Agent Environment

Optional unknown-mechanism discovery is now integrated into the main LangGraph/Runner loop.
It preserves the static Catalog and validates dynamic predicates against host-owned signal/rule
contracts. See [dynamic discovery setup and limits](analysis/dynamic_discovery.md).

Dynamic mechanisms passing the common verification standard return `DISCOVERED_VERIFIED` as the
root-cause result for this investigation. They need no per-mechanism human sufficiency definition;
the result does not automatically promote them to Catalog or assert exhaustive causal proof.

`rca-agent` wraps RCA100 as a controlled, interactive investigation environment. An investigation
policy receives a small event bootstrap payload and can acquire additional facts only through
bounded, typed, read-only tools. Candidate interpretation and final evidence decisions belong to
the internal Agent domain core.

This repository contains the trusted RCA runtime and a GT-free local console:

- repository and trust-boundary skeleton;
- the minimal RCA Environment Contract;
- a host-side RCA100 adapter that keeps dataset locations and case selectors private;
- bounded Metric, Trace, and Topology tools;
- explicit unavailable Log capability;
- query cost and episode budget enforcement;
- neutral observations with trust states.
- a typed Episode Runner and four-state lifecycle;
- append-only hash-chained Transcript persistence;
- offline Recorded Replay validation;
- typed answer freezing and a separate post-freeze Evaluator artifact;
- a sanitized child process for Scripted/Fake Policy lifecycle tests.
- an episode-scoped issued-Observation ledger;
- a trust-aware bridge into the internal evidence-controlled domain core;
- Catalog-owned candidate resolution and authoritative decisions.
- a read-only, non-freezing Core assessment preview after each Policy action;
- a provider-neutral LangChain structured-output boundary;
- LangGraph orchestration that can act only through `EpisodeRunner`;
- a no-API-key Scripted Policy integration test covering Tool -> Observation -> Core -> Freeze.
- candidate-scoped evidence binding without duplicating episode-global Observations;
- threshold-free per-candidate Core previews and discriminating-signal views;
- a Core-derived Graph candidate frontier for active, inconclusive, supported, contradicted, and
  confirmed candidates;
- a deterministic `slowSQL` versus `dbNetworkLatency` multi-candidate integration test.
- bounded Inquiry and result-driven investigation loops;
- dynamic Discovery and Verification Planner contracts;
- a FastAPI presentation host and Vue investigation console.

RCA100 remains an external, private observation directory. It is never bundled with this
repository. The public test and CI path uses synthetic environments; private RCA100 integration and
benchmark runs are separate and are not release-quality or accuracy claims.

## Setup and verification

```bash
uv sync --extra dev
uv run ruff check src backend tests
uv run pytest -q -m 'not requires_rca100'

# Optional local RCA100 integration tests
RCA100_CASES_ROOT=/path/to/RCA100/cases uv run pytest -q -m requires_rca100

# Frontend
cd frontend
npm ci
npm run lint
npm run build
```

The package has no runtime dependency on the frozen sibling `rca-core` repository. The previously
validated runtime domain is maintained under `src/rca_agent/core`: bounded RCA100 observation
access, candidate and EvidenceCondition contracts, capability compilation, candidate proposal and
discovery, evidence assessment, candidate-state aggregation, and safe decisions. Research modules
remain outside Agent Runtime. RCA100 data is not copied into this repository. A host process
supplies the existing `cases` directory when it constructs an environment; that path is never
included in Agent-visible responses.

Private research runs, generated evidence, provider transcripts, and local workspace artifacts are
excluded from the public source distribution. See `.env.example` for host-only configuration.

## Trust states

Every observation carries exactly one of:

- `OBSERVED`: a bounded raw source value;
- `DERIVED`: a deterministic, replayable transformation of observed values;
- `IMPUTED`: an interpolated or filled value;
- `MODEL_INFERRED`: a model estimate;
- `MISSING`: expected source coverage is absent;
- `UNKNOWN`: available values do not support the requested interpretation.

Only trusted `OBSERVED` and replayable `DERIVED` values may later enter confirmation through the
internal Agent domain core. Phase 1 never emits `IMPUTED` or `MODEL_INFERRED` observations.

Policy proposals may name a trusted `candidate_id`, but cannot provide conditions, thresholds,
strengths, sufficiency rules, or a final status. At answer freeze, the Core evaluates only issued
observations explicitly referenced by submitted proposals.

Observations remain immutable episode-global facts. `UPDATE_CANDIDATE_EVIDENCE` explicitly attaches
issued Observation references to one registered hypothesis. The Core assesses each Candidate only
against its own attached references; a fact attached to Candidate A never enters Candidate B unless
Policy explicitly attaches the same reference to B as well.

## LangChain and LangGraph boundary

`LangChainPolicyAdapter` accepts exactly one closed `PolicyAction` as structured output. Its schema
contains no final-decision, threshold, or evidence-weight field. `LangGraphInvestigation` supplies
the model only sanitized bootstrap state, candidate options, prior action results, and the Core's
read-only assessment preview.

The preview exposes only safe per-candidate status, satisfied/missing/contradicting/unavailable
signal names, required capabilities, and discriminating query gaps. It never exposes condition
operators, thresholds, evidence strengths or weights, winner-margin values, or sufficiency rules.
The Graph's Candidate Frontier is reconstructed from this preview after every action and has no
independent decision authority.

The graph never calls `RcaEnvironment` or the evidence controller directly. Tool execution,
budget/max-round enforcement, observation issuance, proposal admission, transcript recording, and
answer freeze all remain inside `EpisodeRunner`. A confirmed preview causes the graph to request
stop; the Runner then recomputes and freezes the authoritative Core decision.

## DeepSeek model binding

The accuracy-first provider binding uses `deepseek-v4-pro` with thinking enabled and high reasoning
effort. `create_deepseek_policy()` binds that model to the same closed structured Policy schema; it
does not grant the model additional tools or decision authority. The provider package reads
`DEEPSEEK_API_KEY` from the host process environment. Secrets must not be committed to this
repository. DeepSeek uses JSON mode for the envelope because thinking mode rejects forced
`tool_choice`; the returned object is still validated against the same Pydantic schema.
