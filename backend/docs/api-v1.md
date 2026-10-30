# RCA Agent Investigation API v1

This is a GT-free API for the RCA investigation console. The API is a
presentation host plus a bounded local job runner: it does not write a Case or
score a Case. The `scripted-trace-demo` profile is deterministic and does not
invoke a model; `qwen-investigation` is an optional host-configured provider
profile.

The presentation layer is a projection of the host-owned `EpisodeRunner`. It
does not own workflow, Evidence admission, Core decision, stopping, Recovery,
or Answer Freeze. `schema_version` is currently `1`.

## Safety contract

The API must never return Ground Truth, answer-key content, evaluator output,
dataset paths, provider credentials, Core thresholds or hidden chain-of-thought.
`case_id` is used by the host to build an Environment; it is never sent to the
Agent as Case metadata.

## Endpoints

### `GET /api/v1/health`

Returns whether the configured RCA100 Case directory is available.

### `GET /api/v1/cases?query=`

Lists Cases discovered from the configured RCA100 directory. Cases are dynamic;
the API does not embed a fixed benchmark manifest.

Each item includes public alert metadata, modality availability, readiness and
the count of recorded runs. `readiness` is one of `ready`, `incomplete`, or
`invalid`.

### `GET /api/v1/cases/{case_id}`

Returns one public `task.json` projection plus its run count. A Case with
missing Observation files remains visible as `incomplete`.

### `GET /api/v1/cases/{case_id}/runs`

Returns recorded runs for a Case, newest first. A run summary includes
`episode_id`, `answer_digest`, `replay_status`, `recovery_status` and
`stop_reason`. `final_status` is populated only when the Answer is frozen and
Recorded Replay is valid. It never contains evaluation results.

### `GET /api/v1/runs/{run_id}`

Returns the presentation projection used by the Vue console:

```text
run + case + candidates + timeline + evidence + final_decision + replay + recovery + audit
```

The projection is derived only from the stored Manifest, Transcript and frozen
AnswerArtifact. `evidence` contains neutral Observation values. Candidate
admission is represented only by explicit observation references recorded in
the Answer; the presenter does not infer a Candidate binding from text.

`final_decision` is returned only when the AnswerArtifact is valid, Recorded
Replay is valid, and Recovery is `FROZEN`. If Replay is invalid, Answer is not
frozen, or Recovery is `RETRYABLE`/`CORRUPTED`, safe timeline and status
metadata remain available, but no authoritative RCA conclusion is returned.

The `replay` object contains only validity, event count, digest, Answer digest
and a safe error code. The `recovery` object exposes the explicit Recovery
state and pending operation references. `audit` is generated through the
redacted `AuditRecord` projection and excludes secrets, paths, raw prompts,
evaluator-only data and raw trace/log identifiers.

### `POST /api/v1/cases/{case_id}/runs`

Starts one asynchronous run and returns HTTP 202. The only supported request is:

```json
{"profile_id": "scripted-trace-demo"}
```

Supported server-owned profiles are:

```json
{"profile_id": "scripted-trace-demo"}
```

This profile calls no model. It exercises the real RCA100 Environment, Runner,
bounded Trace tools, Transcript, frozen Answer and Recorded Replay.

```json
{"profile_id": "qwen-investigation"}
```

This profile runs the configured Qwen thinking model through the existing
Inquiry, handoff, Discovery, Verification Planner, LangGraph, EpisodeRunner and
Core path.
The server must have `RCA_LOCAL_API_KEY`; otherwise creation returns HTTP 503.
The credential is never accepted in an API request and is never written to the
Run directory. Arbitrary provider settings, URLs, keys, query languages,
budgets, Core controls and holdout selectors are rejected by the request
schema.

### `GET /api/v1/runs/{run_id}/status`

Returns the persistent job state. The existing job status transitions are:

```text
queued -> running -> succeeded | failed
```

`recovery_status` is independent from job status and is one of `RESUMABLE`,
`RETRYABLE`, `TERMINAL`, `FROZEN`, `CORRUPTED` or `NOT_AVAILABLE`.
`RETRYABLE` is a host recovery plan; it does not mean that a pending Query,
adapter call or Handoff has already been replayed or completed.

If the local host restarts during a queued or running job, the manifest is
recovered as `failed` with `HostRestarted` and `recovery_status=RETRYABLE`.
The API does not guess a pending result or report the job as successfully
recovered.

## Run directory contract

```text
backend/var/runs/{run_id}/
├── manifest.json
├── transcript.jsonl
├── answer.json
└── metrics.json
```

`manifest.json` contains public run metadata only. `transcript.jsonl` uses the
existing append-only Transcript format. `answer.json` is a frozen
`AnswerArtifact`. `metrics.json` contains host-owned operational counters only;
it cannot affect Core decision, Evidence admission, stopping or Answer Freeze.
The server validates the Answer digest and runs Recorded Replay before producing
a trusted final decision.

## Future live-run controls

Cancellation and incremental event streaming are intentionally not implemented
in this increment:

```text
GET  /api/v1/runs/{run_id}/events
```

The eventual create request will accept only a server-approved `profile_id`.
It will not accept an API key, model URL, arbitrary Core configuration, Ground
Truth selector or raw query.
