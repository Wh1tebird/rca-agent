# RCA Agent Investigation API v1

This is a GT-free API for the RCA investigation console. The API is a
presentation host plus a bounded local job runner: it does not write a Case or
score a Case. The `scripted-trace-demo` profile is deterministic and does not
invoke a model; `qwen-investigation` is an optional host-configured provider
profile.

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

Returns recorded runs for a Case, newest first. A run summary includes the
final recorded decision string and Replay validity, never evaluation results.

### `GET /api/v1/runs/{run_id}`

Returns the presentation projection used by the Vue console:

```text
run + case + candidates + timeline + evidence + final_decision
```

The projection is derived only from the stored Manifest, Transcript and frozen
AnswerArtifact. If Replay is invalid, the forensic timeline remains available,
but `final_decision` is absent and cannot be presented as authoritative.

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

Returns the persistent job state. State transitions are:

```text
queued -> running -> succeeded | failed
```

If the local host restarts during a queued or running job, the manifest is
recovered as `failed` with `HostRestarted`; an interrupted job is never silently
reported as successful.

## Run directory contract

```text
backend/var/runs/{run_id}/
├── manifest.json
├── transcript.jsonl
└── answer.json
```

`manifest.json` contains public run metadata only. `transcript.jsonl` uses the
existing append-only Transcript format. `answer.json` is a frozen
`AnswerArtifact`. The server validates the Answer digest and runs Recorded
Replay before producing a trusted final decision.

## Future live-run controls

Cancellation and incremental event streaming are intentionally not implemented
in this increment:

```text
GET  /api/v1/runs/{run_id}/events
```

The eventual create request will accept only a server-approved `profile_id`.
It will not accept an API key, model URL, arbitrary Core configuration, Ground
Truth selector or raw query.
