# RCA Agent web backend

The backend is a GT-free host and presentation layer. It reuses the Agent
Environment, Runner, Transcript and Replay contracts; it never imports
evaluator code or serves Ground Truth. It supports a bounded scripted smoke
profile and a server-configured Qwen thinking profile.

## Start locally

```bash
uv run uvicorn backend.app.main:app --reload
```

The API is available at `http://127.0.0.1:8000/docs`.

The frontend defaults to Mock data. To use this API:

```bash
cd frontend
VITE_RCA_DATA_MODE=api npm run dev
```

The API-mode frontend starts `qwen-investigation` by default. Configure the
provider only in the backend process:

```bash
RCA_LOCAL_API_KEY='...' \
RCA_LOCAL_BASE_URL='https://provider.example/v1' \
uv run uvicorn backend.app.main:app --reload
```

For a trusted HTTP endpoint, the host must also explicitly set
`RCA_LOCAL_ALLOW_INSECURE_HTTP=true`. Use
`VITE_RCA_RUN_PROFILE=scripted-trace-demo` to keep the frontend in no-model
demo mode. API callers cannot supply provider credentials, URLs, budgets,
timeouts or Core settings.

## Generate a local demonstration Run

```bash
uv run python -m backend.scripts.record_demo_run --case-id t021
```

The helper calls no model and is not a benchmark. It creates one ignored local
Run using the real RCA100 Environment, EpisodeRunner, Transcript, frozen Answer
Artifact and Recorded Replay.
