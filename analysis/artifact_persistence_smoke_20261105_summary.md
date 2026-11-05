# Artifact Persistence smoke test

The deterministic scripted t021 run produced a frozen Answer and a nine-event
JSONL Transcript without invoking a model or production telemetry.

- Answer digest verification: `verified`
- Transcript head and digest chain: `valid`
- Replay status: `valid`
- Offline Evaluator reconstruction: succeeded
- Outcome: `not_evaluable`
- Outcome detail: `ground_truth_not_loaded`
- Token usage: no model calls, so no provider usage was recorded
- Redaction check: no prompt, secret, filesystem path, or raw trace identifier

The persisted artifact directory contains `manifest.json`, `frozen_answer.json`,
`transcript.jsonl`, `transcript_metadata.json`, `evaluator_input.json`,
`usage.json`, and `evaluation_report.json`. The generated directory remains
outside the commit. This summary records only the smoke result.
