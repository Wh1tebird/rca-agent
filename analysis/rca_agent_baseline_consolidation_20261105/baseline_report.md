# RCA Agent Baseline Consolidation

Generated: 2026-11-05

This report consolidates existing artifacts only. No model calls or new cases were run.

## Profile manifest

| Profile | Provider/model | Reasoning | Acquisition | Budget | Max rounds | Inquiry limit | Retry | Config digest |
|---|---|---|---|---:|---:|---:|---:|---|
| deepseek_inquiry_bounded | deepseek/deepseek-flash | disabled | inquiry | 6 | 3 | 6 | 0 | `22f35c5119e20804617f5eb6e2706569f96155ffb26579af2453d2ffb49e78a5` |
| deepseek_fixed_legacy | deepseek/deepseek-flash | disabled | fixed | 20 | 20 | 10 | 0 | `3daeaf1a2e56c32ee0ac9d2f76707fb35503bdfd17d42c864f6a19f35082154b` |

## Profile comparison

| Metric | inquiry bounded | fixed legacy |
|---|---:|---:|
| Cases | 10 | 10 |
| Discovery completed | 0 | 8 |
| Candidate admission | 0 | 8 |
| Cases with Observation refs | 0 | 8 |
| Observation refs total | 0 | 57 |
| VERIFIED | 0 | 0 |
| Replay valid | 9 | 9 |
| Query total | 25 | 122 |
| Model calls total | 7 | 44 |
| Investigation steps total | 90 | 392 |
| Wall time total (s) | 252.78794399999998 | 1102.190884 |

## Status and failure taxonomy

| Profile | Status counts | Failure taxonomy |
|---|---|---|
| deepseek_inquiry_bounded | `{"CASE_SETUP_ERROR": 1, "INCONCLUSIVE": 8, "QUERY_FAILED": 1}` | `{"CASE_SETUP_ERROR": 1, "QUERY_FAILED": 1, "CAPABILITY_UNAVAILABLE": 1}` |
| deepseek_fixed_legacy | `{"CASE_SETUP_ERROR": 1, "INCONCLUSIVE": 6, "SUPPORTED_HYPOTHESIS": 2, "QUERY_FAILED": 1}` | `{"CASE_SETUP_ERROR": 1, "POLICY_INVALID_ACTION_UNATTRIBUTED": 2, "QUERY_FAILED": 1, "CAPABILITY_UNAVAILABLE": 1}` |

## Metric semantics

- Fixed profile `inquiry_rounds=0` means no Inquiry acquisition rounds; it does not mean no investigation. Discovery rounds, verification rounds and total graph steps are reported separately.
- `SUPPORTED_HYPOTHESIS` is not `VERIFIED`; VERIFIED count is zero in both profiles.
- `CASE_SETUP_ERROR` and `CAPABILITY_UNAVAILABLE` are infrastructure/data strata and are excluded from model-failure interpretation.
- `UNATTRIBUTED` is retained when safe artifacts do not identify provider, schema, conversion, profile mapping or admission as the cause.
- Token usage and per-call durations are unavailable and were not estimated.

## Existing result notes

- t001 remains a metadata setup error because the structured alert time is empty.
- t062 remains capability/fixture limited because the exact alert span is absent from the trace fixture.
- t074 and t095 results are retained as diagnostic artifacts; their problems are not treated as direct model capability failures.
- No case reached VERIFIED in these consolidated profiles.

## Post-freeze evaluator

Attempted against existing artifacts only. Status: `not_evaluable`. The evaluator requires a live frozen `EpisodeRunner`; the saved benchmark JSON does not contain a reconstructable live Runner/transcript, so no new Ground Truth was loaded.

## Limitations

- This is an operational and pipeline reliability baseline, not a formal outcome-accuracy baseline.
- Provider token accounting is unavailable.
- Existing artifact schemas do not preserve enough raw policy payload detail to resolve every invalid action.
- The two profiles use different acquisition and budget envelopes and must remain separate.

## Next steps

- Preserve these two manifests as independent benchmark profiles.
- Add a separate artifact-to-evaluator adapter only if it can preserve post-freeze isolation and replay authenticity.
- Do not count supported or compiled candidates as verified outcomes.
