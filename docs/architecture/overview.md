# RCA Agent 1.0 架构说明

RCA Agent 1.0 的目标是提供一个可审计、可 Replay、可安全替换 model-facing 能力的 RCA Runtime。Runtime 通过 typed、read-only、entity-scoped、time-bounded、costed 的 Capability 获取 Observation，再由 host-owned Core 进行 Candidate-scoped Evidence admission 和最终 decision。

1.0 的非目标包括：复制 RCA100 或 Ground Truth；把 evaluator 逻辑放进 Runtime；让模型或 LangGraph 拥有 decision、budget 或 stop authority；接入未经隔离的真实 telemetry；重写 Inquiry、Discovery、Formalizer、Planner 或 Core 的职责。

## Stable host-owned layer

`EpisodeRunner` 是唯一的 episode lifecycle owner 和状态写入入口。它负责 admission、Capability check、Query 执行、Observation issuance、Candidate-scoped Evidence admission、Core decision、budget、stop、Handoff、Transcript、Workspace projection 和 Answer Freeze。`Workspace` 是从 Transcript/Runner projection 得到的派生视图，不能确认、排除或重新激活 Candidate。

`EvidenceProvider` 接受 `EvidenceQuery` 并返回中立的 `Observation`。Query 只能使用 `Capability` 允许的字段和边界；provider 不接收 Candidate，也不能生成 support、contradiction、winner 或 root-cause。`CoreDecisionProvider` 只评估同一 Episode 内已发行、且由 Candidate 显式引用的 Observation。缺失和不可用保持为 `MISSING`、`UNKNOWN`、`OUT_OF_SCOPE` 或 `QUERY_FAILED`，证据不足时保持 `INCONCLUSIVE`。

`Transcript` 是 append-only、hash-chained 的事实源。`Replay` 只读取事件和冻结 artifact，验证 episode_id、sequence、digest chain、budget、Candidate frontier、Workspace projection 和 Answer digest；它不调用模型、EvidenceProvider 或外部服务。`Answer Freeze` 由 EpisodeRunner 执行，冻结后任何 Graph、Workspace、Reasoning Adapter 或 Recovery projection 都不能修改 Answer。

## Replaceable model-facing layer

`ReasoningAdapter` 只返回 typed proposal：Candidate proposal、Query proposal、formalized Hypothesis、verification plan、next action、stop suggestion 或 Handoff suggestion。它不能写 Observation、Transcript、Workspace、Answer，也不能改变 permission、budget、Core decision 或 stop authority。Discovery、Formalizer、Planner 和 Inquiry 的 legacy wrapper 通过这个 facade 继续兼容；未来可以替换成通用 Agent，而无需改变 host-owned layer。

LangGraph 只是 compatibility routing layer。它读取 `RoutingProjection`，调用 Reasoning Adapter，把 proposal 交回 EpisodeRunner，并读取 Runner 返回的 stop/Handoff signal。旧 graph fields 仅作为 compatibility cache，不能成为第二套 lifecycle、Candidate frontier、budget、answer 或 stop reason 事实源。Replay 不需要运行 LangGraph。

## Execution boundary and Recovery

worker 使用 JSON-only typed payload 和 length-prefixed framed protocol。父进程保留唯一状态写入权；ready timeout、request timeout、cancel、异常、crash 和 malformed frame 都会关闭 stream、terminate/kill 并 wait/reap。Late result 在 parent admission 之外被丢弃，因此不能污染 Transcript、Workspace、Observation 或已冻结 Answer。

`EpisodeRecovery` 只做离线检查，状态为 `RESUMABLE`、`RETRYABLE`、`TERMINAL`、`FROZEN` 或 `CORRUPTED`。`RETRYABLE` 表示存在明确的 pending query、adapter call 或 Handoff，需要 host 显式重建 live dependency 后再重试；它不是已经完成的恢复。digest、sequence、episode identity、Answer binding 或 projection 不一致时停止恢复，不猜测事实，也不生成新 Answer。跨进程重启目前不能自动重建 live Environment 或 adapter。

## Audit、Evaluation 和边界

`AuditRecord` 只投影 episode_id、event_id、sequence、previous digest、Capability/provider、opaque query reference、scope digest、time range、cost、adapter reference、Core decision、stop/Handoff、Answer digest、Replay result 和 Failure Taxonomy。它不记录 secret、token、filesystem path、完整 prompt、raw trace/log identifier 或 evaluator-only 数据。

Evaluation 只在 Answer artifact freeze 后运行，加载 Ground Truth、RCA100 case 和标注；Runtime 不导入 evaluator-only loader，也不把评分写回 Observation、Workspace、Candidate 或 Answer。当前测试使用 deterministic fixture 和 offline stub，尚未纳入真实 provider smoke test。

未来模型能力提升时可以替换或删除 LangGraph、legacy Reasoning wrapper 和具体 model adapter；EpisodeRunner、EvidenceProvider、Capability registry、Observation ledger、CoreDecisionProvider、Transcript/Replay、Answer Freeze 和 Evaluation boundary 保持稳定。
