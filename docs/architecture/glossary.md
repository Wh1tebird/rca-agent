# RCA Agent 1.0 术语表

本表只定义稳定术语和边界。代码中的 API 名、类名、事件名和协议字段保持英文。

| 术语 | 定义 | 边界 |
| --- | --- | --- |
| Runtime | 执行一个 Episode 的 host-owned 运行时 | 拥有 admission、query、Observation、Core decision、stop、Handoff 和 Answer Freeze |
| EpisodeRunner | Runtime 的唯一 lifecycle 和状态写入入口 | 负责 Transcript、Workspace projection、Observation ledger 和 Answer artifact 的写入顺序 |
| Observation | 单个 Episode 内不可变、Candidate-neutral 的事实 | 不能包含 support、contradiction、winner 或 root-cause 结论 |
| Evidence | Core 对已 admission 的 Observation 的 Candidate-scoped 使用 | 不等同于 provider 原始返回，也不自动跨 Candidate 传播 |
| Candidate | Catalog 或 Discovery 注册的待验证 Hypothesis 载体 | Candidate 状态只能由 Core 产生 |
| Hypothesis | Policy 或 Reasoning Adapter 提出的候选解释 | 提案不能携带 threshold、weight、sufficiency 或最终 status |
| Capability | host 授予的、带 scope、时间、结果和 cost 边界的能力 | Provider 不能自行扩大 Capability 或 budget |
| QuerySpec | 可序列化的查询意图和边界 | 不接受 SQL、filesystem path 或 raw backend query |
| QueryInstance | 经过 host admission、带稳定 opaque reference 的一次 QuerySpec 执行 | Replay 只验证记录，不重新访问 provider |
| EvidenceProvider | 将 QuerySpec 转成中立 Observation 的 host adapter | 不绑定 Candidate，不产生 Core decision |
| CoreDecisionProvider | 依据已 admission 的 Evidence 生成 authoritative decision 的 host-owned Core | 唯一可产生 VERIFIED、SUPPORTED、INCONCLUSIVE、OUT_OF_SCOPE、QUERY_FAILED、winner 和 stop authority |
| Transcript | append-only、hash-chained 的事实事件记录 | 是 Replay 和 Audit 的事实来源，不接受 late result 的回写 |
| Replay | 不调用模型、Provider 或外部服务的离线一致性验证 | 验证 digest、sequence、预算、projection 和 Answer binding |
| Answer Freeze | EpisodeRunner 对最终 Answer artifact 的不可变冻结动作 | Freeze 后任何 projection 都不能改变 Answer |
| Abstention | Core 在证据不足、冲突或 Candidate 不可区分时保持不确定 | 表现为 `INCONCLUSIVE` 或明确的不可用状态，不强选 winner |
| Handoff | 当前阶段把继续调查所需信号交回 host 的动作 | Handoff 不是 decision，也不能绕过 Runner |
| Reasoning Adapter | model-facing 的 typed proposal 入口 | 只能提出 Candidate、Query、Hypothesis、plan、next action、stop 或 Handoff suggestion |
| Routing Projection | 从 EpisodeRunner 派生给 LangGraph 的只读 routing view | LangGraph 不能把它升级为 authoritative state |
| Recovery | 从完整 Transcript 检查 Episode 是否可继续的离线流程 | `RETRYABLE` 只是恢复计划，不表示已经完成恢复 |
| Audit | 从已批准字段生成的脱敏运行记录 | 不包含 secret、完整 prompt、原始 path、raw trace/log identifier 或 evaluator-only 数据 |
| Failure Taxonomy | 对执行、协议、状态、调查、grounding、contract、Replay 和 adapter 失败的稳定分类 | 用于诊断和 Evaluation，不改变 Runtime decision |

## 状态术语

- `UNKNOWN`：已有值不足以支持请求的解释。
- `MISSING`：预期的数据覆盖不存在。
- `INCONCLUSIVE`：Core 已完成评估，但证据不足、冲突或无法区分 Candidate。
- `OUT_OF_SCOPE`：提交的 Hypothesis 不属于当前可信 Catalog 或能力范围。
- `SUPPORTED`：兼容报告中的简写；代码以 `SUPPORTED_HYPOTHESIS` 或对应 `CandidateEvidenceStatus` 表示。
- `VERIFIED`：兼容报告中的简写；代码以 Core 产生的确认状态表示，Reasoning Adapter 和 LangGraph 都不能产生它。

`IMPUTED` 和 `MODEL_INFERRED` 可以帮助后续探索，但不能成为 critical positive Evidence。
