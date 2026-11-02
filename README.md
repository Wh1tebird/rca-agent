# RCA Agent 1.0

RCA Agent 是一个受控的根因分析调查运行时。1.0 的目标是保持运行时、数据查询、核心决策、事件记录和离线重放的边界稳定，同时允许替换面向模型的提案组件。

当前已完成 P1-1 至 P2-5、文档整理以及 Backend / Frontend 展示层适配。完整架构见[架构说明](docs/architecture/overview.md)，稳定术语见[术语表](docs/architecture/glossary.md)。

## 项目定位

`EpisodeRunner` 是每个调查过程的唯一权威入口，负责准入检查、能力检查、查询执行、Observation 发行、候选证据绑定、核心决策、预算、停止、交接、Transcript、Workspace 投影和答案冻结。

`Observation` 是 `EvidenceProvider` 返回的中立事实，不包含支持、反驳、胜出候选或根因结论。候选证据绑定表示 Candidate 对已发行 Observation 的显式绑定，只能由受主机控制的核心处理。核心决策才能产生最终状态、胜出候选、弃权结果和停止权限。Answer 只有在答案冻结后才是冻结结果。

Discovery、Formalizer、Planner、Inquiry、`ReasoningAdapter` 和 LangGraph 只能提出候选、查询、Hypothesis、验证计划、下一步动作、停止建议或交接建议。它们不能写入 Observation、Transcript、Workspace、Answer、预算、权限或核心决策。

RCA100 是外部私有数据目录，不会复制到本仓库。数据路径、案例选择器、答案标签和评估标注不会进入面向 Agent 的状态。Runtime 不加载评估专用数据。

## 稳定边界

- 运行时负责调查过程、准入、Observation、核心决策、停止、交接和答案冻结。
- `EvidenceProvider` 只能把受实体范围、时间范围、结果数量、成本和预算约束的 `EvidenceQuery` 转成中立 Observation。
- Candidate 的候选证据绑定只使用同一 Episode 内、由 Candidate 显式引用的 Observation ref。
- Evaluation 只在答案 artifact 冻结后读取 Ground Truth、RCA100 case、annotation 和 scoring data，不会修改运行时状态。
- Replay 只读取 Transcript 和冻结 artifact，不调用模型、EvidenceProvider、Environment 或外部服务。
- Workspace、LangGraph 和 Routing Projection 都是派生投影，不能成为第二套事实源。

1.0 不包含 RCA100 或 Ground Truth 复制、运行时评估逻辑、新的工作流引擎，也不让模型获得 Core decision、预算或停止权限。

## 当前能力

- RCA Environment Contract 和主机侧 RCA100 adapter；
- 受边界约束的 Metric、Trace、Topology 查询，以及明确不可用的 Log 能力；
- 查询成本、调查预算和 Capability 边界；
- 带 trust state 的中立 Observation；
- typed `EpisodeRunner`、追加写入的哈希链 Transcript 和 Recorded Replay；
- typed Answer artifact、Answer Freeze 和冻结后的独立 Evaluation artifact；
- 只传 JSON 的分帧 worker 协议，以及超时、取消、终止、强制终止、回收和迟到结果拒绝；
- `EpisodeRecovery` 的 `RESUMABLE`、`RETRYABLE`、`TERMINAL`、`FROZEN` 和 `CORRUPTED` 状态；
- `AuditRecord` 和不参与决策的主机运行指标；
- Candidate 范围内的 Evidence binding、Core assessment preview 和 Candidate Frontier；
- LangChain 结构化输出边界；
- 只能通过 `EpisodeRunner` 执行的 LangGraph 兼容路由层；
- 受边界约束的 Inquiry、Discovery、Formalizer 和 Verification Planner；
- FastAPI Backend 展示主机与 Vue Frontend 调查控制台。

`RETRYABLE` 只表示主机可以制定恢复计划，不表示待处理的 Query、adapter call 或 Handoff 已经完成。跨进程重启目前不能自动重建 live Environment 或 adapter，主机必须显式重建依赖并决定重试或终止。

## Backend 和 Frontend

Backend 是只读展示主机和受控的本地运行执行器。它保留现有 API endpoint 和 server-owned profile：`scripted-trace-demo` 不调用模型，`qwen-investigation` 继续使用现有的主机侧 LangChain/LangGraph 组合。Backend 不重新拥有工作流、Evidence admission、Core decision 或停止权限。

Backend 展示层分别提供 Observation、Candidate Evidence admission、Core decision、Answer Freeze、Replay metadata、Recovery metadata 和安全 Audit 投影。Replay 无效、Answer 未冻结或 Recovery 不可继续时，API 不返回权威根因结论。

Frontend 默认使用 mock data mode，也可以通过 `VITE_RCA_DATA_MODE=api` 读取 Backend API。Frontend 分开展示 Observation、显式 Candidate admission、Core decision 和 Answer Freeze，不通过字符串匹配猜测 Candidate Evidence。`UNKNOWN`、`MISSING`、`INCONCLUSIVE`、`OUT_OF_SCOPE`、`QUERY_FAILED`、`REPLAY_INVALID` 和 `RETRYABLE` 都会明确展示。

Backend API 说明见 [backend/docs/api-v1.md](backend/docs/api-v1.md)，Frontend 说明见 [frontend/README.md](frontend/README.md)。当前继续使用轮询，不引入 SSE 或新的状态机。

## 本地运行和验证

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q

# 可选的本地 RCA100 integration test
RCA100_CASES_ROOT=/path/to/RCA100/cases uv run pytest -q -m requires_rca100

# Frontend
cd frontend
npm ci
npm run lint
npm run build
```

启动 Backend 时，主机通过环境变量或 `backend/app/config.py` 提供 RCA100 cases directory；该路径只留在主机侧。Frontend 可以独立使用 mock mode，也可以设置 `VITE_RCA_DATA_MODE=api` 后连接 `/api/v1`。

本项目运行时不依赖冻结的 sibling `rca-core` repository。运行时核心位于 `src/rca_agent/core`，研究模块不属于 Agent Runtime。私有研究运行、生成的 Evidence、provider Transcript 和本地 Workspace artifact 不属于公开源码分发；主机配置见 `.env.example`。

## Observation 的 trust state

每个 Observation 只有一个 trust state：

- `OBSERVED`：受边界约束的原始来源值；
- `DERIVED`：由已观察值确定性计算、可以重放的值；
- `IMPUTED`：插值或填补得到的值；
- `MODEL_INFERRED`：模型估计值；
- `MISSING`：预期数据不存在；
- `UNKNOWN`：已有数据不足以支持请求的解释。

只有可信的 `OBSERVED` 和可重放的 `DERIVED` 才能进入 Core confirmation。`IMPUTED` 和 `MODEL_INFERRED` 可以帮助后续探索，但不能成为关键正向 Evidence。

Policy proposal 可以引用可信的 `candidate_id`，但不能提供 condition、threshold、strength、sufficiency rule 或 final status。`UPDATE_CANDIDATE_EVIDENCE` 只能把已发行的 Observation ref 显式绑定到一个 Hypothesis；Candidate A 的 Observation 不会自动进入 Candidate B。

## LangChain、LangGraph 和 ReasoningAdapter

`LangChainPolicyAdapter` 只接受一个封闭的 `PolicyAction` structured output，其 schema 不包含 final-decision、threshold 或 evidence-weight。`LangGraphInvestigation` 只向模型提供经过清理的 bootstrap state、Candidate option、之前的 action result 和 Core 的只读 assessment preview。

preview 只公开 Candidate status、已满足或缺失的 signal、不可用 signal、required Capability 和查询缺口，不公开 condition operator、threshold、Evidence strength、winner margin 或 sufficiency rule。Graph 的 Candidate Frontier 每次都从 preview 重建，没有独立决策权限。

Graph 不直接调用 `RcaEnvironment` 或 Evidence controller。查询执行、预算、Observation 发行、proposal admission、Transcript 记录和 Answer Freeze 都留在 `EpisodeRunner`。Graph 只能请求停止，Runner 会重新计算并冻结权威 Core decision。Replay 不需要运行 LangGraph。

未来如果通用 Agent 能自主完成 Hypothesis 生成、查询选择、多轮调查、规划或停止建议，只需替换 `ReasoningAdapter`，无需改变 EpisodeRunner、EvidenceProvider、Capability registry、Observation ledger、CoreDecisionProvider、Transcript/Replay、Answer Freeze 或 Evaluation 边界。

## Recovery、Audit 和验证限制

worker 使用只传 JSON 的 typed payload 和有长度限制的 framed protocol。ready timeout、request timeout、cancel、exception、crash 和 malformed frame 都会关闭 stream、terminate/kill 并 wait/reap。迟到结果会在主机准入边界外被丢弃，不能污染 Transcript、Workspace、Observation 或已冻结的 Answer。

`EpisodeRecovery` 只做离线检查。发现 digest chain、event sequence、episode identity、Answer digest、Workspace projection、Candidate frontier projection、budget、Core binding 或 pending operation identity 不一致时，Recovery 会停止并返回明确状态，不猜测事实，也不生成新的 Answer。

`AuditRecord` 只记录批准的 metadata：episode/event identity、digest chain、Capability/provider、opaque query reference、范围和成本、adapter reference、Core decision、stop/Handoff、Answer digest、Replay result 和 Failure Taxonomy。secret、token、filesystem path、完整 prompt、raw trace/log identifier 和评估专用数据不会进入 Audit。Frontend developer mode 读取的 raw JSON 同样是安全投影。

当前 deterministic test 使用 synthetic fixture、offline stub 和 mock provider。真实 provider smoke test、真实生产 telemetry、外部服务验证和付费模型调用尚未纳入当前离线验证范围，不能用当前测试结果代替。

## DeepSeek model binding

准确性优先的 provider binding 使用 `deepseek-v4-pro`、thinking 和 high reasoning effort。`create_deepseek_policy()` 仍绑定同一个封闭的 structured Policy schema，不授予 model 额外 tools 或 decision authority。provider package 从 host process environment 读取 `DEEPSEEK_API_KEY`，secret 不得提交到 repository。DeepSeek 使用 JSON mode 生成 envelope；返回对象仍会按同一个 Pydantic schema 校验。
