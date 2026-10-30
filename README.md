# RCA Agent 1.0

RCA Agent 是一个受控的 Root Cause Analysis（RCA）investigation runtime。1.0 的目标是让
`Runtime（运行时执行层）`、`EvidenceProvider（返回中立 Observation 的 host adapter）`、
`CoreDecisionProvider（产生权威 Core decision 的 host-owned 组件）`、`Transcript（事实事件记录）`
和 `Replay（离线一致性验证）` 保持稳定，同时让 model-facing layer 可以替换。

当前 1.0 状态包括 P1-1 到 P2-5、Documentation / Comment Hardening，以及 Backend / Frontend
presentation integration adaptation。稳定架构说明见[架构说明](docs/architecture/overview.md)，稳定术语见
[术语表](docs/architecture/glossary.md)。本 README 说明 Runtime、Backend、Frontend 和离线验证边界。

`EpisodeRunner（Episode 生命周期和权威状态写入入口）` 持有 Runtime 的唯一控制权。它负责
admission、Capability check、Query execution、Observation issuance、Evidence admission、Core decision、
budget、stop、Handoff、Transcript、Workspace projection 和 `Answer Freeze（最终 Answer artifact 冻结）`。
`Workspace` 和 `Routing Projection` 都是派生 projection；它们不能成为第二套事实源。

`Observation（EvidenceProvider 返回的中立事实）` 不包含 support、contradiction、winner 或 root-cause
结论。`Evidence admission（Candidate 对已发行 Observation 的显式绑定）` 只由 host-owned Core 根据
Episode 内的 issued reference 处理。`Core decision（Core 产生的权威决策）` 才能产生 VERIFIED、SUPPORTED、
INCONCLUSIVE、OUT_OF_SCOPE、QUERY_FAILED、winner、Abstention 和 stop authority。`Answer` 只在
`Answer Freeze` 后对外作为冻结 artifact 展示。

`ReasoningAdapter（model-facing proposal facade）`、Discovery、Formalizer、Planner 和 LangGraph 只能
提出 Candidate、Query、Hypothesis、verification plan、next action、stop 或 Handoff proposal。它们不能
写 Observation、Transcript、Workspace、Answer、budget、permission 或 Core decision。

## 项目定位和边界

`rca-agent` 将 RCA100 包装成受控的 interactive investigation Environment。Inquiry Policy 只接收精简的
bootstrap payload，并通过 bounded、typed、read-only tools 获取额外事实。RCA100 是外部 private
observation directory，不会复制到本仓库，也不会把 dataset path、case selector、answer label 或
Evaluator annotation 发送到 Agent-visible state。

Runtime、Evidence、Evaluation 和 Replay 分工如下：

- Runtime 负责 Episode lifecycle、admission、Observation、Core decision、stop、Handoff 和 Answer Freeze。
- `EvidenceProvider` 只把受 Capability、entity scope、time range、result size、cost 和 budget 约束的
  `EvidenceQuery` 转成 neutral Observation。
- Evidence admission 由 Core 根据 Candidate 显式引用的 Observation refs 处理；Observation 不会因为 provider
  返回就自动成为 Candidate 的 Evidence。
- `Evaluation` 只在 Answer artifact Freeze 后读取 Ground Truth、RCA100 case、annotation 和 scoring data；
  Runtime 不导入 evaluator-only loader，也不把评分写回 Observation、Workspace、Candidate 或 Answer。
- `Replay` 只读取 Transcript 和冻结 artifact，不调用 model、EvidenceProvider、Environment 或外部服务。

1.0 的非目标包括复制 RCA100/Ground Truth、把 Evaluation 逻辑放入 Runtime、让 model 或 LangGraph 拥有
Core decision/budget/stop authority，以及扩展新的 workflow engine。

## 当前 Runtime 能力

仓库包含 trusted RCA Runtime 和不读取 Ground Truth 的本地 console：

- RCA Environment Contract 和 host-side RCA100 adapter；
- bounded Metric、Trace、Topology tools，以及明确 unavailable 的 Log Capability；
- query cost、episode budget 和 Capability boundary；
- 带 trust state 的 neutral Observation；
- typed `EpisodeRunner`、append-only hash-chained `Transcript` 和 Recorded Replay；
- typed Answer artifact、Answer Freeze 和独立的 post-freeze Evaluation artifact；
- JSON-only framed worker protocol，以及 timeout、cancel、terminate/kill/reap 和 late-result rejection；
- `EpisodeRecovery` 的 `RESUMABLE`、`RETRYABLE`、`TERMINAL`、`FROZEN` 和 `CORRUPTED` 状态；
- `AuditRecord` 和不影响决策的 host-owned operational metrics；
- Candidate-scoped Evidence binding、Core assessment preview 和 Graph Candidate Frontier；
- provider-neutral LangChain structured-output boundary；
- 只能通过 `EpisodeRunner` 执行的 LangGraph compatibility routing layer；
- bounded Inquiry、dynamic Discovery、Formalizer 和 Verification Planner contract；
- FastAPI Backend presentation host 与 Vue Frontend investigation console。

`RETRYABLE` 只表示 host recovery plan，不表示 pending Query、adapter call 或 Handoff 已经完成。跨进程
重启目前不能自动重建 live Environment 或 adapter；host 必须显式重建依赖并决定重试、终止或继续等待。

## Backend / Frontend 使用方式

`Backend` 是只读 presentation host 加上受控的本地 run executor。它保留现有 API endpoint 和 server-owned
profile：`scripted-trace-demo` 不调用 model，`qwen-investigation` 继续使用现有的 host-side LangChain/LangGraph
组合。Backend 不重新拥有 workflow、Evidence admission、Core decision 或 stop authority。

Backend detail projection 会区分 Observation、Candidate Evidence admission、Core decision、Answer Freeze、
Replay metadata、Recovery metadata 和安全 Audit projection。Replay invalid、Answer 未冻结或 Recovery
不可继续时，API 不返回 authoritative final decision；`RETRYABLE` 只返回 recovery plan。

`Frontend` 默认可以使用 mock data mode，也可以通过 `VITE_RCA_DATA_MODE=api` 读取 Backend API。Frontend 将
Observation、显式 Candidate admission、Core decision 和 Answer Freeze 分开显示；它不会通过字符串匹配猜测
Candidate Evidence。Replay invalid、未 Freeze、`UNKNOWN`、`MISSING`、`INCONCLUSIVE`、`OUT_OF_SCOPE`、
`QUERY_FAILED` 和 `RETRYABLE` 都会作为明确状态展示。

Backend API 说明见 [backend/docs/api-v1.md](backend/docs/api-v1.md)，Frontend 说明见
[frontend/README.md](frontend/README.md)。当前轮询方式保持不变，不引入 SSE 或新的 Frontend state machine。

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

启动 Backend 时，host 通过环境变量或 `backend/app/config.py` 提供 RCA100 cases directory；该 path 只留在 host
侧。Frontend 可以在 mock mode 下独立运行，也可以设置 `VITE_RCA_DATA_MODE=api` 后连接 `/api/v1`。

package 在 Runtime 上不依赖冻结的 sibling `rca-core` repository。已验证的 domain slice 位于
`src/rca_agent/core`，Research module 不属于 Agent Runtime。Private research run、generated Evidence、
provider Transcript 和 local Workspace artifact 不属于 public source distribution；host-only configuration
见 `.env.example`。

## Trust state 和 Evidence 边界

每个 Observation 恰有一个 trust state：

- `OBSERVED`：bounded raw source value；
- `DERIVED`：从 observed value 做出的 deterministic、可 Replay transformation；
- `IMPUTED`：interpolated 或 filled value；
- `MODEL_INFERRED`：model estimate；
- `MISSING`：预期的 source coverage 不存在；
- `UNKNOWN`：已有 value 不支持请求的 interpretation。

只有 trusted `OBSERVED` 和可 Replay 的 `DERIVED` value 才能进入 Core confirmation。`IMPUTED` 和
`MODEL_INFERRED` 可以帮助后续 exploration，但不能成为 critical positive Evidence。

Policy proposal 可以引用 trusted `candidate_id`，但不能提供 condition、threshold、strength、sufficiency
rule 或 final status。`UPDATE_CANDIDATE_EVIDENCE` 只能把已发行 Observation reference 显式绑定到一个
已注册 Hypothesis；Candidate A 的 Observation 不会自动进入 Candidate B。

## LangChain、LangGraph 和 ReasoningAdapter boundary

`LangChainPolicyAdapter` 只接受一个 closed `PolicyAction` structured output，其 schema 不包含
final-decision、threshold 或 evidence-weight field。`LangGraphInvestigation` 只向 model 提供 sanitized
bootstrap state、Candidate option、prior action result 和 Core 的 read-only assessment preview。

preview 只公开安全的 Candidate status、satisfied/missing/contradicting/unavailable signal name、required
Capability 和 discriminating query gap，不公开 condition operator、threshold、Evidence strength/weight、
winner margin 或 sufficiency rule。Graph 的 Candidate Frontier 每次都从 preview 重建，没有独立 decision authority。

Graph 不直接调用 `RcaEnvironment` 或 Evidence controller。Tool execution、budget/max-round enforcement、
Observation issuance、proposal admission、Transcript recording 和 Answer Freeze 都留在 `EpisodeRunner`。
Graph 只能请求 stop，Runner 会重新计算并 Freeze authoritative Core decision。Replay 不需要运行 LangGraph。

未来如果更强的通用 Agent 能自主完成 hypothesis generation、query selection、multi-round investigation、
planning 或 stopping suggestion，只需要替换 `ReasoningAdapter`，无需改变 EpisodeRunner、EvidenceProvider、
Capability registry、Observation ledger、CoreDecisionProvider、Transcript/Replay、Answer Freeze 或 Evaluation
boundary。

## Recovery、Audit 和验证限制

worker 使用 JSON-only typed payload 和 bounded framed protocol。ready timeout、request timeout、cancel、
exception、crash 和 malformed frame 都会关闭 stream、terminate/kill 并 wait/reap。late result 在 parent
admission 之外被丢弃，因此不能污染 Transcript、Workspace、Observation 或已冻结的 Answer。

`EpisodeRecovery` 只做离线检查。发现 digest chain、event sequence、episode identity、Answer digest、Workspace
projection、Candidate frontier projection、budget、Core binding 或 pending operation identity 不一致时，
Recovery 会停止并返回明确 failure state，不猜测事实，也不生成新的 Answer。

`AuditRecord` 只记录 approved metadata：episode/event identity、digest chain、Capability/provider、opaque
query reference、scope/time/cost、adapter reference、Core decision、stop/Handoff、Answer digest、Replay
result 和 Failure Taxonomy。secret、token、filesystem path、完整 prompt、raw trace/log identifier 和
Evaluator-only data 不进入 Audit。Frontend developer mode 读取的 raw JSON 也是安全 projection。

当前 deterministic test 使用 synthetic fixture、offline stub 和 mock provider。真实 provider smoke test、
真实生产 telemetry、外部服务验证和付费 model 调用尚未纳入当前离线验证范围；它们不能被当前测试结果替代。

## DeepSeek model binding

accuracy-first provider binding 使用 `deepseek-v4-pro`、thinking 和 high reasoning effort。`create_deepseek_policy()`
仍绑定同一个 closed structured Policy schema，不授予 model 额外 tools 或 decision authority。provider package
从 host process environment 读取 `DEEPSEEK_API_KEY`，secret 不得提交到 repository。DeepSeek 使用 JSON mode
生成 envelope；返回对象仍会按同一个 Pydantic schema 校验。
