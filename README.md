# RCA Agent 1.0

RCA Agent 1.0 的稳定架构说明见 [架构说明](docs/architecture/overview.md)，稳定术语见
[术语表](docs/architecture/glossary.md)。本 README 描述当前 Runtime 的边界、离线验证方式
和可替换的 model-facing layer。

Runtime 的权威状态由 `EpisodeRunner` 持有。`Transcript` 是 append-only、hash-chained 的
事实记录；`Workspace` 和 `RoutingProjection` 都是派生 projection；`CoreDecisionProvider`
负责 Candidate-scoped Evidence admission 后的 authoritative decision；`Answer Freeze` 后
任何 model、LangGraph、Workspace 或 Recovery projection 都不能修改 Answer。

Reasoning Adapter、Discovery、Formalizer、Planner 和 LangGraph 只产生或传递 typed proposal。
worker 使用 JSON-only framed protocol，timeout、cancel、crash 和 late result 都由 parent
terminate/kill/reap，父进程是 Transcript、Observation、Workspace 和 Answer 的唯一写入者。
当前跨进程重启不能自动重建 live Environment 或 adapter；真实 provider smoke test 也不在当前
离线测试范围内。

1.0 的非目标是复制 RCA100/Ground Truth、把 evaluator 逻辑放入 Runtime、让模型拥有
decision/budget/stop authority，或在本阶段扩展新的 workflow engine。

可选的 unknown-mechanism Discovery 已接入 LangGraph/Runner 主循环。它保留静态 Catalog，并根据
host-owned signal/rule contract 验证 dynamic predicate，详见[dynamic Discovery 的设置和限制](analysis/dynamic_discovery.md)。

通过通用 verification standard 的 dynamic mechanism 会以 `DISCOVERED_VERIFIED` 作为本次
Episode 的 root-cause result；这不会自动把它提升到 Catalog，也不声称穷尽因果证明。

`rca-agent` 将 RCA100 包装成受控的 interactive investigation Environment。Inquiry Policy 接收
精简的事件 bootstrap payload，只能通过 bounded、typed、read-only tools 获取额外事实。Candidate
interpretation 和最终 Evidence decision 属于内部 Agent domain core。

仓库包含 trusted RCA Runtime 和不读取 Ground Truth 的本地 console：

- repository 与 trust boundary 的基础结构；
- RCA Environment Contract；
- 保持 dataset location 和 case selector 私有的 host-side RCA100 adapter；
- bounded Metric、Trace、Topology tools，以及明确 unavailable 的 Log Capability；
- query cost 和 episode budget enforcement；
- 带 trust state 的 neutral Observation；
- typed `EpisodeRunner` 和 Episode lifecycle；
- append-only、hash-chained `Transcript`；
- 不调用模型或外部服务的 Recorded Replay；
- typed Answer artifact 和独立的 post-freeze Evaluation artifact；
- 用于 Scripted/Fake Policy lifecycle test 的受控 child process；
- episode-scoped issued-Observation ledger；
- 连接内部 evidence-controlled domain core 的 trust-aware bridge；
- Catalog-owned Candidate resolution 和 authoritative decision；
- 每次 Policy action 后只读、不可 Freeze 的 Core assessment preview；
- provider-neutral 的 LangChain structured-output boundary；
- 只能通过 `EpisodeRunner` 执行的 LangGraph compatibility routing layer；
- 覆盖 Tool -> Observation -> Core -> Freeze 的 no-API-key Scripted Policy test；
- Candidate-scoped Evidence binding，不复制 episode-global Observation；
- threshold-free 的 Candidate Core preview 和 discriminating-signal view；
- 从 Core preview 派生的 Graph Candidate Frontier；
- bounded Inquiry 和 result-driven investigation loop；
- dynamic Discovery、Formalizer 和 Verification Planner contract；
- FastAPI presentation host 与 Vue investigation console。

RCA100 仍是外部 private observation directory，不会打包进仓库。公开 test 和 CI 使用 synthetic
Environment；private RCA100 integration 和 benchmark run 与发布质量、准确率声明分离。

## Setup 和 verification

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

package 在 Runtime 上不依赖冻结的 sibling `rca-core` repository。已验证的 domain slice 位于
`src/rca_agent/core`，包含 bounded RCA100 observation access、Candidate 和 EvidenceCondition
contract、Capability compilation、Candidate proposal/Discovery、Evidence assessment、
Candidate-state aggregation 和 safe decision。Research module 不属于 Agent Runtime。RCA100 data
不会复制到本仓库；host 在构造 Environment 时提供已有的 `cases` directory，该 path 不会进入
Agent-visible response。

Private research run、generated Evidence、provider Transcript 和 local Workspace artifact 不属于
public source distribution。host-only configuration 见 `.env.example`。

## Trust state

每个 Observation 恰有一个 trust state：

- `OBSERVED`：bounded raw source value；
- `DERIVED`：从 observed value 做出的 deterministic、可 Replay transformation；
- `IMPUTED`：interpolated 或 filled value；
- `MODEL_INFERRED`：model estimate；
- `MISSING`：预期的 source coverage 不存在；
- `UNKNOWN`：已有 value 不支持请求的 interpretation。

只有 trusted `OBSERVED` 和可 Replay 的 `DERIVED` value 才能进入 Core confirmation。`IMPUTED` 和
`MODEL_INFERRED` 可以帮助后续 exploration，但不能成为 critical positive Evidence。

Policy proposal 可以引用 trusted `candidate_id`，但不能提供 condition、threshold、strength、
sufficiency rule 或 final status。Answer Freeze 时，Core 只评估已发行且由 proposal 显式引用的
Observation。

Observation 是 immutable 的 episode-global fact。`UPDATE_CANDIDATE_EVIDENCE` 将已发行 Observation
reference 显式绑定到一个已注册 Hypothesis；Candidate A 的 fact 不会自动进入 Candidate B，除非
Policy 也显式把同一 reference 绑定到 B。

## LangChain、LangGraph 和 Reasoning Adapter boundary

`LangChainPolicyAdapter` 只接受一个 closed `PolicyAction` structured output，其 schema 不包含
final-decision、threshold 或 evidence-weight field。`LangGraphInvestigation` 只向 model 提供
sanitized bootstrap state、Candidate option、prior action result 和 Core 的 read-only assessment
preview。

preview 只公开安全的 Candidate status、satisfied/missing/contradicting/unavailable signal name、
required Capability 和 discriminating query gap，不公开 condition operator、threshold、Evidence
strength/weight、winner margin 或 sufficiency rule。Graph 的 Candidate Frontier 每次都从 preview
重建，没有独立 decision authority。

Graph 不直接调用 `RcaEnvironment` 或 Evidence controller。Tool execution、budget/max-round
enforcement、Observation issuance、proposal admission、Transcript recording 和 Answer Freeze 都
留在 `EpisodeRunner`。Graph 只能请求 stop，Runner 会重新计算并 Freeze authoritative Core
decision。Replay 不需要运行 LangGraph。

`ReasoningAdapter` 是可替换的 model-facing facade，只能返回 Candidate、Query、Hypothesis、
verification plan、next action、stop 或 Handoff proposal。未来可以替换成通用 Agent，而不改变
EpisodeRunner、EvidenceProvider、Capability registry、Observation ledger、CoreDecisionProvider、
Transcript/Replay、Answer Freeze 或 Evaluation boundary。

## Recovery、Audit 和 provider boundary

worker 使用 JSON-only typed payload 和 bounded framed protocol。ready timeout、request timeout、
cancel、exception、crash 和 malformed frame 都会关闭 stream、terminate/kill 并 wait/reap。late
result 在 parent admission 之外被丢弃，因此不能污染 Transcript、Workspace、Observation 或已
冻结的 Answer。

`EpisodeRecovery` 只做离线检查，并返回 `RESUMABLE`、`RETRYABLE`、`TERMINAL`、`FROZEN` 或
`CORRUPTED`。`RETRYABLE` 是 host 的 pending-operation plan，不表示已经完成恢复；跨进程重启
当前不能自动重建 live Environment 或 adapter。

`AuditRecord` 只记录 approved metadata：episode/event identity、digest chain、Capability/provider、
opaque query reference、scope/time/cost、adapter reference、Core decision、stop/Handoff、Answer
digest、Replay result 和 Failure Taxonomy。secret、token、filesystem path、完整 prompt、raw
trace/log identifier 和 evaluator-only data 不进入 Audit。

Evaluation 仅在 Answer artifact Freeze 后加载 Ground Truth、RCA100 case、annotation 和 scoring
data。Runtime 不导入 evaluator-only loader，也不把 Evaluation result 写回 Observation、Workspace、
Candidate 或 Answer。metrics 只用于诊断，不改变 Runtime decision。

## DeepSeek model binding

accuracy-first provider binding 使用 `deepseek-v4-pro`、thinking 和 high reasoning effort。
`create_deepseek_policy()` 仍绑定同一个 closed structured Policy schema，不授予 model 额外 tools
或 decision authority。provider package 从 host process environment 读取 `DEEPSEEK_API_KEY`，
secret 不得提交到 repository。DeepSeek 使用 JSON mode 生成 envelope；返回对象仍会按同一个
Pydantic schema 校验。当前 deterministic test 不调用付费 model、真实 telemetry 或外部服务；
真实 provider smoke test 尚未纳入本阶段。
