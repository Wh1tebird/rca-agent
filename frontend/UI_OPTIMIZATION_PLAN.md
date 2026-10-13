# RCA Investigation Console UI 优化计划

更新时间：2026-09-21  
实施模型建议：Codex Terra，中等推理强度

## 1. 本轮目标

在不修改 RCA 算法、Core 判定、Qwen Prompt、后端运行链路和 API 契约的前提下，
把现有前端打磨成适合演示、截图和技术汇报的 RCA 调查工作台。

用户打开一个真实 Run 后，应能在 10 秒内看懂：

1. 发生了什么故障；
2. AI 提出过哪些方向；
3. 哪些方向进入正式验证；
4. 查询获得了哪些关键证据；
5. Core 淘汰、保留或确认了哪些候选；
6. 最终为什么是 VERIFIED、SUPPORTED 或 INCONCLUSIVE。

本轮以真实 Qwen t030 Run 为主要验收样本：

```text
run-live-87c10830a6214f2f
5 个 Inquiry round
14 次查询
2 个正式 Mechanism
1 个 FAILED
1 个 UNRESOLVED
最终 SUPPORTED_HYPOTHESIS
Replay VALID
```

## 2. 明确不做的内容

- 不修改 Core 验证标准、阈值和证据充分性规则；
- 不修改 Inquiry、Discovery、Planner 或模型 Prompt；
- 不增加 Agent、Benchmark 或 GT 评分；
- 不重新设计后端 API；
- 不把前端展示状态反写为 Core 状态；
- 不根据 Observation 自行推断 Support/Refutation；
- 不做移动端优先设计；
- 不做账号、权限、数据库、部署和流式事件系统。

## 3. 当前页面主要问题

### 3.1 信息很多，但阅读顺序不够明确

真实 Run 可能包含 40 多个 Timeline 事件和几十条 Observation。当前页面逐条平铺后，
用户容易看到“查了很多”，却看不出调查阶段和候选如何变化。

### 3.2 三种候选概念容易混在一起

页面需要更明确地区分：

- `Tentative Idea`：Inquiry 曾提出，但没有进入正式验证；
- `Formal Mechanism`：Discovery 已形式化；
- `Core State`：Core 给出的 FAILED、UNRESOLVED、SUPPORTED 或 VERIFIED 状态。

### 3.3 Evidence 数量大，但关键证据不突出

当前 Evidence Inspector 容易显示第一条记录，而不是当前 Candidate 或 Timeline Step 对应的证据。
Metric、Trace、实体发现和缺失数据也需要更清楚的视觉分类。

### 3.4 最终结论与调查过程联系不足

Final Decision 应明确回答：

- 当前最可信方向是什么；
- 哪个候选已经失败；
- 还缺什么证据；
- 为什么没有 VERIFIED；
- Replay 是否有效。

### 3.5 运行中的反馈还比较弱

真实 Qwen Run 可能持续几十秒。现在只有一行状态文字，用户难以区分等待、模型调查、查询和
Replay 校验阶段。

## 4. 推荐的信息架构

页面保持单页工作台，不改成聊天界面。

```text
┌──────────────── Incident Summary ────────────────┐
│ 告警 / 服务 / 时间窗 / Run / Replay / 查询数 / 耗时 │
└───────────────────────────────────────────────────┘

┌──────── Candidate Rail ────────┬──── Investigation Flow ────┬──── Evidence Inspector ────┐
│ 正式候选                       │ 调查阶段                     │ 当前选中证据                  │
│ - FAILED                       │ 1. Incident                  │ 摘要 / 状态 / 数据源          │
│ - UNRESOLVED                   │ 2. Inquiry rounds            │ Metric / Trace 展示           │
│                                │ 3. Discovery                 │ 关联 Candidate                │
│ Inquiry 提到但未正式验证       │ 4. Verification              │ Developer refs                │
│ - NOT_PROPOSED                 │ 5. Core Decision             │                                │
└────────────────────────────────┴──────────────────────────────┴───────────────────────────────┘

┌──────────────────────────── Final Decision ───────────────────────────────────────────────┐
│ 当前结论 / 支持信息 / 失败候选 / 缺失证据 / 未解决 alternatives / 为什么未 VERIFIED       │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

桌面端建议宽度比例：

```text
Candidate 23% / Timeline 47% / Evidence 30%
```

在较窄桌面窗口中，Evidence Inspector 改为右侧 Drawer，不把 Timeline 压得过窄。

## 5. 具体改造任务

### 5.1 顶部与 Case/Run 控制区

目标：减少工具栏占高，让当前 Case 和 Run 状态一眼可见。

- 合并 Case 选择、Run 选择和“开始调查”为一条紧凑控制栏；
- 显示当前运行 Profile，例如 `Qwen Thinking` 或 `Scripted Demo`；
- 将 Replay、Run 状态、查询数、耗时做成统一的小型状态单元；
- 运行中显示阶段式进度：创建任务、调查中、冻结答案、Replay 检查；
- 失败时展示安全错误码和“重新运行”入口，不展示异常堆栈；
- 清理重复渲染的 Final Decision。

### 5.2 Candidate Rail

目标：先展示正式候选，再展示 Inquiry 想过但未正式验证的方向。

- 增加顶部统计：Formalized、Failed、Unresolved、Verified 数量；
- 分成两个区块：
  - `Formal Candidates`；
  - `Explored Ideas`；
- Candidate Card 固定展示：
  - 简短名称；
  - 生命周期：Mentioned / Formalized / Compiled；
  - Core 状态 Badge；
  - 一句话机制摘要；
- `FAILED` 使用低饱和红色，表达“已被证据否定”，而不是系统报错；
- `UNRESOLVED` 使用琥珀色，表达“仍缺证据”；
- `NOT_PROPOSED` 使用灰色并降低视觉权重；
- 点击 Candidate 后，Timeline 和 Evidence 只高亮关联项，不隐藏完整调查历史；
- 不把 Final Decision 的整体状态复制到每个 Candidate。

### 5.3 Investigation Timeline

目标：把事件日志变成可读的调查故事，这是本轮最重要的优化。

- 默认按阶段分组，而不是将所有事件同等平铺：
  1. Incident；
  2. Inquiry；
  3. Discovery；
  4. Verification；
  5. Core Decision；
- 将一次 Inquiry round 的“候选更新 + 1～2 个查询 + Observation”组合成一个 Round Card；
- Round Card 折叠态只显示：候选变化、选择的查询、关键结果；
- 展开后再显示完整 Query、Observation refs 和原始结构化数据；
- Discovery 阶段突出：
  - 哪些 Inquiry Idea 被 discarded；
  - 哪些 Mechanism 被 frozen；
  - 哪些 Plan 编译成功；
- Verification 阶段用状态迁移表达：

```text
UNRESOLVED → 新查询 → FAILED
UNRESOLVED → 新查询 → SUPPORTED
```

- Core Decision 卡片必须显示 blockers、missing evidence 和 unresolved alternatives；
- 普通模式不展示隐藏思维链，只展示现有结构化 summary；
- Developer Mode 保留原始事件 Payload、引用 ID 和序列号。

### 5.4 Evidence Inspector

目标：优先解释“这条数据是什么”，而不是先展示 JSON。

- 顶部显示 Signal、实体、数据状态、时间范围和查询工具；
- Metric 有时间序列时使用 ECharts；无序列时使用数值摘要卡；
- Trace 使用阶段耗时条形图或占比条，不伪装成 Metric 折线图；
- `MISSING`、`UNKNOWN`、`QUERY_FAILED` 分别展示，不能统一显示为 0；
- 选中 Candidate 后，优先定位它引用的 Evidence；
- 选中 Timeline Query 后，显示该次 Query 返回的全部 Observation，并支持前后切换；
- 当前 API 没有权威 evidence role 时继续显示 `Context`，不得由前端猜测 Support 或 Refutation；
- Developer Mode 才展示 `query_ref`、`observation_ref`、`capability_ref` 和 Raw JSON。

### 5.5 Final Decision

目标：让非研发人员能读懂 Core 的谨慎结论。

- 第一行只显示最终状态和一句人话结论；
- 第二层分四列或四块：
  - 当前支持；
  - 已否定候选；
  - 缺失证据；
  - 未解决竞争解释；
- 单独显示“为什么没有 VERIFIED”；
- 显示 Replay VALID/INVALID，说明结论是否可重放；
- `SUPPORTED_HYPOTHESIS` 不翻译成“已确认根因”；
- 如果 Answer 与 Candidate 状态存在差异，分别标注 Overall Decision 和 Candidate State。

### 5.6 Developer Mode

- 普通模式默认隐藏所有内部 ID 和原始 JSON；
- Developer Mode 增加右侧详情 Tabs：
  - Summary；
  - References；
  - Raw JSON；
- 原始 JSON 支持复制；
- 所有字段来自现有安全 API 投影，不读取本地 Run 文件；
- 继续禁止显示 GT、密钥、数据目录和隐藏 chain-of-thought。

### 5.7 视觉规范

- 延续浅色 Observability 产品风格，不使用大面积纯黑或 AI 渐变；
- 主色采用低饱和蓝，状态色固定：
  - VERIFIED：绿色；
  - SUPPORTED：蓝色；
  - UNRESOLVED：琥珀色；
  - FAILED：低饱和红；
  - INCONCLUSIVE：紫灰色；
  - NOT_PROPOSED：中性灰；
- 卡片边框、间距、圆角和阴影统一为设计 Token；
- 减少无意义大标题和大留白，保持 APM 控制台的信息密度；
- 1440px 和 1920px 下均不得横向溢出；
- 页面应能在 1440×900 截图中同时看到 Incident、候选、Timeline 主流程和 Final Decision。

## 6. 推荐组件调整

优先修改现有组件，不进行大规模重写：

| 文件 | 改造内容 |
| --- | --- |
| `src/App.vue` | 页面布局、运行状态、选择联动、移除重复 Final Decision |
| `src/components/IncidentHeader.vue` | 紧凑 Incident/Run 摘要 |
| `src/components/CandidateSidebar.vue` | Formal/Explored 分组及状态统计 |
| `src/components/InvestigationTimeline.vue` | 阶段分组、Round Card、折叠详情 |
| `src/components/EvidenceInspector.vue` | 多 Observation 浏览、Trace/Metric 差异展示 |
| `src/components/FinalDecision.vue` | 人话结论和四类依据 |
| `src/components/StatusBadge.vue` | 统一状态语义和颜色 |
| `src/components/MetricChart.vue` | 空值、缺失值和无序列状态 |
| `src/styles/base.css` | Design Token、三栏响应式和打印样式 |

允许增加少量纯展示组件，例如：

```text
InvestigationPhase.vue
InquiryRoundCard.vue
CandidateLifecycle.vue
RunProgress.vue
```

不要为每种 Event 创建一个独立组件，也不要引入新的状态管理框架。

## 7. 实施顺序

### 阶段 A：信息层级与布局

1. 修正重复区域和基础三栏比例；
2. Candidate 分组；
3. Timeline 阶段分组；
4. Final Decision 重排；
5. 完成 1440px / 1920px 静态验收。

### 阶段 B：交互联动

1. Candidate → Timeline/Evidence 高亮；
2. Timeline → Evidence 精确选择；
3. Inquiry Round 折叠/展开；
4. Developer Mode Tabs 和 JSON 复制；
5. 运行中进度和失败状态。

### 阶段 C：真实数据验收与视觉收尾

1. 使用真实 t030 Qwen Run 验证长 Timeline 和大量 Evidence；
2. 使用 scripted t021 Run 验证无 Candidate、少数据场景；
3. 使用 Mock 模式验证离线演示；
4. 检查空数据、Replay INVALID、Run FAILED 等状态；
5. 调整字体、间距、状态色、图表和打印样式；
6. 生成 1440×900 和 1920×1080 截图进行视觉检查。

## 8. 验收标准

- 不修改后端、Core、模型 Prompt、API Schema 和数据文件；
- Mock/API 两种模式均可运行；
- t030 的 5 轮 Inquiry 不再以 40 多条同权事件平铺；
- 正式 Candidate 与未正式验证的 Idea 明确分区；
- FAILED 和 UNRESOLVED Candidate 与真实 Core 记录一致；
- 选择 Candidate、Timeline 和 Evidence 时联动正确；
- 缺失数据不显示为 0，不由前端编造 Evidence role；
- 普通模式不显示内部 ID 和 Raw JSON；
- Developer Mode 可查看现有安全投影；
- 1440px 与 1920px 页面无横向溢出；
- `npm run build` 通过；
- `npm run lint` 通过；
- 后端 Python 测试不应因纯 UI 修改发生变化。

## 9. Terra 执行提示

执行时先读取：

```text
AGENTS.md
PROJECT_CONTEXT.md
HANDOFF.md
frontend/UI_OPTIMIZATION_PLAN.md
```

然后检查真实 Run：

```text
backend/var/runs/run-live-87c10830a6214f2f/
```

该 Run 只用于验证展示效果，不读取 GT，不重新调用模型。严格按阶段 A → B → C 实施；
不要顺手修改算法、Core 或后端契约。完成后汇报修改文件、三种数据场景的视觉结果、构建结果，
以及仍受当前 API 数据缺口限制的 UI 项目。

