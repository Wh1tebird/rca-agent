# RCA Investigation Console Demo

一个独立于 Python 后端的 Vue 3 Mock Demo，用于展示 RCA Agent 的候选、查询、证据、Core
裁决与追加调查链路。第一版不访问真实接口，也不改变现有 Agent Runtime。

## 运行

```bash
cd frontend
npm install
npm run dev
```

生产构建与检查：

```bash
npm run build
npm run lint
```

## 页面信息架构

1. Incident Header：事故、服务、Operation、时间窗口、严重级别、当前状态与调查成本。
2. Candidate Frontier：区分 Inquiry 提到、正式 Mechanism、Compiled Candidate 与 Core 状态。
3. Investigation Timeline：串联 Inquiry、Query、Observation、Discovery、Planner、Core 和追加查询。
4. Evidence Inspector：展示 Metric、Trace、Query Result 及其证据角色和关联 Candidate。
5. Final Decision：用普通语言解释支持证据、缺失证据、未决替代原因和未 VERIFIED 原因。
6. Developer Mode：显示内部引用、计划版本、Replay 状态和冻结后的原始 JSON。

## 组件拆分

- `IncidentHeader.vue`：事故摘要。
- `CandidateSidebar.vue`：候选根因及三阶段状态。
- `InvestigationTimeline.vue`：调查过程与 Core 反馈闭环。
- `EvidenceInspector.vue`：证据详情和候选联动。
- `MetricChart.vue`：ECharts 指标趋势。
- `FinalDecision.vue`：最终权威裁决。
- `StatusBadge.vue`：统一状态视觉语言。
- `mock/rcaCase.ts`：完整 Mock Case，也是后续 API ViewModel 的参考结构。

## 后端接入建议

第一版建议只提供面向展示的只读 View API，不让前端拼装 Core 语义：

- `GET /api/rca/cases/{caseId}`：事故摘要和冻结状态。
- `GET /api/rca/cases/{caseId}/candidates`：Inquiry、Mechanism、编译和 Core 状态。
- `GET /api/rca/cases/{caseId}/timeline`：按 Transcript 投影出的安全时间线。
- `GET /api/rca/cases/{caseId}/evidence/{evidenceId}`：图表点、Observation 和证据归属。
- `GET /api/rca/cases/{caseId}/decision`：最终裁决及支持、缺失和替代机制。
- `GET /api/rca/cases/{caseId}/developer`：受权限控制的引用、版本、Replay 和原始冻结 JSON。
- `GET /api/rca/cases/{caseId}/events`：可选 SSE，用于调查运行中的增量更新。

后端应直接返回脱敏后的 reasoning summary；不要把模型隐藏 chain-of-thought、Core 私有阈值、数据集
Ground Truth 或原始凭据暴露给前端。
