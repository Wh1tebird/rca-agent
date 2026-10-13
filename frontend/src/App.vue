<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { Document, Refresh, Search, Setting } from '@element-plus/icons-vue'

import CandidateSidebar from '@/components/CandidateSidebar.vue'
import EvidenceInspector from '@/components/EvidenceInspector.vue'
import FinalDecision from '@/components/FinalDecision.vue'
import IncidentHeader from '@/components/IncidentHeader.vue'
import InvestigationTimeline from '@/components/InvestigationTimeline.vue'
import { createDataSource, type CaseOption, type RunOption } from '@/data/rcaDataSource'
import type { Candidate, RcaCase, TimelineStep } from '@/types/rca'

const dataSource = createDataSource()
const developerMode = ref(false)
const rawDrawerOpen = ref(false)
const cases = ref<CaseOption[]>([])
const runs = ref<RunOption[]>([])
const selectedCaseId = ref('')
const selectedRunId = ref('')
const activeCase = ref<RcaCase | null>(null)
const selectedCandidateId = ref('')
const selectedTimelineId = ref('')
const selectedEvidenceId = ref('')
const loading = ref(true)
const loadError = ref('')
const actionError = ref('')
const startingRun = ref(false)
const runProgress = ref('')
const sourceMode = import.meta.env.VITE_RCA_DATA_MODE === 'api' ? 'API' : 'Mock'

const selectedCase = computed(() => cases.value.find((item) => item.caseId === selectedCaseId.value))
const selectedEvidence = computed(() => activeCase.value?.evidence.find((item) => item.id === selectedEvidenceId.value) ?? activeCase.value?.evidence[0])
const runProfileLabel = computed(() => sourceMode === 'Mock' ? '演示数据' : 'Qwen Thinking')

function selectCandidate(candidate: Candidate) {
  selectedCandidateId.value = candidate.id
  const relatedStep = activeCase.value?.timeline.find((step) => candidateMatchesStep(candidate, step))
  if (relatedStep) {
    selectedTimelineId.value = relatedStep.id
    selectedEvidenceId.value = relatedStep.evidenceId ?? candidate.evidenceIds[0] ?? selectedEvidenceId.value
  } else selectedEvidenceId.value = candidate.evidenceIds[0] ?? selectedEvidenceId.value
}

function selectTimeline(step: TimelineStep) {
  selectedTimelineId.value = step.id
  if (step.evidenceId) selectedEvidenceId.value = step.evidenceId
  if (step.candidateIds?.[0]) selectedCandidateId.value = step.candidateIds[0]
}

function candidateMatchesStep(candidate: Candidate, step: TimelineStep) {
  if (step.candidateIds?.includes(candidate.id)) return true
  return Boolean(candidate.mechanismId && JSON.stringify(step.developer).includes(candidate.mechanismId))
}

async function selectCase(caseId: string) {
  loading.value = true
  loadError.value = ''
  actionError.value = ''
  runProgress.value = ''
  selectedCaseId.value = caseId
  try {
    runs.value = await dataSource.listRuns(caseId)
    selectedRunId.value = runs.value[0]?.runId ?? ''
    activeCase.value = selectedRunId.value ? await dataSource.loadRun(selectedRunId.value) : await dataSource.emptyCase(caseId)
    resetSelection()
  } catch (error) {
    activeCase.value = null
    loadError.value = error instanceof Error ? error.message : '无法加载调查记录。'
  } finally { loading.value = false }
}

async function selectRun(runId: string) {
  if (!runId) return selectCase(selectedCaseId.value)
  loading.value = true
  loadError.value = ''
  actionError.value = ''
  selectedRunId.value = runId
  try {
    activeCase.value = await dataSource.loadRun(runId)
    resetSelection()
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '无法加载调查记录。'
  } finally { loading.value = false }
}

function resetSelection() {
  selectedCandidateId.value = activeCase.value?.candidates[0]?.id ?? ''
  selectedTimelineId.value = activeCase.value?.timeline.at(-1)?.id ?? ''
  selectedEvidenceId.value = activeCase.value?.evidence[0]?.id ?? ''
}

async function startInvestigation() {
  if (!selectedCaseId.value || selectedCase.value?.readiness !== 'ready') return
  startingRun.value = true
  actionError.value = ''
  runProgress.value = '正在创建调查任务…'
  try {
    const created = await dataSource.createRun(selectedCaseId.value)
    selectedRunId.value = created.runId
    for (let attempt = 0; attempt < 3600; attempt += 1) {
      const current = await dataSource.getRunStatus(created.runId)
      runProgress.value = current.status === 'queued' ? '等待执行' : 'AI 正在调查并记录过程'
      if (current.status === 'succeeded') {
        activeCase.value = await dataSource.loadRun(created.runId)
        runs.value = await dataSource.listRuns(selectedCaseId.value)
        if (selectedCase.value) selectedCase.value.runCount = runs.value.length
        resetSelection()
        runProgress.value = '调查完成，重放检查通过'
        return
      }
      if (current.status === 'failed' || current.status === 'cancelled') throw new Error(`调查 ${current.status}: ${current.errorCode ?? '未知错误'}`)
      await new Promise((resolve) => window.setTimeout(resolve, 1000))
    }
    throw new Error('调查等待超时')
  } catch (error) {
    const message = error instanceof Error ? error.message : ''
    actionError.value = message.includes('not configured')
      ? '当前后端没有配置 Qwen 密钥，暂时不能新建真实调查；历史调查仍可正常查看。'
      : (message || '无法启动调查，请稍后重试。')
  } finally { startingRun.value = false }
}

onMounted(async () => {
  try {
    cases.value = await dataSource.listCases()
    const initial = cases.value.find((item) => item.runCount > 0) ?? cases.value.find((item) => item.readiness === 'ready') ?? cases.value[0]
    if (initial) await selectCase(initial.caseId)
    else loadError.value = '没有发现可展示的 RCA 案例。'
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '无法加载 RCA 案例列表。'
  } finally { loading.value = false }
})
</script>

<template>
  <div class="app-shell">
    <aside class="side-nav">
      <div class="brand-group"><div class="brand-mark"><span></span><span></span><span></span></div><div class="brand-copy"><strong>RCA Agent</strong><span>根因调查工作台</span></div></div>
      <span class="nav-label">工作区</span>
      <nav class="primary-nav" aria-label="主导航"><button class="active" type="button"><i>◎</i>调查记录</button><button type="button"><i>▣</i>案例库</button><button type="button"><i>◇</i>查询能力</button><button type="button"><i>↻</i>重放检查</button></nav>
      <span class="nav-label">系统</span>
      <nav class="primary-nav"><button type="button"><i>⌁</i>数据源</button><button type="button"><i>⚙</i>设置</button></nav>
      <div class="nav-status"><i></i><span>本地 API 已连接<small>{{ runProfileLabel }} · RCA100</small></span></div>
    </aside>

    <div class="console-shell">
      <header class="app-bar">
        <div class="global-search"><el-icon><Search /></el-icon><span>搜索案例、候选根因、证据或 Trace…</span></div>
        <div class="app-actions"><div class="environment-chip"><i></i>{{ sourceMode }} · {{ runProfileLabel }}</div><div class="replay-chip"><el-icon><Refresh /></el-icon>重放 {{ activeCase?.replayStatus ?? '—' }}</div><div class="mode-switch"><el-icon><Setting /></el-icon><span>开发者模式</span><el-switch v-model="developerMode" size="small" /></div><el-button v-if="developerMode" :icon="Document" plain @click="rawDrawerOpen = true">原始 JSON</el-button></div>
      </header>

      <main class="console-content">
        <section class="case-controls" aria-label="案例和运行记录选择">
          <el-select v-model="selectedCaseId" placeholder="选择 RCA100 案例" :loading="loading" filterable @change="selectCase"><el-option v-for="item in cases" :key="item.caseId" :label="`${item.caseId} · ${item.title}`" :value="item.caseId"><span>{{ item.caseId }} · {{ item.title }}</span><small>{{ item.readiness }} · {{ item.runCount }} 次调查</small></el-option></el-select>
          <el-select v-model="selectedRunId" placeholder="选择历史调查" :disabled="runs.length === 0 || loading" @change="selectRun"><el-option v-for="item in runs" :key="item.runId" :label="`${item.finalStatus ?? item.status} · ${item.startedAt}`" :value="item.runId" /></el-select>
          <el-button type="primary" :loading="startingRun" :disabled="selectedCase?.readiness !== 'ready' || loading" @click="startInvestigation">开始新调查</el-button>
          <span v-if="runProgress" class="run-progress"><i></i>{{ runProgress }}</span>
        </section>
        <div v-if="actionError" class="action-alert"><strong>启动失败</strong><span>{{ actionError }}</span><button type="button" @click="actionError = ''">×</button></div>

        <div v-if="loading" class="state-panel">正在加载调查数据…</div>
        <div v-else-if="loadError" class="state-panel error">{{ loadError }}</div>
        <template v-else-if="activeCase">
          <IncidentHeader :incident="activeCase" />
          <nav class="detail-tabs" aria-label="调查详情导航"><button type="button">总览</button><button class="active" type="button">调查过程</button><button type="button">证据</button><button type="button">指标</button><button type="button">日志</button><button type="button">Trace</button><button type="button">重放</button></nav>
          <div v-if="selectedEvidence" class="workspace-grid">
            <CandidateSidebar :candidates="activeCase.candidates" :selected-id="selectedCandidateId" :developer-mode="developerMode" @select="selectCandidate" />
            <InvestigationTimeline :steps="activeCase.timeline" :selected-id="selectedTimelineId" :developer-mode="developerMode" @select="selectTimeline" />
            <EvidenceInspector :evidence="selectedEvidence" :evidence-options="activeCase.evidence" :candidates="activeCase.candidates" :developer-mode="developerMode" @select-evidence="selectedEvidenceId = $event" />
          </div>
          <div v-else class="state-panel">该案例还没有可展示的查询证据。</div>
          <FinalDecision :decision="activeCase.finalDecision" :candidates="activeCase.candidates" :replay-status="activeCase.replayStatus" />
          <footer class="page-footer"><span>RCA Agent · {{ sourceMode }} 数据</span><span>证据可追溯 · Core 负责裁决 · 支持重放</span></footer>
        </template>
      </main>
    </div>

    <el-drawer v-model="rawDrawerOpen" title="冻结后的调查 JSON" size="52%" class="raw-drawer"><div class="drawer-hint">开发者模式仅展示安全投影，不包含 Ground Truth、密钥或隐藏思维链。</div><pre>{{ JSON.stringify(activeCase, null, 2) }}</pre></el-drawer>
  </div>
</template>
