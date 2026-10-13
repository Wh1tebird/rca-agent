<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowLeft, ArrowRight, CircleCheck, CopyDocument, DataLine, Link, Warning } from '@element-plus/icons-vue'

import MetricChart from './MetricChart.vue'
import StatusBadge from './StatusBadge.vue'
import type { Candidate, Evidence } from '@/types/rca'

const props = defineProps<{
  evidence: Evidence
  evidenceOptions: Evidence[]
  candidates: Candidate[]
  developerMode: boolean
}>()
const emit = defineEmits<{ selectEvidence: [id: string] }>()
const developerTab = ref<'summary' | 'references' | 'raw'>('summary')

const relatedCandidates = computed(() => props.candidates.filter((candidate) =>
  props.evidence.candidateIds.includes(candidate.id),
))
const evidenceIndex = computed(() => Math.max(0, props.evidenceOptions.findIndex((item) => item.id === props.evidence.id)))

function moveEvidence(offset: number) {
  const next = props.evidenceOptions[evidenceIndex.value + offset]
  if (next) emit('selectEvidence', next.id)
}

async function copyRaw() {
  await navigator.clipboard?.writeText(JSON.stringify(props.evidence.raw, null, 2))
}

const roleLabel: Record<Evidence['role'], string> = {
  support: '支持证据',
  refutation: '反驳证据',
  alternative: '竞争证据',
  context: '背景信息',
}
</script>

<template>
  <aside class="panel evidence-panel">
    <div class="panel-heading">
      <div>
        <span class="eyebrow">证据检查器</span>
        <h2>查到了什么</h2>
      </div>
      <el-icon class="panel-icon"><DataLine /></el-icon>
    </div>

    <div class="evidence-typebar">
      <span class="kind">{{ evidence.kind.toUpperCase() }}</span>
      <span class="role" :class="evidence.role">{{ roleLabel[evidence.role] }}</span>
    </div>
    <div v-if="evidenceOptions.length > 1" class="evidence-pager">
      <button type="button" :disabled="evidenceIndex === 0" aria-label="上一条证据" @click="moveEvidence(-1)"><el-icon><ArrowLeft /></el-icon></button>
      <span>第 {{ evidenceIndex + 1 }} / {{ evidenceOptions.length }} 条证据</span>
      <button type="button" :disabled="evidenceIndex >= evidenceOptions.length - 1" aria-label="下一条证据" @click="moveEvidence(1)"><el-icon><ArrowRight /></el-icon></button>
    </div>

    <h3 class="evidence-title">{{ evidence.title }}</h3>
    <div class="metric-hero">
      <strong>{{ evidence.value }}</strong><span>{{ evidence.unit }}</span>
      <em :class="{ missing: evidence.queryStatus === 'MISSING' }">{{ evidence.delta }}</em>
    </div>
    <div v-if="evidence.baseline" class="baseline-line">正常基线 <strong>{{ evidence.baseline }}</strong></div>

    <MetricChart v-if="evidence.points" :points="evidence.points" :unit="evidence.unit" />
    <div v-else class="trace-visual" :class="evidence.queryStatus.toLowerCase()">
      <el-icon><Warning v-if="evidence.queryStatus === 'MISSING'" /><CircleCheck v-else /></el-icon>
      <div><span>{{ evidence.queryStatus }}</span><strong>{{ evidence.observation }}</strong></div>
    </div>

    <dl class="evidence-metadata">
      <div><dt>数据源</dt><dd>{{ evidence.source }}</dd></div>
      <div><dt>时间范围</dt><dd>{{ evidence.timeRange }}</dd></div>
      <div><dt>查询状态</dt><dd><StatusBadge :status="evidence.queryStatus" compact /></dd></div>
    </dl>

    <div class="evidence-note">
      <span>这条证据说明什么</span>
      <p>{{ evidence.interpretation }}</p>
    </div>

    <div class="related-candidates">
      <span>影响哪些候选</span>
      <div><span v-for="candidate in relatedCandidates" :key="candidate.id">{{ candidate.name }}</span></div>
    </div>

    <div v-if="developerMode" class="developer-evidence">
      <div class="developer-tabs"><button v-for="tab in ['summary', 'references', 'raw'] as const" :key="tab" :class="{ active: developerTab === tab }" type="button" @click="developerTab = tab">{{ tab }}</button></div>
      <template v-if="developerTab === 'summary'"><p>此面板仅展示已冻结 Run 的安全投影，不包含 GT、密钥或隐藏思维链。</p></template>
      <template v-else-if="developerTab === 'references'"><div><el-icon><Link /></el-icon><span>query_ref</span><code>{{ evidence.queryRef }}</code></div><div><el-icon><Link /></el-icon><span>observation_ref</span><code>{{ evidence.observationRef }}</code></div><div><el-icon><Link /></el-icon><span>capability_ref</span><code>{{ evidence.capabilityRef }}</code></div></template>
      <template v-else><button class="copy-raw" type="button" @click="copyRaw"><el-icon><CopyDocument /></el-icon>复制 JSON</button><pre>{{ JSON.stringify(evidence.raw, null, 2) }}</pre></template>
    </div>
  </aside>
</template>
