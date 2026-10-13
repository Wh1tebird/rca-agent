<script setup lang="ts">
import { computed } from 'vue'
import { Check, Close, Connection, DocumentChecked, FolderChecked, Search } from '@element-plus/icons-vue'

import StatusBadge from './StatusBadge.vue'
import type { Candidate } from '@/types/rca'

const props = defineProps<{
  candidates: Candidate[]
  selectedId: string
  developerMode: boolean
}>()

const emit = defineEmits<{ select: [candidate: Candidate] }>()
const formalCandidates = computed(() => props.candidates.filter((candidate) => candidate.formalized))
const exploredIdeas = computed(() => props.candidates.filter((candidate) => !candidate.formalized))
const summary = computed(() => ({
  formalized: formalCandidates.value.length,
  failed: formalCandidates.value.filter((candidate) => candidate.status === 'FAILED').length,
  unresolved: formalCandidates.value.filter((candidate) => candidate.status === 'UNRESOLVED').length,
  verified: formalCandidates.value.filter((candidate) => candidate.status === 'VERIFIED').length,
}))
</script>

<template>
  <aside class="panel candidate-panel">
    <div class="panel-heading"><div><span class="eyebrow">候选工作区</span><h2>AI 怀疑什么</h2></div><span class="count-pill">{{ candidates.length }}</span></div>
    <div class="candidate-stats"><span><strong>{{ summary.formalized }}</strong> 正式候选</span><span class="failed"><strong>{{ summary.failed }}</strong> 已排除</span><span class="unresolved"><strong>{{ summary.unresolved }}</strong> 待补证</span><span class="verified"><strong>{{ summary.verified }}</strong> 已确认</span></div>
    <div class="candidate-list">
      <section v-if="formalCandidates.length" class="candidate-group">
        <div class="candidate-group-heading"><el-icon><FolderChecked /></el-icon><span>正式候选</span><small>已经进入 Core 验证</small></div>
        <button v-for="candidate in formalCandidates" :key="candidate.id" class="candidate-card" :class="[{ selected: selectedId === candidate.id }, `candidate-${candidate.status.toLowerCase()}`]" type="button" @click="emit('select', candidate)">
          <div class="candidate-topline"><h3>{{ candidate.name }}</h3><StatusBadge :status="candidate.status" compact /></div>
          <p class="candidate-mechanism">{{ candidate.mechanism }}</p>
          <div class="pipeline-state"><span :class="{ on: candidate.mentioned }"><el-icon><Check /></el-icon>AI 提出</span><i></i><span :class="{ on: candidate.formalized }"><el-icon><DocumentChecked /></el-icon>已形式化</span><i></i><span :class="{ on: candidate.compiled }"><el-icon><Connection /></el-icon>可验证</span></div>
          <div class="candidate-summary">{{ candidate.summary }}</div>
          <div v-if="developerMode" class="developer-id"><span>candidate_id</span>{{ candidate.id }}<template v-if="candidate.mechanismId"><span>mechanism_id</span>{{ candidate.mechanismId }}</template></div>
        </button>
      </section>
      <section v-if="exploredIdeas.length" class="candidate-group explored-group">
        <div class="candidate-group-heading"><el-icon><Search /></el-icon><span>探索过的方向</span><small>AI 想到过，但未正式验证</small></div>
        <button v-for="candidate in exploredIdeas" :key="candidate.id" class="candidate-card" :class="[{ selected: selectedId === candidate.id }, `candidate-${candidate.status.toLowerCase()}`]" type="button" @click="emit('select', candidate)">
          <div class="candidate-topline"><h3>{{ candidate.name }}</h3><StatusBadge :status="candidate.status" compact /></div>
          <p class="candidate-mechanism">{{ candidate.mechanism }}</p>
          <div class="pipeline-state"><span class="on"><el-icon><Check /></el-icon>AI 提出</span><i></i><span><el-icon><Close /></el-icon>未形式化</span></div>
          <div class="candidate-summary">{{ candidate.summary }}</div>
          <div v-if="developerMode" class="developer-id"><span>hypothesis_id</span>{{ candidate.id }}</div>
        </button>
      </section>
      <div v-if="candidates.length === 0" class="candidate-empty">本次 Run 没有形成可展示的候选机制。</div>
    </div>
  </aside>
</template>
