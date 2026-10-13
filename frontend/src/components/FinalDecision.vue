<script setup lang="ts">
import { computed } from 'vue'
import { CircleCheck, CircleClose, InfoFilled, QuestionFilled } from '@element-plus/icons-vue'

import StatusBadge from './StatusBadge.vue'
import type { Candidate, RcaCase } from '@/types/rca'

const props = defineProps<{
  decision: RcaCase['finalDecision']
  candidates: Candidate[]
  replayStatus: RcaCase['replayStatus']
}>()
const failedCandidates = computed(() => props.candidates.filter((candidate) => candidate.formalized && candidate.status === 'FAILED'))
const unresolvedCandidates = computed(() => props.candidates.filter((candidate) => candidate.formalized && candidate.status === 'UNRESOLVED'))
</script>

<template>
  <section class="final-decision">
    <div class="decision-summary">
      <span class="eyebrow">Core 最终裁决 · 重放 {{ replayStatus }}</span>
      <div class="decision-status-row">
        <h2>最终调查结论</h2><StatusBadge :status="decision.status" />
      </div>
      <p>{{ decision.explanation }}</p>
      <div class="not-verified"><el-icon><InfoFilled /></el-icon><span><strong>为什么还不能确认根因</strong>{{ decision.whyNotVerified }}</span></div>
    </div>
    <div class="decision-columns">
      <div class="decision-list support">
        <h3><el-icon><CircleCheck /></el-icon>支持证据</h3>
        <ul><li v-for="item in decision.supporting" :key="item">{{ item }}</li></ul>
      </div>
      <div class="decision-list missing">
        <h3><el-icon><CircleClose /></el-icon>缺失证据</h3>
        <ul><li v-for="item in decision.missing" :key="item">{{ item }}</li></ul>
      </div>
      <div class="decision-list alternative">
        <h3><el-icon><QuestionFilled /></el-icon>尚未排除的原因</h3>
        <ul><li v-for="item in decision.alternatives" :key="item">{{ item }}</li></ul>
      </div>
      <div class="decision-list candidate-state">
        <h3><el-icon><CircleClose /></el-icon>候选状态</h3>
        <ul>
          <li v-for="candidate in failedCandidates" :key="candidate.id"><strong>已排除</strong> · {{ candidate.name }}</li>
          <li v-for="candidate in unresolvedCandidates" :key="candidate.id"><strong>待补证</strong> · {{ candidate.name }}</li>
          <li v-if="failedCandidates.length + unresolvedCandidates.length === 0">当前没有已形式化的候选状态变化。</li>
        </ul>
      </div>
    </div>
  </section>
</template>
