<script setup lang="ts">
import { computed, ref } from 'vue'
import { ArrowDown, ArrowUp, Cpu, DataLine, Search, Setting } from '@element-plus/icons-vue'

import StatusBadge from './StatusBadge.vue'
import type { TimelineStep } from '@/types/rca'

const props = defineProps<{
  steps: TimelineStep[]
  selectedId: string
  developerMode: boolean
}>()
const emit = defineEmits<{ select: [step: TimelineStep] }>()
const expanded = ref(new Set<string>())

const actorIcon = computed(() => ({
  SYSTEM: Setting,
  AI: Cpu,
  RUNNER: Search,
  CORE: DataLine,
}))

function toggle(step: TimelineStep) {
  const next = new Set(expanded.value)
  if (next.has(step.id)) next.delete(step.id)
  else next.add(step.id)
  expanded.value = next
  emit('select', step)
}

type Phase = { id: string; label: string; detail: string; steps: TimelineStep[] }

const phases = computed<Phase[]>(() => {
  const grouped = new Map<string, Phase>()
  for (const step of props.steps) {
    const phase = phaseFor(step)
    const current = grouped.get(phase.id) ?? { ...phase, steps: [] }
    current.steps.push(step)
    grouped.set(phase.id, current)
  }
  return [...grouped.values()]
})

function phaseFor(step: TimelineStep): Omit<Phase, 'steps'> {
  const title = step.title.toLowerCase()
  if (step.actor === 'SYSTEM') return { id: 'incident', label: '01 · 告警进入', detail: '确定故障与调查范围' }
  if (title.includes('inquiry')) return { id: 'inquiry', label: '02 · 初步调查', detail: '提出候选并选择查询' }
  if (title.includes('discovery') || title.includes('机制') || title.includes('mechanism')) return { id: 'discovery', label: '03 · 形成机制', detail: '把方向变成可验证假设' }
  if (step.actor === 'CORE' || step.decision) return { id: 'decision', label: '05 · Core 裁决', detail: '判断证据是否足够' }
  return { id: 'verification', label: '04 · 查询取证', detail: '执行查询并补齐证据' }
}

function togglePhase(phase: Phase) {
  const next = new Set(expanded.value)
  const ids = phase.steps.map((step) => step.id)
  const allExpanded = ids.every((id) => next.has(id))
  for (const id of ids) {
    if (allExpanded) next.delete(id)
    else next.add(id)
  }
  expanded.value = next
}
</script>

<template>
  <section class="panel timeline-panel">
    <div class="panel-heading timeline-heading">
      <div>
        <span class="eyebrow">调查主链路</span>
        <h2>AI 查了什么，Core 怎么判断</h2>
      </div>
      <div class="timeline-key">
        <span><i class="actor-color ai"></i>AI 分析</span>
        <span><i class="actor-color runner"></i>查询执行</span>
        <span><i class="actor-color core"></i>Core 裁决</span>
      </div>
    </div>

    <div class="timeline-list">
      <section v-for="phase in phases" :key="phase.id" class="timeline-phase">
        <button class="phase-heading" type="button" @click="togglePhase(phase)">
          <span><strong>{{ phase.label }}</strong><small>{{ phase.detail }}</small></span><em>{{ phase.steps.length }} 个步骤</em>
        </button>
        <article v-for="step in phase.steps" :key="step.id" class="timeline-item" :class="[{ selected: selectedId === step.id }, `step-${step.status}`]">
          <div class="timeline-rail">
            <span class="sequence">{{ String(step.sequence).padStart(2, '0') }}</span>
            <span class="actor-icon" :class="step.actor.toLowerCase()">
              <el-icon><component :is="actorIcon[step.actor]" /></el-icon>
            </span>
            <i class="rail-line"></i>
          </div>

          <button class="timeline-content" type="button" @click="toggle(step)">
            <div class="timeline-meta">
              <span class="actor-label">{{ step.actor }}</span>
              <span>{{ step.time }}</span>
              <StatusBadge v-if="step.decision" :status="step.decision" compact />
            </div>
            <div class="timeline-title-row">
              <h3>{{ step.title }}</h3>
              <el-icon><ArrowUp v-if="expanded.has(step.id)" /><ArrowDown v-else /></el-icon>
            </div>
            <p>{{ step.summary }}</p>

            <div v-if="expanded.has(step.id)" class="timeline-detail">
              <div v-if="step.reasoning" class="detail-row reasoning">
                <span>分析摘要</span><p>{{ step.reasoning }}</p>
              </div>
              <div v-if="step.query" class="detail-row query">
                <span>查询内容</span><code>{{ step.query }}</code>
              </div>
              <div v-if="step.observation" class="detail-row">
                <span>查询结果</span><p>{{ step.observation }}</p>
              </div>
              <div v-if="step.gap" class="detail-row gap">
                <span>还缺什么</span><p>{{ step.gap }}</p>
              </div>
              <div v-if="developerMode" class="raw-inline">
                <span>事件原始数据</span><pre>{{ JSON.stringify(step.developer, null, 2) }}</pre>
              </div>
            </div>
          </button>
        </article>
      </section>
    </div>
  </section>
</template>
