<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ status: string; compact?: boolean }>()

const normalized = computed(() => props.status.toLowerCase().replaceAll('_', '-'))
const labels: Record<string, string> = {
  VERIFIED: '已确认',
  SUPPORTED: '有证据支持',
  SUPPORTED_HYPOTHESIS: '假设获支持',
  UNRESOLVED: '待补证',
  FAILED: '已排除',
  INCONCLUSIVE: '证据不足',
  OUT_OF_SCOPE: '超出范围',
  QUERY_FAILED: '查询失败',
  UNKNOWN: '未知',
  MISSING: '缺失',
  NOT_PROPOSED: '仅被提及',
  SUCCESS: '查询成功',
  EMPTY: '无结果',
  CAPABILITY_UNAVAILABLE: '能力不可用',
  COVERAGE_INSUFFICIENT: '覆盖不足',
  REJECTED: '查询被拒绝',
  BUDGET_EXHAUSTED: '预算耗尽',
  INVALID: '无效',
  VALID: '重放有效',
  NOT_AVAILABLE: '暂不可用',
  REPLAY_INVALID: 'Replay 无效',
  RETRYABLE: '可重试恢复',
  RESUMABLE: '可继续',
  FROZEN: 'Answer 已冻结',
  CORRUPTED: '记录损坏',
  TERMINAL: '已终止',
}
const label = computed(() => labels[props.status] ?? props.status.replaceAll('_', ' '))
</script>

<template>
  <span class="status-badge" :class="[`status-${normalized}`, { compact }]">
    <span class="status-dot"></span>{{ label }}
  </span>
</template>
