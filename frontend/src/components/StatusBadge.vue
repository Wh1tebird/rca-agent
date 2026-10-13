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
  NOT_PROPOSED: '仅被提及',
  SUCCESS: '查询成功',
  MISSING: '数据缺失',
  INVALID: '无效',
  VALID: '重放有效',
  NOT_AVAILABLE: '暂不可用',
}
const label = computed(() => labels[props.status] ?? props.status.replaceAll('_', ' '))
</script>

<template>
  <span class="status-badge" :class="[`status-${normalized}`, { compact }]">
    <span class="status-dot"></span>{{ label }}
  </span>
</template>
