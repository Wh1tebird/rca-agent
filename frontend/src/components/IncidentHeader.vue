<script setup lang="ts">
import { Clock, Search, Timer } from '@element-plus/icons-vue'

import StatusBadge from './StatusBadge.vue'
import type { RcaCase } from '@/types/rca'

defineProps<{ incident: RcaCase }>()
</script>

<template>
  <section class="incident-shell">
    <div class="incident-primary">
      <div class="incident-title-row">
        <span class="incident-id">#{{ incident.caseId ?? incident.id }}</span>
        <StatusBadge :status="incident.status" />
      </div>
      <h1>{{ incident.title }}</h1>
      <div class="operation-line"><span class="service-chip">{{ incident.service }}</span><span class="operation-name">{{ incident.operation }}</span></div>
    </div>

    <div class="incident-facts">
      <div class="fact"><el-icon><Clock /></el-icon><div><span>告警时间</span><strong>{{ incident.occurredAt }}</strong></div></div>
      <div class="fact"><el-icon><Timer /></el-icon><div><span>调查窗口</span><strong>{{ incident.window }}</strong></div></div>
      <div class="fact"><el-icon><Search /></el-icon><div><span>调查规模</span><strong>{{ incident.queryCount }} 次查询 · {{ incident.duration }}</strong></div></div>
      <div class="fact severity-fact"><span>级别</span><strong>{{ incident.severity }}</strong></div>
    </div>
  </section>
</template>
