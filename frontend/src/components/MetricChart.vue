<script setup lang="ts">
import * as echarts from 'echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

import type { MetricPoint } from '@/types/rca'

const props = defineProps<{ points: MetricPoint[]; unit?: string }>()
const chartElement = ref<HTMLDivElement>()
let chart: echarts.ECharts | undefined
let observer: ResizeObserver | undefined

function render() {
  if (!chartElement.value) return
  chart ??= echarts.init(chartElement.value)
  chart.setOption({
    animationDuration: 500,
    grid: { left: 42, right: 14, top: 24, bottom: 28 },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#182432',
      borderWidth: 0,
      textStyle: { color: '#fff', fontSize: 11 },
    },
    xAxis: {
      type: 'category',
      data: props.points.map((point) => point.time),
      boundaryGap: false,
      axisLine: { lineStyle: { color: '#2b3d52' } },
      axisTick: { show: false },
      axisLabel: { color: '#6f8299', fontSize: 9, interval: 2 },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#1d2d40' } },
      axisLabel: { color: '#6f8299', fontSize: 9 },
    },
    series: [
      {
        name: '当前值', type: 'line', smooth: 0.28, symbol: 'none',
        data: props.points.map((point) => point.value),
        lineStyle: { color: '#4f8cff', width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(79, 140, 255, .3)' },
            { offset: 1, color: 'rgba(79, 140, 255, .02)' },
          ]),
        },
        markLine: {
          symbol: 'none', label: { formatter: '正常基线', color: '#79879a', fontSize: 10 },
          lineStyle: { color: '#60738a', type: 'dashed' },
          data: [{ yAxis: props.points[0]?.baseline ?? 0 }],
        },
      },
    ],
  })
}

onMounted(() => {
  render()
  observer = new ResizeObserver(() => chart?.resize())
  if (chartElement.value) observer.observe(chartElement.value)
})
watch(() => props.points, render, { deep: true })
onBeforeUnmount(() => {
  observer?.disconnect()
  chart?.dispose()
})
</script>

<template><div ref="chartElement" class="metric-chart" role="img" aria-label="指标趋势图"></div></template>
