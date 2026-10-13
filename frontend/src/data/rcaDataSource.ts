import { mockRcaCase } from '@/mock/rcaCase'
import type { Candidate, Evidence, RcaCase, TimelineStep } from '@/types/rca'

export interface CaseOption {
  caseId: string
  title: string
  alertEntity: string
  readiness: 'ready' | 'incomplete' | 'invalid'
  runCount: number
}

export interface RunOption {
  runId: string
  status: string
  replayStatus: string
  startedAt: string
  finalStatus: string | null
  queryCount: number
  errorCode?: string | null
}

export interface StartedRun {
  runId: string
  status: string
}

export interface RcaDataSource {
  listCases(): Promise<CaseOption[]>
  listRuns(caseId: string): Promise<RunOption[]>
  loadRun(runId: string): Promise<RcaCase>
  emptyCase(caseId: string): Promise<RcaCase>
  createRun(caseId: string): Promise<StartedRun>
  getRunStatus(runId: string): Promise<RunOption>
}

type ApiCase = {
  case_id: string
  title: string
  alert_entity: string
  alert_entity_type: string
  triggered_at: string
  window_start: string
  window_end: string
  readiness: CaseOption['readiness']
  run_count: number
}

type ApiRun = {
  run_id: string
  status: string
  replay_status: string
  started_at: string
  final_status: string | null
  query_count: number
  error_code?: string | null
}

type ApiRunView = {
  run: ApiRun & { case_id: string; duration_ms: number | null }
  case: ApiCase
  candidates: Array<{
    id: string; name: string; mechanism: string; status: Candidate['status']; mentioned: boolean
    formalized: boolean; compiled: boolean; mechanism_id: string | null; summary: string
    evidence_ids: string[]
  }>
  timeline: Array<{
    id: string; sequence: number; time: string; actor: TimelineStep['actor']; title: string
    summary: string; status: TimelineStep['status']; reasoning: string | null; query: string | null
    observation: string | null; decision: string | null; gap: string | null; evidence_id: string | null
    candidate_ids: string[]; developer: Record<string, unknown>
  }>
  evidence: Array<{
    id: string; title: string; kind: Evidence['kind']; role: Evidence['role']; value: string
    unit: string | null; source: string; time_range: string | null; query_status: string
    observation: string; interpretation: string; candidate_ids: string[]; query_ref: string | null
    observation_ref: string | null; capability_ref: string | null; points: Evidence['points']; raw: Record<string, unknown>
  }>
  final_decision: {
    status: string; explanation: string; supporting: string[]; missing: string[]
    alternatives: string[]; why_not_verified: string
  } | null
}

export class MockRcaDataSource implements RcaDataSource {
  async listCases(): Promise<CaseOption[]> {
    return [{
      caseId: 'mock-demo', title: mockRcaCase.title, alertEntity: mockRcaCase.service,
      readiness: 'ready', runCount: 1,
    }]
  }

  async listRuns(): Promise<RunOption[]> {
    return [{
      runId: 'mock-run', status: 'succeeded', replayStatus: mockRcaCase.replayStatus,
      startedAt: mockRcaCase.occurredAt, finalStatus: mockRcaCase.status, queryCount: mockRcaCase.queryCount,
    }]
  }

  async loadRun(): Promise<RcaCase> {
    return mockRcaCase
  }

  async emptyCase(): Promise<RcaCase> {
    return mockRcaCase
  }

  async createRun(): Promise<StartedRun> {
    return { runId: 'mock-run', status: 'succeeded' }
  }

  async getRunStatus(): Promise<RunOption> {
    return (await this.listRuns())[0]!
  }
}

export class ApiRcaDataSource implements RcaDataSource {
  constructor(private readonly baseUrl = import.meta.env.VITE_RCA_API_BASE ?? '/api/v1') {}

  async listCases(): Promise<CaseOption[]> {
    const response = await this.fetchJson<{ items: ApiCase[] }>('/cases')
    return response.items.map((item) => ({
      caseId: item.case_id, title: item.title, alertEntity: item.alert_entity,
      readiness: item.readiness, runCount: item.run_count,
    }))
  }

  async listRuns(caseId: string): Promise<RunOption[]> {
    const response = await this.fetchJson<{ items: ApiRun[] }>(`/cases/${encodeURIComponent(caseId)}/runs`)
    return response.items.map((item) => ({
      runId: item.run_id, status: item.status, replayStatus: item.replay_status,
      startedAt: item.started_at, finalStatus: item.final_status, queryCount: item.query_count,
      errorCode: item.error_code,
    }))
  }

  async loadRun(runId: string): Promise<RcaCase> {
    return toRcaCase(await this.fetchJson<ApiRunView>(`/runs/${encodeURIComponent(runId)}`))
  }

  async emptyCase(caseId: string): Promise<RcaCase> {
    const item = await this.fetchJson<ApiCase>(`/cases/${encodeURIComponent(caseId)}`)
    return {
      id: `case-${item.case_id}`, caseId: item.case_id, title: item.title,
      service: item.alert_entity, operation: item.alert_entity_type,
      occurredAt: item.triggered_at, window: formatWindow(item.window_start, item.window_end),
      severity: 'UNKNOWN', status: item.readiness === 'ready' ? 'NOT_RUN' : 'DATA_INCOMPLETE',
      queryCount: 0, duration: '—', replayStatus: 'NOT_AVAILABLE', candidates: [], timeline: [], evidence: [],
      finalDecision: {
        status: 'NOT_RUN', explanation: '该 RCA100 Case 尚无已保存的调查记录。', supporting: [], missing: [], alternatives: [],
        whyNotVerified: '尚未执行并冻结 AnswerArtifact，因此不存在可由 Replay 验证的结论。',
      },
    }
  }

  async createRun(caseId: string): Promise<StartedRun> {
    const profileId = import.meta.env.VITE_RCA_RUN_PROFILE ?? 'qwen-investigation'
    const response = await fetch(`${this.baseUrl}/cases/${encodeURIComponent(caseId)}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: profileId }),
    })
    if (!response.ok) {
      const payload = await response.json().catch(() => ({})) as { detail?: string }
      throw new Error(payload.detail ?? `无法启动调查（HTTP ${response.status}）`)
    }
    const payload = await response.json() as { run_id: string; status: string }
    return { runId: payload.run_id, status: payload.status }
  }

  async getRunStatus(runId: string): Promise<RunOption> {
    const item = await this.fetchJson<ApiRun>(`/runs/${encodeURIComponent(runId)}/status`)
    return {
      runId: item.run_id, status: item.status, replayStatus: item.replay_status,
      startedAt: item.started_at, finalStatus: item.final_status, queryCount: item.query_count,
      errorCode: item.error_code,
    }
  }

  private async fetchJson<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`)
    if (!response.ok) throw new Error(`API request failed (${response.status})`)
    return response.json() as Promise<T>
  }
}

export function createDataSource(): RcaDataSource {
  return import.meta.env.VITE_RCA_DATA_MODE === 'api' ? new ApiRcaDataSource() : new MockRcaDataSource()
}

function toRcaCase(value: ApiRunView): RcaCase {
  const decision = value.final_decision
  return {
    id: value.run.run_id, caseId: value.run.case_id, title: value.case.title,
    service: value.case.alert_entity, operation: value.case.alert_entity_type,
    occurredAt: value.case.triggered_at, window: formatWindow(value.case.window_start, value.case.window_end),
    severity: 'UNKNOWN', status: decision?.status ?? value.run.status.toUpperCase(),
    queryCount: value.run.query_count, duration: formatDuration(value.run.duration_ms),
    replayStatus: toReplayStatus(value.run.replay_status),
    candidates: value.candidates.map((item) => ({
      id: item.id, name: item.name, mechanism: item.mechanism, status: item.status,
      mentioned: item.mentioned, formalized: item.formalized, compiled: item.compiled,
      mechanismId: item.mechanism_id ?? undefined, summary: item.summary, evidenceIds: item.evidence_ids,
    })),
    timeline: value.timeline.map((item) => ({
      id: item.id, sequence: item.sequence, time: item.time, actor: item.actor, title: item.title,
      summary: item.summary, status: item.status, reasoning: item.reasoning ?? undefined,
      query: item.query ?? undefined, observation: item.observation ?? undefined,
      decision: item.decision ?? undefined, gap: item.gap ?? undefined,
      evidenceId: item.evidence_id ?? undefined, candidateIds: item.candidate_ids, developer: item.developer,
    })),
    evidence: value.evidence.map((item) => ({
      id: item.id, title: item.title, kind: toEvidenceKind(item.kind), role: toEvidenceRole(item.role), value: item.value,
      unit: item.unit ?? undefined, source: item.source, timeRange: item.time_range ?? '—',
      queryStatus: toQueryStatus(item.query_status), observation: item.observation,
      interpretation: item.interpretation, candidateIds: item.candidate_ids,
      query: item.query_ref ?? '—', queryRef: item.query_ref ?? '—',
      observationRef: item.observation_ref ?? '—', capabilityRef: item.capability_ref ?? '—',
      points: item.points && item.points.length > 0 ? item.points : undefined, raw: item.raw,
    })),
    finalDecision: decision ? {
      status: decision.status, explanation: decision.explanation, supporting: decision.supporting,
      missing: decision.missing, alternatives: decision.alternatives, whyNotVerified: decision.why_not_verified,
    } : {
      status: 'REPLAY_INVALID', explanation: '该记录的 Replay 无效，不能作为可信 RCA 结论展示。',
      supporting: [], missing: [], alternatives: [], whyNotVerified: '需要先修复或重新生成完整的 Run Transcript。',
    },
  }
}

function toReplayStatus(value: string): RcaCase['replayStatus'] {
  if (value === 'valid') return 'VALID'
  if (value === 'invalid') return 'INVALID'
  return 'NOT_AVAILABLE'
}

function toQueryStatus(value: string): Evidence['queryStatus'] {
  return value === 'SUCCESS' ? 'SUCCESS' : value === 'MISSING' ? 'MISSING' : 'FAILED'
}

function toEvidenceKind(value: string): Evidence['kind'] {
  return value === 'metric' || value === 'trace' || value === 'log' ? value : 'query'
}

function toEvidenceRole(value: string): Evidence['role'] {
  return value === 'support' || value === 'refutation' || value === 'alternative' ? value : 'context'
}

function formatWindow(start: string, end: string): string {
  return `${start} — ${end}`
}

function formatDuration(value: number | null): string {
  if (value === null) return '—'
  return `${(value / 1000).toFixed(1)}s`
}
