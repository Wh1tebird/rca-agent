import { mockRcaCase } from '@/mock/rcaCase'
import type {
  AuditRecord,
  Candidate,
  Evidence,
  RecoveryMetadata,
  RcaCase,
  ReplayMetadata,
  TimelineStep,
} from '@/types/rca'

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
  recoveryStatus: string
  episodeId?: string
  answerDigest?: string
  stopReason?: string
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
  schema_version: number
  run_id: string
  status: string
  replay_status: string
  recovery_status: string
  episode_id: string | null
  answer_digest: string | null
  stop_reason: string | null
  started_at: string
  final_status: string | null
  query_count: number
  error_code?: string | null
}

type ApiRunView = {
  run: ApiRun & { case_id: string; duration_ms: number | null }
  schema_version: number
  episode_id: string | null
  answer_digest: string | null
  replay: {
    schema_version: number; status: string; valid: boolean; event_count: number
    head_digest: string | null; answer_digest: string | null; error_code: string | null
  }
  recovery: {
    schema_version: number; status: string; episode_id: string | null; event_count: number
    next_sequence: number; head_digest: string | null; budget_remaining: number | null
    answer_digest: string | null; stop_reason: string | null
    pending_operations: Array<{
      operation_id: string; kind: 'query' | 'adapter_call' | 'handoff'
      disposition: 'retryable' | 'terminate'; source_event_id: string
    }>
    replay_valid: boolean; failure_category: string | null; failure_kind: string | null
  }
  audit: Array<{
    episode_id: string; event_id: string; sequence: number; previous_digest: string
    capability: string | null; provider: string | null; query_ref: string | null
    entity_scope_digest: string | null; time_range: Record<string, unknown> | null
    cost_units: number | null; adapter_ref: string | null; core_decision: string | null
    stop_reason: string | null; handoff: string | null; answer_artifact_digest: string | null
    replay_result: string | null; failure_category: string | null
  }>
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
    data_status: string; observation: string; observation_note: string; admission_context: string | null
    interpretation: string; candidate_ids: string[]; query_ref: string | null
    observation_ref: string | null; capability_ref: string | null; points: Evidence['points']; raw: Record<string, unknown>
  }>
  final_decision: {
    status: string; explanation: string; supporting: string[]; missing: string[]
    alternatives: string[]; why_not_verified: string; authoritative: boolean; answer_frozen: boolean
    answer_digest: string | null; replay_status: string
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
      recoveryStatus: mockRcaCase.recoveryStatus,
      episodeId: mockRcaCase.episodeId, answerDigest: mockRcaCase.answerDigest,
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
      runId: item.run_id, status: item.status, replayStatus: item.replay_status, recoveryStatus: item.recovery_status,
      episodeId: item.episode_id ?? undefined, answerDigest: item.answer_digest ?? undefined,
      stopReason: item.stop_reason ?? undefined,
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
      queryCount: 0, duration: '—', schemaVersion: 1, replayStatus: 'NOT_AVAILABLE', recoveryStatus: 'NOT_AVAILABLE',
      candidates: [], timeline: [], evidence: [], replay: emptyReplayMetadata(), recovery: emptyRecoveryMetadata(), audit: [],
      finalDecision: {
        status: 'NOT_RUN', explanation: '该 RCA100 Case 尚无已保存的调查记录。', supporting: [], missing: [], alternatives: [],
        whyNotVerified: '尚未执行并冻结 AnswerArtifact，因此不存在可由 Replay 验证的结论。',
        authoritative: false, answerFrozen: false, replayStatus: 'NOT_AVAILABLE',
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
      runId: item.run_id, status: item.status, replayStatus: item.replay_status, recoveryStatus: item.recovery_status,
      episodeId: item.episode_id ?? undefined, answerDigest: item.answer_digest ?? undefined,
      stopReason: item.stop_reason ?? undefined,
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
    severity: 'UNKNOWN', status: decision?.status ?? fallbackDecisionStatus(value),
    schemaVersion: value.schema_version, episodeId: value.episode_id ?? undefined,
    answerDigest: value.answer_digest ?? undefined,
    recoveryStatus: toRecoveryStatus(value.recovery.status),
    stopReason: value.run.stop_reason ?? value.recovery.stop_reason ?? undefined,
    queryCount: value.run.query_count, duration: formatDuration(value.run.duration_ms),
    replayStatus: toReplayStatus(value.run.replay_status),
    replay: toReplayMetadata(value.replay), recovery: toRecoveryMetadata(value.recovery),
    audit: value.audit.map(toAuditRecord),
    candidates: value.candidates.map((item) => ({
      id: item.id, name: item.name, mechanism: item.mechanism, status: toCandidateStatus(item.status),
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
      queryStatus: toQueryStatus(item.query_status), dataStatus: item.data_status, observation: item.observation,
      observationNote: item.observation_note, admissionContext: item.admission_context ?? undefined,
      interpretation: item.interpretation, candidateIds: item.candidate_ids,
      query: item.query_ref ?? '—', queryRef: item.query_ref ?? '—',
      observationRef: item.observation_ref ?? '—', capabilityRef: item.capability_ref ?? '—',
      points: item.points && item.points.length > 0 ? item.points : undefined, raw: item.raw,
    })),
    finalDecision: decision ? {
      status: decision.status, explanation: decision.explanation, supporting: decision.supporting,
      missing: decision.missing, alternatives: decision.alternatives, whyNotVerified: decision.why_not_verified,
      authoritative: decision.authoritative, answerFrozen: decision.answer_frozen,
      answerDigest: decision.answer_digest ?? undefined, replayStatus: toReplayStatus(decision.replay_status),
    } : {
      status: fallbackDecisionStatus(value), explanation: fallbackDecisionExplanation(value),
      supporting: [], missing: [], alternatives: [], whyNotVerified: fallbackDecisionExplanation(value),
      authoritative: false, answerFrozen: false, replayStatus: toReplayStatus(value.run.replay_status),
    },
  }
}

function toReplayStatus(value: string): RcaCase['replayStatus'] {
  if (value === 'valid') return 'VALID'
  if (value === 'invalid') return 'INVALID'
  return 'NOT_AVAILABLE'
}

function toQueryStatus(value: string): Evidence['queryStatus'] {
  const normalized = value.toUpperCase() as Evidence['queryStatus']
  const known: Evidence['queryStatus'][] = [
    'SUCCESS', 'EMPTY', 'MISSING', 'UNKNOWN', 'QUERY_FAILED', 'CAPABILITY_UNAVAILABLE',
    'COVERAGE_INSUFFICIENT', 'REJECTED', 'BUDGET_EXHAUSTED', 'FAILED',
  ]
  return known.includes(normalized) ? normalized : 'FAILED'
}

function toEvidenceKind(value: string): Evidence['kind'] {
  return value === 'metric' || value === 'trace' || value === 'log' ? value : 'query'
}

function toEvidenceRole(value: string): Evidence['role'] {
  return value === 'observation' || value === 'admission' || value === 'support'
    || value === 'refutation' || value === 'alternative' ? value : 'context'
}

function toCandidateStatus(value: string): Candidate['status'] {
  const normalized = value.toUpperCase()
  if (normalized === 'CONFIRMED_ROOT_CAUSE' || normalized === 'DISCOVERED_VERIFIED') return 'VERIFIED'
  if (normalized === 'SUPPORTED_HYPOTHESIS' || normalized === 'MULTIPLE_SUPPORTED') return 'SUPPORTED'
  if (normalized === 'OUT_OF_SCOPE' || normalized === 'QUERY_FAILED' || normalized === 'UNKNOWN' || normalized === 'MISSING') return normalized
  if (normalized === 'INCONCLUSIVE' || normalized === 'FAILED' || normalized === 'SUPPORTED' || normalized === 'VERIFIED' || normalized === 'UNRESOLVED' || normalized === 'NOT_PROPOSED') return normalized
  return 'UNRESOLVED'
}

function toRecoveryStatus(value: string): RcaCase['recoveryStatus'] {
  const normalized = value.toUpperCase()
  return ['RESUMABLE', 'RETRYABLE', 'TERMINAL', 'FROZEN', 'CORRUPTED'].includes(normalized)
    ? normalized as RcaCase['recoveryStatus'] : 'NOT_AVAILABLE'
}

function toReplayMetadata(value: ApiRunView['replay']): ReplayMetadata {
  return {
    schemaVersion: value.schema_version, status: toReplayStatus(value.status), valid: value.valid,
    eventCount: value.event_count, headDigest: value.head_digest ?? undefined,
    answerDigest: value.answer_digest ?? undefined, errorCode: value.error_code ?? undefined,
  }
}

function toRecoveryMetadata(value: ApiRunView['recovery']): RecoveryMetadata {
  return {
    schemaVersion: value.schema_version, status: toRecoveryStatus(value.status), episodeId: value.episode_id ?? undefined,
    eventCount: value.event_count, nextSequence: value.next_sequence, headDigest: value.head_digest ?? undefined,
    budgetRemaining: value.budget_remaining ?? undefined, answerDigest: value.answer_digest ?? undefined,
    stopReason: value.stop_reason ?? undefined,
    pendingOperations: value.pending_operations.map((item) => ({
      operationId: item.operation_id, kind: item.kind, disposition: item.disposition, sourceEventId: item.source_event_id,
    })), replayValid: value.replay_valid, failureCategory: value.failure_category ?? undefined,
    failureKind: value.failure_kind ?? undefined,
  }
}

function toAuditRecord(value: ApiRunView['audit'][number]): AuditRecord {
  return {
    episodeId: value.episode_id, eventId: value.event_id, sequence: value.sequence, previousDigest: value.previous_digest,
    capability: value.capability ?? undefined, provider: value.provider ?? undefined, queryRef: value.query_ref ?? undefined,
    entityScopeDigest: value.entity_scope_digest ?? undefined, timeRange: value.time_range ?? undefined,
    costUnits: value.cost_units ?? undefined, adapterRef: value.adapter_ref ?? undefined,
    coreDecision: value.core_decision ?? undefined, stopReason: value.stop_reason ?? undefined,
    handoff: value.handoff ?? undefined, answerArtifactDigest: value.answer_artifact_digest ?? undefined,
    replayResult: value.replay_result ?? undefined, failureCategory: value.failure_category ?? undefined,
  }
}

function fallbackDecisionStatus(value: ApiRunView): string {
  if (value.run.replay_status === 'invalid') return 'REPLAY_INVALID'
  if (value.recovery.status === 'retryable') return 'RETRYABLE'
  if (value.recovery.status === 'corrupted') return 'CORRUPTED'
  if (value.recovery.status === 'frozen') return 'INCONCLUSIVE'
  return value.run.status.toUpperCase()
}

function fallbackDecisionExplanation(value: ApiRunView): string {
  if (value.run.replay_status === 'invalid') return '该记录的 Replay 无效，不能展示 authoritative RCA conclusion。'
  if (value.recovery.status === 'retryable') return '当前记录包含 pending operation；RETRYABLE 只是 host recovery plan。'
  if (value.recovery.status === 'corrupted') return '记录完整性校验失败，不能继续恢复或生成新的 Answer。'
  if (value.recovery.status !== 'frozen') return 'Answer 尚未 Freeze，因此当前没有 authoritative RCA conclusion。'
  return '当前 Run 没有可展示的 authoritative RCA conclusion。'
}

function emptyReplayMetadata(): ReplayMetadata {
  return { schemaVersion: 1, status: 'NOT_AVAILABLE', valid: false, eventCount: 0 }
}

function emptyRecoveryMetadata(): RecoveryMetadata {
  return { schemaVersion: 1, status: 'NOT_AVAILABLE', eventCount: 0, nextSequence: 1, pendingOperations: [], replayValid: false }
}

function formatWindow(start: string, end: string): string {
  return `${start} — ${end}`
}

function formatDuration(value: number | null): string {
  if (value === null) return '—'
  return `${(value / 1000).toFixed(1)}s`
}
