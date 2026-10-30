export type CandidateStatus =
  | 'VERIFIED'
  | 'SUPPORTED'
  | 'UNRESOLVED'
  | 'FAILED'
  | 'INCONCLUSIVE'
  | 'OUT_OF_SCOPE'
  | 'QUERY_FAILED'
  | 'UNKNOWN'
  | 'MISSING'
  | 'NOT_PROPOSED'

export type EvidenceKind = 'metric' | 'trace' | 'query' | 'log'
export type EvidenceRole = 'observation' | 'admission' | 'support' | 'refutation' | 'alternative' | 'context'
export type QueryStatus =
  | 'SUCCESS'
  | 'EMPTY'
  | 'MISSING'
  | 'UNKNOWN'
  | 'QUERY_FAILED'
  | 'CAPABILITY_UNAVAILABLE'
  | 'COVERAGE_INSUFFICIENT'
  | 'REJECTED'
  | 'BUDGET_EXHAUSTED'
  | 'FAILED'
export type ReplayStatus = 'VALID' | 'INVALID' | 'NOT_AVAILABLE'
export type RecoveryStatus = 'RESUMABLE' | 'RETRYABLE' | 'TERMINAL' | 'FROZEN' | 'CORRUPTED' | 'NOT_AVAILABLE'

export interface Candidate {
  id: string
  name: string
  mechanism: string
  status: CandidateStatus
  mentioned: boolean
  formalized: boolean
  compiled: boolean
  mechanismId?: string
  summary: string
  evidenceIds: string[]
}

export interface MetricPoint {
  time: string
  value: number
  baseline: number
}

export interface Evidence {
  id: string
  title: string
  kind: EvidenceKind
  role: EvidenceRole
  value: string
  baseline?: string
  delta?: string
  unit?: string
  source: string
  timeRange: string
  queryStatus: QueryStatus
  dataStatus: string
  observation: string
  observationNote: string
  admissionContext?: string
  interpretation: string
  candidateIds: string[]
  query: string
  queryRef: string
  observationRef: string
  capabilityRef: string
  points?: MetricPoint[]
  raw: Record<string, unknown>
}

export interface TimelineStep {
  id: string
  sequence: number
  time: string
  actor: 'SYSTEM' | 'AI' | 'RUNNER' | 'CORE'
  title: string
  summary: string
  status: 'neutral' | 'running' | 'success' | 'warning' | 'failed'
  reasoning?: string
  query?: string
  observation?: string
  decision?: string
  gap?: string
  evidenceId?: string
  candidateIds?: string[]
  developer: Record<string, unknown>
}

export interface RcaCase {
  id: string
  caseId?: string
  title: string
  service: string
  operation: string
  occurredAt: string
  window: string
  severity: 'SEV-1' | 'SEV-2' | 'SEV-3' | 'UNKNOWN'
  status: string
  queryCount: number
  duration: string
  schemaVersion: number
  episodeId?: string
  answerDigest?: string
  replayStatus: ReplayStatus
  recoveryStatus: RecoveryStatus
  stopReason?: string
  replay: ReplayMetadata
  recovery: RecoveryMetadata
  audit: AuditRecord[]
  candidates: Candidate[]
  timeline: TimelineStep[]
  evidence: Evidence[]
  finalDecision: {
    status: string
    explanation: string
    supporting: string[]
    missing: string[]
    alternatives: string[]
    whyNotVerified: string
    authoritative: boolean
    answerFrozen: boolean
    answerDigest?: string
    replayStatus: ReplayStatus
  }
}

export interface ReplayMetadata {
  schemaVersion: number
  status: ReplayStatus
  valid: boolean
  eventCount: number
  headDigest?: string
  answerDigest?: string
  errorCode?: string
}

export interface PendingOperation {
  operationId: string
  kind: 'query' | 'adapter_call' | 'handoff'
  disposition: 'retryable' | 'terminate'
  sourceEventId: string
}

export interface RecoveryMetadata {
  schemaVersion: number
  status: RecoveryStatus
  episodeId?: string
  eventCount: number
  nextSequence: number
  headDigest?: string
  budgetRemaining?: number
  answerDigest?: string
  stopReason?: string
  pendingOperations: PendingOperation[]
  replayValid: boolean
  failureCategory?: string
  failureKind?: string
}

export interface AuditRecord {
  episodeId: string
  eventId: string
  sequence: number
  previousDigest: string
  capability?: string
  provider?: string
  queryRef?: string
  entityScopeDigest?: string
  timeRange?: Record<string, unknown>
  costUnits?: number
  adapterRef?: string
  coreDecision?: string
  stopReason?: string
  handoff?: string
  answerArtifactDigest?: string
  replayResult?: string
  failureCategory?: string
}
