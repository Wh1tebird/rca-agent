export type CandidateStatus =
  | 'VERIFIED'
  | 'SUPPORTED'
  | 'UNRESOLVED'
  | 'FAILED'
  | 'INCONCLUSIVE'
  | 'NOT_PROPOSED'

export type EvidenceKind = 'metric' | 'trace' | 'query' | 'log'
export type EvidenceRole = 'support' | 'refutation' | 'alternative' | 'context'

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
  queryStatus: 'SUCCESS' | 'MISSING' | 'FAILED'
  observation: string
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
  replayStatus: 'VALID' | 'INVALID' | 'NOT_AVAILABLE'
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
  }
}
