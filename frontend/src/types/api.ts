// API types — generated from src/schemas/api.py

export interface DocumentUploadResponse {
  id: string
  filename: string
  doc_type: string
  collection: string
  chunk_count: number
  created_at: string
}

export interface DocumentResponse {
  id: string
  filename: string
  doc_type: string
  collection: string
  chunk_count: number
  created_at: string
  metadata: Record<string, unknown>
}

export interface Citation {
  chunk_id: string
  text: string
  score: number
  source: string | null
  page: number | null
  section: string | null
  law_article: string | null
}

export interface SearchRequest {
  query: string
  collection?: string
  limit?: number
  session_id?: string
  filters?: Record<string, string>
}

export interface SearchResponse {
  answer: string
  query: string
  citations: Citation[]
  collection: string
  latency_ms: number
}

export interface ValidateRequest {
  session_id: string
  document_id?: string
}

export interface ValidationIssue {
  severity: 'critical' | 'warning' | 'info'
  description: string
  law_article: string | null
  citation: Citation | null
}

export interface ValidateResponse {
  session_id: string
  status: 'compliant' | 'non_compliant' | 'review_required'
  issues: ValidationIssue[]
  summary: string
  checked_articles: string[]
}

export interface CollectionStat {
  name: string
  point_count: number
  vector_size: number
  status: string
}

export interface CollectionsStatsResponse {
  collections: CollectionStat[]
}

// WebSocket message types
export interface WsChatMessage {
  query: string
  session_id: string
  collection?: string
}

export interface WsTokenMessage {
  type: 'token'
  content: string
}

export interface WsCitationMessage {
  type: 'citation'
  citations: Citation[]
}

export interface WsErrorMessage {
  type: 'error'
  message: string
}

export interface WsDoneMessage {
  type: 'done'
  latency_ms: number
}

export type WsIncomingMessage =
  | WsTokenMessage
  | WsCitationMessage
  | WsErrorMessage
  | WsDoneMessage
