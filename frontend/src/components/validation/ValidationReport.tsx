import { useState } from 'react'
import type { Citation, ValidateResponse, ValidationIssue } from '../../types/api'

// ─── Status config ────────────────────────────────────────────────────────────

const STATUS_CONFIG: Record<
  ValidateResponse['status'],
  { label: string; icon: string; banner: string; text: string }
> = {
  compliant: {
    label: 'Соответствует требованиям',
    icon: '✓',
    banner: 'bg-green-50 border-green-200',
    text: 'text-green-700',
  },
  non_compliant: {
    label: 'Критические нарушения',
    icon: '✗',
    banner: 'bg-red-50 border-red-200',
    text: 'text-red-700',
  },
  review_required: {
    label: 'Требуется проверка',
    icon: '⚠',
    banner: 'bg-yellow-50 border-yellow-200',
    text: 'text-yellow-700',
  },
}

// ─── Severity config ──────────────────────────────────────────────────────────

const SEVERITY_CONFIG: Record<
  ValidationIssue['severity'],
  { label: string; icon: string; card: string; badge: string }
> = {
  critical: {
    label: 'Критично',
    icon: '🔴',
    card: 'bg-red-50 border-red-200',
    badge: 'bg-red-100 text-red-700',
  },
  warning: {
    label: 'Предупреждение',
    icon: '🟡',
    card: 'bg-yellow-50 border-yellow-200',
    badge: 'bg-yellow-100 text-yellow-700',
  },
  info: {
    label: 'Инфо',
    icon: '🔵',
    card: 'bg-blue-50 border-blue-200',
    badge: 'bg-blue-100 text-blue-700',
  },
}

// ─── CitationBlock ────────────────────────────────────────────────────────────

function CitationBlock({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false)
  const pct = citation.score ? Math.round(citation.score * 100) : null

  return (
    <div className="mt-1.5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs text-blue-600 hover:text-blue-800 transition-colors"
      >
        {open ? '▲' : '▼'} Источник{pct != null ? ` · ${pct}%` : ''}
        {citation.law_article ? ` · ${citation.law_article}` : ''}
      </button>
      {open && (
        <blockquote className="mt-1 text-xs text-gray-600 border-l-2 border-blue-200 pl-2 italic leading-relaxed">
          {citation.text}
        </blockquote>
      )}
    </div>
  )
}

// ─── IssueCard ────────────────────────────────────────────────────────────────

function IssueCard({ issue }: { issue: ValidationIssue }) {
  const cfg = SEVERITY_CONFIG[issue.severity]
  return (
    <div className={`rounded-md border px-2.5 py-2 ${cfg.card}`}>
      <div className="flex items-start gap-1.5">
        <span className="text-xs mt-0.5 shrink-0">{cfg.icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-1.5 mb-1">
            <span className={`text-xs font-semibold rounded px-1.5 py-0.5 ${cfg.badge}`}>
              {cfg.label}
            </span>
            {issue.law_article && (
              <span className="text-xs bg-white/70 border border-gray-200 rounded px-1.5 py-0.5 font-mono text-gray-600">
                {issue.law_article}
              </span>
            )}
          </div>
          <p className="text-xs leading-relaxed text-gray-700 whitespace-pre-line">
            {issue.description}
          </p>
          {issue.citation && <CitationBlock citation={issue.citation} />}
        </div>
      </div>
    </div>
  )
}

// ─── SeveritySummary ──────────────────────────────────────────────────────────

function SeveritySummary({ issues }: { issues: ValidationIssue[] }) {
  const counts = {
    critical: issues.filter((i) => i.severity === 'critical').length,
    warning: issues.filter((i) => i.severity === 'warning').length,
    info: issues.filter((i) => i.severity === 'info').length,
  }
  const items = (
    [
      ['critical', 'красных', counts.critical],
      ['warning', 'жёлтых', counts.warning],
      ['info', 'синих', counts.info],
    ] as const
  ).filter(([, , n]) => n > 0)

  if (items.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(([sev, , n]) => (
        <span
          key={sev}
          className={`text-xs rounded-full px-2 py-0.5 font-medium ${SEVERITY_CONFIG[sev].badge}`}
        >
          {n} {SEVERITY_CONFIG[sev].label.toLowerCase()}
        </span>
      ))}
    </div>
  )
}

// ─── ValidationReport ─────────────────────────────────────────────────────────

interface ValidationReportProps {
  report: ValidateResponse
  onClose: () => void
}

export function ValidationReport({ report, onClose }: ValidationReportProps) {
  const statusCfg = STATUS_CONFIG[report.status]

  return (
    <div className="text-xs space-y-3">
      {/* Header */}
      <div className={`flex items-start justify-between gap-2 rounded-lg border p-2.5 ${statusCfg.banner}`}>
        <div className="flex items-center gap-1.5">
          <span className={`font-bold text-sm ${statusCfg.text}`}>{statusCfg.icon}</span>
          <span className={`font-semibold ${statusCfg.text}`}>{statusCfg.label}</span>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 transition-colors shrink-0"
          aria-label="Закрыть"
        >
          ✕
        </button>
      </div>

      {/* Summary */}
      {report.summary && (
        <p className="text-gray-600 leading-relaxed">{report.summary}</p>
      )}

      {/* Severity summary */}
      {report.issues.length > 0 && <SeveritySummary issues={report.issues} />}

      {/* Checked articles */}
      {report.checked_articles.length > 0 && (
        <div>
          <p className="text-gray-500 font-medium mb-1">Проверено статей:</p>
          <div className="flex flex-wrap gap-1">
            {report.checked_articles.map((art) => (
              <span
                key={art}
                className="bg-gray-100 border border-gray-200 rounded px-1.5 py-0.5 font-mono text-gray-600"
              >
                {art}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Issues */}
      {report.issues.length === 0 ? (
        <p className="text-gray-500 text-center py-2">Замечаний не выявлено.</p>
      ) : (
        <div className="space-y-2">
          {report.issues.map((issue, i) => (
            <IssueCard key={i} issue={issue} />
          ))}
        </div>
      )}
    </div>
  )
}
