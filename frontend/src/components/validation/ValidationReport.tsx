import type { ValidateResponse, ValidationIssue } from '../../types/api'

interface ValidationReportProps {
  report: ValidateResponse
  onClose: () => void
}

const SEVERITY_STYLE: Record<ValidationIssue['severity'], string> = {
  critical: 'bg-red-50 border-red-200 text-red-800',
  warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  info: 'bg-blue-50 border-blue-200 text-blue-800',
}

const SEVERITY_LABEL: Record<ValidationIssue['severity'], string> = {
  critical: 'Критично',
  warning: 'Предупреждение',
  info: 'Инфо',
}

export function ValidationReport({ report, onClose }: ValidationReportProps) {
  const isCompliant = report.status === 'compliant'

  return (
    <div className="text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className={`font-semibold ${isCompliant ? 'text-green-600' : 'text-red-600'}`}>
          {isCompliant ? '✓ Соответствует' : '✗ Несоответствие'}
        </span>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 transition-colors"
        >
          ✕
        </button>
      </div>

      {report.summary && (
        <p className="text-gray-600 mb-2">{report.summary}</p>
      )}

      {report.issues.length === 0 ? (
        <p className="text-gray-500">Замечаний нет.</p>
      ) : (
        <ul className="space-y-1.5 max-h-64 overflow-y-auto">
          {report.issues.map((issue, i) => (
            <li
              key={i}
              className={`border rounded-md px-2 py-1.5 ${SEVERITY_STYLE[issue.severity]}`}
            >
              <div className="font-medium">{SEVERITY_LABEL[issue.severity]}</div>
              <p className="mt-0.5">{issue.description}</p>
              {issue.law_article && (
                <p className="mt-0.5 opacity-70">{issue.law_article}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
