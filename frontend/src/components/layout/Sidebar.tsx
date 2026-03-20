import { useState } from 'react'
import { useValidation } from '../../hooks/useValidation'
import { useSessionStore } from '../../store/sessionStore'
import type { ValidateResponse } from '../../types/api'
import { UploadPanel } from '../documents/UploadPanel'
import { ValidationReport } from '../validation/ValidationReport'

interface ReportState {
  docId: string
  filename: string
  data: ValidateResponse
}

export function Sidebar() {
  const documents = useSessionStore((s) => s.documents)
  const { mutateAsync: validate } = useValidation()

  const [pendingDocId, setPendingDocId] = useState<string | null>(null)
  const [report, setReport] = useState<ReportState | null>(null)
  const [validateError, setValidateError] = useState<string | null>(null)

  async function handleValidate(docId: string, filename: string) {
    setPendingDocId(docId)
    setValidateError(null)
    setReport(null)
    try {
      const result = await validate(docId)
      setReport({ docId, filename, data: result })
    } catch (err) {
      setValidateError(err instanceof Error ? err.message : 'Ошибка при проверке')
    } finally {
      setPendingDocId(null)
    }
  }

  return (
    <aside className="w-72 border-r border-gray-200 bg-gray-50 flex flex-col overflow-hidden">
      <UploadPanel />

      <div className="p-3 border-b border-gray-200 shrink-0">
        <h2 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
          Загруженные документы
        </h2>
      </div>

      {/* Document list — shrinks when report is visible */}
      <div className={`overflow-y-auto ${report ? 'max-h-48 shrink-0' : 'flex-1'}`}>
        {documents.length === 0 ? (
          <p className="text-xs text-gray-400 p-4">Нет загруженных документов</p>
        ) : (
          <ul className="divide-y divide-gray-100">
            {documents.map((doc) => {
              const isPending = pendingDocId === doc.id
              const isActive = report?.docId === doc.id

              return (
                <li
                  key={doc.id}
                  className={`p-3 transition-colors ${isActive ? 'bg-blue-50' : ''}`}
                >
                  <p
                    className="text-xs font-medium text-gray-800 truncate"
                    title={doc.filename}
                  >
                    {doc.filename}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {doc.chunk_count} чанков · {doc.collection}
                  </p>
                  <button
                    onClick={() => void handleValidate(doc.id, doc.filename)}
                    disabled={pendingDocId !== null}
                    className="mt-1.5 text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50 transition-colors"
                  >
                    {isPending ? (
                      <span className="inline-flex items-center gap-1">
                        <span className="animate-spin">⟳</span> Проверка…
                      </span>
                    ) : (
                      'Проверить соответствие'
                    )}
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {/* Validation error */}
      {validateError && (
        <div className="mx-3 mb-2 shrink-0 rounded-md bg-red-50 border border-red-200 px-2.5 py-2">
          <div className="flex items-start justify-between gap-1">
            <p className="text-xs text-red-700">{validateError}</p>
            <button
              onClick={() => setValidateError(null)}
              className="text-red-400 hover:text-red-600 shrink-0"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Validation report panel */}
      {report && (
        <div className="flex-1 border-t border-gray-200 bg-white flex flex-col overflow-hidden min-h-0">
          <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 shrink-0">
            <div className="min-w-0">
              <p className="text-xs font-semibold text-gray-700">Отчёт о соответствии</p>
              <p className="text-xs text-gray-400 truncate" title={report.filename}>
                {report.filename}
              </p>
            </div>
          </div>
          <div className="overflow-y-auto flex-1 p-3">
            <ValidationReport report={report.data} onClose={() => setReport(null)} />
          </div>
        </div>
      )}
    </aside>
  )
}
