import { useState } from 'react'
import { useSessionStore } from '../../store/sessionStore'
import { useValidation } from '../../hooks/useValidation'
import type { ValidateResponse } from '../../types/api'
import { ValidationReport } from '../validation/ValidationReport'
import { UploadPanel } from '../documents/UploadPanel'

export function Sidebar() {
  const documents = useSessionStore((s) => s.documents)
  const { mutateAsync: validate, isPending } = useValidation()
  const [report, setReport] = useState<ValidateResponse | null>(null)

  async function handleValidate(docId: string) {
    try {
      const result = await validate(docId)
      setReport(result)
    } catch {
      // error surfaced by mutation status
    }
  }

  return (
    <aside className="w-72 border-r border-gray-200 bg-gray-50 flex flex-col overflow-hidden">
      <UploadPanel />

      <div className="p-3 border-b border-gray-200">
        <h2 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
          Загруженные документы
        </h2>
      </div>

      {documents.length === 0 ? (
        <p className="text-xs text-gray-400 p-4">Нет загруженных документов</p>
      ) : (
        <ul className="flex-1 overflow-y-auto divide-y divide-gray-100">
          {documents.map((doc) => (
            <li key={doc.id} className="p-3">
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
                onClick={() => void handleValidate(doc.id)}
                disabled={isPending}
                className="mt-1.5 text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50 transition-colors"
              >
                {isPending ? 'Проверка…' : 'Проверить соответствие'}
              </button>
            </li>
          ))}
        </ul>
      )}

      {report && (
        <div className="border-t border-gray-200 p-3 bg-white">
          <ValidationReport report={report} onClose={() => setReport(null)} />
        </div>
      )}
    </aside>
  )
}
