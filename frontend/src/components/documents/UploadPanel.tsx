import { useRef, useState } from 'react'
import { useUploadDocument } from '../../hooks/useDocuments'
import { useSessionStore } from '../../store/sessionStore'

const COLLECTIONS = [
  { value: 'current_package', label: 'Текущий пакет' },
  { value: 'normative_base', label: 'Нормативная база' },
  { value: 'deal_precedents', label: 'Прецеденты сделок' },
  { value: 'reference_templates', label: 'Шаблоны' },
]

export function UploadPanel() {
  const [collection, setCollection] = useState('current_package')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { mutateAsync: upload, isPending, error } = useUploadDocument()
  const setSessionId = useSessionStore((s) => s.setSessionId)
  const sessionId = useSessionStore((s) => s.sessionId)

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return
    if (!sessionId) {
      setSessionId(crypto.randomUUID())
    }
    for (const file of Array.from(files)) {
      await upload({ file })
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    void handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="p-4 border-b border-gray-200 bg-white">
      <h2 className="text-sm font-semibold text-gray-700 mb-3">Загрузка документа</h2>

      <select
        value={collection}
        onChange={(e) => setCollection(e.target.value)}
        className="w-full mb-3 text-sm border border-gray-200 rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {COLLECTIONS.map((c) => (
          <option key={c.value} value={c.value}>
            {c.label}
          </option>
        ))}
      </select>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          className="hidden"
          onChange={(e) => void handleFiles(e.target.files)}
        />
        {isPending ? (
          <p className="text-xs text-blue-600">Загрузка…</p>
        ) : (
          <>
            <p className="text-xs text-gray-500">Перетащите файл или нажмите</p>
            <p className="text-xs text-gray-400 mt-0.5">PDF, DOCX, TXT · до 50 МБ</p>
          </>
        )}
      </div>

      {error instanceof Error && (
        <p className="mt-2 text-xs text-red-500">{error.message}</p>
      )}
    </div>
  )
}
