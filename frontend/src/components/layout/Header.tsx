import { useSessionStore } from '../../store/sessionStore'

export function Header() {
  const { sessionId, clearSession } = useSessionStore()

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-gray-200 bg-white">
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center">
          <span className="text-white text-xs font-bold">R</span>
        </div>
        <span className="font-semibold text-gray-900">RAG Pipeline</span>
        <span className="text-xs text-gray-400 hidden sm:block">
          Кредитная документация
        </span>
      </div>
      <div className="flex items-center gap-3">
        {sessionId && (
          <span className="text-xs text-gray-500 font-mono">
            Session: {sessionId.slice(0, 8)}…
          </span>
        )}
        <button
          onClick={clearSession}
          className="text-xs text-gray-500 hover:text-gray-800 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
        >
          Новая сессия
        </button>
      </div>
    </header>
  )
}
