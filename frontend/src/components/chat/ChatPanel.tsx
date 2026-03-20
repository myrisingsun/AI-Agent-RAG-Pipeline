import { useEffect, useRef, useState } from 'react'
import { useChat } from '../../hooks/useChat'
import { useSessionStore } from '../../store/sessionStore'
import { MessageBubble } from './MessageBubble'

export function ChatPanel() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const { messages, connected, error, sendMessage } = useChat({ sessionId })
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const isStreaming = messages.some((m) => m.streaming)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const query = input.trim()
    if (!query || isStreaming) return
    sendMessage(query)
    setInput('')
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as unknown as React.FormEvent)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Status bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-200 bg-white">
        <span
          className={`w-2 h-2 rounded-full ${
            connected ? 'bg-green-500' : 'bg-red-400'
          }`}
        />
        <span className="text-xs text-gray-500">
          {connected ? 'Подключено' : 'Нет соединения'}
        </span>
        {error && (
          <span className="ml-auto text-xs text-red-500 truncate">{error}</span>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-3">
              <span className="text-blue-600 text-xl">?</span>
            </div>
            <p className="text-gray-500 text-sm">
              Задайте вопрос по загруженным документам
            </p>
            <p className="text-gray-400 text-xs mt-1">
              Shift+Enter для переноса строки
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="px-4 py-3 border-t border-gray-200 bg-white"
      >
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Введите вопрос…"
            rows={2}
            disabled={!connected || isStreaming}
            className="flex-1 resize-none rounded-xl border border-gray-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:bg-gray-50"
          />
          <button
            type="submit"
            disabled={!connected || isStreaming || !input.trim()}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            {isStreaming ? '…' : 'Отправить'}
          </button>
        </div>
      </form>
    </div>
  )
}
