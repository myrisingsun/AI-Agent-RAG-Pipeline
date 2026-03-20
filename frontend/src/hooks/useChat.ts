import { useCallback, useEffect, useRef, useState } from 'react'
import type { Citation, WsIncomingMessage } from '../types/api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
}

interface UseChatOptions {
  sessionId?: string | null
}

export function useChat({ sessionId }: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const assistantIdRef = useRef<string | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setError('WebSocket connection failed')

    ws.onmessage = (event: MessageEvent) => {
      const msg: WsIncomingMessage = JSON.parse(event.data as string)

      if (msg.type === 'token') {
        setMessages((prev) => {
          const id = assistantIdRef.current
          if (!id) return prev
          return prev.map((m) =>
            m.id === id ? { ...m, content: m.content + msg.content } : m
          )
        })
      } else if (msg.type === 'citation') {
        setMessages((prev) => {
          const id = assistantIdRef.current
          if (!id) return prev
          return prev.map((m) =>
            m.id === id ? { ...m, citations: msg.citations } : m
          )
        })
      } else if (msg.type === 'done') {
        // Capture id synchronously BEFORE clearing the ref,
        // so the setMessages callback always sees the correct id.
        const id = assistantIdRef.current
        assistantIdRef.current = null
        if (id) {
          setMessages((prev) =>
            prev.map((m) => (m.id === id ? { ...m, streaming: false } : m))
          )
        }
      } else if (msg.type === 'error') {
        assistantIdRef.current = null
        setError(msg.message)
      }
    }

    return () => ws.close()
  }, [])

  const sendMessage = useCallback(
    (query: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setError('Not connected')
        return
      }

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: query,
      }

      const assistantId = crypto.randomUUID()
      assistantIdRef.current = assistantId
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        streaming: true,
      }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setError(null)

      // Backend WsChatMessage: { query, session_id (required), collection }
      wsRef.current.send(
        JSON.stringify({
          query,
          session_id: sessionId ?? 'anonymous',
          collection: 'current_package',
        })
      )
    },
    [sessionId]
  )

  return { messages, connected, error, sendMessage }
}
