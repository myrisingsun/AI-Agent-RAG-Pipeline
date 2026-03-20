import { useMutation } from '@tanstack/react-query'
import type { ValidateResponse } from '../types/api'
import { useSessionStore } from '../store/sessionStore'

async function validateDocument(
  documentId: string,
  sessionId: string
): Promise<ValidateResponse> {
  const res = await fetch('/api/v1/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, document_id: documentId }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Validation failed (${res.status})`)
  }
  return res.json() as Promise<ValidateResponse>
}

export function useValidation() {
  const sessionId = useSessionStore((s) => s.sessionId)
  return useMutation({
    mutationFn: (documentId: string) => {
      if (!sessionId) throw new Error('Нет активной сессии')
      return validateDocument(documentId, sessionId)
    },
  })
}
