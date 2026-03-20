import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { DocumentResponse, DocumentUploadResponse } from '../types/api'
import { useSessionStore } from '../store/sessionStore'

async function uploadDocument(
  file: File,
  sessionId?: string | null
): Promise<DocumentUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('doc_type', 'contract')
  if (sessionId) form.append('session_id', sessionId)

  const res = await fetch('/api/v1/documents/upload', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Upload failed (${res.status})`)
  }
  return res.json() as Promise<DocumentUploadResponse>
}

async function fetchDocument(docId: string): Promise<DocumentResponse> {
  const res = await fetch(`/api/v1/documents/${docId}`)
  if (!res.ok) {
    const body = await res.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Fetch failed (${res.status})`)
  }
  return res.json() as Promise<DocumentResponse>
}

export function useUploadDocument() {
  const addDocument = useSessionStore((s) => s.addDocument)
  const sessionId = useSessionStore((s) => s.sessionId)
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ file }: { file: File }) =>
      uploadDocument(file, sessionId),
    onSuccess: (data) => {
      addDocument(data)
      void queryClient.invalidateQueries({ queryKey: ['collections'] })
    },
  })
}

export function useDocument(docId: string | null) {
  return useQuery({
    queryKey: ['document', docId],
    queryFn: () => fetchDocument(docId!),
    enabled: !!docId,
  })
}
