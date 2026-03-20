import { create } from 'zustand'
import type { DocumentUploadResponse } from '../types/api'

interface SessionState {
  sessionId: string | null
  documents: DocumentUploadResponse[]
  setSessionId: (id: string) => void
  addDocument: (doc: DocumentUploadResponse) => void
  clearSession: () => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  documents: [],
  setSessionId: (id) => set({ sessionId: id }),
  addDocument: (doc) => set((state) => ({ documents: [...state.documents, doc] })),
  clearSession: () => set({ sessionId: null, documents: [] }),
}))
