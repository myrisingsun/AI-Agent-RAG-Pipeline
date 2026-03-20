import type { Citation } from '../../types/api'

interface CitationCardProps {
  citations: Citation[]
}

export function CitationCard({ citations }: CitationCardProps) {
  if (citations.length === 0) return null

  return (
    <div className="mt-2 space-y-1.5">
      <p className="text-xs text-gray-500 font-medium">Источники:</p>
      {citations.map((c) => (
        <div
          key={c.chunk_id}
          className="bg-blue-50 border border-blue-100 rounded-md px-3 py-2"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-blue-800 truncate">
              {c.source ?? 'Документ'}
              {c.page != null && ` · стр. ${c.page}`}
              {c.law_article && ` · ${c.law_article}`}
            </span>
            <span className="text-xs text-blue-500 shrink-0">
              {(c.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-xs text-blue-700 mt-0.5 line-clamp-2">{c.text}</p>
        </div>
      ))}
    </div>
  )
}
