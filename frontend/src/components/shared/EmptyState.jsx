import { FileQuestion } from 'lucide-react'

export default function EmptyState({ message = 'No data available.' }) {
  return (
    <div className="empty-state">
      <FileQuestion size={32} color="var(--text-muted)" />
      <p className="empty-state-text">{message}</p>
    </div>
  )
}
