import { BarChart2, Newspaper } from 'lucide-react'

export default function TopNav({ page, onNavigate }) {
  return (
    <nav className="top-nav">
      <span className="nav-brand">Earnings Intelligence</span>
      <button
        className={`nav-tab ${page === 'dashboard' ? 'active' : ''}`}
        onClick={() => onNavigate('dashboard')}
      >
        <BarChart2 size={14} />
        Dashboard
      </button>
      <button
        className={`nav-tab ${page === 'news' ? 'active' : ''}`}
        onClick={() => onNavigate('news')}
      >
        <Newspaper size={14} />
        News Intelligence
      </button>
    </nav>
  )
}
