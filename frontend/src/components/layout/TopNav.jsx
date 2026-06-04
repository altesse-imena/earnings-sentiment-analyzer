import { BarChart2, Menu, Newspaper, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', Icon: BarChart2 },
  { id: 'news', label: 'Newsfeed Prediction', Icon: Newspaper },
]

export default function TopNav({ page, onNavigate }) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef(null)

  function handleNavigate(id) {
    onNavigate(id)
    setOpen(false)
  }

  // Close on outside click
  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    if (open) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  return (
    <nav className="top-nav" ref={menuRef}>
      <span className="nav-brand">Market Sentiment Analysis Assistant</span>

      {/* Desktop tabs */}
      <div className="nav-tabs-desktop">
        {NAV_ITEMS.map(({ id, label, Icon }) => (
          <button
            key={id}
            className={`nav-tab ${page === id ? 'active' : ''}`}
            onClick={() => handleNavigate(id)}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* Mobile hamburger */}
      <button
        className="nav-hamburger"
        onClick={() => setOpen((v) => !v)}
        aria-label="Toggle menu"
      >
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Mobile dropdown */}
      {open && (
        <div className="nav-mobile-menu">
          {NAV_ITEMS.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`nav-mobile-item ${page === id ? 'active' : ''}`}
              onClick={() => handleNavigate(id)}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>
      )}
    </nav>
  )
}
