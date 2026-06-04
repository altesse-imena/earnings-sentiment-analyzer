import { SlidersHorizontal } from 'lucide-react'

export default function Header({ ticker, date, onOpenDrawer }) {
  return (
    <div className="header">
      <div className="header-top-row">
        <div className="header-tagline">
          FinBERT sentiment correlated with post-earnings price movement
        </div>
        <button className="header-filter-btn" onClick={onOpenDrawer} aria-label="Open filters">
          <SlidersHorizontal size={16} />
          {ticker && <span className="header-filter-label">{ticker}</span>}
        </button>
      </div>
      {(ticker || date) && (
        <div className="header-pills">
          {ticker && <span className="header-pill accent">{ticker}</span>}
          {date && <span className="header-pill">{date}</span>}
        </div>
      )}
      <hr className="header-divider" />
    </div>
  )
}
