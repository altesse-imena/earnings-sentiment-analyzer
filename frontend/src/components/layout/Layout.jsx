import { useState } from 'react'
import Dashboard from '../dashboard/Dashboard'
import EmptyState from '../shared/EmptyState'
import Header from './Header'
import Sidebar from './Sidebar'

export default function Layout({ ticker, date, tickers, dates, onTickerChange, onDateChange }) {
  const [drawerOpen, setDrawerOpen] = useState(false)

  function handleTickerChange(t) {
    onTickerChange(t)
    setDrawerOpen(false)
  }

  function handleDateChange(d) {
    onDateChange(d)
    setDrawerOpen(false)
  }

  return (
    <div className="layout">
      {/* Backdrop — mobile only */}
      {drawerOpen && (
        <div className="drawer-backdrop" onClick={() => setDrawerOpen(false)} />
      )}

      <Sidebar
        ticker={ticker}
        date={date}
        tickers={tickers}
        dates={dates}
        onTickerChange={handleTickerChange}
        onDateChange={handleDateChange}
        drawerOpen={drawerOpen}
        onClose={() => setDrawerOpen(false)}
      />

      <main className="layout-main">
        <Header
          ticker={ticker}
          date={date}
          onOpenDrawer={() => setDrawerOpen(true)}
        />
        {ticker && date ? (
          <Dashboard ticker={ticker} date={date} />
        ) : (
          <EmptyState message="Select a ticker and date to view analysis." />
        )}
      </main>
    </div>
  )
}
