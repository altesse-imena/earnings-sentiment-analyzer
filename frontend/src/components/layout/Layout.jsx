import Dashboard from '../dashboard/Dashboard'
import NewsPanel from '../realtime/NewsPanel'
import EmptyState from '../shared/EmptyState'
import Header from './Header'
import Sidebar from './Sidebar'

export default function Layout({ ticker, date, tickers, dates, onTickerChange, onDateChange }) {
  return (
    <div className="layout">
      <Sidebar
        ticker={ticker}
        date={date}
        tickers={tickers}
        dates={dates}
        onTickerChange={onTickerChange}
        onDateChange={onDateChange}
      />
      <main className="layout-main">
        <Header ticker={ticker} date={date} />
        {ticker && date ? (
          <>
            <Dashboard ticker={ticker} date={date} />
            <NewsPanel ticker={ticker} />
          </>
        ) : (
          <EmptyState message="Select a ticker and date to view analysis." />
        )}
      </main>
    </div>
  )
}
