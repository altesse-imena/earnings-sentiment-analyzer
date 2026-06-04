import { useEffect, useState } from 'react'
import TopNav from './components/layout/TopNav'
import Layout from './components/layout/Layout'
import LivePriceTicker from './components/realtime/LivePriceTicker'
import NewsPage from './pages/NewsPage'
import { useDates } from './hooks/useDashboard'
import { useTickers } from './hooks/useTickers'

export default function App() {
  const { data: tickers = [] } = useTickers()
  const [ticker, setTicker] = useState(null)
  const [date, setDate] = useState(null)
  const [page, setPage] = useState('dashboard')

  const { data: dates = [] } = useDates(ticker)

  useEffect(() => {
    if (!ticker && tickers.length > 0) {
      setTicker(tickers[0])
    }
  }, [tickers, ticker])

  useEffect(() => {
    if (dates.length > 0) {
      setDate(dates[0])
    }
  }, [dates])

  function handleTickerChange(t) {
    setTicker(t)
    setDate(null)
  }

  return (
    <div className="app-wrapper">
      <TopNav page={page} onNavigate={setPage} />
      <LivePriceTicker />
      {page === 'dashboard' ? (
        <Layout
          ticker={ticker}
          date={date}
          tickers={tickers}
          dates={dates}
          onTickerChange={handleTickerChange}
          onDateChange={setDate}
        />
      ) : (
        <NewsPage />
      )}
    </div>
  )
}
