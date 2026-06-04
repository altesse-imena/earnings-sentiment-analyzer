import { useEffect, useState } from 'react'
import Layout from './components/layout/Layout'
import LivePriceTicker from './components/realtime/LivePriceTicker'
import { useDates } from './hooks/useDashboard'
import { useTickers } from './hooks/useTickers'

export default function App() {
  const { data: tickers = [] } = useTickers()
  const [ticker, setTicker] = useState(null)
  const [date, setDate] = useState(null)

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
      <LivePriceTicker />
      <Layout
        ticker={ticker}
        date={date}
        tickers={tickers}
        dates={dates}
        onTickerChange={handleTickerChange}
        onDateChange={setDate}
      />
    </div>
  )
}
