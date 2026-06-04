import { TrendingDown, TrendingUp } from 'lucide-react'
import { usePriceStream } from '../../hooks/usePriceStream'

const TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

function statusDotColor(status) {
  if (status === 'connected') return 'var(--positive)'
  if (status === 'connecting') return 'var(--warning)'
  return 'var(--negative)'
}

export default function LivePriceTicker() {
  const { prices, status } = usePriceStream()

  return (
    <div className="live-ticker-bar">
      {TICKERS.map((ticker) => {
        const data = prices[ticker]
        const change = data?.change ?? 0
        const isPos = change > 0
        const isNeg = change < 0
        const changeColor = isPos
          ? 'var(--positive)'
          : isNeg
            ? 'var(--negative)'
            : 'var(--text-muted)'

        return (
          <div key={ticker} className="ticker-item">
            <span className="ticker-symbol">{ticker}</span>
            {data ? (
              <>
                <span className="ticker-price" style={{ color: 'var(--text-primary)' }}>
                  ${data.price.toFixed(2)}
                </span>
                <span className="ticker-change" style={{ color: changeColor }}>
                  {isPos ? <TrendingUp size={13} /> : isNeg ? <TrendingDown size={13} /> : null}
                  {change > 0 ? '+' : ''}{change.toFixed(2)} ({data.change_pct > 0 ? '+' : ''}{data.change_pct.toFixed(2)}%)
                </span>
              </>
            ) : (
              <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>—</span>
            )}
          </div>
        )
      })}

      <div className="ticker-status">
        <span className="status-dot" style={{ background: statusDotColor(status) }} />
        {status === 'connected' ? 'Live' : status === 'connecting' ? 'Connecting...' : 'Reconnecting...'}
      </div>
    </div>
  )
}
