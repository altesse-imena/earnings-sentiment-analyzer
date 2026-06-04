import { AlertTriangle, TrendingDown, TrendingUp, Minus } from 'lucide-react'
import { useRecommendation } from '../../hooks/useRecommendation'
import Badge from '../shared/Badge'

const ACTION_META = {
  'BUY CALLS':  { color: 'var(--positive)',  Icon: TrendingUp },
  'ACCUMULATE': { color: 'var(--positive)',  Icon: TrendingUp },
  'HOLD':       { color: 'var(--text-muted)', Icon: Minus },
  'REDUCE':     { color: 'var(--negative)',  Icon: TrendingDown },
  'BUY PUTS':   { color: 'var(--negative)',  Icon: TrendingDown },
  'AVOID':      { color: 'var(--negative)',  Icon: TrendingDown },
}

function convictionColor(c) {
  if (c === 'HIGH') return 'var(--positive)'
  if (c === 'MEDIUM') return 'var(--warning)'
  return 'var(--text-muted)'
}

export default function InvestmentRecommendation({ ticker }) {
  const { data, isLoading, isError } = useRecommendation(ticker)

  return (
    <aside className="rec-panel">
      <div className="rec-panel-title">
        <AlertTriangle size={14} color="var(--warning)" />
        Investment Recommendation
      </div>

      {isLoading && (
        <div className="rec-loading">
          <div className="loader" />
          Generating recommendation...
        </div>
      )}

      {isError && (
        <div style={{ fontSize: '0.775rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          Recommendation unavailable. Ensure FINNHUB_API_KEY and ANTHROPIC_API_KEY are set.
        </div>
      )}

      {!isLoading && !isError && data && (() => {
        const meta = ACTION_META[data.action] ?? ACTION_META['HOLD']
        const { color, Icon } = meta

        return (
          <>
            <div className="rec-action" style={{ color }}>{data.action}</div>

            <div className="rec-meta-row">
              <Badge text={`${data.conviction} CONVICTION`} color={convictionColor(data.conviction)} />
            </div>

            <div className="rec-section-label">Timeframe</div>
            <div className="rec-timeframe">{data.timeframe}</div>

            <hr className="rec-divider" />

            <div className="rec-section-label">Rationale</div>
            <div className="rec-rationale">{data.rationale}</div>

            {data.key_catalysts?.length > 0 && (
              <>
                <hr className="rec-divider" />
                <div className="rec-section-label">Key Catalysts</div>
                <ul className="rec-list">
                  {data.key_catalysts.map((c, i) => (
                    <li key={i} className="rec-list-item">{c}</li>
                  ))}
                </ul>
              </>
            )}

            {data.risk_factors?.length > 0 && (
              <>
                <hr className="rec-divider" />
                <div className="rec-section-label">Risk Factors</div>
                <ul className="rec-list">
                  {data.risk_factors.map((r, i) => (
                    <li key={i} className="rec-list-item">{r}</li>
                  ))}
                </ul>
              </>
            )}

            {data.disclaimer && (
              <div className="rec-disclaimer">{data.disclaimer}</div>
            )}
          </>
        )
      })()}
    </aside>
  )
}
