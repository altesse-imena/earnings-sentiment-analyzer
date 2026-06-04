import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import Card from '../shared/Card'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border)',
      borderRadius: '6px',
      padding: '8px 12px',
      fontSize: '0.8rem',
      color: 'var(--text-primary)',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(p.dataKey === 'sentiment' ? 3 : 2) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function PriceChart({ prices, eventDate, overallSentiment }) {
  if (!prices || prices.length === 0) return null

  const data = prices.map((row) => ({
    date: row.date,
    price: row.close,
    sentiment: row.date === eventDate ? overallSentiment : null,
  }))

  const priceMin = Math.min(...data.map((d) => d.price))
  const priceMax = Math.max(...data.map((d) => d.price))
  const padding = (priceMax - priceMin) * 0.05

  return (
    <Card className="mb-1">
      <div className="card-header">
        <span className="card-title">5-Day Price Window</span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            yAxisId="price"
            domain={[priceMin - padding, priceMax + padding]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
          />
          <YAxis
            yAxisId="sentiment"
            orientation="right"
            domain={[-1, 1]}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}
          />
          {eventDate && (
            <ReferenceLine
              x={eventDate}
              yAxisId="price"
              stroke="var(--text-muted)"
              strokeDasharray="4 2"
              label={{ value: 'Earnings', fill: 'var(--text-muted)', fontSize: 11 }}
            />
          )}
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="price"
            name="Price"
            stroke="var(--text-primary)"
            strokeWidth={2}
            dot={{ fill: 'var(--text-primary)', r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line
            yAxisId="sentiment"
            type="monotone"
            dataKey="sentiment"
            name="Sentiment"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={{ fill: 'var(--accent)', r: 4 }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  )
}
