import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
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
      <div>{payload[0].value.toFixed(3)}</div>
    </div>
  )
}

export default function SectionChart({ sentiment }) {
  const data = [
    { name: 'Prepared', value: sentiment?.prepared_sentiment ?? 0 },
    { name: 'Q&A', value: sentiment?.qa_sentiment ?? 0 },
    { name: 'Shift', value: sentiment?.tone_shift ?? 0 },
  ]

  return (
    <Card>
      <div className="card-header">
        <span className="card-title">Section Sentiment</span>
      </div>
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Bar dataKey="value" radius={[3, 3, 0, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={entry.value >= 0 ? 'var(--accent)' : 'var(--negative)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}
