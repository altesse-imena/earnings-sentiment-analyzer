import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Card from '../shared/Card'

const CustomTooltip = ({ active, payload }) => {
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
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{payload[0].payload.feature}</div>
      <div>{payload[0].value.toFixed(4)}</div>
    </div>
  )
}

export default function ShapChart({ features }) {
  if (!features || features.length === 0) return null

  const n = features.length

  return (
    <Card>
      <div className="card-header">
        <span className="card-title">Feature Importance (SHAP)</span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          layout="vertical"
          data={features}
          margin={{ top: 4, right: 16, left: 8, bottom: 0 }}
        >
          <XAxis
            type="number"
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={130}
            tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <Bar dataKey="mean_abs_shap" radius={[0, 3, 3, 0]}>
            {features.map((_, i) => (
              <Cell
                key={i}
                fill={`rgba(99, 91, 255, ${0.25 + 0.75 * (i / Math.max(n - 1, 1))})`}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}
