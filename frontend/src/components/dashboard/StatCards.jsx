import { Activity, ArrowUpDown, Mic, TrendingUp } from 'lucide-react'

function sentimentColor(v) {
  if (v == null) return 'var(--text-dim)'
  if (v > 0.05) return 'var(--positive)'
  if (v < -0.05) return 'var(--negative)'
  return 'var(--text-dim)'
}

function StatCard({ icon: Icon, label, value, sub, color }) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <span className="stat-card-label">{label}</span>
        <Icon size={16} color="var(--text-muted)" />
      </div>
      <div className="stat-card-value" style={{ color }}>
        {value}
      </div>
      {sub && <div className="stat-card-sub">{sub}</div>}
    </div>
  )
}

export default function StatCards({ sentiment, prediction }) {
  const overall = sentiment?.overall_sentiment
  const ceo = sentiment?.ceo_sentiment
  const toneShift = sentiment?.tone_shift
  const conf = prediction?.confidence

  return (
    <div className="stat-cards-grid mb-1">
      <StatCard
        icon={Activity}
        label="Overall Sentiment"
        value={overall != null ? overall.toFixed(3) : '—'}
        sub="All speakers, full call"
        color={sentimentColor(overall)}
      />
      <StatCard
        icon={Mic}
        label="CEO Tone"
        value={ceo != null ? ceo.toFixed(3) : '—'}
        sub="Executive prepared remarks"
        color={sentimentColor(ceo)}
      />
      <StatCard
        icon={ArrowUpDown}
        label="Tone Shift"
        value={toneShift != null ? toneShift.toFixed(3) : '—'}
        sub="Q&A minus prepared"
        color={sentimentColor(toneShift)}
      />
      <StatCard
        icon={TrendingUp}
        label="Model Confidence"
        value={conf != null ? (conf * 100).toFixed(1) + '%' : '—'}
        sub={prediction ? `Direction: ${prediction.direction}` : 'Model not trained'}
        color={conf != null && conf > 0.6 ? 'var(--positive)' : 'var(--text-primary)'}
      />
    </div>
  )
}
