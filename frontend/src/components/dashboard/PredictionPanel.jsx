import Card from '../shared/Card'

export default function PredictionPanel({ prediction }) {
  if (!prediction) {
    return (
      <Card>
        <div className="card-header">
          <span className="card-title">Direction Prediction</span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', paddingTop: '0.5rem' }}>
          Model not trained yet. Run the pipeline to generate predictions.
        </div>
      </Card>
    )
  }

  const isUp = prediction.direction === 'UP'
  const dirColor = isUp ? 'var(--positive)' : 'var(--negative)'
  const conf = prediction.confidence ?? 0

  return (
    <Card>
      <div className="card-header">
        <span className="card-title">Direction Prediction</span>
      </div>

      <div className="prediction-direction" style={{ color: dirColor }}>
        {prediction.direction}
      </div>

      <div className="prediction-actual">
        Confidence: {(conf * 100).toFixed(1)}%
      </div>

      <div className="confidence-track">
        <div
          className="confidence-fill"
          style={{
            width: `${conf * 100}%`,
            background: dirColor,
          }}
        />
      </div>

      {prediction.actual_change != null && (
        <div className="prediction-actual">
          Actual 48h change: {prediction.actual_change > 0 ? '+' : ''}{prediction.actual_change.toFixed(2)}%
        </div>
      )}

      {prediction.top_feature && (
        <div className="prediction-driver">
          Top driver: {prediction.top_feature}
        </div>
      )}
    </Card>
  )
}
