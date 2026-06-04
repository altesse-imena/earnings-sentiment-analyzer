import { Play, X } from 'lucide-react'
import { useState } from 'react'
import { triggerProcess, triggerTrain } from '../../api/endpoints'
import { useModelReport } from '../../hooks/useDashboard'

export default function Sidebar({
  ticker, date, tickers, dates,
  onTickerChange, onDateChange,
  drawerOpen, onClose,
}) {
  const { data: report } = useModelReport()
  const [running, setRunning] = useState(false)

  async function handleRunPipeline() {
    setRunning(true)
    try {
      await triggerProcess()
      await triggerTrain()
    } catch {
      // silently fail — server logs have details
    } finally {
      setRunning(false)
    }
  }

  return (
    <aside className={`sidebar ${drawerOpen ? 'drawer-open' : ''}`}>
      {/* Close button — mobile drawer only */}
      <button className="sidebar-close-btn" onClick={onClose} aria-label="Close">
        <X size={18} />
      </button>

      <div>
        <div className="section-label">Ticker</div>
        <div className="select-wrap">
          <select
            className="sidebar-select"
            value={ticker || ''}
            onChange={(e) => onTickerChange(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <div className="section-label">Earnings Date</div>
        <div className="select-wrap">
          <select
            className="sidebar-select"
            value={date || ''}
            onChange={(e) => onDateChange(e.target.value)}
            disabled={!ticker || dates.length === 0}
          >
            {dates.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
      </div>

      <div>
        <button
          className="sidebar-btn"
          onClick={handleRunPipeline}
          disabled={running}
        >
          <Play size={14} />
          {running ? 'Running...' : 'Run Pipeline'}
        </button>
      </div>

      {report && (
        <div>
          <div className="section-label">Model Performance</div>
          <div>
            <div className="sidebar-metric-row">
              <span className="sidebar-metric-label">Accuracy</span>
              <span className="sidebar-metric-value">
                {report.accuracy != null ? (report.accuracy * 100).toFixed(1) + '%' : '—'}
              </span>
            </div>
            <div className="sidebar-metric-row">
              <span className="sidebar-metric-label">AUC-ROC</span>
              <span className="sidebar-metric-value">
                {report.auc_roc != null ? report.auc_roc.toFixed(3) : '—'}
              </span>
            </div>
            <div className="sidebar-metric-row">
              <span className="sidebar-metric-label">CV AUC</span>
              <span className="sidebar-metric-value">
                {report.cv_auc_mean != null ? report.cv_auc_mean.toFixed(3) : '—'}
              </span>
            </div>
            <div className="sidebar-metric-row">
              <span className="sidebar-metric-label">Samples</span>
              <span className="sidebar-metric-value">{report.n_samples ?? '—'}</span>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
