import { ExternalLink, RefreshCw, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { useNews } from '../hooks/useNews'
import Badge from '../components/shared/Badge'
import InvestmentRecommendation from '../components/realtime/InvestmentRecommendation'

function sentimentColor(s) {
  if (!s) return 'var(--text-muted)'
  const l = s.toLowerCase()
  if (l === 'positive') return 'var(--positive)'
  if (l === 'negative') return 'var(--negative)'
  return 'var(--text-muted)'
}

function signalColor(s) {
  if (!s) return 'var(--text-muted)'
  const l = s.toLowerCase()
  if (l === 'positive' || l === 'bullish') return 'var(--positive)'
  if (l === 'negative' || l === 'bearish') return 'var(--negative)'
  return 'var(--text-muted)'
}

function signalClass(s) {
  if (!s) return ''
  const l = s.toLowerCase()
  if (l === 'positive' || l === 'bullish') return 'bullish'
  if (l === 'negative' || l === 'bearish') return 'bearish'
  return ''
}

function SentimentDistribution({ articles }) {
  if (!articles || articles.length === 0) return null
  const pos = articles.filter(a => a.llm_sentiment?.toLowerCase() === 'positive').length
  const neg = articles.filter(a => a.llm_sentiment?.toLowerCase() === 'negative').length
  const neu = articles.length - pos - neg
  const total = articles.length

  return (
    <div className="sentiment-dist">
      <div className="sentiment-dist-label">Article Sentiment Breakdown</div>
      <div className="sentiment-dist-bar">
        {pos > 0 && (
          <div
            className="sentiment-dist-seg"
            style={{ width: `${(pos / total) * 100}%`, background: 'var(--positive)' }}
          />
        )}
        {neu > 0 && (
          <div
            className="sentiment-dist-seg"
            style={{ width: `${(neu / total) * 100}%`, background: 'var(--text-dim)' }}
          />
        )}
        {neg > 0 && (
          <div
            className="sentiment-dist-seg"
            style={{ width: `${(neg / total) * 100}%`, background: 'var(--negative)' }}
          />
        )}
      </div>
      <div className="sentiment-dist-legend">
        <div className="sentiment-dist-legend-item">
          <span className="legend-dot" style={{ background: 'var(--positive)' }} />
          Positive ({pos})
        </div>
        <div className="sentiment-dist-legend-item">
          <span className="legend-dot" style={{ background: 'var(--text-dim)' }} />
          Neutral ({neu})
        </div>
        <div className="sentiment-dist-legend-item">
          <span className="legend-dot" style={{ background: 'var(--negative)' }} />
          Negative ({neg})
        </div>
      </div>
    </div>
  )
}

function AiSignalCard({ data }) {
  const color = signalColor(data.aggregate_signal)
  const cls = signalClass(data.aggregate_signal)

  return (
    <div className={`ai-signal-card ${cls}`}>
      <div className="ai-signal-top">
        <div>
          <div
            className="section-label"
            style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
          >
            <Sparkles size={12} color="var(--accent)" />
            AI Signal
          </div>
          <div className="ai-signal-direction" style={{ color }}>
            {data.aggregate_signal}
          </div>
        </div>
        <div className="ai-signal-meta">
          <span className="ai-signal-conf-label">Confidence</span>
          <Badge
            text={data.aggregate_confidence}
            color={color}
          />
        </div>
      </div>
      {data.llm_prediction && (
        <div className="ai-signal-prediction">{data.llm_prediction}</div>
      )}
    </div>
  )
}

function ArticleCard({ article }) {
  return (
    <a
      href={article.url}
      target="_blank"
      rel="noopener noreferrer"
      className="article-card"
    >
      <div className="article-card-body">
        <div className="article-card-meta">
          <span className="article-card-source">{article.source}</span>
          {article.published_at && (
            <>
              <span className="article-card-dot" />
              <span>{article.published_at}</span>
            </>
          )}
        </div>
        <div className="article-card-headline">{article.headline}</div>
        {article.summary && (
          <div className="article-card-summary">{article.summary}</div>
        )}
        {article.llm_reasoning && (
          <div className="article-card-reasoning">"{article.llm_reasoning}"</div>
        )}
      </div>
      <div className="article-card-right">
        <Badge
          text={article.llm_sentiment}
          color={sentimentColor(article.llm_sentiment)}
        />
        <span className="article-card-conf">{article.llm_confidence}</span>
        <ExternalLink size={13} color="var(--text-dim)" style={{ marginTop: '0.25rem' }} />
      </div>
    </a>
  )
}

const NEWS_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']

export default function NewsPage() {
  const [activeTicker, setActiveTicker] = useState(NEWS_TICKERS[0])

  const ticker = activeTicker
  const { data, isLoading, isError, refetch, isFetching } = useNews(ticker)

  return (
    <div className="news-page">
      <div className="news-page-header" style={{ maxWidth: '1200px', margin: '0 auto 1.25rem' }}>
        <div className="news-page-title">Newsfeed Prediction</div>
        <div className="news-page-sub">
          Live news feed analyzed by Claude — sentiment scored per article with an aggregate directional signal
        </div>
      </div>

      <div className="ticker-pills" style={{ maxWidth: '1200px', margin: '0 auto 1.25rem' }}>
        {NEWS_TICKERS.map((t) => (
          <button
            key={t}
            className={`ticker-pill ${ticker === t ? 'active' : ''}`}
            onClick={() => setActiveTicker(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="news-page-columns" style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div>
          {isLoading && (
            <div className="news-loading">
              <div className="loader" />
              Analyzing news with Claude...
            </div>
          )}

          {isError && (
            <div className="news-error">
              News unavailable — check that <strong>FINNHUB_API_KEY</strong> and{' '}
              <strong>ANTHROPIC_API_KEY</strong> are set in <code>.env</code> and restart the backend.
            </div>
          )}

          {!isLoading && !isError && data && (
            <>
              <AiSignalCard data={data} />

              {data.articles.length > 0 && (
                <SentimentDistribution articles={data.articles} />
              )}

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <span className="section-label" style={{ margin: 0 }}>
                  {data.articles.length} article{data.articles.length !== 1 ? 's' : ''} · last 7 days
                </span>
                <button
                  className="icon-btn"
                  onClick={() => refetch()}
                  disabled={isFetching}
                  style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}
                >
                  <RefreshCw
                    size={13}
                    style={{ animation: isFetching ? 'spin 0.7s linear infinite' : 'none' }}
                  />
                  Refresh
                </button>
              </div>

              {data.articles.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', padding: '1rem 0' }}>
                  No recent articles found for {ticker}.
                </div>
              ) : (
                data.articles.map((article, i) => (
                  <ArticleCard key={i} article={article} />
                ))
              )}
            </>
          )}
        </div>

        <InvestmentRecommendation ticker={ticker} />
      </div>
    </div>
  )
}
