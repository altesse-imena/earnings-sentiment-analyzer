import { RefreshCw } from 'lucide-react'
import { useNews } from '../../hooks/useNews'
import Badge from '../shared/Badge'
import Card from '../shared/Card'
import Loader from '../shared/Loader'

function sentimentColor(s) {
  if (!s) return 'var(--text-muted)'
  const lower = s.toLowerCase()
  if (lower === 'positive') return 'var(--positive)'
  if (lower === 'negative') return 'var(--negative)'
  return 'var(--text-muted)'
}

export default function NewsPanel({ ticker }) {
  const { data, isLoading, isError, refetch, isFetching } = useNews(ticker)

  return (
    <Card className="mt-1">
      <div className="card-header">
        <span className="card-title">News Sentiment — {ticker}</span>
        <button className="icon-btn" onClick={() => refetch()} disabled={isFetching}>
          <RefreshCw size={16} style={{ animation: isFetching ? 'spin 0.7s linear infinite' : 'none' }} />
        </button>
      </div>

      {isLoading && <Loader />}

      {isError && (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          News unavailable — check FINNHUB_API_KEY in .env
        </div>
      )}

      {!isLoading && !isError && data && (
        <>
          {data.articles.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
              No recent news found for {ticker}.
            </div>
          ) : (
            data.articles.map((article, i) => (
              <a
                key={i}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="news-article-row"
              >
                <div className="news-article-body">
                  <div className="news-meta">{article.source} · {article.published_at}</div>
                  <div className="news-headline">{article.headline}</div>
                  {article.summary && (
                    <div className="news-summary">{article.summary}</div>
                  )}
                </div>
                <div className="news-badge-col">
                  <Badge
                    text={article.llm_sentiment}
                    color={sentimentColor(article.llm_sentiment)}
                  />
                  <span className="news-confidence">{article.llm_confidence}</span>
                </div>
              </a>
            ))
          )}

          {data.llm_prediction && (
            <div className="ai-signal-box">
              <div className="ai-signal-label">
                AI Signal — {data.aggregate_signal}
                {data.aggregate_confidence && ` (${data.aggregate_confidence})`}
              </div>
              <div className="ai-signal-text">{data.llm_prediction}</div>
            </div>
          )}
        </>
      )}
    </Card>
  )
}
