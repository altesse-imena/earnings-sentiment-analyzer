export default function Header({ ticker, date }) {
  return (
    <div className="header">
      <div className="header-tagline">
        FinBERT sentiment correlated with post-earnings price movement
      </div>
      {(ticker || date) && (
        <div className="header-pills">
          {ticker && <span className="header-pill accent">{ticker}</span>}
          {date && <span className="header-pill">{date}</span>}
        </div>
      )}
      <hr className="header-divider" />
    </div>
  )
}
