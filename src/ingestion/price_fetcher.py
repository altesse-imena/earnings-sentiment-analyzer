"""
Fetches historical stock price data using yfinance.
Focuses on the 5-day window around each earnings date.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/raw/cache"))
PRICE_DIR = Path("data/raw/prices")


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.csv"


def fetch_price_window(
    ticker: str,
    event_date: str,
    days_before: int = 2,
    days_after: int = 5,
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a ticker in a window around an event date.

    Args:
        ticker: Stock ticker symbol.
        event_date: Date of the earnings event (YYYY-MM-DD).
        days_before: Calendar days before the event to include.
        days_after: Calendar days after the event to include.

    Returns:
        DataFrame with OHLCV data, or None on failure.
    """
    cache_key = f"price_{ticker}_{event_date}_b{days_before}_a{days_after}"
    cache_file = _cache_path(cache_key)

    if cache_file.exists():
        logger.debug(f"Price cache hit: {cache_key}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        dt = datetime.strptime(event_date, "%Y-%m-%d")
        start = (dt - timedelta(days=days_before + 5)).strftime("%Y-%m-%d")  # buffer for weekends
        end = (dt + timedelta(days=days_after + 5)).strftime("%Y-%m-%d")

        logger.info(f"Fetching prices for {ticker} around {event_date}")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            logger.warning(f"No price data returned for {ticker} around {event_date}")
            return None

        df.index = pd.to_datetime(df.index).tz_localize(None)
        event_dt = pd.Timestamp(event_date)

        # Filter to the actual window (trading days only)
        trading_days = df.index.tolist()
        days_before_trading = [d for d in trading_days if d <= event_dt][-days_before:] if days_before > 0 else []
        days_after_trading = [d for d in trading_days if d > event_dt][:days_after]
        event_day = [d for d in trading_days if d == event_dt]

        window = sorted(set(days_before_trading + event_day + days_after_trading))
        df_window = df.loc[df.index.isin(window)].copy()
        df_window["ticker"] = ticker
        df_window["event_date"] = event_date

        df_window.to_csv(cache_file)
        return df_window

    except Exception as e:
        logger.error(f"Failed to fetch prices for {ticker} on {event_date}: {e}")
        return None


def compute_price_change(df: pd.DataFrame, event_date: str, hours: int = 48) -> dict:
    """
    Compute price change metrics relative to the earnings event.

    Returns a dict with close_on_event, close_48h, pct_change_48h, direction.
    """
    if df is None or df.empty:
        return {}

    event_dt = pd.Timestamp(event_date)
    trading_days = sorted(df.index.tolist())

    # Price at close on event day (or last day before if no trading that day)
    pre_event = [d for d in trading_days if d <= event_dt]
    post_event = [d for d in trading_days if d > event_dt]

    if not pre_event or not post_event:
        return {}

    close_event = df.loc[pre_event[-1], "Close"]

    # 48-hour post: use the second trading day after the event
    target_days = post_event[:2]
    if not target_days:
        return {}
    close_48h = df.loc[target_days[-1], "Close"]

    pct_change = (close_48h - close_event) / close_event * 100

    return {
        "close_event": round(float(close_event), 4),
        "close_48h": round(float(close_48h), 4),
        "pct_change_48h": round(float(pct_change), 4),
        "direction": 1 if pct_change > 0 else 0,
        "event_date": event_date,
        "post_date_48h": target_days[-1].strftime("%Y-%m-%d"),
    }


def fetch_prices_for_tickers(tickers: list[str], event_dates: dict[str, list[str]]) -> pd.DataFrame:
    """
    Fetch price windows for multiple tickers and their earnings dates.

    Args:
        tickers: List of ticker symbols.
        event_dates: Dict mapping ticker → list of earnings dates.

    Returns:
        DataFrame with one row per ticker-event with price change metrics.
    """
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for ticker in tickers:
        dates = event_dates.get(ticker, [])
        if not dates:
            logger.warning(f"No event dates provided for {ticker}")
            continue

        for date in dates:
            df = fetch_price_window(ticker, date)
            if df is not None:
                metrics = compute_price_change(df, date)
                if metrics:
                    metrics["ticker"] = ticker
                    records.append(metrics)
                    # Save full window CSV
                    out = PRICE_DIR / f"{ticker}_{date}.csv"
                    df.to_csv(out)

    result = pd.DataFrame(records)
    if not result.empty:
        out_path = PRICE_DIR / "all_price_changes.csv"
        result.to_csv(out_path, index=False)
        logger.info(f"Saved price changes to {out_path}")
    return result


if __name__ == "__main__":
    # Example usage
    sample_events = {
        "AAPL": ["2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02"],
        "MSFT": ["2023-01-24", "2023-04-25", "2023-07-25", "2023-10-24"],
        "NVDA": ["2023-02-22", "2023-05-24", "2023-08-23", "2023-11-21"],
    }

    logger.info("Starting price fetch for sample tickers")
    df = fetch_prices_for_tickers(list(sample_events.keys()), sample_events)
    logger.info(f"Fetched price data:\n{df}")
