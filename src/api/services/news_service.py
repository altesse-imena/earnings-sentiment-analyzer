from datetime import datetime, timedelta

import httpx

FINNHUB_BASE = "https://finnhub.io/api/v1"


async def fetch_news(ticker: str, api_key: str, days_back: int = 7) -> list[dict]:
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": api_key,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{FINNHUB_BASE}/company-news", params=params)
        resp.raise_for_status()
        articles = resp.json()

    return articles[:15]
