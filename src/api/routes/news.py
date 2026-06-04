import os
from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.api.services.llm_service import aggregate_articles, annotate_article
from src.api.services.news_service import fetch_news

router = APIRouter()


@router.get("/news/{ticker}")
async def get_news(ticker: str):
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY not configured")

    try:
        raw_articles = await fetch_news(ticker, api_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"News fetch failed: {e}")

    if not raw_articles:
        return {
            "ticker": ticker,
            "articles": [],
            "aggregate_signal": "NEUTRAL",
            "aggregate_confidence": "LOW",
            "llm_prediction": "No recent news found.",
        }

    articles_out = []
    for article in raw_articles[:5]:
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        annotation = await annotate_article(headline, summary)

        ts = article.get("datetime", 0)
        published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""

        articles_out.append(
            {
                "headline": headline,
                "summary": summary,
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published_at": published,
                "llm_sentiment": annotation.get("sentiment", "neutral").upper(),
                "llm_confidence": annotation.get("confidence", "LOW"),
                "llm_reasoning": annotation.get("reasoning", ""),
            }
        )

    aggregate = await aggregate_articles(raw_articles)

    return {
        "ticker": ticker,
        "articles": articles_out,
        "aggregate_signal": aggregate.get("aggregate_signal", "NEUTRAL"),
        "aggregate_confidence": aggregate.get("aggregate_confidence", "LOW"),
        "llm_prediction": aggregate.get("llm_prediction", ""),
    }
