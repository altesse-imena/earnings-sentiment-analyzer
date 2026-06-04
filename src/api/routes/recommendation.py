import os

from fastapi import APIRouter, HTTPException, Request

from src.api.services.llm_service import investment_recommendation
from src.api.services.news_service import fetch_news
from src.api.routes.sentiment import safe_float

router = APIRouter()


@router.get("/recommendation/{ticker}")
async def get_recommendation(ticker: str, request: Request):
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    if not finnhub_key:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY not configured")

    try:
        articles = await fetch_news(ticker, finnhub_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"News fetch failed: {e}")

    # Annotate a quick aggregate signal from the raw article list
    pos = sum(1 for a in articles if a.get("llm_sentiment", "").lower() == "positive")
    neg = sum(1 for a in articles if a.get("llm_sentiment", "").lower() == "negative")
    if pos > neg:
        agg_signal, agg_conf = "POSITIVE", "MEDIUM"
    elif neg > pos:
        agg_signal, agg_conf = "NEGATIVE", "MEDIUM"
    else:
        agg_signal, agg_conf = "NEUTRAL", "LOW"

    # Grab the most recent sentiment row for this ticker
    df = request.app.state.df_features
    sentiment_row = None
    if not df.empty:
        subset = df[df["ticker"] == ticker].sort_values("event_date", ascending=False)
        if not subset.empty:
            row = subset.iloc[0]
            sentiment_row = {
                col: safe_float(row.get(col))
                for col in ["overall_sentiment", "ceo_sentiment", "tone_shift"]
            }

    rec = await investment_recommendation(
        ticker=ticker,
        articles=articles,
        aggregate_signal=agg_signal,
        aggregate_confidence=agg_conf,
        sentiment_row=sentiment_row,
    )

    return {"ticker": ticker, **rec}
