import math

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

SENTIMENT_FIELDS = [
    "overall_sentiment",
    "ceo_sentiment",
    "cfo_sentiment",
    "analyst_sentiment",
    "prepared_sentiment",
    "qa_sentiment",
    "tone_shift",
    "sentiment_volatility",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "ceo_sentence_count",
    "cfo_sentence_count",
    "analyst_sentence_count",
    "total_sentences",
]


def safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


@router.get("/sentiment/{ticker}/{date}")
def get_sentiment(ticker: str, date: str, request: Request):
    df = request.app.state.df_features
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    mask = (df["ticker"] == ticker) & (df["event_date"] == date)
    rows = df[mask]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker} on {date}")

    row = rows.iloc[0]
    return {field: safe_float(row.get(field, 0.0)) for field in SENTIMENT_FIELDS}
