import asyncio
import json
import os

import anthropic

MODEL = "claude-sonnet-4-20250514"

_PER_ARTICLE_SYSTEM = """You are a financial news sentiment analyst. Analyze the given news article headline and summary.
Return ONLY a JSON object with exactly this structure and no other text:
{"sentiment": "positive" | "negative" | "neutral", "confidence": "HIGH" | "MEDIUM" | "LOW", "reasoning": "<one sentence>"}"""

_AGGREGATE_SYSTEM = """You are a financial analyst assessing investor sentiment. Given news headlines and summaries
about a company, return ONLY a JSON object with exactly this structure and no other text:
{"aggregate_signal": "POSITIVE" | "NEGATIVE" | "NEUTRAL", "aggregate_confidence": "HIGH" | "MEDIUM" | "LOW",
"llm_prediction": "<one paragraph assessing likely investor sentiment toward executive statements>"}"""


def _get_client() -> anthropic.Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _call_claude(system: str, user: str) -> str:
    client = _get_client()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _safe_parse(raw: str, fallback: dict) -> dict:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return fallback


async def annotate_article(headline: str, summary: str) -> dict:
    user = f"Headline: {headline}\nSummary: {summary}"
    try:
        raw = await asyncio.to_thread(_call_claude, _PER_ARTICLE_SYSTEM, user)
        return _safe_parse(
            raw,
            {"sentiment": "neutral", "confidence": "LOW", "reasoning": "Analysis unavailable."},
        )
    except Exception:
        return {"sentiment": "neutral", "confidence": "LOW", "reasoning": "Analysis unavailable."}


_RECOMMENDATION_SYSTEM = """You are a professional equity analyst providing actionable investment guidance.
Given recent news headlines, article sentiment scores, and earnings call sentiment data for a stock, produce a
concise investment recommendation. Return ONLY a JSON object with exactly this structure and no other text:
{
  "action": "BUY CALLS" | "BUY PUTS" | "ACCUMULATE" | "HOLD" | "REDUCE" | "AVOID",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "timeframe": "<e.g. 1–2 weeks, 1–3 months>",
  "rationale": "<2–3 sentences explaining the recommendation>",
  "risk_factors": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "key_catalysts": ["<catalyst 1>", "<catalyst 2>"],
  "disclaimer": "This is AI-generated analysis for informational purposes only and does not constitute financial advice."
}"""


async def investment_recommendation(
    ticker: str,
    articles: list[dict],
    aggregate_signal: str,
    aggregate_confidence: str,
    sentiment_row: dict | None,
) -> dict:
    headlines = "\n".join(
        f"- [{a.get('llm_sentiment','?')}] {a.get('headline','')}" for a in articles[:5]
    )
    sentiment_summary = ""
    if sentiment_row:
        sentiment_summary = (
            f"\nEarnings call sentiment (most recent): "
            f"overall={sentiment_row.get('overall_sentiment', 'n/a')}, "
            f"ceo={sentiment_row.get('ceo_sentiment', 'n/a')}, "
            f"tone_shift={sentiment_row.get('tone_shift', 'n/a')}"
        )

    user = (
        f"Ticker: {ticker}\n"
        f"Aggregate news signal: {aggregate_signal} (confidence: {aggregate_confidence})\n"
        f"Recent headlines:\n{headlines}"
        f"{sentiment_summary}"
    )
    fallback = {
        "action": "HOLD",
        "conviction": "LOW",
        "timeframe": "unclear",
        "rationale": "Insufficient data to generate a recommendation.",
        "risk_factors": [],
        "key_catalysts": [],
        "disclaimer": "This is AI-generated analysis for informational purposes only and does not constitute financial advice.",
    }
    try:
        raw = await asyncio.to_thread(_call_claude, _RECOMMENDATION_SYSTEM, user)
        return _safe_parse(raw, fallback)
    except Exception:
        return fallback


async def aggregate_articles(articles: list[dict]) -> dict:
    lines = "\n".join(
        f"- {a.get('headline', '')} | {a.get('summary', '')[:200]}" for a in articles[:5]
    )
    user = f"News articles:\n{lines}"
    try:
        raw = await asyncio.to_thread(_call_claude, _AGGREGATE_SYSTEM, user)
        return _safe_parse(
            raw,
            {
                "aggregate_signal": "NEUTRAL",
                "aggregate_confidence": "LOW",
                "llm_prediction": "Aggregate analysis unavailable.",
            },
        )
    except Exception:
        return {
            "aggregate_signal": "NEUTRAL",
            "aggregate_confidence": "LOW",
            "llm_prediction": "Aggregate analysis unavailable.",
        }
