# Earnings Sentiment Analyzer

> NLP pipeline that analyzes earnings call transcripts and correlates executive sentiment with post-earnings stock price movement — now with a FastAPI backend, React frontend, real-time price streaming, and LLM-annotated news feed.

---

## Architecture

```
Pipeline (edgar_fetcher / sentiment_pipeline / train)
     |
     v
FastAPI (src/api/main.py) :8000
     |-- REST /api/tickers, /api/sentiment, /api/prediction, /api/prices, /api/shap
     |-- REST /api/model/report, /api/news/{ticker}
     |-- POST /api/pipeline/{ingest,process,train}
     |-- WebSocket /ws/prices  (live AAPL/MSFT/GOOGL/AMZN/NVDA prices every 30s)
     v
React/Vite (frontend/) :3000
     |-- Dashboard: stat cards, speaker/section charts, price chart, SHAP chart, prediction
     |-- LivePriceTicker: real-time prices with auto-reconnect WebSocket
     |-- NewsPanel: Finnhub articles + LLM sentiment badges (Claude)
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | SEC EDGAR Full-Text Search API, `yfinance` |
| NLP | `transformers` (FinBERT) |
| Feature Engineering | `pandas`, `numpy` |
| Modeling | `lightgbm`, `shap`, `scikit-learn` |
| Backend | `FastAPI`, `uvicorn`, `httpx` |
| LLM Layer | `anthropic` (claude-sonnet-4-20250514) |
| News Feed | Finnhub free API |
| Frontend | React 19, Vite, Recharts, TanStack Query, lucide-react |
| Infra | Python 3.11+, Node 18+, `python-dotenv` |

---

## Setup

### Prerequisites

- Python 3.11+
- Node 18+

### Installation

```bash
git clone https://github.com/altesse-imena/earnings-sentiment-analyzer.git
cd earnings-sentiment-analyzer

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and FINNHUB_API_KEY in .env

cd frontend && npm install && cd ..
```

### Running

**Backend:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend && npm run dev
```

Then open `http://localhost:3000`.

---

### Running the data pipeline (optional)

```bash
# 1. Fetch transcripts and price data
python src/ingestion/edgar_fetcher.py --tickers AAPL MSFT NVDA --years 2023 2024

# 2. Run FinBERT sentiment scoring and feature engineering
python src/processing/sentiment_pipeline.py

# 3. Train LightGBM model and generate SHAP report
python src/modeling/train.py
```

Or use the **Run Pipeline** button in the sidebar of the React app to trigger all three stages via the API.

---

## Real-time & LLM Features

### Live Price Streaming

The WebSocket endpoint `/ws/prices` polls yfinance every 30 seconds for AAPL, MSFT, GOOGL, AMZN, and NVDA. The `LivePriceTicker` component at the top of the page receives these updates and displays price, change, and percent change. The React hook uses exponential backoff (1s base, 30s max, ±20% jitter) to auto-reconnect on disconnect.

### Finnhub News Feed

`GET /api/news/{ticker}` fetches the last 7 days of company news from Finnhub's free API, then runs each article through Claude for per-article sentiment classification (POSITIVE / NEGATIVE / NEUTRAL, with HIGH / MEDIUM / LOW confidence and a one-sentence reasoning). A second aggregate Claude call produces a paragraph-length directional prediction based on the news as a whole.

### LLM Sentiment Layer

All LLM calls use `claude-sonnet-4-20250514` via the Anthropic SDK. Prompts instruct Claude to return structured JSON only. All responses are JSON-parsed with a safe fallback to neutral if parsing fails. The synchronous Anthropic SDK calls are wrapped in `asyncio.to_thread` to avoid blocking the FastAPI event loop.

---

## Project Structure

```
earnings-sentiment-analyzer/
├── src/
│   ├── api/                   # FastAPI backend
│   │   ├── main.py            # App entry point, lifespan, CORS
│   │   ├── routes/            # One file per resource
│   │   └── services/          # news_service, llm_service, price_stream
│   ├── ingestion/             # SEC EDGAR + yfinance fetchers
│   ├── processing/            # FinBERT pipeline + feature engineering
│   └── modeling/              # LightGBM + SHAP training
├── frontend/                  # React/Vite app
│   └── src/
│       ├── components/        # layout/, dashboard/, realtime/, shared/
│       ├── hooks/             # usePriceStream, useDashboard, useNews, useTickers
│       ├── api/               # axios client + endpoint functions
│       └── styles/globals.css # Design tokens
├── data/
│   ├── raw/                   # Transcripts, price CSVs, cache
│   └── processed/             # Feature matrices, model, reports
├── tests/
│   ├── test_api.py            # FastAPI TestClient tests (14 cases)
│   └── test_processing.py     # Processing unit tests
├── .env.example
└── requirements.txt
```

---

## Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.71 |
| Accuracy | 0.69 |
| CV AUC (5-fold) | 0.68 ± 0.04 |
| Top Feature | `tone_shift` (Q&A − prepared remarks) |
| Tickers | AAPL, MSFT, NVDA |

**Key finding:** `tone_shift` — the delta between Q&A sentiment and prepared-remarks sentiment — is the strongest single predictor of 48h post-earnings price direction.

---

## License

MIT
