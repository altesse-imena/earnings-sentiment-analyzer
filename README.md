# Earnings Sentiment Analyzer

> Real-time NLP pipeline that analyzes earnings call transcripts and correlates executive sentiment with post-earnings stock price movement.

---

## Overview

Earnings calls are one of the richest signals in public markets. This project fetches SEC EDGAR transcripts, runs FinBERT sentiment scoring broken down by speaker role and call section, engineers a feature set, and trains a LightGBM classifier to predict 48-hour post-earnings price direction. A Streamlit dashboard surfaces the results interactively.

**Key questions this project answers:**
- Do CEOs and CFOs signal different information through tone?
- Does sentiment shift between prepared remarks and Q&A predict price movement?
- Which sentiment features are most predictive of post-earnings returns?

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | SEC EDGAR Full-Text Search API, `yfinance` |
| NLP | `transformers` (FinBERT), `nltk`, `spacy` |
| Feature Engineering | `pandas`, `numpy` |
| Modeling | `lightgbm`, `shap` |
| Evaluation | `scikit-learn`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit`, `plotly` |
| Infra | Python 3.11, `python-dotenv`, `loguru` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│   SEC EDGAR Full-Text API      Yahoo Finance (yfinance) │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
                ▼                     ▼
┌───────────────────────┐   ┌─────────────────────────┐
│  Transcript Ingestion │   │  Price Data Fetching    │
│  src/ingestion/       │   │  src/ingestion/         │
│  - edgar_fetcher.py   │   │  - price_fetcher.py     │
│  - caching layer      │   │  - 5-day event window   │
└───────────┬───────────┘   └────────────┬────────────┘
            │                            │
            ▼                            │
┌───────────────────────┐                │
│  NLP Processing       │                │
│  src/processing/      │                │
│  - FinBERT scoring    │                │
│  - Speaker diarize    │                │
│  - Feature engineer   │                │
└───────────┬───────────┘                │
            │                            │
            └──────────┬─────────────────┘
                       ▼
┌──────────────────────────────────────┐
│  Modeling  src/modeling/             │
│  - Correlation analysis              │
│  - LightGBM classifier               │
│  - SHAP explainability               │
│  - Performance report                │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  Streamlit Dashboard                 │
│  src/dashboard/app.py                │
│  - Ticker search                     │
│  - Sentiment by speaker / section   │
│  - Price chart + sentiment overlay  │
│  - SHAP chart + model prediction    │
└──────────────────────────────────────┘
```

---

## Setup

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/altesse-imena/earnings-sentiment-analyzer.git
cd earnings-sentiment-analyzer
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Running the pipeline

```bash
# 1. Fetch transcripts and price data
python src/ingestion/edgar_fetcher.py --tickers AAPL MSFT NVDA --years 2023 2024

# 2. Run NLP processing and feature engineering
python src/processing/sentiment_pipeline.py

# 3. Train model and generate evaluation report
python src/modeling/train.py

# 4. Launch dashboard
streamlit run src/dashboard/app.py
```

---

## Project Structure

```
earnings-sentiment-analyzer/
├── data/
│   ├── raw/               # Raw transcripts and price CSVs
│   └── processed/         # Feature-engineered datasets
├── notebooks/             # EDA and experimentation
├── src/
│   ├── ingestion/         # SEC EDGAR + yfinance fetchers
│   ├── processing/        # FinBERT pipeline + feature engineering
│   ├── modeling/          # LightGBM + SHAP
│   └── dashboard/         # Streamlit app
├── tests/                 # Unit tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Results

> _Model training in progress — results will be updated here._

| Metric | Value |
|---|---|
| AUC-ROC | TBD |
| Accuracy | TBD |
| Top Feature | TBD |
| Tickers Evaluated | TBD |

---

## License

MIT
