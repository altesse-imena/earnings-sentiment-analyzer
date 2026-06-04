import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    model,
    news,
    pipeline,
    prediction,
    prices,
    realtime,
    sentiment,
    shap,
    tickers,
)
from src.api.services.price_stream import price_broadcast_loop

load_dotenv()

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = PROCESSED_DIR / "model/lgbm_sentiment.pkl"
FEATURES_PATH = PROCESSED_DIR / "call_features.csv"
MATRIX_PATH = PROCESSED_DIR / "feature_matrix.csv"
REPORTS_DIR = PROCESSED_DIR / "reports"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    app.state.df_features = pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame()
    app.state.df_matrix = pd.read_csv(MATRIX_PATH) if MATRIX_PATH.exists() else pd.DataFrame()

    report_path = REPORTS_DIR / "performance_report.json"
    app.state.report = json.loads(report_path.read_text()) if report_path.exists() else {}

    shap_path = REPORTS_DIR / "shap_values.csv"
    app.state.shap_df = pd.read_csv(shap_path) if shap_path.exists() else pd.DataFrame()

    try:
        app.state.price_task = asyncio.create_task(price_broadcast_loop())
    except RuntimeError:
        app.state.price_task = None

    yield

    if app.state.price_task is not None:
        app.state.price_task.cancel()
        try:
            await app.state.price_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Earnings Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickers.router, prefix="/api")
app.include_router(sentiment.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(prices.router, prefix="/api")
app.include_router(shap.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(news.router, prefix="/api")
app.include_router(realtime.router)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
