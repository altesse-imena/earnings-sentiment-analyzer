import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel

from src.ingestion.edgar_fetcher import run_ingestion
from src.modeling.train import run_training
from src.processing.sentiment_pipeline import run_pipeline

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = PROCESSED_DIR / "model/lgbm_sentiment.pkl"
FEATURES_PATH = PROCESSED_DIR / "call_features.csv"
MATRIX_PATH = PROCESSED_DIR / "feature_matrix.csv"
REPORTS_DIR = PROCESSED_DIR / "reports"

router = APIRouter()


class IngestRequest(BaseModel):
    tickers: list[str]
    years: list[int]


def _train_and_refresh(app):
    run_training()
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
    if FEATURES_PATH.exists():
        app.state.df_features = pd.read_csv(FEATURES_PATH)
    if MATRIX_PATH.exists():
        app.state.df_matrix = pd.read_csv(MATRIX_PATH)
    report_path = REPORTS_DIR / "performance_report.json"
    if report_path.exists():
        app.state.report = json.loads(report_path.read_text())
    shap_path = REPORTS_DIR / "shap_values.csv"
    if shap_path.exists():
        app.state.shap_df = pd.read_csv(shap_path)


def _process_and_refresh(app):
    run_pipeline()
    if FEATURES_PATH.exists():
        app.state.df_features = pd.read_csv(FEATURES_PATH)


@router.post("/pipeline/ingest")
def trigger_ingest(body: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_ingestion, body.tickers, body.years)
    return {"status": "started", "message": "Ingestion running in background", "tickers": body.tickers, "years": body.years}


@router.post("/pipeline/process")
def trigger_process(background_tasks: BackgroundTasks, request: Request):
    background_tasks.add_task(_process_and_refresh, request.app)
    return {"status": "started", "message": "Processing running in background"}


@router.post("/pipeline/train")
def trigger_train(background_tasks: BackgroundTasks, request: Request):
    background_tasks.add_task(_train_and_refresh, request.app)
    return {"status": "started", "message": "Training running in background"}
