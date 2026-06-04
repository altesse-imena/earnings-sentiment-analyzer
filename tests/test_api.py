import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_tickers_returns_list(client):
    resp = client.get("/api/tickers")
    assert resp.status_code == 200
    body = resp.json()
    assert "tickers" in body
    assert isinstance(body["tickers"], list)


def test_dates_for_valid_ticker(client):
    resp = client.get("/api/tickers/AAPL/dates")
    assert resp.status_code == 200
    body = resp.json()
    assert "dates" in body
    assert isinstance(body["dates"], list)


def test_dates_for_unknown_ticker_returns_empty(client):
    resp = client.get("/api/tickers/ZZZZ/dates")
    assert resp.status_code == 200
    assert resp.json()["dates"] == []


def test_sentiment_valid(client):
    resp = client.get("/api/sentiment/AAPL/2023-08-03")
    assert resp.status_code == 200
    body = resp.json()
    assert "overall_sentiment" in body
    assert isinstance(body["overall_sentiment"], float)


def test_sentiment_missing_returns_404(client):
    resp = client.get("/api/sentiment/FAKE/1999-01-01")
    assert resp.status_code == 404


def test_prediction_valid(client):
    resp = client.get("/api/prediction/AAPL/2023-08-03")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        body = resp.json()
        assert "direction" in body
        assert body["direction"] in ("UP", "DOWN")


def test_prediction_missing_returns_404(client):
    resp = client.get("/api/prediction/FAKE/1999-01-01")
    assert resp.status_code == 404


def test_prices_valid(client):
    resp = client.get("/api/prices/AAPL/2023-08-03")
    assert resp.status_code == 200
    body = resp.json()
    assert "prices" in body
    assert isinstance(body["prices"], list)
    if body["prices"]:
        row = body["prices"][0]
        assert "date" in row and "close" in row


def test_prices_missing_returns_empty(client):
    resp = client.get("/api/prices/FAKE/1999-01-01")
    assert resp.status_code == 200
    assert resp.json()["prices"] == []


def test_shap_returns_features(client):
    resp = client.get("/api/shap")
    assert resp.status_code == 200
    body = resp.json()
    assert "features" in body
    assert isinstance(body["features"], list)


def test_model_report_returns_json(client):
    resp = client.get("/api/model/report")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)


def test_pipeline_ingest_starts(client):
    resp = client.post(
        "/api/pipeline/ingest",
        json={"tickers": ["AAPL"], "years": [2023]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_pipeline_process_starts(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.api.routes.pipeline._process_and_refresh", lambda app: None)
        resp = client.post("/api/pipeline/process")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_pipeline_train_starts(client):
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("src.api.routes.pipeline._train_and_refresh", lambda app: None)
        resp = client.post("/api/pipeline/train")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"
