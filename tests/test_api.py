"""
API endpoint tests using FastAPI TestClient.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.server import app
from utils.sample_data import generate_sample_data


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def csv_bytes(sample_df):
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_upload_and_train(client, csv_bytes):
    resp = client.post(
        "/upload",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["rows"] > 0

    train_resp = client.post(
        "/train",
        json={
            "forecast_horizon": 7,
            "volatility_window": 7,
            "shap_threshold": 0.001,
            "arima_order": [1, 0, 1],
        },
    )
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "success"
    assert "metrics" in train_data
    assert "xgboost" in train_data["metrics"]


def test_metrics_after_train(client, csv_bytes):
    client.post("/upload", files={"file": ("m.csv", csv_bytes, "text/csv")})
    client.post(
        "/train",
        json={
            "forecast_horizon": 7,
            "volatility_window": 7,
            "shap_threshold": 0.001,
            "arima_order": [1, 0, 1],
        },
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "metrics" in resp.json()


def test_forecast_endpoint(client, csv_bytes):
    client.post("/upload", files={"file": ("f.csv", csv_bytes, "text/csv")})
    client.post(
        "/train",
        json={
            "forecast_horizon": 5,
            "volatility_window": 7,
            "shap_threshold": 0.001,
            "arima_order": [1, 0, 1],
        },
    )
    resp = client.get("/forecast/P001", params={"model_name": "hybrid", "horizon": 5})
    assert resp.status_code == 200
    assert "forecast" in resp.json()


def test_upload_invalid_extension(client):
    resp = client.post(
        "/upload",
        files={"file": ("bad.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
