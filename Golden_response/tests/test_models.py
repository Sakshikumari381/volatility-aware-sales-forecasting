"""
Tests for forecasting models and metrics.
"""

import numpy as np
import pytest

from evaluation.metrics import compute_all_metrics, rmsle
from models.baseline_models import ARIMAForecaster, NaiveBaseline, SARIMAForecaster
from models.hybrid_model import HybridXGBARIMA
from models.model_utils import time_based_split
from preprocessing.preprocessing_pipeline import PreprocessingPipeline


def test_naive_baseline():
    X = np.arange(20).reshape(-1, 1)
    y = np.linspace(10, 30, 20)
    model = NaiveBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 20
    assert preds[-1] == pytest.approx(y[-1])


def test_arima_fits():
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, 50)) + 100
    X = np.arange(len(y)).reshape(-1, 1)
    model = ARIMAForecaster(order=(1, 0, 1))
    model.fit(X, y)
    preds = model.predict(X, steps=10)
    assert len(preds) == 10


def test_hybrid_model(sample_df):
    pipeline = PreprocessingPipeline()
    processed, feature_cols = pipeline.run(sample_df, save=False)
    train_df, val_df, test_df = time_based_split(processed, "date", 0.6, 0.2)
    target = "sales_quantity"
    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target].values.astype(np.float32)
    model = HybridXGBARIMA(arima_order=(1, 0, 1), use_tuning=False)
    model.fit(X, y, feature_names=feature_cols)
    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_metrics():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 28])
    m = compute_all_metrics(y_true, y_pred)
    assert "rmsle" in m
    assert "rmse" in m
    assert "r2" in m
    assert rmsle(y_true, y_pred) >= 0


def test_time_split_no_shuffle(sample_df):
    pipeline = PreprocessingPipeline()
    processed, _ = pipeline.run(sample_df, save=False)
    train, val, test = time_based_split(processed, "date")
    assert train["date"].max() <= val["date"].min() or len(val) == 0
    assert len(train) + len(val) + len(test) == len(processed)
