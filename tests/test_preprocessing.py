"""
Tests for preprocessing pipeline.
"""

import pandas as pd
import pytest

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.validation import DataValidator


def test_validation_schema(sample_df):
    validator = DataValidator()
    ok, errors = validator.run_all_checks(sample_df)
    assert ok is True
    assert errors == []


def test_validation_empty():
    validator = DataValidator()
    ok, errors = validator.validate_not_empty(pd.DataFrame())
    assert ok is False


def test_feature_engineering(sample_df):
    pipeline = PreprocessingPipeline()
    cleaned = pipeline.clean(sample_df)
    featured = FeatureEngineer().transform(cleaned)
    assert "lag_1" in featured.columns
    assert "lag_7" in featured.columns
    assert "lag_14" in featured.columns
    assert "rolling_mean_7" in featured.columns
    assert "weekend" in featured.columns
    assert "promo_x_lag_1" in featured.columns


def test_preprocessing_pipeline_run(sample_df):
    pipeline = PreprocessingPipeline()
    processed, features = pipeline.run(sample_df, save=False)
    assert len(processed) > 0
    assert len(features) > 0
    assert "sales_quantity" in processed.columns


def test_chronological_sort(sample_df):
    pipeline = PreprocessingPipeline()
    sorted_df = pipeline.sort_chronologically(sample_df)
    products = sorted_df.groupby("product_id", observed=True)
    for _, group in products:
        dates = group["date"].values
        assert (dates == sorted(dates)).all()
