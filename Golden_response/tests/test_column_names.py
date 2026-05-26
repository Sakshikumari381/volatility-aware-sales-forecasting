"""Tests for CSV column name normalization."""

import pandas as pd
import pytest

from preprocessing.column_names import normalize_columns
from preprocessing.data_loader import read_sales_csv
from utils.sample_data import generate_sample_data


def test_normalize_date_alias():
    df = generate_sample_data(n_products=1, days=5)
    df = df.rename(columns={"date": "Order Date"})
    normalized = normalize_columns(df)
    assert "date" in normalized.columns


def test_read_sales_csv_from_buffer():
    df = generate_sample_data(n_products=1, days=5)
    df = df.rename(columns={"sales_quantity": "Qty", "promotional_flag": "Promo"})
    buf = df.to_csv(index=False)
    from io import StringIO

    loaded = read_sales_csv(StringIO(buf))
    assert "date" in loaded.columns
    assert "sales_quantity" in loaded.columns
    assert "day_of_week" in loaded.columns


def test_missing_date_raises_clear_error():
    df = pd.DataFrame({"product_id": ["P1"], "sales": [10]})
    from io import StringIO

    with pytest.raises(ValueError, match="Missing required columns"):
        read_sales_csv(StringIO(df.to_csv(index=False)))
