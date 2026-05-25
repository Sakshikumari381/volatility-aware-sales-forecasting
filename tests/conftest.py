"""
Pytest fixtures shared across tests.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.sample_data import generate_sample_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset."""
    return generate_sample_data(n_products=2, days=60)


@pytest.fixture
def sample_csv(tmp_path, sample_df) -> Path:
    """Write sample CSV to temp directory."""
    path = tmp_path / "test_sales.csv"
    sample_df.to_csv(path, index=False)
    return path
