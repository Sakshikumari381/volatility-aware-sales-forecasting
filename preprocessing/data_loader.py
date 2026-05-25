"""
Efficient data loading with parquet caching for large datasets.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, Optional, Union

import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from preprocessing.column_names import missing_required_columns, normalize_columns

logger = setup_logging(__name__)


def read_sales_csv(
    source: Union[str, Path, BinaryIO, BytesIO],
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """
    Load a sales CSV, normalize column names, parse dates, and fill derived fields.
    """
    settings = settings or get_settings()
    if isinstance(source, (str, Path)) and Path(source).exists():
        df = pd.read_csv(source)
    elif isinstance(source, str):
        df = pd.read_csv(StringIO(source))
    else:
        df = pd.read_csv(source)
    df = normalize_columns(df)

    missing = missing_required_columns(df, settings.required_columns)
    if missing:
        found = ", ".join(df.columns.astype(str).tolist()) or "(none)"
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {found}. "
            "Expected at least: date, product_id, category, sales_quantity, "
            "price, promotional_flag (names are case-insensitive)."
        )

    date_col = settings.date_column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"Could not parse any values in '{date_col}' column.")

    df["product_id"] = df["product_id"].astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)

    if "day_of_week" not in df.columns:
        df["day_of_week"] = df[date_col].dt.dayofweek.astype("int8")
    if "month" not in df.columns:
        df["month"] = df[date_col].dt.month.astype("int8")

    for col in ("promotional_flag", "day_of_week", "month"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    return df


class DataLoader:
    """
    Load CSV datasets with optional parquet cache for fast reloads.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def _cache_path(self, source_path: Path) -> Path:
        """Derive parquet cache path from source file stem."""
        cache_name = f"{source_path.stem}_cache.parquet"
        return self.settings.data_cache_dir / cache_name

    def load_csv(
        self,
        file_path: Union[str, Path],
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load CSV with vectorized parsing; cache as parquet when enabled.

        Parameters
        ----------
        file_path : str or Path
            Path to CSV file.
        use_cache : bool
            Whether to read/write parquet cache.
        force_reload : bool
            Ignore cache and reload from CSV.

        Returns
        -------
        pd.DataFrame
            Raw dataframe.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        cache_path = self._cache_path(path)
        if use_cache and not force_reload and cache_path.exists():
            logger.info("Loading cached parquet: %s", cache_path)
            return pd.read_parquet(cache_path)

        logger.info("Loading CSV: %s", path)
        df = read_sales_csv(path, self.settings)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Cached dataset to %s", cache_path)

        return df

    def load_processed(self, filename: str = "processed_data.parquet") -> pd.DataFrame:
        """Load preprocessed parquet from processed data directory."""
        path = self.settings.data_processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Processed data not found: {path}")
        return pd.read_parquet(path)

    def save_processed(
        self, df: pd.DataFrame, filename: str = "processed_data.parquet"
    ) -> Path:
        """Persist processed dataframe as parquet."""
        path = self.settings.data_processed_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved processed data to %s", path)
        return path

    def save_upload(self, content: bytes, filename: str) -> Path:
        """Save uploaded file bytes to raw data directory."""
        path = self.settings.data_raw_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        logger.info("Saved upload to %s", path)
        return path
