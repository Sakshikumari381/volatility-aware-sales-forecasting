"""
Dataset schema and quality validation.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)


class DataValidator:
    """Validate input CSV structure and basic data quality."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check required columns exist.

        Returns
        -------
        tuple
            (is_valid, list of error messages)
        """
        errors: List[str] = []
        missing = [
            col
            for col in self.settings.required_columns
            if col not in df.columns
        ]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        return len(errors) == 0, errors

    def validate_not_empty(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Ensure dataframe has rows."""
        if df is None or df.empty:
            return False, ["Dataset is empty."]
        return True, []

    def validate_dates(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate date column parseability and non-null ratio."""
        errors: List[str] = []
        col = self.settings.date_column
        if col not in df.columns:
            return False, [f"Date column '{col}' not found."]

        dates = pd.to_datetime(df[col], errors="coerce")
        invalid_count = dates.isna().sum()
        if invalid_count == len(df):
            errors.append("All dates are invalid.")
        elif invalid_count > 0:
            pct = 100 * invalid_count / len(df)
            logger.warning("%.2f%% rows have invalid dates.", pct)

        return len(errors) == 0, errors

    def validate_numeric_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check sales and price are non-negative where present."""
        errors: List[str] = []
        for col in ("sales_quantity", "price"):
            if col in df.columns:
                if (df[col] < 0).any():
                    errors.append(f"Negative values found in '{col}'.")
        return len(errors) == 0, errors

    def run_all_checks(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Run full validation pipeline."""
        all_errors: List[str] = []
        checks = [
            self.validate_not_empty,
            self.validate_schema,
            self.validate_dates,
            self.validate_numeric_ranges,
        ]
        for check in checks:
            ok, errs = check(df)
            if not ok:
                all_errors.extend(errs)
        return len(all_errors) == 0, all_errors

    @staticmethod
    def coerce_types(df: pd.DataFrame, settings: Optional[Settings] = None) -> pd.DataFrame:
        """Normalize dtypes after load."""
        settings = settings or get_settings()
        out = df.copy()
        out[settings.date_column] = pd.to_datetime(
            out[settings.date_column], errors="coerce"
        )
        out = out.dropna(subset=[settings.date_column])
        for col in ("sales_quantity", "price"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "promotional_flag" in out.columns:
            out["promotional_flag"] = (
                out["promotional_flag"].fillna(0).astype(np.int8)
            )
        return out
