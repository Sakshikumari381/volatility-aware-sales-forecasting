"""
Modular feature engineering for volatility-aware sales forecasting.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)


class FeatureEngineer:
    """
    Generate lag, rolling, volatility, seasonal, and interaction features.
    """

    LAG_PERIODS: List[int] = [1, 7, 14]
    ROLLING_WINDOWS: List[int] = [7, 14]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        volatility_window: Optional[int] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.volatility_window = volatility_window or self.settings.volatility_window
        self.target = self.settings.target_column
        self.date_col = self.settings.date_column
        self.product_col = self.settings.product_column

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create t-1, t-7, t-14 lag features per product."""
        out = df.sort_values([self.product_col, self.date_col]).copy()
        for lag in self.LAG_PERIODS:
            col = f"lag_{lag}"
            out[col] = out.groupby(self.product_col, observed=True)[self.target].shift(
                lag
            )
        return out

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean (7, 14) and rolling std per product."""
        out = df.sort_values([self.product_col, self.date_col]).copy()
        grouped = out.groupby(self.product_col, observed=True)[self.target]

        for window in self.ROLLING_WINDOWS:
            out[f"rolling_mean_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )
        out["rolling_std_7"] = grouped.transform(
            lambda s: s.rolling(window=7, min_periods=1).std()
        )
        return out

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling volatility (coefficient of variation) per product."""
        out = df.sort_values([self.product_col, self.date_col]).copy()
        w = self.volatility_window

        def _volatility(series: pd.Series) -> pd.Series:
            roll_mean = series.rolling(window=w, min_periods=2).mean()
            roll_std = series.rolling(window=w, min_periods=2).std()
            vol = roll_std / roll_mean.replace(0, np.nan)
            return vol.fillna(0)

        out[f"rolling_volatility_{w}"] = out.groupby(
            self.product_col, observed=True
        )[self.target].transform(_volatility)
        return out

    def add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Day of week, month, weekend indicator."""
        out = df.copy()
        if self.date_col in out.columns:
            out["day_of_week"] = out[self.date_col].dt.dayofweek.astype(np.int8)
            out["month"] = out[self.date_col].dt.month.astype(np.int8)
        out["weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
        return out

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """promotional_flag × lag features."""
        out = df.copy()
        if "promotional_flag" not in out.columns:
            out["promotional_flag"] = 0
        for lag in self.LAG_PERIODS:
            lag_col = f"lag_{lag}"
            if lag_col in out.columns:
                out[f"promo_x_{lag_col}"] = (
                    out["promotional_flag"] * out[lag_col]
                )
        return out

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional price-derived features."""
        out = df.copy()
        if "price" in out.columns:
            out["log_price"] = np.log1p(out["price"].clip(lower=0))
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned, sorted input data.

        Returns
        -------
        pd.DataFrame
            Dataframe with all engineered features.
        """
        logger.info("Starting feature engineering.")
        out = self.add_lag_features(df)
        out = self.add_rolling_features(out)
        out = self.add_volatility_features(out)
        out = self.add_seasonal_features(out)
        out = self.add_interaction_features(out)
        out = self.add_price_features(out)
        logger.info("Feature engineering complete. Shape: %s", out.shape)
        return out

    @staticmethod
    def get_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
        """
        Return model feature column names (exclude ids, dates, target).
        """
        exclude = {
            target,
            "date",
            "product_id",
            "category",
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeric_cols if c not in exclude]
