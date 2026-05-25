"""
End-to-end preprocessing: cleaning, gap handling, features, alignment.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.validation import DataValidator

logger = setup_logging(__name__)


class PreprocessingPipeline:
    """
    Orchestrates validation, cleaning, gap filling, and feature engineering.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        volatility_window: Optional[int] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.validator = DataValidator(self.settings)
        self.loader = DataLoader(self.settings)
        self.feature_engineer = FeatureEngineer(
            self.settings, volatility_window=volatility_window
        )

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric NaNs with 0 after gap logic; categorical with mode."""
        out = df.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].fillna(0)
        if "category" in out.columns:
            out["category"] = out["category"].astype(str).fillna("unknown")
        return out

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate product-date rows keeping last."""
        key = [self.settings.product_column, self.settings.date_column]
        before = len(df)
        out = df.drop_duplicates(subset=key, keep="last")
        removed = before - len(out)
        if removed > 0:
            logger.info("Removed %d duplicate rows.", removed)
        return out

    def sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by product_id then date."""
        return df.sort_values(
            [self.settings.product_column, self.settings.date_column]
        ).reset_index(drop=True)

    def forward_fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill small gaps per product; mask large consecutive gaps.
        """
        out = df.copy()
        target = self.settings.target_column
        product = self.settings.product_column
        max_gap = self.settings.max_ffill_gap
        large_thresh = self.settings.large_gap_threshold

        filled_parts = []
        for pid, group in out.groupby(product, observed=True):
            g = group.sort_values(self.settings.date_column).copy()
            sales = g[target].copy()
            is_na = sales.isna()
            if not is_na.any():
                filled_parts.append(g)
                continue

            gap_groups = (is_na != is_na.shift()).cumsum()
            for _, idx in sales[is_na].groupby(gap_groups[is_na]):
                gap_len = len(idx)
                if gap_len <= max_gap:
                    sales.loc[idx.index] = np.nan
                else:
                    sales.loc[idx.index] = np.nan

            sales = sales.ffill(limit=max_gap)
            run_length = sales.isna().astype(int).groupby(
                (sales.notna()).cumsum()
            ).transform("sum")
            large_mask = run_length >= large_thresh
            sales.loc[large_mask & sales.isna()] = np.nan
            g[target] = sales
            filled_parts.append(g)

        result = pd.concat(filled_parts, ignore_index=True)
        return result

    def align_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent daily frequency per product where possible."""
        return self.sort_chronologically(df)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline without features."""
        is_valid, errors = self.validator.run_all_checks(df)
        if not is_valid:
            raise ValueError("; ".join(errors))

        out = DataValidator.coerce_types(df, self.settings)
        out = self.remove_duplicates(out)
        out = self.sort_chronologically(out)
        out = self.forward_fill_gaps(out)
        out = self.handle_missing_values(out)
        out = self.align_time_index(out)
        return out

    def run(
        self,
        df: pd.DataFrame,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, list]:
        """
        Execute full preprocessing and feature engineering.

        Returns
        -------
        tuple
            (processed_df, feature_column_names)
        """
        cleaned = self.clean(df)
        featured = self.feature_engineer.transform(cleaned)
        feature_cols = FeatureEngineer.get_feature_columns(
            featured, self.settings.target_column
        )
        featured = featured.dropna(subset=feature_cols + [self.settings.target_column])

        if save:
            self.loader.save_processed(featured)

        logger.info(
            "Preprocessing done. Rows=%d, features=%d",
            len(featured),
            len(feature_cols),
        )
        return featured, feature_cols

    def run_from_path(self, file_path: str, save: bool = True) -> Tuple[pd.DataFrame, list]:
        """Load CSV, preprocess, return result."""
        raw = self.loader.load_csv(file_path)
        return self.run(raw, save=save)
