"""
Model comparison tables and ranked summaries.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.logging_config import setup_logging

logger = setup_logging(__name__)


class ModelComparator:
    """Build comparison tables and rank models by RMSLE."""

    METRIC_COLUMNS = ["rmsle", "rmse", "r2"]

    def build_table(self, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create comparison DataFrame from per-model metrics dict.
        """
        rows = []
        for model_name, model_metrics in metrics.items():
            row = {"model": model_name}
            row.update(model_metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty and "rmsle" in df.columns:
            df["rank"] = df["rmsle"].rank(method="min").astype(int)
            df = df.sort_values("rank")
        return df

    def ranked_summary(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Return models sorted by rank (best RMSLE first)."""
        if "rank" in comparison_df.columns:
            return comparison_df.sort_values("rank")
        return comparison_df.sort_values("rmsle")

    @staticmethod
    def save_table(df: pd.DataFrame, path: Path) -> None:
        """Save comparison table to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved comparison table to %s", path)

    def to_dict(self, comparison_df: pd.DataFrame) -> Dict:
        """Convert to JSON-serializable dict."""
        return comparison_df.to_dict(orient="records")
