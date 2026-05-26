"""
Forecast and volatility visualization saved as PNG.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)
sns.set_theme(style="whitegrid")


class ForecastPlotter:
    """Generate and save forecast-related charts."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.output_dir = self.settings.outputs_charts_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def _next_path(self, name: str) -> Path:
        self._counter += 1
        return self.output_dir / f"{name}_{self._counter}.png"

    def plot_actual_vs_predicted(
        self,
        dates: Union[np.ndarray, pd.Series],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        filename: Optional[str] = None,
    ) -> Path:
        """Line chart of actual vs predicted sales."""
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(y_true))
        ax.plot(x, y_true, label="Actual", linewidth=2)
        ax.plot(x, y_pred, label="Predicted", linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Sales Quantity")
        ax.legend()
        fig.tight_layout()
        path = (
            self.output_dir / filename
            if filename
            else self._next_path("actual_vs_pred")
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved chart: %s", path)
        return path

    def plot_rolling_volatility(self, df: pd.DataFrame, filename: str = "rolling_volatility.png") -> Path:
        """Plot rolling volatility by product (aggregated mean)."""
        vol_cols = [c for c in df.columns if "rolling_volatility" in c]
        if not vol_cols:
            return self.output_dir / filename
        col = vol_cols[0]
        agg = (
            df.groupby(self.settings.date_column, observed=True)[col]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(agg[self.settings.date_column], agg[col], color="crimson")
        ax.set_title("Rolling Volatility (Mean Across Products)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_residuals(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        title: str = "Residual Analysis",
        filename: Optional[str] = None,
    ) -> Path:
        """Residual distribution and scatter."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual")
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_title("Residuals vs Predicted")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        fig.suptitle(title)
        fig.tight_layout()
        path = (
            self.output_dir / filename
            if filename
            else self._next_path("residuals")
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_forecast_trend(
        self,
        df: pd.DataFrame,
        horizon: int = 14,
        filename: str = "forecast_trend.png",
    ) -> Path:
        """Aggregate sales trend with simple moving average."""
        target = self.settings.target_column
        agg = (
            df.groupby(self.settings.date_column, observed=True)[target]
            .sum()
            .reset_index()
            .sort_values(self.settings.date_column)
        )
        agg["ma_7"] = agg[target].rolling(7, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(agg[self.settings.date_column], agg[target], label="Daily Sales", alpha=0.6)
        ax.plot(
            agg[self.settings.date_column],
            agg["ma_7"],
            label="7-day MA",
            linewidth=2,
        )
        ax.set_title(f"Sales Trend (Horizon context: {horizon} days)")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
