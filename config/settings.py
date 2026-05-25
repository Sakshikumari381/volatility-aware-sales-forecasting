"""
Application settings loaded from environment variables and defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Central configuration for paths, API, and ML defaults."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed"
    )
    data_cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    saved_models_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "saved_models"
    )
    outputs_charts_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "charts"
    )
    outputs_shap_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "shap"
    )
    outputs_reports_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "reports"
    )
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "API_BASE_URL", "http://127.0.0.1:8000"
        )
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "forecasting.log")
    )

    required_columns: Tuple[str, ...] = (
        "date",
        "product_id",
        "category",
        "sales_quantity",
        "price",
        "promotional_flag",
    )
    derived_columns: Tuple[str, ...] = ("day_of_week", "month")

    target_column: str = "sales_quantity"
    date_column: str = "date"
    product_column: str = "product_id"

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    max_ffill_gap: int = 3
    large_gap_threshold: int = 7

    volatility_window: int = 7
    forecast_horizon: int = 14
    shap_threshold: float = 0.01
    shap_top_k: int = 15

    arima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_order: Tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)

    xgb_param_grid: dict = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    )

    random_state: int = 42
    n_jobs: int = -1

    def ensure_directories(self) -> None:
        """Create all required project directories if missing."""
        for path in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_cache_dir,
            self.saved_models_dir,
            self.outputs_charts_dir,
            self.outputs_shap_dir,
            self.outputs_reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
