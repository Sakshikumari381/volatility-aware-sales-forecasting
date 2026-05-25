"""
Model utilities: time splits, persistence, hyperparameter tuning, plugin registry.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = setup_logging(__name__)


class BaseForecaster(ABC):
    """Abstract base for pluggable forecasters (LightGBM, LSTM extensions)."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseForecaster":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseForecaster":
        return joblib.load(path)


def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation/test split without shuffling.

    Parameters
    ----------
    df : pd.DataFrame
        Time-sorted dataframe.
    date_col : str
        Date column name.
    train_ratio, val_ratio : float
        Split ratios (test = remainder).

    Returns
    -------
    tuple of DataFrames
        train, validation, test
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    if n < 10:
        raise ValueError("Insufficient rows for time-based split.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test


def split_by_product(
    df: pd.DataFrame,
    product_id: str,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """Filter dataframe for a single product."""
    settings = settings or get_settings()
    col = settings.product_column
    subset = df[df[col].astype(str) == str(product_id)].copy()
    if subset.empty:
        raise ValueError(f"No data for product_id={product_id}")
    return subset.sort_values(settings.date_column)


def save_artifact(obj: Any, filename: str, settings: Optional[Settings] = None) -> Path:
    """Persist object with joblib."""
    settings = settings or get_settings()
    path = settings.saved_models_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact: %s", path)
    return path


def load_artifact(filename: str, settings: Optional[Settings] = None) -> Any:
    """Load joblib artifact."""
    settings = settings or get_settings()
    path = settings.saved_models_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_splits: int = 3,
    settings: Optional[Settings] = None,
) -> XGBRegressor:
    """
    Hyperparameter tuning with TimeSeriesSplit GridSearchCV.
    """
    settings = settings or get_settings()
    param_grid = param_grid or settings.xgb_param_grid

    base = XGBRegressor(
        objective="reg:squarederror",
        random_state=settings.random_state,
        n_jobs=settings.n_jobs,
        verbosity=0,
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        base,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("Best XGB params: %s", search.best_params_)
    return search.best_estimator_


def select_features_by_shap(
    feature_names: List[str],
    shap_values: np.ndarray,
    threshold: float,
) -> List[str]:
    """
    Keep features whose mean |SHAP| exceeds threshold.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    selected = [
        f for f, v in zip(feature_names, mean_abs) if v >= threshold
    ]
    if not selected:
        top_idx = np.argmax(mean_abs)
        selected = [feature_names[top_idx]]
    return selected


class ModelRegistry:
    """Registry for extensible model plugins."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        cls._models[name] = model_class

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered.")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())
