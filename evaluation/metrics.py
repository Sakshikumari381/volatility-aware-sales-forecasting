"""
Forecast evaluation metrics: RMSLE, RMSE, R².
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score as sklearn_r2


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error.
    """
    y_true = np.maximum(np.asarray(y_true, dtype=float), 0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(sklearn_r2(y_true, y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSLE, RMSE, and R² for predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
