"""
Hybrid XGBoost + ARIMA residual correction model.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from xgboost import XGBRegressor

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from models.baseline_models import ARIMAForecaster
from models.model_utils import BaseForecaster, ModelRegistry, tune_xgboost

logger = setup_logging(__name__)


class HybridXGBARIMA(BaseForecaster):
    """
    Train XGBoost first, then ARIMA on residuals.
    final_prediction = xgb_pred + arima_residual_pred
    """

    name = "hybrid"

    def __init__(
        self,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
        settings: Optional[Settings] = None,
        use_tuning: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.arima_order = arima_order
        self.use_tuning = use_tuning
        self.xgb_model: Optional[XGBRegressor] = None
        self.arima_model: Optional[ARIMAForecaster] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "HybridXGBARIMA":
        self.feature_names = feature_names or []

        if self.use_tuning and len(X) >= 30:
            self.xgb_model = tune_xgboost(X, y, settings=self.settings)
        else:
            self.xgb_model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.settings.random_state,
                n_jobs=self.settings.n_jobs,
                verbosity=0,
            )
            self.xgb_model.fit(X, y)

        xgb_pred = self.xgb_model.predict(X)
        residuals = y - xgb_pred

        self.arima_model = ARIMAForecaster(order=self.arima_order)
        self.arima_model.fit(X, residuals)
        logger.info("Hybrid model training complete.")
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self.xgb_model is None:
            raise RuntimeError("Hybrid model not fitted.")
        xgb_pred = self.xgb_model.predict(X)
        steps = len(X)
        arima_pred = self.arima_model.predict(X, steps=steps) if self.arima_model else 0
        if isinstance(arima_pred, np.ndarray) and len(arima_pred) != len(xgb_pred):
            arima_pred = np.resize(arima_pred, len(xgb_pred))
        return xgb_pred + arima_pred

    def predict_xgb_only(self, X: np.ndarray) -> np.ndarray:
        return self.xgb_model.predict(X)

    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.xgb_model.predict(X)


ModelRegistry.register("hybrid", HybridXGBARIMA)
