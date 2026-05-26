"""
Baseline forecasting models: Naive, ARIMA, SARIMA.
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from models.model_utils import BaseForecaster, ModelRegistry

logger = setup_logging(__name__)


class NaiveBaseline(BaseForecaster):
    """Last-value (lag-1) naive forecaster."""

    name = "naive"

    def __init__(self) -> None:
        self.last_value: float = 0.0
        self.history: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "NaiveBaseline":
        self.history = y.copy()
        self.last_value = float(y[-1]) if len(y) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        n = len(X) if X is not None else 1
        return np.full(n, self.last_value)


class ARIMAForecaster(BaseForecaster):
    """ARIMA univariate forecaster."""

    name = "arima"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        settings: Optional[Settings] = None,
    ) -> None:
        self.order = order
        self.settings = settings or get_settings()
        self.model_result: Any = None
        self.fitted_values: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ARIMAForecaster":
        series = pd.Series(y).astype(float)
        try:
            model = ARIMA(series, order=self.order)
            self.model_result = model.fit()
            self.fitted_values = self.model_result.fittedvalues.values
        except Exception as exc:
            logger.warning("ARIMA fit failed (%s). Using mean fallback.", exc)
            self.model_result = None
            self.fitted_values = np.full(len(y), np.nanmean(y))
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        steps = kwargs.get("steps", len(X) if X is not None else 1)
        if self.model_result is None:
            return np.zeros(steps)
        try:
            forecast = self.model_result.forecast(steps=steps)
            return np.asarray(forecast)
        except Exception as exc:
            logger.warning("ARIMA predict failed: %s", exc)
            return np.zeros(steps)

    def predict_in_sample(self) -> np.ndarray:
        """Return fitted in-sample values aligned to training length."""
        if self.fitted_values is not None:
            return np.nan_to_num(self.fitted_values, nan=0.0)
        return np.array([])


class SARIMAForecaster(BaseForecaster):
    """SARIMA seasonal forecaster."""

    name = "sarima"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        settings: Optional[Settings] = None,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.settings = settings or get_settings()
        self.model_result: Any = None
        self.fitted_values: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "SARIMAForecaster":
        series = pd.Series(y).astype(float)
        try:
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_result = model.fit(disp=False)
            self.fitted_values = self.model_result.fittedvalues.values
        except Exception as exc:
            logger.warning("SARIMA fit failed (%s). Using mean fallback.", exc)
            self.model_result = None
            self.fitted_values = np.full(len(y), np.nanmean(y))
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        steps = kwargs.get("steps", len(X) if X is not None else 1)
        if self.model_result is None:
            return np.zeros(steps)
        try:
            forecast = self.model_result.forecast(steps=steps)
            return np.asarray(forecast)
        except Exception as exc:
            logger.warning("SARIMA predict failed: %s", exc)
            return np.zeros(steps)

    def predict_in_sample(self) -> np.ndarray:
        if self.fitted_values is not None:
            return np.nan_to_num(self.fitted_values, nan=0.0)
        return np.array([])


ModelRegistry.register("naive", NaiveBaseline)
ModelRegistry.register("arima", ARIMAForecaster)
ModelRegistry.register("sarima", SARIMAForecaster)
