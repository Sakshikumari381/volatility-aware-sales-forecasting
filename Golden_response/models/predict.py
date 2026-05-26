"""
Prediction service loading persisted models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from models.model_utils import load_artifact, split_by_product
from preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = setup_logging(__name__)


class PredictionService:
    """Load trained models and generate forecasts per product."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._models: Dict[str, Any] = {}
        self._feature_cols: List[str] = []
        self._metadata: Dict[str, Any] = {}

    def load_models(self) -> None:
        """Load all persisted model artifacts."""
        from models.train import TrainingOrchestrator

        self._models["xgboost"] = load_artifact(TrainingOrchestrator.ARTIFACT_XGB)
        self._models["hybrid"] = load_artifact(TrainingOrchestrator.ARTIFACT_HYBRID)
        self._models["arima"] = load_artifact(TrainingOrchestrator.ARTIFACT_ARIMA)
        self._models["sarima"] = load_artifact(TrainingOrchestrator.ARTIFACT_SARIMA)
        self._models["naive"] = load_artifact(TrainingOrchestrator.ARTIFACT_NAIVE)
        self._feature_cols = load_artifact(TrainingOrchestrator.ARTIFACT_FEATURES)
        self._metadata = load_artifact(TrainingOrchestrator.ARTIFACT_META)
        logger.info("Models loaded from disk.")

    def _ensure_loaded(self) -> None:
        if not self._models:
            self.load_models()

    def forecast_product(
        self,
        product_id: str,
        model_name: str = "hybrid",
        horizon: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate forecast for a single product.

        Returns
        -------
        dict
            JSON-serializable forecast result.
        """
        self._ensure_loaded()
        horizon = horizon or self.settings.forecast_horizon

        if df is None:
            from preprocessing.data_loader import DataLoader

            loader = DataLoader(self.settings)
            df = loader.load_processed()

        product_df = split_by_product(df, product_id, self.settings)
        pipeline = PreprocessingPipeline(self.settings)
        featured, _ = pipeline.run(product_df, save=False)

        feature_cols = self._feature_cols or [
            c
            for c in featured.columns
            if c in featured.select_dtypes(include=[np.number]).columns
            and c != self.settings.target_column
        ]
        available = [c for c in feature_cols if c in featured.columns]
        X = featured[available].values.astype(np.float32)

        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        if model_name == "xgboost":
            preds = model.predict(X[-horizon:])
        elif model_name == "hybrid":
            preds = model.predict(X[-horizon:])
        else:
            preds = model.predict(X, steps=horizon)

        preds = np.maximum(preds, 0).tolist()
        dates = (
            featured[self.settings.date_column]
            .iloc[-horizon:]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        return {
            "product_id": str(product_id),
            "model": model_name,
            "horizon": horizon,
            "dates": dates,
            "predictions": preds,
            "actual_last": float(
                featured[self.settings.target_column].iloc[-horizon:].mean()
            ),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return stored evaluation metrics from training."""
        self._ensure_loaded()
        return self._metadata.get("metrics", {})
