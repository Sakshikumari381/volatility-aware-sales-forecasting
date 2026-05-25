"""
Training orchestration for all forecasting models with SHAP pruning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from config.logging_config import setup_logging
from config.settings import Settings, get_settings
from evaluation.comparison import ModelComparator
from evaluation.metrics import compute_all_metrics
from models.baseline_models import ARIMAForecaster, NaiveBaseline, SARIMAForecaster
from models.hybrid_model import HybridXGBARIMA
from models.model_utils import (
    save_artifact,
    select_features_by_shap,
    time_based_split,
    tune_xgboost,
)
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from visualization.plots import ForecastPlotter
from visualization.shap_visualizer import SHAPVisualizer

logger = setup_logging(__name__)


@dataclass
class TrainingState:
    """Holds trained models and metadata."""

    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    comparison_table: Optional[pd.DataFrame] = None
    processed_df: Optional[pd.DataFrame] = None
    shap_selected_features: List[str] = field(default_factory=list)


class TrainingOrchestrator:
    """
    Full ML training pipeline: preprocess, train, SHAP prune, evaluate, persist.
    """

    ARTIFACT_XGB = "xgboost_model.joblib"
    ARTIFACT_ARIMA = "arima_model.joblib"
    ARTIFACT_SARIMA = "sarima_model.joblib"
    ARTIFACT_NAIVE = "naive_model.joblib"
    ARTIFACT_HYBRID = "hybrid_model.joblib"
    ARTIFACT_META = "training_metadata.joblib"
    ARTIFACT_FEATURES = "feature_columns.joblib"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        forecast_horizon: int = 14,
        volatility_window: int = 7,
        shap_threshold: float = 0.01,
        arima_order: Tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self.settings = settings or get_settings()
        self.forecast_horizon = forecast_horizon
        self.volatility_window = volatility_window
        self.shap_threshold = shap_threshold
        self.arima_order = arima_order
        self.state = TrainingState()
        self.plotter = ForecastPlotter(self.settings)
        self.shap_viz = SHAPVisualizer(self.settings)

    def _prepare_splits(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, ...]:
        train_df, val_df, test_df = time_based_split(
            df,
            self.settings.date_column,
            self.settings.train_ratio,
            self.settings.val_ratio,
        )
        target = self.settings.target_column

        def _xy(split: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            X = split[feature_cols].values.astype(np.float32)
            y = split[target].values.astype(np.float32)
            return X, y

        X_train, y_train = _xy(train_df)
        X_val, y_val = _xy(val_df)
        X_test, y_test = _xy(test_df)
        return X_train, y_train, X_val, y_val, X_test, y_test, test_df

    def _train_xgboost_with_shap(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_cols: List[str],
    ) -> Tuple[XGBRegressor, List[str], np.ndarray]:
        model = tune_xgboost(X_train, y_train, settings=self.settings)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            selected = select_features_by_shap(
                feature_cols, shap_values, self.shap_threshold
            )
            self.shap_viz.save_summary(shap_values, feature_cols, "xgboost_summary")
            self.shap_viz.save_top_features(
                shap_values, feature_cols, top_k=self.settings.shap_top_k
            )
            logger.info("SHAP selected %d / %d features.", len(selected), len(feature_cols))
        except Exception as exc:
            logger.warning("SHAP failed (%s). Using all features.", exc)
            selected = feature_cols
            shap_values = np.zeros((len(X_train), len(feature_cols)))

        if len(selected) < len(feature_cols):
            idx = [feature_cols.index(f) for f in selected]
            X_train_sel = X_train[:, idx]
            X_val_sel = X_val[:, idx]
            model = tune_xgboost(X_train_sel, y_train, settings=self.settings)
            return model, selected, shap_values

        return model, selected, shap_values

    def train_all(
        self,
        df: Optional[pd.DataFrame] = None,
        file_path: Optional[str] = None,
    ) -> TrainingState:
        """
        Run full training pipeline on dataframe or CSV path.
        """
        pipeline = PreprocessingPipeline(
            self.settings, volatility_window=self.volatility_window
        )
        if df is None:
            if file_path is None:
                raise ValueError("Either df or file_path must be provided.")
            processed, feature_cols = pipeline.run_from_path(file_path)
        else:
            processed, feature_cols = pipeline.run(df)

        self.state.processed_df = processed
        self.state.feature_columns = feature_cols

        X_train, y_train, X_val, y_val, X_test, y_test, test_df = self._prepare_splits(
            processed, feature_cols
        )

        xgb_model, selected_features, _ = self._train_xgboost_with_shap(
            X_train, y_train, X_val, y_val, feature_cols
        )
        self.state.shap_selected_features = selected_features

        idx = [feature_cols.index(f) for f in selected_features]
        X_train_sel = X_train[:, idx]
        X_val_sel = X_val[:, idx]
        X_test_sel = X_test[:, idx]

        xgb_model.fit(
            np.vstack([X_train_sel, X_val_sel]),
            np.concatenate([y_train, y_val]),
        )
        self.state.models["xgboost"] = xgb_model

        naive = NaiveBaseline()
        naive.fit(X_train, y_train)
        self.state.models["naive"] = naive

        arima = ARIMAForecaster(order=self.arima_order, settings=self.settings)
        arima.fit(X_train, y_train)
        self.state.models["arima"] = arima

        sarima = SARIMAForecaster(
            order=self.settings.arima_order,
            seasonal_order=self.settings.sarima_seasonal_order,
            settings=self.settings,
        )
        sarima.fit(X_train, y_train)
        self.state.models["sarima"] = sarima

        hybrid = HybridXGBARIMA(
            arima_order=self.arima_order,
            settings=self.settings,
            use_tuning=False,
        )
        hybrid.xgb_model = xgb_model
        hybrid.feature_names = selected_features
        xgb_pred_train = xgb_model.predict(X_train_sel)
        residuals = y_train - xgb_pred_train
        hybrid.arima_model = ARIMAForecaster(order=self.arima_order)
        hybrid.arima_model.fit(X_train_sel, residuals)
        self.state.models["hybrid"] = hybrid

        predictions: Dict[str, np.ndarray] = {
            "xgboost": xgb_model.predict(X_test_sel),
            "naive": naive.predict(X_test),
            "arima": self._align_length(arima.predict(X_test, steps=len(y_test)), y_test),
            "sarima": self._align_length(
                sarima.predict(X_test, steps=len(y_test)), y_test
            ),
            "hybrid": hybrid.predict(X_test_sel),
        }

        for name, preds in predictions.items():
            self.state.metrics[name] = compute_all_metrics(y_test, preds)

        comparator = ModelComparator()
        self.state.comparison_table = comparator.build_table(
            self.state.metrics
        )
        comparator.save_table(
            self.state.comparison_table,
            self.settings.outputs_reports_dir / "model_comparison.csv",
        )

        self._save_artifacts(selected_features)
        self._generate_charts(y_test, predictions, test_df)
        logger.info("Training pipeline completed successfully.")
        return self.state

    @staticmethod
    def _align_length(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(preds) == len(y):
            return preds
        if len(preds) > len(y):
            return preds[-len(y) :]
        padded = np.full(len(y), preds[-1] if len(preds) else 0.0)
        padded[-len(preds) :] = preds
        return padded

    def _save_artifacts(self, feature_cols: List[str]) -> None:
        save_artifact(self.state.models.get("xgboost"), self.ARTIFACT_XGB)
        save_artifact(self.state.models.get("arima"), self.ARTIFACT_ARIMA)
        save_artifact(self.state.models.get("sarima"), self.ARTIFACT_SARIMA)
        save_artifact(self.state.models.get("naive"), self.ARTIFACT_NAIVE)
        save_artifact(self.state.models.get("hybrid"), self.ARTIFACT_HYBRID)
        save_artifact(feature_cols, self.ARTIFACT_FEATURES)
        save_artifact(
            {
                "metrics": self.state.metrics,
                "shap_features": self.state.shap_selected_features,
                "forecast_horizon": self.forecast_horizon,
            },
            self.ARTIFACT_META,
        )

    def _generate_charts(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        test_df: pd.DataFrame,
    ) -> None:
        dates = test_df[self.settings.date_column].values
        for model_name, preds in predictions.items():
            self.plotter.plot_actual_vs_predicted(
                dates, y_true, preds, title=f"{model_name} - Test Set"
            )
            self.plotter.plot_residuals(preds, y_true, title=f"{model_name} Residuals")
        if self.state.processed_df is not None:
            self.plotter.plot_rolling_volatility(self.state.processed_df)
            self.plotter.plot_forecast_trend(
                self.state.processed_df, horizon=self.forecast_horizon
            )
