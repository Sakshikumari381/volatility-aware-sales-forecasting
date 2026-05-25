"""
FastAPI backend for volatility-aware sales forecasting.
"""

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config.logging_config import setup_logging
from config.settings import get_settings
from evaluation.comparison import ModelComparator
from models.predict import PredictionService
from models.train import TrainingOrchestrator
from preprocessing.data_loader import DataLoader
from preprocessing.validation import DataValidator
from reporting.reporter import ForecastReporter

logger = setup_logging(__name__)
settings = get_settings()

app = FastAPI(
    title="Volatility-Aware Sales Forecasting API",
    description="AI-powered e-commerce demand forecasting with explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_training_lock = asyncio.Lock()
_last_training_state: Optional[Dict[str, Any]] = None
_uploaded_file_path: Optional[Path] = None


class TrainRequest(BaseModel):
    """Training request parameters."""

    forecast_horizon: int = Field(default=14, ge=1, le=90)
    volatility_window: int = Field(default=7, ge=3, le=30)
    shap_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    arima_order: Tuple[int, int, int] = Field(default=(1, 1, 1))

    @field_validator("arima_order")
    @classmethod
    def validate_arima_order(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("arima_order must be three non-negative integers.")
        return v


class ForecastQuery(BaseModel):
    model_config = {"protected_namespaces": ()}
    forecast_model: str = Field(default="hybrid")
    horizon: int = Field(default=14, ge=1, le=90)


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "volatility-aware-forecasting"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload CSV sales dataset for training.
    """
    global _uploaded_file_path

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        loader = DataLoader(settings)
        path = loader.save_upload(content, file.filename)

        df = loader.load_csv(path, force_reload=True)
        validator = DataValidator(settings)
        is_valid, errors = validator.run_all_checks(df)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"errors": errors})

        _uploaded_file_path = path
        preview_df = df.head(10).copy()
        for col in preview_df.columns:
            preview_df[col] = preview_df[col].map(
                lambda x: "" if pd.isna(x) else str(x)
            )
        preview = preview_df.to_dict(orient="records")

        return {
            "status": "success",
            "message": "File uploaded successfully.",
            "filename": file.filename,
            "path": str(path),
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train")
async def train_models(request: TrainRequest) -> Dict[str, Any]:
    """
    Train all forecasting models via ML pipeline (async wrapper).
    """
    global _last_training_state, _uploaded_file_path

    async with _training_lock:
        try:
            if _uploaded_file_path is None:
                processed_path = settings.data_processed_dir / "processed_data.parquet"
                if processed_path.exists():
                    file_path = str(processed_path)
                    use_parquet = True
                else:
                    raw_files = list(settings.data_raw_dir.glob("*.csv"))
                    if not raw_files:
                        raise HTTPException(
                            status_code=400,
                            detail="No dataset uploaded. POST /upload first.",
                        )
                    file_path = str(raw_files[0])
                    use_parquet = False
            else:
                file_path = str(_uploaded_file_path)
                use_parquet = False

            orchestrator = TrainingOrchestrator(
                settings=settings,
                forecast_horizon=request.forecast_horizon,
                volatility_window=request.volatility_window,
                shap_threshold=request.shap_threshold,
                arima_order=request.arima_order,
            )

            loop = asyncio.get_event_loop()
            if use_parquet:
                df = pd.read_parquet(file_path)
                state = await loop.run_in_executor(
                    None, partial(orchestrator.train_all, df=df)
                )
            else:
                state = await loop.run_in_executor(
                    None, partial(orchestrator.train_all, file_path=file_path)
                )

            reporter = ForecastReporter(settings)
            reporter.generate_html_report(
                state.metrics,
                state.comparison_table,
                extra={
                    "forecast_horizon": request.forecast_horizon,
                    "volatility_window": request.volatility_window,
                    "shap_threshold": request.shap_threshold,
                },
            )
            reporter.generate_text_summary(state.metrics)

            comparator = ModelComparator()
            response = {
                "status": "success",
                "message": "Training completed.",
                "metrics": state.metrics,
                "comparison": comparator.to_dict(state.comparison_table),
                "shap_features": state.shap_selected_features,
                "feature_count": len(state.feature_columns),
                "charts_dir": str(settings.outputs_charts_dir),
                "shap_dir": str(settings.outputs_shap_dir),
            }
            _last_training_state = response
            return response

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Training failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/forecast/{product_id}")
async def get_forecast(
    product_id: str,
    model_name: str = "hybrid",
    horizon: int = 14,
) -> Dict[str, Any]:
    """
    Generate forecast for a specific product.
    """
    try:
        service = PredictionService(settings)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                service.forecast_product,
                product_id=product_id,
                model_name=model_name,
                horizon=horizon,
            ),
        )
        return {"status": "success", "forecast": result}
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Models not trained. Call POST /train first.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Return evaluation metrics from last training run.
    """
    try:
        if _last_training_state and "metrics" in _last_training_state:
            return {
                "status": "success",
                "metrics": _last_training_state["metrics"],
                "comparison": _last_training_state.get("comparison", []),
                "charts_dir": _last_training_state.get(
                    "charts_dir", str(settings.outputs_charts_dir)
                ),
                "shap_dir": _last_training_state.get(
                    "shap_dir", str(settings.outputs_shap_dir)
                ),
            }
        service = PredictionService(settings)
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, service.get_metrics)
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="No metrics available. Train models first.",
            )
        settings.ensure_directories()
        return {
            "status": "success",
            "metrics": metrics,
            "charts_dir": str(settings.outputs_charts_dir),
            "shap_dir": str(settings.outputs_shap_dir),
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="No trained models found.",
        ) from exc
    except Exception as exc:
        logger.exception("Metrics retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def create_app() -> FastAPI:
    """Factory for uvicorn."""
    settings.ensure_directories()
    return app
