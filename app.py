"""
Streamlit dashboard for volatility-aware sales forecasting.
All training is delegated to the FastAPI backend — no direct model training here.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config.settings import get_settings
from preprocessing.data_loader import read_sales_csv

settings = get_settings()
API_BASE = settings.api_base_url.rstrip("/")

st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_request(
    method: str,
    endpoint: str,
    json_body: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Call FastAPI with user-friendly error handling."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, params=params, timeout=timeout)
        elif method.upper() == "POST":
            if files:
                resp = requests.post(url, files=files, timeout=timeout)
            else:
                resp = requests.post(url, json=json_body, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API error ({resp.status_code}): {detail}")

        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API at {API_BASE}. "
            "Start the backend: python main.py api"
        )


def safe_image(path: str, caption: str | None = None) -> None:
    """Display image (compatible with all Streamlit versions)."""
    st.image(path, caption=caption)


def render_sidebar() -> Dict[str, Any]:
    """Sidebar controls for pipeline parameters."""
    st.sidebar.header("Pipeline Controls")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 90, 14)
    volatility_window = st.sidebar.slider("Volatility Window", 3, 30, 7)
    shap_threshold = st.sidebar.slider("SHAP Threshold", 0.0, 0.1, 0.01, 0.001)
    arima_p = st.sidebar.number_input("ARIMA p", 0, 5, 1)
    arima_d = st.sidebar.number_input("ARIMA d", 0, 2, 1)
    arima_q = st.sidebar.number_input("ARIMA q", 0, 5, 1)
    model_choice = st.sidebar.selectbox(
        "Forecast Model",
        ["hybrid", "xgboost", "arima", "sarima", "naive"],
    )
    product_id = st.sidebar.text_input("Product ID for forecast", "P001")
    return {
        "forecast_horizon": forecast_horizon,
        "volatility_window": volatility_window,
        "shap_threshold": shap_threshold,
        "arima_order": [int(arima_p), int(arima_d), int(arima_q)],
        "model_choice": model_choice,
        "product_id": product_id,
    }


def main() -> None:
    st.title("Volatility-Aware AI Sales Forecasting")
    st.markdown(
        "E-commerce demand forecasting with explainable ML. "
        "Training runs on the **FastAPI** backend only."
    )

    try:
        requests.get(f"{API_BASE}/health", timeout=3).raise_for_status()
    except requests.exceptions.RequestException:
        st.error(
            f"**Backend not running.** Start it first in a separate terminal:\n\n"
            f"`cd {settings.project_root}` then `python main.py api`\n\n"
            f"Expected API at {API_BASE}"
        )
        st.stop()

    params = render_sidebar()

    tab_upload, tab_pipeline, tab_results, tab_explain = st.tabs(
        ["Upload Data", "Run Pipeline", "Results", "Explainability"]
    )

    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "train_result" not in st.session_state:
        st.session_state.train_result = None

    with tab_upload:
        st.subheader("Upload Sales Dataset")
        st.caption(
            "Required columns: date, product_id, category, sales_quantity, price, "
            "promotional_flag. Names are case-insensitive (e.g. `Date`, `Qty`, `SKU`)."
        )
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                uploaded.seek(0)
                df = read_sales_csv(uploaded)
                st.session_state.uploaded_df = df
                st.success(f"Loaded {len(df):,} rows locally for preview.")

                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                with st.spinner("Uploading to API..."):
                    result = api_request("POST", "/upload", files=files)
                st.success(result.get("message", "Upload complete."))
                st.dataframe(df.head(20), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    daily = df.groupby("date")["sales_quantity"].sum().reset_index()
                    fig = px.line(daily, x="date", y="sales_quantity", title="Daily Sales")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if "category" in df.columns:
                        cat = df.groupby("category")["sales_quantity"].sum().reset_index()
                        fig2 = px.bar(cat, x="category", y="sales_quantity", title="By Category")
                        st.plotly_chart(fig2, use_container_width=True)
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    with tab_pipeline:
        st.subheader("Run ML Pipeline")
        st.info("This triggers training on the FastAPI server (XGBoost, ARIMA, SARIMA, Naive, Hybrid).")
        if st.button("Run Pipeline", type="primary"):
            try:
                body = {
                    "forecast_horizon": params["forecast_horizon"],
                    "volatility_window": params["volatility_window"],
                    "shap_threshold": params["shap_threshold"],
                    "arima_order": params["arima_order"],
                }
                with st.spinner("Training models via API (may take a few minutes)..."):
                    result = api_request("POST", "/train", json_body=body, timeout=1200)
                st.session_state.train_result = result
                st.success(result.get("message", "Training complete."))
            except Exception as exc:
                st.error(str(exc))

    with tab_results:
        st.subheader("Forecast Results & Metrics")
        result = st.session_state.train_result
        if result is None:
            try:
                cached = api_request("GET", "/metrics")
                if cached.get("metrics"):
                    st.session_state.train_result = cached
                    result = cached
            except Exception:
                pass
        if result is None:
            st.warning("Run the pipeline first to see metrics.")
        else:
            metrics = result.get("metrics", {})
            if metrics:
                rows = [{"model": k, **v} for k, v in metrics.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            comparison = result.get("comparison", [])
            if comparison:
                st.markdown("### Model Ranking")
                st.dataframe(pd.DataFrame(comparison), use_container_width=True)

            charts_dir = settings.outputs_charts_dir
            if charts_dir.exists() and any(charts_dir.glob("*.png")):
                st.markdown("### Saved Charts")
                for img in sorted(charts_dir.glob("*.png"))[:8]:
                    safe_image(str(img), caption=img.name)

        if st.button("Get Product Forecast"):
            try:
                fc = api_request(
                    "GET",
                    f"/forecast/{params['product_id']}",
                    params={
                        "model_name": params["model_choice"],
                        "horizon": params["forecast_horizon"],
                    },
                )
                forecast = fc.get("forecast", {})
                if forecast.get("dates") and forecast.get("predictions"):
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=forecast["dates"],
                            y=forecast["predictions"],
                            mode="lines+markers",
                            name="Forecast",
                        )
                    )
                    fig.update_layout(
                        title=f"Forecast — {params['product_id']} ({params['model_choice']})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    with tab_explain:
        st.subheader("Explainability & Volatility")
        result = st.session_state.train_result
        shap_dir = settings.outputs_shap_dir
        if result:
            shap_dir = Path(result.get("shap_dir", shap_dir))
            features = result.get("shap_features", [])
            if features:
                st.markdown("**SHAP-selected features:**")
                st.write(", ".join(features))

        if shap_dir.exists():
            for img in sorted(shap_dir.glob("*.png")):
                safe_image(str(img), caption=img.name)
        else:
            st.info("SHAP plots appear after training.")

        df = st.session_state.uploaded_df
        if df is not None and "sales_quantity" in df.columns:
            st.markdown("### Volatility Preview")
            vol_df = df.copy()
            vol_df["date"] = pd.to_datetime(vol_df["date"])
            vol_df = vol_df.sort_values("date")
            vol_df["rolling_vol"] = (
                vol_df.groupby("product_id")["sales_quantity"]
                .transform(lambda s: s.rolling(7, min_periods=2).std())
            )
            agg = vol_df.groupby("date")["rolling_vol"].mean().reset_index()
            fig = px.line(agg, x="date", y="rolling_vol", title="7-day Rolling Volatility")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
