# Volatility-Aware Sales Forecasting System
---

## Prompt
I'm a data scientist building a volatility-aware sales forecasting system for e-commerce and retail platforms. This document defines exactly what to build, how it should behave, and what I expect at the end. Read it fully before writing a single line of code.


---

## Context and Role

As a Data Scientist you will be developing the retail analytics platform. You need to design an interpretable and scalable volatility-based system for forecasting sales in order to minimize the difference between real forecasting and predicted forecasting. The solution will be expected to handle classical e-commerce transaction data, perform feature engineering, build a hybrid machine learning model pipeline, and produce a daily forecast of sales.

---

## Objective

Retail and e-commerce platforms lose money in two ways: overstocking products that don't move, and understocking products that sell out. Both problems come from bad forecasts. Bad forecasts come from models that ignore volatility — a product that sells 10 units most weeks but occasionally spikes to 200 needs a different treatment than one that sells 50 units every single week without fail.

The core engineering challenge here is building a hybrid model that handles both patterns. XGBoost captures non-linear feature relationships. ARIMA corrects the residuals XGBoost leaves behind. SHAP keeps the feature space honest by pruning noise. The system needs to be interpretable enough that a business analyst can understand why a forecast changed, and scalable enough to handle thousands of SKUs without row-by-row processing.

Everything else is secondary to getting those four things right: **volatility-aware features**, a **clean hybrid pipeline**, **residual correction**, and **meaningful explainability**.

---

## Tech Stack

> Use exactly this. Don't substitute anything.

| Tool | Version / Purpose |
|------|-------------------|
| **Python** | 3.11+ — handles 5M+ daily observations; interpreter speed gains directly reduce pipeline runtime |
| **Pandas + NumPy** | Time-series resampling, per-SKU groupby, forward-fill gap handling, chronological sorting across products |
| **Scikit-learn** | Preprocessing, splitting, metrics computation |
| **XGBoost** | Primary ML regressor |
| **Statsmodels** | Baseline models — ARIMA and SARIMA — used for initial computation |
| **SHAP** | Explainability of feature importance and pruning |
| **Matplotlib + Seaborn** | Saving all visualizations as PNG |
| **Streamlit** | Interactive UI dashboard |
| **FastAPI** | REST API — runs independently from Streamlit |
| **Argparse** | CLI entry point |
| **Logging** | Configurable levels: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| **JSON** | Report output |
| **Parquet** | Processed data cache |

---

## Performance Targets

> These are hard requirements, not suggestions.

- The pipeline must handle up to **5 million daily observations** across thousands of SKUs without row-by-row iteration — vectorized pandas/numpy only.
- Processed data must be **cached in Parquet format** — rerunning the pipeline on the same input should skip re-preprocessing entirely.
- All random operations must use a **fixed seed** set at module initialization — results must be fully reproducible across runs.
- The model architecture must be designed so that swapping XGBoost for LightGBM, or ARIMA for LSTM, **requires changing one module** — not touching the pipeline.

---

## Project Structure

```
ai-demand-forecasting/
├── main.py                  # CLI entry point — argparse, full pipeline execution
├── app.py                   # Streamlit UI dashboard
├── api.py                   # FastAPI REST API — runs independently
├── data_loader.py           # CSV ingestion, schema validation, date parsing
├── feature_engineering.py   # All feature creation — lags, rolling stats, volatility
├── models.py                # Model definitions — XGBoost, ARIMA, SARIMA, Naive
├── evaluation.py            # Metrics, comparison tables, best-model selection
├── visualization.py         # All plots — saved to /outputs/plots/ as PNG
├── reporter.py              # JSON report generation
├── baseline.py              # Standalone baseline models (ARIMA, SARIMA, Naive)
├── data/
│   └── transactions.csv     # Input dataset
├── outputs/
│   ├── plots/               # All saved PNG charts
│   ├── reports/             # JSON metric reports
│   └── cache/               # Parquet-cached processed data
├── .env.example
├── requirements.txt         # Pinned versions
└── README.md
```

> Modules do one thing. `feature_engineering.py` doesn't train models. `models.py` doesn't generate plots. `app.py` calls services — it doesn't contain business logic. If a function is doing two unrelated things, split it.

---

## Input Data

### Transaction Dataset (CSV)

```
date, product_id, category, sales_quantity, price, promotional_flag, day_of_week, month
```

| Field | Type | Notes |
|-------|------|-------|
| `date` | DATE | ISO-8601 format — `YYYY-MM-DD` |
| `product_id` | VARCHAR | SKU identifier |
| `category` | VARCHAR | Product category |
| `sales_quantity` | INT | Target variable — must be `>= 0` |
| `price` | DECIMAL | Must be `> 0` |
| `promotional_flag` | BOOLEAN | `0` or `1` |
| `day_of_week` | INT | `0–6` |
| `month` | INT | `1–12` |

> Invalid rows — negative quantities, unparseable dates, prices of zero or below — are **dropped and logged**. They are never silently passed through.

---

## Data Preprocessing

### Cleaning

- Drop exact duplicate rows
- Parse and validate all dates — rows with unparseable dates are logged and dropped, not coerced
- Sort chronologically by `product_id` then `date`
- Forward-fill small gaps (up to 3 consecutive missing values per series)
- Drop series segments with more than 3 consecutive missing values — don't interpolate over large holes

### Feature Engineering

All features are created in `feature_engineering.py`. Nothing else touches this logic.

**Lag features** — per `product_id`:

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales at `t-1` |
| `lag_7` | Sales at `t-7` |
| `lag_14` | Sales at `t-14` |

**Rolling statistics** — per `product_id`:

| Feature | Description |
|---------|-------------|
| `rolling_mean_7` | 7-day rolling mean |
| `rolling_mean_14` | 14-day rolling mean |
| `rolling_std_7` | 7-day rolling standard deviation |
| `rolling_std_14` | 14-day rolling standard deviation |

**Volatility:**

| Feature | Description |
|---------|-------------|
| `volatility` | Rolling standard deviation over a configurable window (default: 7 days) — this is the primary volatility signal |

**Seasonal features:**

| Feature | Description |
|---------|-------------|
| `day_of_week` | Already in input — keep as-is |
| `month` | Already in input — keep as-is |
| `is_weekend` | `1` if `day_of_week` is `5` or `6`, else `0` |

**Promotion interactions:**

| Feature | Description |
|---------|-------------|
| `promo_x_lag1` | `promotional_flag * lag_1` |
| `promo_x_lag7` | `promotional_flag * lag_7` |
| `promo_x_lag14` | `promotional_flag * lag_14` |

> All features must be on the same time index before any model sees them. Rows with NaN from lag/rolling calculations are **dropped** — no imputation.

---

## Model Pipeline

### 1. XGBoost Regressor

Train on all engineered features. **Time-based split only** — no shuffling, ever.

**Split:**

| Set | Range |
|-----|-------|
| Train | First 70% of timeline per product |
| Validation | Next 15% |
| Test | Final 15% |

**Hyperparameters to tune:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

Use validation set for early stopping. Test set is for final evaluation only — it's never touched during training or tuning.

### 2. SHAP Feature Selection

- Compute SHAP values on the trained XGBoost model using the validation set
- Drop features whose mean absolute SHAP value falls below a configurable threshold (default: `0.01`)
- Log which features were kept and which were pruned
- Retrain XGBoost on the pruned feature set — this retrained model is the one used downstream

### 3. ARIMA Residual Correction

Compute residuals on the training set:

```
r_t = y_t - ŷ_ML
```

Train ARIMA on those residuals. Final prediction:

```
ŷ_final = ŷ_ML + r̂_t
```

ARIMA order is configurable: `p`, `d`, `q` — defaults `(1, 1, 1)`. If ARIMA fails to converge, log a warning and fall back to `ŷ_final = ŷ_ML`. Don't crash the pipeline.

### 4. Baselines (`baseline.py`)

| Model | Description |
|-------|-------------|
| **Naive** | Yesterday's sales as today's forecast |
| **ARIMA standalone** | Trained directly on `sales_quantity` — no residual correction |
| **SARIMA** | Trained with seasonal order `(1, 1, 1, 7)` — weekly seasonality |

All baselines are evaluated on the same test split as the hybrid model.

---

## Evaluation

All models are evaluated on the held-out test set only.

| Metric | Notes |
|--------|-------|
| **RMSLE** | Primary metric — penalizes underprediction more than overprediction |
| **RMSE** | Secondary |
| **R²** | Goodness of fit |

Generate a comparison table across all models (Naive, ARIMA, SARIMA, Hybrid). Best model is selected by **lowest RMSLE**. Log the winner.

---

## API

FastAPI. Runs independently from Streamlit — `uvicorn api:app` should start it without touching `app.py`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Accepts a CSV file, validates schema, caches to Parquet |
| `POST` | `/train` | Triggers the full pipeline for all products |
| `GET` | `/forecast/{product_id}` | Returns forecast JSON for a specific product |
| `GET` | `/metrics` | Returns the last evaluation metrics for all models |

Every response is a structured JSON object. Success responses have a `data` key. Error responses have `error_code`, `message`, and `status_code`. No raw strings, no bare arrays, no `200` status for failures.

Request validation via **Pydantic v2**. If the uploaded CSV is missing required columns, return a `400` with a list of missing fields — don't crash.

---

## UI (Streamlit)

Four sections, in this order:

### 1. File Upload

- Upload CSV
- On upload: display a preview table and a sales trend line chart for a user-selected `product_id`

### 2. Model Training

- Sidebar controls: forecast horizon, volatility window size, SHAP pruning threshold, ARIMA order (`p`, `d`, `q`)
- A single **"Run Pipeline"** button that executes everything end-to-end
- Progress indicator while running

### 3. Results & Forecasting

- Line chart: actual vs. baseline predictions vs. hybrid model forecast
- Model comparison table with RMSLE, RMSE, R² for all models

### 4. Explainability

- SHAP feature importance bar chart (top 15 features)
- Rolling volatility chart for the selected product

---

## Outputs

### PNG Charts (saved to `outputs/plots/`)

| File | Description |
|------|-------------|
| `actual_vs_predicted_all_models.png` | Actual sales overlaid with all model predictions |
| `rolling_volatility_{product_id}.png` | Per-product volatility over time |
| `shap_top15.png` | Top 15 features by mean absolute SHAP value |
| `arima_residuals.png` | Actual vs. fitted residuals from the ARIMA correction step |

### JSON Reports (saved to `outputs/reports/`)

| File | Description |
|------|-------------|
| `metrics_report.json` | Evaluation metrics for all models, best model flagged |
| `forecast_{product_id}.json` | Forecast output per product, date-indexed |

---

## Error Handling

Every error path is logged. Stack traces never reach the user — display a clean message instead.

| Error | Behavior |
|-------|----------|
| Missing or unreadable CSV | Log `ERROR`, return message: `"File could not be read."` |
| Missing required columns | Log `ERROR`, list missing columns, halt pipeline |
| Invalid date formats | Log `WARNING`, drop affected rows, continue |
| Negative sales quantities | Log `WARNING`, drop affected rows, continue |
| ARIMA convergence failure | Log `WARNING`, fall back to ML-only prediction |
| SHAP computation error | Log `WARNING`, skip pruning, use full feature set |

Logging is configurable via `LOG_LEVEL` env var — `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default is `INFO`. **No `print` statements in production code** — logging only.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `RANDOM_SEED` | `42` | Global random seed for reproducibility |
| `DEFAULT_VOLATILITY_WINDOW` | `7` | Rolling window size for volatility calculation |
| `SHAP_PRUNING_THRESHOLD` | `0.01` | Minimum mean absolute SHAP value to keep a feature |
| `ARIMA_ORDER` | `1,1,1` | ARIMA (p, d, q) order |
| `FORECAST_HORIZON_DAYS` | `30` | Number of days to forecast |
| `CACHE_DIR` | `outputs/cache` | Directory for Parquet-cached data |
| `PLOT_OUTPUT_DIR` | `outputs/plots` | Directory for PNG chart output |
| `REPORT_OUTPUT_DIR` | `outputs/reports` | Directory for JSON report output |

---

## Deliverables

A **complete, working codebase** — not a skeleton. Every file listed in the project structure must exist and contain real, runnable code.

- All source modules under the root directory
- `data/transactions.csv` — a sample dataset with at least 5 products, 365 days each, including some promotional periods and volatility spikes
- `outputs/` directory with at least one run's worth of PNGs and JSON reports committed
- `requirements.txt` — pinned versions; `pip install -r requirements.txt` works on a clean Python 3.11 env
- `.env.example` — every variable above with a one-line comment
- `README.md` — local setup, how to run the CLI, how to start the Streamlit app, how to start the API, how to run a full pipeline end-to-end
