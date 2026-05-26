# AI Demand Forecasting for E-Commerce

> A Volatility-Aware and Explainable Forecasting Framework with Uncertainty Estimation

---

## Overview

This project develops an end-to-end demand forecasting pipeline on the **Olist Brazilian E-Commerce Public Dataset**. It benchmarks classical statistical models against a machine learning approach, with a focus on:

- Handling **volatile and irregular demand patterns**
- Providing **interpretable predictions** using SHAP
- Quantifying **forecast uncertainty** via quantile regression
- Performing **rigorous statistical validation** using the Diebold-Mariano test

---

## Models Used

| Model | Type | Role |
|---|---|---|
| SARIMAX | Statistical | Baseline 1 |
| Facebook Prophet | Statistical | Baseline 2 |
| XGBoost | Machine Learning | Main Model |

---

## Key Innovations

- **CV-7 Volatility Feature** — Coefficient of variation over a 7-day rolling window to capture demand instability
- **Quantile Forecasting** — Uncertainty-aware predictions (10th, 50th, 90th percentile)
- **SHAP Explainability** — Feature importance and contribution analysis for model transparency
- **Diebold-Mariano Test** — Statistical significance testing to validate XGBoost's superiority over baselines
- **Leakage Prevention** — Formal verification step to ensure no data leakage in feature engineering

---

## Project Structure

```
ai-demand-forecasting/
│
├── ai_demand_forecasting.ipynb   # Main notebook (end-to-end pipeline)
│
├── data/                         # Olist dataset CSVs
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Notebook Sections

1. Imports & Configuration
2. Data Loading
3. Data Preprocessing
4. Daily Sales Aggregation
5. Exploratory Data Analysis (EDA + Stationarity Test + Seasonal Decomposition)
6. Feature Engineering
7. Leakage Prevention — Formal Verification
8. Time-Based Train-Test Split
9. Time-Series Cross-Validation
10. Baseline Model 1 — SARIMAX
11. Baseline Model 2 — Facebook Prophet
12. Main Model — XGBoost Regressor
13. Results Comparison
14. Actual vs Predicted Sales Plot
15. Quantile Forecasting — Uncertainty Estimation
16. Statistical Validation — Diebold-Mariano Test
17. SHAP Explainability
18. Residual Analysis
19. Final Summary & Conclusion

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-demand-forecasting.git
cd ai-demand-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook ai_demand_forecasting.ipynb
```

> The notebook reads CSVs from the `data/` folder by default. No path changes needed if you follow the structure above.

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset**
Available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

It contains ~100K orders from 2016–2018 across multiple Brazilian marketplaces, with information on orders, products, payments, reviews, customers, sellers, and geolocation.

---

## Requirements

See `requirements.txt` for full list. Key libraries:

- `pandas`, `numpy`
- `xgboost`
- `prophet`
- `statsmodels`
- `scikit-learn`
- `shap`
- `matplotlib`

---

## Author

**Sakshi Kumari**
B.Tech Computer Science Engineering, Galgotias University
[LinkedIn](https://linkedin.com) • [GitHub](https://github.com)
