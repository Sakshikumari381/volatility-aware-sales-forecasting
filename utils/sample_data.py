"""
Generate synthetic e-commerce sales dataset for demos and tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings


def generate_sample_data(
    n_products: int = 5,
    days: int = 120,
    output_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create synthetic sales data matching required schema.
    """
    rng = np.random.default_rng(seed)
    settings = get_settings()
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    rows = []

    for pid in range(1, n_products + 1):
        base = 50 + pid * 10
        trend = np.linspace(0, 20, days)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
        noise = rng.normal(0, 8, days)
        sales = np.maximum(base + trend + seasonal + noise, 0).round(2)
        price = 19.99 + pid + rng.normal(0, 1, days)
        promo = (rng.random(days) < 0.15).astype(int)

        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "product_id": f"P{pid:03d}",
                    "category": f"Category_{pid % 3 + 1}",
                    "sales_quantity": sales[i],
                    "price": round(float(price[i]), 2),
                    "promotional_flag": int(promo[i]),
                    "day_of_week": d.dayofweek,
                    "month": d.month,
                }
            )

    df = pd.DataFrame(rows)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    settings = get_settings()
    path = settings.data_raw_dir / "sample_sales.csv"
    generate_sample_data(output_path=path)
    print(f"Sample data written to {path}")
