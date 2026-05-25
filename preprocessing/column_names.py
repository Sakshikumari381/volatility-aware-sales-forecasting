"""
Normalize CSV column headers to the canonical schema expected by the pipeline.
"""

import re
from typing import Dict, List, Union

import pandas as pd

# Canonical name -> accepted aliases (after normalization)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": [
        "date",
        "dt",
        "order_date",
        "transaction_date",
        "sale_date",
        "sales_date",
        "datetime",
        "timestamp",
        "period",
        "day",
    ],
    "product_id": [
        "product_id",
        "productid",
        "product",
        "sku",
        "item_id",
        "item",
        "product_code",
    ],
    "category": ["category", "cat", "product_category", "product_cat"],
    "sales_quantity": [
        "sales_quantity",
        "sales",
        "quantity",
        "qty",
        "units",
        "units_sold",
        "demand",
        "volume",
    ],
    "price": ["price", "unit_price", "sales_price", "retail_price"],
    "promotional_flag": [
        "promotional_flag",
        "promo",
        "promotion",
        "is_promo",
        "promotional",
        "on_promotion",
    ],
    "day_of_week": ["day_of_week", "dow", "weekday"],
    "month": ["month", "mon"],
}


def _normalize_key(name: str) -> str:
    """Lowercase header and convert spaces/dashes to underscores."""
    key = str(name).strip().lstrip("\ufeff")
    key = re.sub(r"[\s\-]+", "_", key)
    return key.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names when a known alias is present.

    Headers are trimmed; matching is case-insensitive.
    """
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff") for c in out.columns]

    key_to_original = {_normalize_key(c): c for c in out.columns}
    rename: Dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            norm = _normalize_key(alias)
            if norm in key_to_original:
                original = key_to_original[norm]
                if original not in rename.values():
                    rename[original] = canonical
                break

    if rename:
        out = out.rename(columns=rename)
    return out


def missing_required_columns(
    df: pd.DataFrame, required: tuple[str, ...]
) -> List[str]:
    """Return canonical column names that are still missing after normalization."""
    return [col for col in required if col not in df.columns]
