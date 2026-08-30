"""
FlightWise — Cleaning & Feature Engineering
---------------------------------------------
Takes raw TSA checkpoint data and produces a model-ready feature table.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SCHOOL_HOLIDAY_RANGES = [
    ("2022-07-20", "2022-09-01"),
    ("2022-12-20", "2023-01-03"),
    ("2023-07-20", "2023-09-01"),
    ("2023-12-20", "2024-01-03"),
    ("2024-07-20", "2024-09-01"),
    ("2024-12-20", "2025-01-03"),
    ("2025-07-20", "2025-09-01"),
    ("2025-12-20", "2026-01-03"),
    ("2026-07-20", "2026-09-01"),
]


def load_latest_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("tsa_checkpoint_*.csv"))
    if not files:
        raise FileNotFoundError("No raw TSA data found. Run collect_tsa_data.py first.")
    latest = files[-1]
    logger.info(f"Loading {latest}")
    return pd.read_csv(latest, parse_dates=["date"])


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_school_holiday"] = 0
    for start, end in SCHOOL_HOLIDAY_RANGES:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "is_school_holiday"] = 1
    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str = "checkpoint_travel_numbers"
) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["lag_7d"] = df[target_col].shift(7)
    df["lag_14d"] = df[target_col].shift(14)
    df["rolling_mean_7d"] = df[target_col].shift(1).rolling(window=7).mean()
    df["rolling_mean_30d"] = df[target_col].shift(1).rolling(window=30).mean()
    df["rolling_std_7d"] = df[target_col].shift(1).rolling(window=7).std()
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(
        f"Dropped {before - len(df)} rows with insufficient history for lag features"
    )
    return df


def save_processed(df: pd.DataFrame, filename: str = "features_v1.csv") -> Path:
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} feature rows to {out_path}")
    return out_path


if __name__ == "__main__":
    raw = load_latest_raw()
    features = build_feature_table(raw)
    save_processed(features)
