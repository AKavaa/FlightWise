"""
FlightWise — TSA Checkpoint Data Collector
--------------------------------------------
Downloads daily TSA checkpoint passenger volumes and saves as raw CSV.

TSA publishes this data at:
https://www.tsa.gov/travel/passenger-volumes

This script is designed to run on a schedule (daily cron job) so the
dataset stays current. For the dissertation build, run it once to
pull the full historical archive, then daily thereafter to append
fresh rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TSA_SOURCE_NOTE = (
    "Live source: https://www.tsa.gov/travel/passenger-volumes "
    "— replace fetch_live() body with the real scrape/export call once "
    "the exact export endpoint is confirmed."
)


def fetch_live() -> pd.DataFrame:
    """
    Fetches the latest TSA checkpoint data.

    In production this hits the TSA passenger volumes page and parses
    the published table. Swap this implementation once you've confirmed
    the exact export format (TSA changes its page structure occasionally,
    so this is intentionally isolated into one function).
    """
    logger.info(TSA_SOURCE_NOTE)
    raise NotImplementedError(
        "Wire this up to the live TSA export once you've inspected the "
        "current page structure. Use generate_sample_data() below to "
        "develop and test the rest of the pipeline in the meantime."
    )


def generate_sample_data(
    start_date: str = "2022-01-01", end_date: str = "2026-08-01"
) -> pd.DataFrame:
    """
    Generates a realistic synthetic TSA-style dataset for development.

    This mirrors the real schema (date, throughput) so every downstream
    step — cleaning, feature engineering, XGBoost training — can be built
    and tested today, before the live scraper is finalised.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rng = np.random.default_rng(seed=42)

    base = 2_100_000
    values = []
    for d in dates:
        dow_factor = {0: 0.92, 1: 0.88, 2: 0.90, 3: 0.97, 4: 1.12, 5: 1.05, 6: 1.15}[
            d.weekday()
        ]
        month_factor = {
            1: 0.95,
            2: 0.90,
            3: 1.00,
            4: 1.02,
            5: 1.05,
            6: 1.10,
            7: 1.15,
            8: 1.13,
            9: 0.95,
            10: 0.97,
            11: 1.05,
            12: 1.10,
        }[d.month]
        noise = rng.normal(1.0, 0.04)
        values.append(base * dow_factor * month_factor * noise)

    df = pd.DataFrame(
        {"date": dates, "checkpoint_travel_numbers": [int(v) for v in values]}
    )
    return df


def save_raw(df: pd.DataFrame, filename: str = None) -> Path:
    """Saves raw data with a timestamped filename for traceability."""
    if filename is None:
        filename = f"tsa_checkpoint_{datetime.now().strftime('%Y%m%d')}.csv"
    out_path = RAW_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    logger.info(
        "Generating development dataset (swap for fetch_live() once TSA scraper is confirmed)"
    )
    df = generate_sample_data()
    save_raw(df)
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
