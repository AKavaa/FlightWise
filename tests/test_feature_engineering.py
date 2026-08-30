"""
Basic tests for the FlightWise data pipeline.

These are intentionally lightweight — the goal for a dissertation project
isn't 100% coverage, it's demonstrating that core transformations are
correct and won't silently break as the pipeline grows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.clean_and_engineer import (
    add_calendar_features,
    add_holiday_features,
    add_lag_features,
)


def make_sample_df(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    values = np.linspace(1_000_000, 2_000_000, n_days)
    return pd.DataFrame({"date": dates, "checkpoint_travel_numbers": values})


def test_calendar_features_add_expected_columns():
    df = make_sample_df()
    result = add_calendar_features(df)
    expected_cols = {
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    }
    assert expected_cols.issubset(result.columns)


def test_day_of_week_is_correct_range():
    df = make_sample_df()
    result = add_calendar_features(df)
    assert result["day_of_week"].min() >= 0
    assert result["day_of_week"].max() <= 6


def test_weekend_flag_matches_day_of_week():
    df = make_sample_df()
    result = add_calendar_features(df)
    weekends = result[result["is_weekend"] == 1]
    assert set(weekends["day_of_week"].unique()).issubset({5, 6})


def test_holiday_flag_is_binary():
    df = make_sample_df()
    result = add_holiday_features(df)
    assert set(result["is_school_holiday"].unique()).issubset({0, 1})


def test_lag_features_shift_correctly():
    df = make_sample_df()
    result = add_lag_features(df)
    # lag_7d at row 10 should equal the raw value at row 3 (10 - 7)
    raw_value_at_3 = df.loc[3, "checkpoint_travel_numbers"]
    lag_value_at_10 = result.loc[10, "lag_7d"]
    assert lag_value_at_10 == raw_value_at_3


def test_lag_features_produce_nan_for_early_rows():
    df = make_sample_df()
    result = add_lag_features(df)
    # first 7 rows can't have a 7-day lag — must be NaN, not silently wrong
    assert result.loc[0, "lag_7d"] != result.loc[0, "lag_7d"]  # NaN check
