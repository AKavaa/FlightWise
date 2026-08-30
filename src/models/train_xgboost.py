"""
FlightWise — XGBoost Airport Busyness Model
----------------------------------------------
Trains a gradient-boosted regression model to predict daily airport
checkpoint volume, evaluates it against a naive baseline, and logs
everything to MLflow for versioning and comparison.

Run: python src/models/train_xgboost.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "checkpoint_travel_numbers"
FEATURE_COLS = [
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_school_holiday",
    "lag_7d",
    "lag_14d",
    "rolling_mean_7d",
    "rolling_mean_30d",
    "rolling_std_7d",
]

mlflow.set_tracking_uri(f"sqlite:///{MODELS_DIR / 'mlflow.db'}")
mlflow.set_experiment("flightwise-airport-busyness")


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features_v1.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No processed features found. Run clean_and_engineer.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def time_aware_split(df: pd.DataFrame, test_size: float = 0.15):
    """
    Time series data must NOT be split randomly — that would let the model
    train on future dates and test on past dates, leaking information.
    Split chronologically instead: earliest data trains, most recent tests.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    logger.info(
        f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df)} rows)"
    )
    logger.info(
        f"Test:  {test_df['date'].min().date()} → {test_df['date'].max().date()} ({len(test_df)} rows)"
    )
    return train_df, test_df


def naive_baseline_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """
    The baseline every model must beat: 'today will look like the same
    day last week'. A model is only worth using if it clearly outperforms
    this trivial, zero-training-cost heuristic.
    """
    return test_df["lag_7d"].values


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    logger.info(
        f"[{label}] MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.3f}  MAPE={mape:.2f}%"
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train():
    df = load_features()
    train_df, test_df = time_aware_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    with mlflow.start_run(run_name="xgboost_v1"):
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params, objective="reg:squarederror")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        baseline_pred = naive_baseline_predictions(test_df)

        model_metrics = evaluate(y_test.values, y_pred, "XGBoost")
        baseline_metrics = evaluate(
            y_test.values, baseline_pred, "Naive baseline (same day last week)"
        )

        improvement_pct = (1 - model_metrics["mae"] / baseline_metrics["mae"]) * 100
        logger.info(
            f"XGBoost improves MAE over naive baseline by {improvement_pct:.1f}%"
        )

        for k, v in model_metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{k}", v)
        mlflow.log_metric("improvement_over_baseline_pct", improvement_pct)

        importance = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        logger.info("\nTop features:\n" + importance.head(8).to_string())

        mlflow.xgboost.log_model(model, "model")
        model_path = MODELS_DIR / "xgboost_v1.json"
        model.save_model(model_path)

        results = {
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "improvement_over_baseline_pct": improvement_pct,
            "feature_importance": importance.to_dict(),
            "trained_on_rows": len(train_df),
            "tested_on_rows": len(test_df),
        }
        with open(MODELS_DIR / "xgboost_v1_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nModel saved to {model_path}")
        logger.info(f"Results saved to {MODELS_DIR / 'xgboost_v1_results.json'}")

    return model, results


if __name__ == "__main__":
    train()
