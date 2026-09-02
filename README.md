# FlightWise

AI-powered airport crowd prediction and flight recommendation system.
Final year project — BSc Computing (Software Development), UCLan Cyprus.

FlightWise predicts how crowded an airport and flight will be on a given
date, using machine learning trained on real historical passenger data,
and helps travellers find their quietest window to fly.

## Project status

🚧 In active development. Current milestone: data pipeline + baseline
XGBoost model (see `models/xgboost_v1_results.json` for latest metrics).

## Architecture

```
Raw data (TSA/BTS) → collection → cleaning → feature engineering
   → XGBoost training → evaluation + SHAP → MLflow versioning
   → FastAPI serving → Next.js dashboard
```

## Repository structure

```
flightwise/
├── data/
│   ├── raw/            # untouched source data (gitignored, regenerable)
│   └── processed/      # feature-engineered tables ready for training
├── src/
│   ├── pipeline/        # data collection + cleaning + feature engineering
│   ├── models/          # training scripts (XGBoost, LSTM later)
│   └── api/             # FastAPI serving layer
├── models/              # saved model artefacts + MLflow tracking db
├── tests/                # pytest test suite
├── notebooks/            # exploratory analysis (not production code)
└── .github/workflows/    # CI: lint + test on every push
```

## Setup

```bash
git clone <repo-url>
cd flightwise
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install                 # auto-formats/lints on every commit
```

## Running the pipeline

```bash
# 1. Collect raw data
python src/pipeline/collect_tsa_data.py

# 2. Clean + engineer features
python src/pipeline/clean_and_engineer.py

# 3. Train the model
python src/models/train_xgboost.py

# 4. Run tests
pytest tests/ -v
```

## Model tracking

Training runs are logged with MLflow. To view the experiment dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///models/mlflow.db
```

## Data sources

- **TSA Checkpoint Travel Numbers** — daily US airport passenger
  throughput, published by the Transportation Security Administration.
  https://www.tsa.gov/travel/passenger-volumes
- **BTS T-100 Segment Data** — commercial flight passenger counts and
  load factors, Bureau of Transportation Statistics.
  https://www.transtats.bts.gov
- **Amadeus for Developers** — live flight search and booking data.
  https://developers.amadeus.com

