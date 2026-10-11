# FlightWise repository guide

This guide explains the files already present. You do not need to understand them all at once. The first learning path is: source data, feature table, baseline, model, then evaluation. The web application comes later.

## The main flow

```text
Source data -> data/raw/ -> cleaning and features -> data/processed/
            -> training and evaluation -> models/ -> future API and web app
```

`raw` means a copy of data as received from a source. `processed` means a table changed into the format a model needs. A model artifact is the saved result of training, not the data used to train it. Generated files in these directories are ignored by Git; the `.gitkeep` files only keep the directories in the repository.

## Read these files in this order

1. `src/pipeline/collect_tsa_data.py`: shows where passenger data would enter the project. Its current command creates *synthetic* sample rows, and the real fetch function is unfinished. Read the `generate_sample_data` and `fetch_live` functions separately.
2. `src/pipeline/clean_and_engineer.py`: turns a dated passenger-count table into calendar and lag features. A lag is a count from an earlier day. We must check whether every lag would be known when a future prediction is requested.
3. `src/models/train_xgboost.py`: divides rows by time, trains XGBoost, and compares it with a previous-week baseline. Its old saved scores came from synthetic data and are not real-world evidence.
4. `tests/test_feature_engineering.py`: checks a few transformations, such as whether a seven-day lag really points seven rows back.

Read one file, run or inspect one small example, and explain its input and output before moving to the next.

## Supporting files

| File or directory | Why it exists |
| --- | --- |
| `requirements.txt` | Lists Python libraries to install; some are reserved for later parts of the proposal |
| `.env.example` | Shows the names of future API settings without containing real keys |
| `.env` | Local secrets; ignored by Git and never needed for the first data audit |
| `.gitignore` | Prevents local secrets, environments, generated data, and model artifacts from being committed |
| `.pre-commit-config.yaml` | Defines formatting and lint checks before a commit |
| `.github/workflows/ci.yml` | Runs checks on GitHub after a push or pull request |
| `src/__init__.py`, `src/pipeline/__init__.py`, `src/models/__init__.py`, `tests/__init__.py` | Empty Python package marker files; they contain no application logic |
| `notebooks/` | Optional place for exploratory analysis; no notebook is part of the tracked code yet |
| `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | Local environment or generated caches; not source code |
| `.vscode/`, `.kilo/` | Local editor or assistant settings; not part of the model |

## What is intentionally absent

The previous untracked API experiment has been archived outside the repository. We will design the real API after defining what a prediction takes as input and returns as output. The frontend, LSTM comparison, and external integrations are later milestones.
