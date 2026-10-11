# FlightWise

FlightWise is a final-year computing project about helping travellers find quieter times to fly. The idea is to analyse historical passenger traffic, forecast airport busyness, add route-level historical context, and present the results alongside flight search in a web application. Predictions should explain what they measure and how uncertain they are.

## Project goals

- Compare forecasting approaches, including a simple baseline, XGBoost, and LSTM.
- Explore an interpretable Quiet Score based on supported airport and route signals.
- Let travellers compare dates and understand why a date may be quieter.
- Show available flight offers for a selected route and date.

## Proposed stack

| Part | Tools | Role |
| --- | --- | --- |
| Data and modelling | Python, pandas, scikit-learn, XGBoost, an LSTM framework | Prepare data, train models, and evaluate forecasts |
| Explanation | SHAP | Examine which features influence model predictions |
| Backend | FastAPI | Serve predictions and connect external services |
| Frontend | React or Next.js | Display the calendar, score, explanations, and flights |
| Data and services | TSA, BTS T-100, airport-level data, Open-Meteo, Amadeus | Provide historical targets, context, weather, and flight offers |

The final choice of each data source and model depends on evaluation. A planned tool in this table is not necessarily implemented in the repository yet.

## How to read this repository

Start with the [repository guide](docs/repository-guide.md) for a plain-language tour of every file group. The [data audit](docs/data-audit.md) records current evidence, open questions, and the next research steps.

