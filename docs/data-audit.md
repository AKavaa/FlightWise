# Milestone 1: data and prediction audit

This is the starting point for deciding exactly what FlightWise will predict. It records what has been verified so far and what remains open. It is not a replacement for the approved proposal.

## Implementation checkpoint (11 October 2026)

| Part | What exists in the repository | What is still needed |
| --- | --- | --- |
| Passenger data | A reproducible synthetic-data generator and feature-engineering functions; no working data files | Download, inspect, and document real passenger observations |
| Updates | No live collector or scheduled job | Decide whether updates are needed, then implement and monitor a real source |
| Machine learning | XGBoost training code and a previous-week baseline | Train on real data, evaluate the intended forecast horizon, and save a validated model |
| User-facing prediction | No active API endpoint or frontend | Define the prediction input/output, then build the service and interface |
| Other proposal components | No LSTM comparison, SHAP result, Quiet Score, or external API integration | Add these in separate milestones after the target and data are defined |

The previous model artifact and its metrics were generated from synthetic data and were archived outside the working repository. They are not evidence of real-world accuracy.

## Source-to-claim check

| Source | What we can establish now | Possible FlightWise use | Limit to respect |
| --- | --- | --- | --- |
| [TSA checkpoint travel numbers](https://www.tsa.gov/travel/passenger-volumes) | Published table has a date and passenger number for each day | US-wide daily travel-volume forecast and seasonal context | The public table does not provide airport or hour columns |
| [BTS T-100 segment data](https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EEE&QO_anzr=Nv4+Pn44vr4+f6n6v56vp5+%28S14z+HE+g4nssvp%29-+Nyy+Pn44vr45&Yv0x=D) | Monthly reports contain route/segment passenger and capacity information | Historical route load-factor context or a monthly route estimate | Does not reveal actual occupancy of a particular future flight |
| [Embry-Riddle airport passenger dataset](https://datacommons.erau.edu/datasets/4dsy9vxxgx/1) | 974 daily observations for each of ATL, DEN, ORD, LAX, and DFW, from 15 February 2020 to 15 October 2022; CC BY 4.0 | First candidate for a real airport-level daily prediction target | Five airports and a pandemic-era period only; assess how well findings generalise to current travel |
| [Open-Meteo historical forecast data](https://open-meteo.com/en/docs/historical-forecast-api) | Historical weather forecasts are available | Optional weather features and an ablation study | A forecast must use information available on the prediction date |
| [Amadeus Flight Offers Search](https://admin.developers.amadeus.com/self-service/apis-docs/guides/developer-guides/faq/) | Searches flight offers; pricing and booking are separate API steps | Show available flights after a date is selected | Search results do not establish how full a flight will be; booking redirect needs verification |

## Prediction definitions to decide

1. **Airport busyness:** Which airport, what unit (passengers, percentage of typical traffic, or a relative category), and how many days ahead? Find a real airport-level target before claiming an airport-specific forecast. A national TSA total is a separate target.
2. **Route load factor:** Define whether this is a historical monthly average or a forecast of that average. It must not be presented as the occupancy of a specific flight.
3. **Quiet Score:** Specify how the airport and route signals are combined, how each is normalised, and what a score of 0 or 100 means. The interface must label historical estimates and forecasts separately.
4. **Hourly view:** Identify an hourly passenger-count target or describe the view as a schedule-based proxy. Do not infer hourly passenger counts from a daily or monthly total without validation.

## First practical tasks

- Download a small sample of the Embry-Riddle dataset and record its files, column meanings, row count, missing values, and licence attribution. Its published description and coverage have now been verified, but its data files have not yet been inspected.
- Obtain a small sample from each other candidate passenger dataset and record source URL, download date, licence, column meanings, date range, coverage, missing values, and row count.
- Choose one first target and prediction horizon that the data supports.
- Define a simple baseline and a chronological train/validation/test split for that horizon.
- Only after the real data is loaded, evaluate XGBoost and LSTM against the same target and splits.

## Existing prototype caveats

- `src/pipeline/collect_tsa_data.py` currently writes seeded synthetic data when run directly; `fetch_live()` raises `NotImplementedError`.
- The old saved XGBoost metrics were calculated on that synthetic data and have been archived outside the working repository.
- Lag features in `clean_and_engineer.py` use previous observed targets. For a forecast made multiple days ahead, validation must ensure those values would actually be known at prediction time.
- A previous untracked API prototype was archived before this milestone. Its input schema requested engineered features from the caller, so it was not yet a usable traveller-facing API.
