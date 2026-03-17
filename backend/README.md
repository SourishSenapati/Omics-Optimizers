# Omics Optimizers: Twin Engine Backend

This is the predictive core of the Epidemiological Digital Twin. It leverages data harmonization and mechanistic differential modeling to transform reactive surveillance into proactive forecasting.

## Architecture

- Ingestion Layer: Multi-source fetching from Disease.sh and ProMED-mail.
- Harmonization Core: Logic for heterogeneous data-flow normalization.
- Intelligence Layer: PINN engine for SIR model parameter identification and forecasting.

## Setup

1. pip install -r requirements.txt
2. python main.py

## Endpoints

- GET /api/surveillance?disease=covid-19: Returns unified global stats, real-time alerts, and mathematical predictions.
