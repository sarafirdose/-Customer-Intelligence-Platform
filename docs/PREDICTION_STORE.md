# Prediction Store & Auditability Guide

## Overview
The Prediction Store (`backend/ml/prediction_store.py`) persists all model prediction events, latencies, confidence levels, model versions, and feature hashes to `logs/prediction_store.jsonl` for offline analytics, auditing, and replay.

## Structure
Every prediction record includes:
- `prediction_id`: Unique identifier.
- `timestamp`: ISO-8601 UTC timestamp.
- `customer_id`: Account ID.
- `request_id`: Request correlation ID.
- `model_version`: Serving model version.
- `churn_probability` & `predicted_ltv`: Predictions.
- `latency_ms`: Inferred response time.
- `feature_hash`: SHA256 feature hash.

## Query API
`GET /api/v1/predictions/history?customer_id=C100&limit=50`
