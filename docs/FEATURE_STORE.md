# Feature Store & Schema Validation Guide

## Overview
The Feature Store (`backend/ml/feature_store.py`) ensures training-serving consistency, validates incoming customer payloads against versioned schemas (`v1.0`), and applies default values for missing attributes.

## Features Managed
- **Numeric Features**: `tenure_months`, `monthly_charges`, `total_charges`, `senior_citizen`.
- **Categorical Features**: `contract_type`, `payment_method`, `internet_service`, `paperless_billing`, `partner`, `dependents`, service options.

## Python Usage
```python
from backend.ml.feature_store import feature_store

# Validate payload
is_valid, errors = feature_store.validate_features(customer_payload)

# Apply defaults
clean_payload = feature_store.apply_defaults(customer_payload)
```
