# SHAP Explainability Service Guide

## Overview
The Explainability Service (`backend/ml/explain.py`) provides global feature importance rankings and local customer SHAP explanations, including waterfall plot data, force plot data, and top positive/negative feature drivers.

## REST Endpoint
`GET /api/v1/explain/{customer_id}`

## Response Schema
```json
{
  "explanation": {
    "customer_id": "C100",
    "base_value": 0.26,
    "predicted_probability": 0.68,
    "top_positive_drivers": [{"feature": "contract_type", "impact": 0.25}],
    "top_negative_drivers": [{"feature": "tenure_months", "impact": -0.12}],
    "waterfall_data": [...],
    "force_plot_data": {...}
  },
  "global_importance": [...]
}
```
