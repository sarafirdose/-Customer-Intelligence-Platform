# Continuous Retraining Pipeline Guide

## Overview
The Retraining Pipeline (`backend/ml/retraining.py`) automates end-to-end retraining:
1. **Data Ingest & Feature Validation**
2. **Train & Evaluate** (accuracy, ROC-AUC, F1, Brier score)
3. **Baseline Metric Comparison** vs active production model
4. **Registry Registration & Staging Promotion**
5. **Auto-Promote to Production** (optional, if ROC-AUC improves)

## REST API Trigger
`POST /api/v1/retraining/run`
Body: `{"auto_promote": true}`
