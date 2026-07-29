# Canary Deployment Guide

## Overview
Canary deployment allows gradual traffic shifting (10% -> 25% -> 50% -> 100%) to test candidate models under live traffic with automated rollback capability.

## Canary Workflow
1. Retrain new model -> Promoted to `staging` (`v1.1.0`).
2. Start Canary at 10% traffic: `POST /api/v1/deployment/canary` `{"target_version": "v1.1.0", "percentage": 10}`.
3. Monitor latency, error rate, and prediction distribution.
4. Advance to 25%, 50%, then 100%.
5. At 100%, candidate model is automatically promoted to `production`.

## Rollback Trigger
If error rate > 5% or latency > 500ms, trigger instant rollback: `POST /api/v1/deployment/rollback`.
