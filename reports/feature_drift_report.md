# Feature Drift Report

**Run Date**: 2026-08-18T12:38:58.525660+00:00
**Dataset Rows**: 200
**Overall Severity**: 🚨 Critical
**PSI Warning Threshold**: 0.1
**PSI Critical Threshold**: 0.25

---

## Numerical Feature Drift (PSI)

| Feature | PSI | Severity | Baseline μ | Prod μ | Baseline σ | Prod σ |
|---|---|---|---|---|---|---|
| tenure_months | 2.0278 | 🚨 Critical | 32.3711 | 31.5296 | 24.5595 | 13.1247 |
| monthly_charges | 0.0213 | ✅ Normal | 64.7617 | 66.0867 | 30.09 | 27.8202 |
| total_charges | 3.7243 | 🚨 Critical | 2279.7343 | 2218.7886 | 2266.7945 | 997.5918 |
| charges_ratio | 7.8456 | 🚨 Critical | 5.7706 | 5.7283 | 8.7224 | 2.026 |
| total_services | 7.0081 | 🚨 Critical | 3.3629 | 4.02 | 2.062 | 1.9999 |
| total_charges_log | 0.4070 | 🚨 Critical | 6.9325 | 6.9629 | 1.5694 | 0.9553 |

---

## Categorical Feature Drift (Max Proportional Shift)

### contract_type — ✅ Normal (max shift: 0.0248)

| Category | Baseline % | Production % | Shift |
|---|---|---|---|
| Month-to-month | 55.02% | 57.50% | 0.0248 |
| One year | 20.91% | 20.50% | 0.0041 |
| Two year | 24.07% | 22.00% | 0.0207 |

### payment_method — ✅ Normal (max shift: 0.0542)

| Category | Baseline % | Production % | Shift |
|---|---|---|---|
| Electronic check | 33.58% | 31.50% | 0.0208 |
| Bank transfer (automatic) | 21.92% | 16.50% | 0.0542 |
| Mailed check | 22.89% | 27.50% | 0.0461 |
| Credit card (automatic) | 21.61% | 24.50% | 0.0289 |

### internet_service — ✅ Normal (max shift: 0.0737)

| Category | Baseline % | Production % | Shift |
|---|---|---|---|
| DSL | 34.37% | 27.00% | 0.0737 |
| Fiber optic | 43.96% | 45.00% | 0.0104 |
| No | 21.67% | 28.00% | 0.0633 |

### tenure_group — ⚠️ Warning (max shift: 0.1154)

| Category | Baseline % | Production % | Shift |
|---|---|---|---|
| 0-12m | 31.04% | 19.50% | 0.1154 |
| 48-60m | 11.81% | 22.50% | 0.1069 |
| 12-24m | 14.54% | 16.50% | 0.0196 |
| 60m+ | 19.98% | 18.50% | 0.0148 |
| 24-48m | 22.63% | 23.00% | 0.0037 |

---

*Generated automatically by drift.py*