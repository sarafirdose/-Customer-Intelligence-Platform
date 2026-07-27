# Unified Customer Intelligence scoring Methodology

The Unified **Customer Intelligence Score (0-100)** is computed as a weighted index of value and risk metrics:

## 1. Scoring Formula
```text
Score = 30% * (1.0 - Churn Probability) * 100
      + 30% * (log(Predicted LTV) / log(Max LTV)) * 100
      + 20% * (Tenure / 72 months) * 100
      + 20% * (Services / 8 services) * 100
```

## 2. Category Share Breakdown
- **Good** (Count: 2221): average score of 70.0
- **Moderate** (Count: 2023): average score of 50.2
- **Poor** (Count: 1541): average score of 30.8
- **Excellent** (Count: 1177): average score of 88.4
- **Critical** (Count: 81): average score of 18.9
