# Feature Recommendation Report

Based on statistical significance, Mutual Information, Pearson/Spearman correlation, and Cramér's V association tests:

## 1. High-Value Predictor Features (Strong Association)
- **contract_type**: Highest Cramér's V (0.410). Month-to-month contracts have extreme churn susceptibility.
- **tenure_months**: Strongly correlated with churn. Longer tenure reduces likelihood of cancellation.
- **internet_service**: Fiber optic users churn at a significantly higher rate compared to DSL and No internet users.
- **online_security** & **tech_support**: Security and support features strongly mitigate churn risk.
- **payment_method**: Electronic check payment represents a high churn segment.

## 2. Low-Information / Weak Attributes (To Drop or Avoid)
- **gender**: Cramér's V of 0.0083 and Chi-Square p-value > 0.05. No statistical variance in churn between Male/Female.
- **phone_service**: Extremely low mutual info score. Churn rates do not fluctuate with basic phone line presence.

## 3. Redundant / Highly Collinear Features
- **total_charges**: Strongly correlated with `tenure_months` (Pearson r = 0.826) and `monthly_charges` (Pearson r = 0.651). Represents multicollinearity risk. Use scaling, regularization, or feature engineering (e.g. Ratio variables) to avoid instability.

## 4. Candidate Engineered Features for Phase 3
- **Tenure Bins**: Grouping tenure into segments (e.g. `0-12 months`, `12-24 months`, `24-48 months`, `48+ months`) to capture non-linear retention rates.
- **Total Services Count**: Count of active communication & support services (online security, backup, protection, tech support, streaming) to model account stickiness.
- **Charges Ratio**: Ratio of Monthly Charges to Tenure to isolate billing velocity impact.
- **Automatic Payment Flag**: Binary indicator for automatic payment methods (Credit Card / Bank Transfer) vs manual check methods.
