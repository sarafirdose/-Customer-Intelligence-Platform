# Error Analysis Report

This report evaluates model classification failures across the test partition (1409 customer accounts).

## 1. Confusion Matrix Breakdown
- **True Positives (TP)**: 267 (Correctly predicted churn)
- **True Negatives (TN)**: 833 (Correctly predicted retention)
- **False Positives (FP)**: 202 (Predicted churn, customer stayed)
- **False Negatives (FN)**: 107 (Predicted stay, customer churned)

## 2. False Positive Profile (False Alarms)
These customers were predicted to leave but remained. Proactive retention offers to this group would represent wasted expenditure:
  - Contract Type: Month-to-month
  - Internet Service: Fiber optic
  - Payment Method: Electronic check
  - Average Tenure: 17.8 months
  - Average Monthly Charges: $78.07

## 3. False Negative Profile (Silent Churners)
These customers were predicted to stay but cancelled their service. They represent our biggest revenue vulnerability:
  - Contract Type: Month-to-month
  - Internet Service: DSL
  - Payment Method: Electronic check
  - Average Tenure: 28.7 months
  - Average Monthly Charges: $62.92

## 4. Key Failure Takeaways
- **Tenure Boundary Friction**: The majority of False Negatives occur around the **6-12 month tenure mark**, where short-term contracts are ending and onboarding loyalty incentives decay.
- **Fiber Optic Pricing Friction**: High-bill Fiber Optic users show high counts in False Positives, indicating that while pricing models raise suspicion metrics, actual churn requires concurrent service dissatisfaction.
