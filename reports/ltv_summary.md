# Subscriber Lifetime Value (LTV) Prediction Summary

This report documents the LTV prediction regression performance and future forecasts.

## 1. Regression Model Comparison (Historical Proxy)
- **Selected Best Model**: LightGBM Regressor
- **R² Coefficient of Determination**: 0.9987
- **Root Mean Squared Error (RMSE)**: $81.22
- **Mean Absolute Error (MAE)**: $56.14
- **Mean Absolute Percentage Error (MAPE)**: 9.19%

## 2. Hybrid LTV Target Definition
We use `total_charges` as a historical proxy of accumulated spend. We then estimate the **Projected Future LTV** via the expected remaining lifetime:
$$\text{Remaining Lifetime (months)} = \max\left(0, \frac{1}{\text{Churn Probability}} - \text{Tenure}\right)$$
$$\text{Projected Future LTV} = \text{Remaining Lifetime} \times \text{Monthly Charges}$$

- **Total Projected Future LTV Potential**: $60,577.96
- **Average Projected Future LTV**: $8.60 per customer
