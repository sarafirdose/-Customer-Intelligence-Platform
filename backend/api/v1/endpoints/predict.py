"""
Machine Learning churn and LTV prediction endpoints.

Exposes REST APIs to score customer records and fetch explainability coefficients.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from backend.schemas.customer import (
    CustomerPredictRequest,
    CustomerPredictResponse,
    ExplainResponse,
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=CustomerPredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn probability and LTV for a customer",
)
def predict_customer(request: CustomerPredictRequest) -> CustomerPredictResponse:
    """
    Score a single customer's features to predict churn risk and lifetime value.

    Args:
        request: Customer demographics and behavioral features.

    Returns:
        CustomerPredictResponse: Risk scores, classifications, and estimated LTV.
    """
    # Placeholder implementation yielding mock values
    churn_probability = 0.28
    predicted_ltv = 1500.00
    is_churn = churn_probability > 0.5

    return CustomerPredictResponse(
        customer_id=request.customer_id,
        churn_probability=churn_probability,
        is_churn=is_churn,
        predicted_ltv=predicted_ltv,
        model_version="1.0.0",
    )


@router.get(
    "/explain/{customer_id}",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch explainable AI (SHAP) attributions for a prediction",
)
def explain_prediction(customer_id: str) -> ExplainResponse:
    """
    Calculate or retrieve SHAP explainability values for a customer prediction.

    Args:
        customer_id: Unique string identifying the customer.

    Returns:
        ExplainResponse: Feature attributions and base value offsets.
    """
    # Placeholder implementation yielding mock SHAP attributions
    return ExplainResponse(
        customer_id=customer_id,
        base_value=0.35,
        attributions={
            "tenure_months": -0.15,
            "monthly_charges": 0.08,
            "total_charges": -0.05,
            "contract_type_two_year": -0.12,
            "paperless_billing_yes": 0.02,
        },
        model_version="1.0.0",
    )
