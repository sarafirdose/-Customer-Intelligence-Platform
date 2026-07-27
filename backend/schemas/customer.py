"""
Pydantic Data Schemas for Customer Entity.

Defines schemas for prediction requests, prediction responses, and model
explanations (SHAP value representations).
"""

from typing import Dict
from pydantic import BaseModel, Field


class CustomerPredictRequest(BaseModel):
    """
    Validation schema for scoring customer churn and estimating LTV.
    """

    customer_id: str = Field(
        ...,
        description="Unique customer identifier",
        json_schema_extra={"example": "1234-ABCD"},
    )
    tenure_months: int = Field(
        ...,
        description="Total months the customer has been with the company",
        ge=0,
        json_schema_extra={"example": 24},
    )
    monthly_charges: float = Field(
        ...,
        description="Monthly service cost charged to the customer",
        ge=0.0,
        json_schema_extra={"example": 64.85},
    )
    total_charges: float = Field(
        ...,
        description="Cumulative charges over the lifetime of the customer",
        ge=0.0,
        json_schema_extra={"example": 1556.40},
    )
    contract_type: str = Field(
        ...,
        description="Subscription billing contract type",
        json_schema_extra={"example": "Two year"},
    )
    paperless_billing: str = Field(
        ...,
        description="Whether paperless invoice billing is activated (Yes/No)",
        json_schema_extra={"example": "Yes"},
    )
    internet_service: str = Field(
        ...,
        description="Type of internet connection line (DSL, Fiber optic, No)",
        json_schema_extra={"example": "DSL"},
    )
    tech_support: str = Field(
        ...,
        description="Tech support subscription service state (Yes/No/No internet service)",
        json_schema_extra={"example": "Yes"},
    )


class CustomerPredictResponse(BaseModel):
    """
    Schema for churn prediction and LTV model scoring responses.
    """

    customer_id: str = Field(..., description="Unique customer identifier")
    churn_probability: float = Field(
        ..., description="Predicted probability of churn (0.0 to 1.0)", ge=0.0, le=1.0
    )
    is_churn: bool = Field(
        ..., description="Boolean churn classification based on default thresholds"
    )
    predicted_ltv: float = Field(
        ..., description="Estimated customer lifetime value", ge=0.0
    )
    model_version: str = Field(..., description="Version of the inference model used")


class ExplainResponse(BaseModel):
    """
    Schema for explaining customer scores (SHAP attributions).
    """

    customer_id: str = Field(..., description="Unique customer identifier")
    base_value: float = Field(
        ..., description="Baseline expectation value for predictions"
    )
    attributions: Dict[str, float] = Field(
        ..., description="Map of feature names to SHAP importance weights"
    )
    model_version: str = Field(
        ..., description="Version of the model that generated explanations"
    )
