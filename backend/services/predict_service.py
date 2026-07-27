"""
Customer prediction service.

Orchestrates data fetching, feature mapping, model pipeline inference,
explanations generation, and persistence operations.
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.core.logger import logger
from backend.repositories.customer_repository import CustomerRepository
from backend.schemas.customer import CustomerPredictRequest, CustomerPredictResponse
from backend.models.customer import Customer


class PredictService:
    """
    Service class managing client prediction lifecycle.
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize prediction service with repository.

        Args:
            db: Database session.
        """
        self.repository = CustomerRepository(db)

    def process_and_predict(
        self, request: CustomerPredictRequest
    ) -> CustomerPredictResponse:
        """
        Process incoming features, execute ML models, save history, and cache values.

        Args:
            request: Validated customer scoring request.

        Returns:
            CustomerPredictResponse: Prediction outputs.
        """
        logger.info(f"Incoming prediction request for customer: {request.customer_id}")

        # Check if customer already exists in database
        db_customer = self.repository.get_by_customer_id(request.customer_id)

        # Generate predictions (placeholder calculation)
        churn_prob = 0.28
        ltv = 1500.00
        is_churn = churn_prob > 0.5

        if db_customer is None:
            # Create a new record in our database
            db_customer = Customer(
                customer_id=request.customer_id,
                tenure_months=request.tenure_months,
                monthly_charges=request.monthly_charges,
                total_charges=request.total_charges,
                contract_type=request.contract_type,
                paperless_billing=request.paperless_billing,
                internet_service=request.internet_service,
                tech_support=request.tech_support,
                churn_risk=churn_prob,
                predicted_ltv=ltv,
            )
            self.repository.save(db_customer)
            logger.info(f"Created new customer record in DB for {request.customer_id}")
        else:
            # Update predictions on existing record
            self.repository.update_predictions(
                customer_id=request.customer_id,
                churn_risk=churn_prob,
                predicted_ltv=ltv,
            )
            logger.info(f"Updated predictions cache for customer: {request.customer_id}")

        return CustomerPredictResponse(
            customer_id=request.customer_id,
            churn_probability=churn_prob,
            is_churn=is_churn,
            predicted_ltv=ltv,
            model_version="1.0.0",
        )
