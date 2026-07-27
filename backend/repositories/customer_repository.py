"""
Customer repository module.

Provides database access queries and operations for Customer entities using the
Repository pattern to isolate database interactions.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from backend.models.customer import Customer


class CustomerRepository:
    """
    Repository class executing CRUD operations for Customer entities.
    """

    def __init__(self, db: Session) -> None:
        """
        Initialize the customer repository.

        Args:
            db: Scoped active SQLAlchemy database session.
        """
        self.db = db

    def get_by_id(self, id_: int) -> Optional[Customer]:
        """
        Fetch a customer by their internal database integer key.

        Args:
            id_: Internal database ID.

        Returns:
            Optional[Customer]: The customer if found, else None.
        """
        return self.db.query(Customer).filter(Customer.id == id_).first()

    def get_by_customer_id(self, customer_id: str) -> Optional[Customer]:
        """
        Fetch a customer by their unique alphanumeric customer_id string.

        Args:
            customer_id: Alphanumeric customer key.

        Returns:
            Optional[Customer]: The customer if found, else None.
        """
        return self.db.query(Customer).filter(Customer.customer_id == customer_id).first()

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Customer]:
        """
        Fetch a paginated list of customers.

        Args:
            limit: Maximum count to return.
            offset: Index offset for pagination.

        Returns:
            List[Customer]: List of customer objects.
        """
        return self.db.query(Customer).offset(offset).limit(limit).all()

    def save(self, customer: Customer) -> Customer:
        """
        Persist a new or modified customer record.

        Args:
            customer: The Customer instance to commit.

        Returns:
            Customer: The persisted Customer record.
        """
        self.db.add(customer)
        self.db.commit()
        self.db.refresh(customer)
        return customer

    def update_predictions(
        self, customer_id: str, churn_risk: float, predicted_ltv: float
    ) -> Optional[Customer]:
        """
        Save prediction results to the customer table cache.

        Args:
            customer_id: Target customer key.
            churn_risk: Model predicted probability of churn.
            predicted_ltv: Model estimated LTV.

        Returns:
            Optional[Customer]: The updated Customer record if found.
        """
        customer = self.get_by_customer_id(customer_id)
        if customer is not None:
            customer.churn_risk = churn_risk
            customer.predicted_ltv = predicted_ltv
            self.db.commit()
            self.db.refresh(customer)
        return customer
