"""
Unit tests for database connection and transaction scripts.

Verifies table creation, connection testing utilities, and repository CRUD transactions.
"""

from sqlalchemy.orm import Session
from backend.database.database import test_db_connection
from backend.models.customer import Customer
from backend.models.contract import Contract
from backend.models.service import Service
from backend.models.billing import Billing
from backend.repositories.customer_repository import CustomerRepository


def test_database_connection_check() -> None:
    """
    Test that database connection checking returns True for active DBs.
    """
    assert test_db_connection() is True


def test_customer_model_creation(db_session: Session) -> None:
    """
    Test that database models can be instantiated and saved via repositories.
    """
    repo = CustomerRepository(db_session)

    b = Billing(monthly_charges=85.50, total_charges=3078.00)
    con = Contract(contract_type="One year", paperless_billing="Yes", payment_method="Credit card (automatic)")
    s = Service(phone_service="Yes", multiple_lines="No", internet_service="Fiber optic",
                 online_security="No", online_backup="No", device_protection="No",
                 tech_support="No", streaming_tv="No", streaming_movies="No")

    # Insert record
    new_cust = Customer(
        customer_id="9999-WXYZ",
        gender="Male",
        senior_citizen=0,
        partner="No",
        dependents="No",
        tenure_months=36,
        churn=0,
        billing=b,
        contract=con,
        service=s
    )
    saved = repo.save(new_cust)

    assert saved.id is not None
    assert saved.customer_id == "9999-WXYZ"

    # Query record
    queried = repo.get_by_customer_id("9999-WXYZ")
    assert queried is not None
    assert queried.tenure_months == 36
    assert queried.contract.contract_type == "One year"


def test_customer_dict_serialization() -> None:
    """
    Test table to_dict serialization method.
    """
    b = Billing(monthly_charges=30.0, total_charges=180.0)
    con = Contract(contract_type="Month-to-month", paperless_billing="No", payment_method="Mailed check")
    s = Service(phone_service="Yes", multiple_lines="No", internet_service="DSL",
                 online_security="No", online_backup="No", device_protection="No",
                 tech_support="No", streaming_tv="No", streaming_movies="No")

    cust = Customer(
        customer_id="MOCK-111",
        gender="Male",
        senior_citizen=0,
        partner="No",
        dependents="No",
        tenure_months=6,
        churn=0,
        billing=b,
        contract=con,
        service=s
    )
    # Convert base fields
    c_dict = cust.to_dict()
    assert c_dict["customer_id"] == "MOCK-111"
    assert c_dict["tenure_months"] == 6
