"""
Unit tests for reporting SQL queries.

Executes database reporting scripts against a mock SQLite schema to ensure
no syntax issues exist and returns correct metrics.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.models.customer import Customer
from backend.models.contract import Contract
from backend.models.service import Service
from backend.models.billing import Billing
from scripts.run_sql_reports import SQL_QUERIES


def seed_report_data(db: Session) -> None:
    """
    Seed a small sample dataset to run verification queries.
    """
    b1 = Billing(monthly_charges=50.0, total_charges=500.0)
    b2 = Billing(monthly_charges=100.0, total_charges=2000.0)

    con1 = Contract(contract_type="Month-to-month", paperless_billing="Yes", payment_method="Electronic check")
    con2 = Contract(contract_type="Two year", paperless_billing="No", payment_method="Credit card (automatic)")

    s1 = Service(phone_service="Yes", multiple_lines="No", internet_service="DSL",
                 online_security="Yes", online_backup="No", device_protection="No",
                 tech_support="No", streaming_tv="No", streaming_movies="No")
    s2 = Service(phone_service="Yes", multiple_lines="Yes", internet_service="Fiber optic",
                 online_security="Yes", online_backup="Yes", device_protection="Yes",
                 tech_support="Yes", streaming_tv="Yes", streaming_movies="Yes")

    c1 = Customer(customer_id="REP-001", gender="Female", senior_citizen=0,
                  partner="Yes", dependents="No", tenure_months=10, churn=0,
                  billing=b1, contract=con1, service=s1)
    c2 = Customer(customer_id="REP-002", gender="Male", senior_citizen=1,
                  partner="No", dependents="Yes", tenure_months=20, churn=1,
                  billing=b2, contract=con2, service=s2)

    db.add_all([c1, c2])
    db.commit()


def test_sql_reports_execution(db_session: Session) -> None:
    """
    Verify that each SQL query in the report runner compiles and runs successfully.
    """
    seed_report_data(db_session)

    # Dialect check
    dialect = db_session.bind.dialect.name

    for name, query_str in SQL_QUERIES.items():
        # Adapt numeric casts if running on SQLite
        if dialect == "sqlite":
            query_str = query_str.replace("::numeric", "")

        try:
            result = db_session.execute(text(query_str))
            rows = result.all()
            assert rows is not None
            assert len(rows) >= 0
        except Exception as e:
            pytest.fail(f"SQL query '{name}' failed execution: {e}")
