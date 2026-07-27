"""
Integration tests for the ETL Ingestion Pipeline.

Validates Extract -> Profile -> Validate -> Clean -> Transform -> Load stages,
asserts database insertion, tests incremental loading idempotency, and tests
API endpoints.
"""

import os
from pathlib import Path
import pandas as pd
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.models.customer import Customer
from backend.models.contract import Contract
from backend.models.service import Service
from backend.models.billing import Billing
from backend.models.import_history import ImportHistory
from backend.services.ingestion_service import IngestionService

RAW_DATA_DIR = Path("data/raw")
MOCK_CSV = RAW_DATA_DIR / "telco_customer_churn.csv"


def setup_mock_csv() -> None:
    """
    Generate a mock CSV file to run validation testing.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = pd.DataFrame(
        {
            "customerID": ["9999-INGEST1", "9999-INGEST2"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [12, 24],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["Yes", "No"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [45.85, 95.00],
            "TotalCharges": ["550.20", "2280.00"],
            "Churn": ["No", "Yes"],
        }
    )
    df.to_csv(MOCK_CSV, index=False)


def teardown_mock_csv() -> None:
    """
    Clean up mock CSV files.
    """
    if MOCK_CSV.exists():
        os.remove(MOCK_CSV)


def test_ingestion_pipeline_full_run(db_session: Session) -> None:
    """
    Test that the IngestionService successfully validates, cleans, and seeds ORM databases.
    """
    setup_mock_csv()
    try:
        service = IngestionService(db_session)
        metrics = service.run_pipeline()

        assert metrics["status"] == "success"
        assert metrics["rows_processed"] == 2
        assert metrics["rows_inserted"] == 2
        assert metrics["rows_skipped"] == 0

        # Query Customer table
        customers = db_session.query(Customer).all()
        assert len(customers) == 2

        # Check normalization associations
        c1 = db_session.query(Customer).filter(Customer.customer_id == "9999-INGEST1").first()
        assert c1 is not None
        assert c1.gender == "Female"
        assert c1.contract.contract_type == "Month-to-month"
        assert c1.billing.monthly_charges == 45.85
        assert c1.service.internet_service == "DSL"

        # Check that import logs are cached in import_history table
        history = db_session.query(ImportHistory).all()
        assert len(history) == 1
        assert history[0].status == "success"

        # Test IDEMPOTENCY: Run pipeline again
        metrics_retry = service.run_pipeline()
        assert metrics_retry["status"] == "success"
        assert metrics_retry["rows_inserted"] == 0  # No new rows inserted
        assert metrics_retry["rows_skipped"] == 2   # Both skipped due to pre-existence

    finally:
        teardown_mock_csv()


def test_api_ingestion_endpoints(client: TestClient) -> None:
    """
    Test REST API Ingestion status and trigger routes.
    """
    setup_mock_csv()
    try:
        # 1. Check status endpoint (should show dataset available but database unpopulated initially)
        status_res = client.get("/api/v1/dataset/status")
        assert status_res.status_code == status.HTTP_200_OK
        assert status_res.json()["dataset_available"] is True

        # 2. Trigger Ingestion via API
        ingest_res = client.post("/api/v1/ingest")
        assert ingest_res.status_code == status.HTTP_200_OK
        metrics = ingest_res.json()
        assert metrics["status"] == "success"
        assert metrics["rows_inserted"] == 2

        # 3. Check status endpoint again
        status_res_after = client.get("/api/v1/dataset/status")
        assert status_res_after.json()["database_populated"] is True
        assert status_res_after.json()["record_count"] == 2
        assert status_res_after.json()["last_import_time"] is not None

    finally:
        teardown_mock_csv()
