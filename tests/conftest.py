"""
Configuration and shared fixtures for the pytest suite.

Overrides production configurations to use a local test SQLite database
and exposes mock request clients.
"""

import os
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# 1. Force testing environment before loading configurations
os.environ["ENV"] = "testing"
os.environ["USE_SQLITE_TEST"] = "true"

from backend.api.main import app
from backend.database.database import get_db
from backend.database.models import Base
from backend.models.contract import Contract
from backend.models.service import Service
from backend.models.billing import Billing
from backend.models.customer import Customer
from backend.models.prediction import Prediction, LtvPrediction, Recommendation
from backend.models.import_history import ImportHistory

# Setup temporary SQLite database URL
TEST_DB_URL = "sqlite:///./test_temp.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_db() -> Generator[None, None, None]:
    """
    Session-scoped database table setup and teardown.
    """
    # Create tables in test database
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after session finishes
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists("test_temp.db"):
        os.remove("test_temp.db")


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """
    Fixture providing a clean isolated database session per test.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    Fixture yielding a FastAPI TestClient with overridden database dependencies.
    """

    def _get_test_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    # Override get_db dependency in FastAPI application
    app.dependency_overrides[get_db] = _get_test_db
    with TestClient(app) as test_client:
        yield test_client

    # Reset overrides after test completes
    app.dependency_overrides.clear()
