"""
Analytical Reports Runner.

Automatically writes all reusable SQL query scripts to scripts/sql/ and runs
them against the active database to print analytical tables.
"""

import os
import sys
from pathlib import Path
from sqlalchemy import text

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import logger
from backend.database.database import SessionLocal

SQL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sql"

# Dictionary of all reporting SQL statements
SQL_QUERIES = {
    "total_customers": "SELECT COUNT(*) AS total_customers FROM customers;",
    "churn_count": "SELECT COUNT(*) AS churn_count FROM customers WHERE churn = 1;",
    "avg_monthly_charges": "SELECT ROUND(AVG(monthly_charges)::numeric, 2) AS avg_monthly_charges FROM billings;",
    "avg_tenure": "SELECT ROUND(AVG(tenure_months)::numeric, 2) AS avg_tenure_months FROM customers;",
    "contract_distribution": """
        SELECT contract_type, COUNT(*) AS count
        FROM contracts JOIN customers ON customers.contract_id = contracts.id
        GROUP BY contract_type
        ORDER BY count DESC;
    """,
    "internet_service_distribution": """
        SELECT internet_service, COUNT(*) AS count
        FROM services JOIN customers ON customers.service_id = services.id
        GROUP BY internet_service
        ORDER BY count DESC;
    """,
    "payment_method_distribution": """
        SELECT payment_method, COUNT(*) AS count
        FROM contracts JOIN customers ON customers.contract_id = contracts.id
        GROUP BY payment_method
        ORDER BY count DESC;
    """,
    "top_revenue_customers": """
        SELECT customer_id, total_charges
        FROM customers JOIN billings ON customers.billing_id = billings.id
        ORDER BY total_charges DESC
        LIMIT 10;
    """,
    "gender_distribution": "SELECT gender, COUNT(*) AS count FROM customers GROUP BY gender;",
    "senior_citizen_distribution": "SELECT senior_citizen, COUNT(*) AS count FROM customers GROUP BY senior_citizen;",
}


def write_sql_files() -> None:
    """
    Ensure the scripts/sql/ directory exists and write each query to a file.
    """
    os.makedirs(SQL_DIR, exist_ok=True)
    for name, query in SQL_QUERIES.items():
        file_path = SQL_DIR / f"{name}.sql"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(query.strip() + "\n")
    logger.info(f"Wrote {len(SQL_QUERIES)} SQL files to: {SQL_DIR}")


def run_reports() -> None:
    """
    Execute all SQL query scripts and print formatting tables to stdout.
    """
    db = SessionLocal()
    try:
        # SQLite adaptation if running locally in SQLite test mode
        dialect = db.bind.dialect.name
        logger.info(f"Executing SQL reports on {dialect} database engine...")

        print("\n" + "=" * 50)
        print("CUSTOMER INTELLIGENCE PLATFORM - DATABASE REPORTS")
        print("=" * 50)

        for name, query_str in SQL_QUERIES.items():
            print(f"\n>> Report: {name.upper().replace('_', ' ')}")
            print("-" * 50)

            # SQLite double colon casting fix
            if dialect == "sqlite":
                query_str = query_str.replace("::numeric", "")

            # Execute query
            result = db.execute(text(query_str))

            # Fetch columns and print rows
            keys = result.keys()
            print(" | ".join(keys))
            print("-" * 50)
            rows = result.all()
            for row in rows:
                print(" | ".join(str(val) for val in row))

        print("\n" + "=" * 50)

    except Exception as e:
        logger.error(f"Error running database reports: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    write_sql_files()
    run_reports()
