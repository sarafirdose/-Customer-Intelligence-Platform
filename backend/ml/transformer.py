"""
ETL Ingestion - Transformation Stage (Data Transformer).

Maps pandas DataFrame records into SQLAlchemy ORM object graphs
(Customer, Contract, Service, Billing) ready for bulk insertion.
"""

from typing import List, Tuple
import pandas as pd
from backend.core.logger import logger
from backend.models.billing import Billing
from backend.models.contract import Contract
from backend.models.customer import Customer
from backend.models.service import Service


class DataTransformer:
    """
    Transforms tabular DataFrame records into database-ready ORM objects.
    """

    def __init__(self) -> None:
        """
        Initialize the DataTransformer.
        """
        logger.info("Initializing Data Transformer module.")

    def transform(self, df: pd.DataFrame) -> List[Customer]:
        """
        Transform cleaned DataFrame rows into lists of SQLAlchemy Customer graphs.

        Args:
            df: Cleaned customer DataFrame.

        Returns:
            List[Customer]: Mapped ORM objects list.
        """
        logger.info(f"Transformation stage: Mapping {len(df)} records to ORM entity graphs.")
        orm_customers = []

        for _, row in df.iterrows():
            # 1. Instantiate Billing relationship object
            billing = Billing(
                monthly_charges=float(row["monthly_charges"]),
                total_charges=float(row["total_charges"]),
            )

            # 2. Instantiate Contract relationship object
            contract = Contract(
                contract_type=str(row["contract_type"]),
                paperless_billing=str(row["paperless_billing"]),
                payment_method=str(row["payment_method"]),
            )

            # 3. Instantiate Service relationship object
            service = Service(
                phone_service=str(row["phone_service"]),
                multiple_lines=str(row["multiple_lines"]),
                internet_service=str(row["internet_service"]),
                online_security=str(row["online_security"]),
                online_backup=str(row["online_backup"]),
                device_protection=str(row["device_protection"]),
                tech_support=str(row["tech_support"]),
                streaming_tv=str(row["streaming_tv"]),
                streaming_movies=str(row["streaming_movies"]),
            )

            # 4. Instantiate Customer parent object and join relationships
            customer = Customer(
                customer_id=str(row["customer_id"]),
                gender=str(row["gender"]),
                senior_citizen=int(row["senior_citizen"]),
                partner=str(row["partner"]),
                dependents=str(row["dependents"]),
                tenure_months=int(row["tenure_months"]),
                churn=int(row["churn"]),
                billing=billing,
                contract=contract,
                service=service,
            )

            orm_customers.append(customer)

        logger.info("Transformation stage: Mapped completed successfully.")
        return orm_customers
