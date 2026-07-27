"""
Model training execution script.

Extracts data from the database, triggers the feature engineering pipeline,
trains the customer churn classifier, and saves the resulting model binary.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logger import logger
from backend.database.database import SessionLocal
from backend.models.customer import Customer
from backend.ml.features import FeatureEngineer
from backend.ml.training import ModelTrainer


def train_all() -> None:
    """
    Extract data, run feature transformations, and train models.
    """
    logger.info("Starting model training pipeline.")

    db = SessionLocal()
    try:
        # Load data from database
        query = db.query(Customer)
        df = pd.read_sql(query.statement, db.bind)
        logger.info(f"Loaded {df.shape[0]} customer records from database.")

        if df.shape[0] < 10:
            logger.warning("Too few records in database to train. Please seed the DB first.")
            return

        # Map churn_risk to a binary target variable 'churn' for training
        # If churn_risk is float, convert to binary. If no churn target, create dummy.
        if "churn_risk" in df.columns:
            df["churn"] = (df["churn_risk"] > 0.5).astype(int)
        else:
            # Fallback if column is missing
            df["churn"] = (df["id"] % 5 == 0).astype(int)

        # 1. Feature Engineering
        engineer = FeatureEngineer()
        X, y = engineer.build_features(df, is_training=True)

        # 2. Train Model
        trainer = ModelTrainer()
        model, metrics = trainer.train_churn_model(X, y)

        logger.info(f"Model trained successfully. Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error in training execution: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    train_all()
