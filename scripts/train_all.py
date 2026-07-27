"""
CLI Script to execute the ML training pipeline.

Performs visual logs during training executions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ml.training import run_pipeline


def train() -> None:
    """
    Triggers execution of the ML pipeline.
    """
    print("Loading Dataset...")
    print("Engineering Features...")
    print("Splitting Dataset...")
    
    # run_pipeline handles all evaluation, cross-val, and saving
    run_pipeline()
    
    print("Completed")


if __name__ == "__main__":
    train()
