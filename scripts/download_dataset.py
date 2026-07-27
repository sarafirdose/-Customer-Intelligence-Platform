"""
Download Dataset Utility Script.

Downloads the standard IBM/Kaggle Telco Customer Churn dataset from a public raw URL
and saves it to the local raw data directory.
"""

import os
import urllib.request
from pathlib import Path
import pandas as pd

# Set up logging dummy to match script conventions
print_info = lambda msg: print(f"[INFO] {msg}")
print_err = lambda msg: print(f"[ERROR] {msg}")

# Standard raw dataset URL
DATASET_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "telco_customer_churn.csv"


def download_dataset() -> None:
    """
    Download the dataset from the public repository.

    Saves it locally or generates a mock fallback dataset if offline.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_info(f"Target download folder: {OUTPUT_DIR}")

    try:
        print_info(f"Attempting to download dataset from: {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_FILE)
        print_info(f"Dataset downloaded successfully and saved to {OUTPUT_FILE}")
    except Exception as e:
        print_err(f"Failed to download remote dataset: {e}")
        print_info("Generating a mock dataset with exact Kaggle schema for local development fallback...")

        # Generate local mock dataset matching Kaggle columns exactly
        mock_data = pd.DataFrame(
            {
                "customerID": [f"{i:04d}-MOCK" for i in range(1, 101)],
                "gender": ["Female" if i % 2 == 0 else "Male" for i in range(1, 101)],
                "SeniorCitizen": [0 if i % 3 != 0 else 1 for i in range(1, 101)],
                "Partner": ["Yes" if i % 2 == 0 else "No" for i in range(1, 101)],
                "Dependents": ["No" if i % 3 != 0 else "Yes" for i in range(1, 101)],
                "tenure": [1 + (i % 72) for i in range(1, 101)],
                "PhoneService": ["Yes" if i % 8 != 0 else "No" for i in range(1, 101)],
                "MultipleLines": ["No" if i % 2 == 0 else "Yes" for i in range(1, 101)],
                "InternetService": ["DSL" if i % 3 == 1 else ("Fiber optic" if i % 3 == 2 else "No") for i in range(1, 101)],
                "OnlineSecurity": ["Yes" if i % 4 == 0 else "No" for i in range(1, 101)],
                "OnlineBackup": ["No" if i % 4 == 0 else "Yes" for i in range(1, 101)],
                "DeviceProtection": ["Yes" if i % 4 == 1 else "No" for i in range(1, 101)],
                "TechSupport": ["No" if i % 4 == 2 else "Yes" for i in range(1, 101)],
                "StreamingTV": ["Yes" if i % 2 == 0 else "No" for i in range(1, 101)],
                "StreamingMovies": ["No" if i % 2 == 0 else "Yes" for i in range(1, 101)],
                "Contract": ["Month-to-month" if i % 3 == 0 else ("One year" if i % 3 == 1 else "Two year") for i in range(1, 101)],
                "PaperlessBilling": ["Yes" if i % 3 != 0 else "No" for i in range(1, 101)],
                "PaymentMethod": ["Electronic check" if i % 2 == 0 else "Mailed check" for i in range(1, 101)],
                "MonthlyCharges": [29.95 + (i * 0.5) for i in range(1, 101)],
                "TotalCharges": [str((1 + (i % 72)) * (29.95 + (i * 0.5))) for i in range(1, 101)],
                "Churn": ["Yes" if i % 5 == 0 else "No" for i in range(1, 101)],
            }
        )
        # Inject space in TotalCharges to verify cleaning pipeline casts empty charges correctly
        mock_data.loc[10, "TotalCharges"] = " "
        mock_data.loc[10, "tenure"] = 0

        mock_data.to_csv(OUTPUT_FILE, index=False)
        print_info(f"Mock fallback dataset generated and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    download_dataset()
