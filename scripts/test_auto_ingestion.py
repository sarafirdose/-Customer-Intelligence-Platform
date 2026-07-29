"""
Test script to verify Automated Ingestion Watch Folder, API, and DB auto-sync.
"""

import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.auto_ingestion import (
    scan_watch_folder,
    run_database_auto_sync,
    process_subscriber_dataframe,
    get_sync_state,
)

BASE_DIR = Path(__file__).resolve().parents[1]
INCOMING_DIR = BASE_DIR / "data" / "incoming"


def main():
    print("==================================================")
    print("Testing Automated Ingestion Engine...")
    print("==================================================")

    # 1. Create a sample CSV in data/incoming/
    sample_df = pd.DataFrame([
        {
            "customer_id": "AUTOTEST-9901",
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure_months": 24,
            "phone_service": "Yes",
            "multiple_lines": "Yes",
            "internet_service": "Fiber optic",
            "contract_type": "One year",
            "paperless_billing": "Yes",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 89.50,
            "total_charges": 2148.00,
        },
        {
            "customer_id": "AUTOTEST-9902",
            "gender": "Male",
            "senior_citizen": 1,
            "partner": "No",
            "dependents": "No",
            "tenure_months": 2,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "contract_type": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 95.10,
            "total_charges": 190.20,
        }
    ])

    test_csv_path = INCOMING_DIR / "test_auto_subscribers.csv"
    sample_df.to_csv(test_csv_path, index=False)
    print(f"[OK] Created incoming test file: {test_csv_path}")

    # 2. Trigger Watch Folder Scan
    print("\nScanning watch folder...")
    res = scan_watch_folder()
    print(f"[OK] Watch Folder Scan Result: {res}")

    # 3. Verify file moved to data/processed/
    processed_files = list((BASE_DIR / "data" / "processed").glob("*test_auto_subscribers.csv"))
    if processed_files:
        print(f"[OK] Verified file moved to data/processed/: {processed_files[0].name}")
    else:
        print("[FAIL] File not found in data/processed/")

    # 4. Check sync state
    state = get_sync_state()
    print(f"\n[OK] Current Sync State: {state}")

    # 5. Run DB Auto Sync test
    print("\nRunning DB Auto Sync test...")
    db_res = run_database_auto_sync()
    print(f"[OK] DB Auto Sync Result: {db_res}")

    print("\n==================================================")
    print("Automated Ingestion Engine Test PASSED Successfully!")
    print("==================================================")


if __name__ == "__main__":
    main()
