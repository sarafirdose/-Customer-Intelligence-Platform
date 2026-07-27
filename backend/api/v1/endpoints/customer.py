"""
REST API Endpoints for Customer Intelligence.

Exposes REST APIs to score customer records, fetch hybrid recommendations,
retrieve LTV predictions, check segmentation profiles, and trigger batch analysis.
"""

from typing import Any, Dict, List
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.models.customer import Customer
from backend.models.contract import Contract
from backend.models.service import Service
from backend.models.billing import Billing
from backend.services.predict_service import PredictService
from backend.ml.intelligence import (
    calculate_intelligence_score,
    generate_recommendation_details,
    calculate_rfm,
)

router = APIRouter()

# Schema declarations
class LtvResponse(BaseModel):
    customer_id: str
    historical_ltv_proxy: float
    expected_remaining_lifetime_months: float
    projected_future_ltv: float

class SegmentResponse(BaseModel):
    customer_id: str
    segment: str
    profile: Dict[str, Any]

class IntelligenceResponse(BaseModel):
    customer_id: str
    score: float
    category: str

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[Dict[str, Any]]

class UnifiedIntelligenceResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_ltv: float
    projected_future_ltv: float
    customer_segment: str
    rfm_persona: str
    intelligence_score: float
    intelligence_category: str
    recommendations: List[Dict[str, Any]]

class BatchIntelligenceRequest(BaseModel):
    customer_ids: List[str]


def fetch_customer_sample(db: Session, customer_id: str) -> Dict[str, Any]:
    """
    Fetch a customer from DB and format into a key-value dictionary for ML pipelines.
    """
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer not found with id: {customer_id}"
        )

    # Resolve relationships
    con = customer.contract
    srv = customer.service
    bil = customer.billing

    sample = {
        "customer_id": customer.customer_id,
        "gender": customer.gender,
        "senior_citizen": customer.senior_citizen,
        "partner": customer.partner,
        "dependents": customer.dependents,
        "tenure_months": customer.tenure_months,
        "contract_type": con.contract_type if con else "Month-to-month",
        "paperless_billing": con.paperless_billing if con else "No",
        "payment_method": con.payment_method if con else "Mailed check",
        "phone_service": srv.phone_service if srv else "No",
        "multiple_lines": srv.multiple_lines if srv else "No",
        "internet_service": srv.internet_service if srv else "No",
        "online_security": srv.online_security if srv else "No",
        "online_backup": srv.online_backup if srv else "No",
        "device_protection": srv.device_protection if srv else "No",
        "tech_support": srv.tech_support if srv else "No",
        "streaming_tv": srv.streaming_tv if srv else "No",
        "streaming_movies": srv.streaming_movies if srv else "No",
        "monthly_charges": float(bil.monthly_charges) if bil else 0.0,
        "total_charges": float(bil.total_charges) if bil else 0.0,
    }

    # Count YES services
    service_cols = [
        "phone_service", "multiple_lines", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv", "streaming_movies"
    ]
    sample["total_services"] = sum(1 for col in service_cols if str(sample[col]).strip().lower() == "yes")

    return sample


def run_intelligence_calculations(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run predictions, segmentations, and recommendations on-the-fly.
    """
    # 1. Churn probability
    predict_service = PredictService()
    churn_res = predict_service.predict_proba(sample)
    churn_prob = churn_res["probability"]

    # 2. LTV Predictions
    # Load LTV model from models folder
    import joblib
    base_dir = Path(__file__).resolve().parents[4]
    ltv_model_path = base_dir / "models" / "ltv_model.pkl"
    
    if not ltv_model_path.exists():
        # Fallback path in artifacts registry
        ltv_model_path = base_dir / "artifacts" / "models" / "ltv_model.pkl"

    if ltv_model_path.exists():
        ltv_pipeline = joblib.load(ltv_model_path)
        # Create a single row DF
        from backend.ml.training import engineer_features
        import pandas as pd
        df_row = pd.DataFrame([sample])
        df_eng = engineer_features(df_row)
        predicted_ltv = float(ltv_pipeline.predict(df_eng)[0])
    else:
        # Fallback historical proxy if models are not generated yet
        predicted_ltv = float(sample["total_charges"])

    # Forecast Projected Future LTV
    expected_remaining_lifetime = max(0.0, (1.0 / max(0.01, churn_prob)) - sample["tenure_months"])
    projected_future_ltv = expected_remaining_lifetime * sample["monthly_charges"]

    # 3. Customer Segment K-Means
    seg_model_path = base_dir / "models" / "segmentation_model.pkl"
    if not seg_model_path.exists():
        seg_model_path = base_dir / "artifacts" / "models" / "segmentation_model.pkl"

    segment = "Silver"  # Fallback default
    if seg_model_path.exists():
        seg_pipeline = joblib.load(seg_model_path)
        numeric_cols = ["tenure_months", "monthly_charges", "total_services"]
        
        # Calculate charges ratio
        sample_ratio = sample["monthly_charges"] / (sample["tenure_months"] + 1)
        X_cluster = [[sample["tenure_months"], sample["monthly_charges"], sample["total_services"], sample_ratio]]
        
        # Scale and predict cluster raw ID
        cluster_id = int(seg_pipeline.named_steps["kmeans"].predict(seg_pipeline.named_steps["scaler"].transform(X_cluster))[0])
        # Map cluster ID to name based on historical proxy (Platinum, Gold, Silver, Bronze)
        # For simplicity, map using a standard modulo index (or typical cluster properties)
        names = ["Platinum", "Gold", "Silver", "Bronze"]
        segment = names[cluster_id % len(names)]

    # 4. RFM Persona
    # Map RFM Score to Persona on-the-fly
    r_score = int(min(5, max(1, round((1.0 - churn_prob) * 5))))
    # Binned frequency based on tenure (0-12m -> 1, 12-24m -> 2, etc.)
    f_score = min(5, max(1, sample["tenure_months"] // 15 + 1))
    # Binned monetary based on total charges
    m_score = min(5, max(1, int(sample["total_charges"] // 1500 + 1)))
    
    def get_persona(r, f, m) -> str:
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 4:
            return "Loyal Customers"
        elif r >= 3 and f >= 3 and m < 4:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2:
            return "Lost Customers"
        return "About to Sleep"

    persona = get_persona(r_score, f_score, m_score)

    # 5. Composite Customer Intelligence Score (0-100)
    score, category = calculate_intelligence_score(
        churn_prob, predicted_ltv, sample["tenure_months"], sample["total_services"]
    )

    # 6. Hybrid Recommendation Engine details
    # Generate recommendations
    recs = generate_recommendation_details(
        sample, churn_prob, predicted_ltv, segment, persona
    )

    return {
        "customer_id": sample["customer_id"],
        "churn_probability": churn_prob,
        "predicted_ltv": predicted_ltv,
        "projected_future_ltv": projected_future_ltv,
        "customer_segment": segment,
        "rfm_persona": persona,
        "intelligence_score": score,
        "intelligence_category": category,
        "recommendations": recs,
        "expected_remaining_lifetime_months": expected_remaining_lifetime,
    }


@router.get(
    "/customer/{customer_id}",
    response_model=UnifiedIntelligenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get unified intelligence parameters for a single customer",
)
def get_customer_intelligence(customer_id: str, db: Session = Depends(get_db)) -> UnifiedIntelligenceResponse:
    """
    Score and fetch profile metrics, LTV prediction, K-Means segment, RFM persona,
    unified score, and retention recommendations in a single call.
    """
    sample = fetch_customer_sample(db, customer_id)
    res = run_intelligence_calculations(sample)
    return UnifiedIntelligenceResponse(**res)


@router.get(
    "/customer/{customer_id}/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get recommendations for a customer",
)
def get_customer_recommendations(customer_id: str, db: Session = Depends(get_db)) -> RecommendationResponse:
    """
    Get rule-based retention actions and saved revenue details.
    """
    sample = fetch_customer_sample(db, customer_id)
    res = run_intelligence_calculations(sample)
    return RecommendationResponse(customer_id=customer_id, recommendations=res["recommendations"])


@router.get(
    "/customer/{customer_id}/ltv",
    response_model=LtvResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Customer Lifetime Value predictions",
)
def get_customer_ltv(customer_id: str, db: Session = Depends(get_db)) -> LtvResponse:
    """
    Get LTV historical proxy and remaining lifetime projections.
    """
    sample = fetch_customer_sample(db, customer_id)
    res = run_intelligence_calculations(sample)
    return LtvResponse(
        customer_id=customer_id,
        historical_ltv_proxy=res["predicted_ltv"],
        expected_remaining_lifetime_months=res["expected_remaining_lifetime_months"],
        projected_future_ltv=res["projected_future_ltv"]
    )


@router.get(
    "/customer/{customer_id}/segment",
    response_model=SegmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get K-Means cluster segment for a customer",
)
def get_customer_segment(customer_id: str, db: Session = Depends(get_db)) -> SegmentResponse:
    """
    Get customer K-Means segment name and baseline metrics.
    """
    sample = fetch_customer_sample(db, customer_id)
    res = run_intelligence_calculations(sample)
    
    # Format a profile summary
    profile = {
        "monthly_charges": sample["monthly_charges"],
        "tenure_months": sample["tenure_months"],
        "total_services": sample["total_services"]
    }
    return SegmentResponse(customer_id=customer_id, segment=res["customer_segment"], profile=profile)


@router.get(
    "/customer/{customer_id}/intelligence",
    response_model=IntelligenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get unified Customer Intelligence Score",
)
def get_customer_intelligence_score(customer_id: str, db: Session = Depends(get_db)) -> IntelligenceResponse:
    """
    Get unified score (0-100) and risk category.
    """
    sample = fetch_customer_sample(db, customer_id)
    res = run_intelligence_calculations(sample)
    return IntelligenceResponse(
        customer_id=customer_id,
        score=res["intelligence_score"],
        category=res["intelligence_category"]
    )


@router.post(
    "/customers/batch_intelligence",
    response_model=List[UnifiedIntelligenceResponse],
    status_code=status.HTTP_200_OK,
    summary="Trigger batch scoring and intelligence reports",
)
def batch_score_customers(request: BatchIntelligenceRequest, db: Session = Depends(get_db)) -> List[UnifiedIntelligenceResponse]:
    """
    Score a list of customer IDs and return batch customer intelligence records.
    """
    results = []
    for cid in request.customer_ids:
        try:
            sample = fetch_customer_sample(db, cid)
            res = run_intelligence_calculations(sample)
            results.append(UnifiedIntelligenceResponse(**res))
        except HTTPException:
            # Skip invalid customer IDs in batch processing
            continue
    return results
