# Customer Intelligence Platform - Architecture Design

This document details the system design, request flow, and database schema for the Customer Intelligence Platform (CIP).

---

## 1. System Architecture

The platform is designed around **Clean Architecture** principles to separate core business entities, storage repositories, use cases, and delivery channels.

```mermaid
graph TD
    subgraph Client Layer
        Dash[Streamlit Dashboard]
        API_Client[REST API Client / Superset]
    end

    subgraph Service Delivery Layer
        API[FastAPI HTTP Web Server]
        Auth[Security & Middleware]
    end

    subgraph Business Logic Layer
        Service[PredictService]
        Repo[CustomerRepository]
        MLEngine[PredictionEngine]
        LTV[LtvEngine]
        SHAP[ModelExplainer]
    end

    subgraph Data & Storage Layer
        DB[(PostgreSQL Database)]
        Alembic[Alembic Migrations]
        Registry[Artifacts Model Registry]
    end

    %% Routing Flows
    Dash -->|HTTP POST/GET| API
    API_Client -->|HTTP POST/GET| API
    API --> Auth
    Auth --> Service
    Service --> Repo
    Service --> MLEngine
    Service --> LTV
    Service --> SHAP
    Repo -->|SQLAlchemy ORM| DB
    MLEngine -->|Loads PKL| Registry
    SHAP -->|Loads Explainer| Registry
    Alembic -->|Version Schema| DB
```

---

## 2. Request Lifecycle

The diagram below details the sequence of operations for scoring a customer record:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI Router
    participant Service as PredictService
    participant Repo as CustomerRepository
    participant ML as PredictionEngine
    participant DB as PostgreSQL

    Client->>API: POST /api/v1/predict {customer_id, tenure...}
    API->>API: Request Validation (Pydantic)
    API->>Service: process_and_predict(request)
    Service->>Repo: get_by_customer_id(id)
    Repo->>DB: Query customers table
    DB-->>Repo: Return record (or Null)
    Service->>ML: predict_churn_probability(features)
    ML-->>Service: Return score (e.g. 0.28)
    Service->>Repo: save/update_predictions(customer_id, scores)
    Repo->>DB: UPDATE/INSERT
    DB-->>Repo: Confirm commit
    Service-->>API: Return response payload
    API-->>Client: HTTP 200 OK {is_churn, ltv, model_version...}
```

---

## 3. Database Entity-Relationship (ER) Diagram

The system stores customer telemetry alongside ML prediction cache records:

```mermaid
erDiagram
    CUSTOMER {
        int id PK "Auto-increment primary key"
        string customer_id UK "Unique alphanumeric customer key"
        int tenure_months "Customer duration in months"
        float monthly_charges "Monthly subscription charge"
        float total_charges "Total cumulative charges"
        string contract_type "Billing cycle (Month-to-month, One year, Two year)"
        string paperless_billing "Paperless invoice status (Yes/No)"
        string internet_service "DSL, Fiber optic, or None"
        string tech_support "Tech support status (Yes/No/None)"
        float churn_risk "Cached churn probability"
        float predicted_ltv "Cached LTV estimation"
        datetime created_at "Database insertion timestamp"
        datetime updated_at "Database last modified timestamp"
    }

    MODEL_RUNS {
        int run_id PK "Primary key"
        string model_name "Churn / LTV model identifier"
        string model_version "Model tag (e.g. 1.0.0)"
        float evaluation_accuracy "Validation accuracy metric"
        float evaluation_auc "Validation AUC metric"
        datetime trained_at "Training execution timestamp"
    }

    CUSTOMER ||--o. MODEL_RUNS : "scored by"
```
