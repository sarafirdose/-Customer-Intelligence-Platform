# Customer Intelligence Platform (CIP)

> **AI-Powered Customer Churn Prediction & Lifetime Value (LTV) Engine**
> An enterprise-grade, clean-architecture analytics platform built using FastAPI, PostgreSQL, and Scikit-Learn.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Architectural Pillars](#-key-architectural-pillars)
3. [Repository Directory Structure](#-repository-directory-structure)
4. [Getting Started & Installation](#-getting-started--installation)
5. [Running Locally](#-running-locally)
6. [Dockerization Setup](#-dockerization-setup)
7. [Testing and Quality Assurance](#-testing-and-quality-assurance)
8. [Phase Roadmap](#-phase-roadmap)

---

## 🌟 Project Overview

The **Customer Intelligence Platform (CIP)** predicts subscription customer churn risks, calculates estimated customer lifetime value (LTV), and serves explainable AI metrics (SHAP feature attributions).

This repository represents the initial foundation (**Phase 0**) focusing on structural skeleton initialization, logging setups, testing rigs, container deployment configurations, and SQLAlchemy database integrations.

---

## 🏗️ Key Architectural Pillars

The platform is designed following **Clean Architecture** principles, decoupling concerns into distinct layers:

- **Entity Layer (`backend/models`)**: Defines core database-mapped models (e.g. `Customer` schema).
- **Service Layer (`backend/services`)**: Implements business transactions, orchestrating data engineering and predictions.
- **Repository Layer (`backend/repositories`)**: Isolates raw database query statements from endpoints.
- **API Router Layer (`backend/api`)**: Manages routing, endpoint logic, versioning (`/api/v1/`), and payload schemas (Pydantic validation).
- **ML Pipeline (`backend/ml`)**: Holds core pipeline scripts for data ingestion, training, evaluation, explainability, and feature engineering.

*For detailed sequence diagrams and database design diagrams, see [docs/architecture.md](docs/architecture.md).*

---

## 📂 Repository Directory Structure

```text
Customer-Intelligence-Platform/
├── .github/                    # GitHub actions workflows and templates
├── alembic/                    # Database migration configuration scripts
├── artifacts/                  # Registry containing serialized models, encoders, and scalers
├── backend/                    # Core Python API and ML package
│   ├── api/                    # API Routing and endpoint versions
│   ├── core/                   # Shared configurations and loggers
│   ├── database/               # Session creators and engine connections
│   ├── ml/                     # Machine learning workflows (ingestion, training, explanation)
│   ├── models/                 # SQLAlchemy DB models
│   ├── repositories/           # Database transaction repositories
│   ├── schemas/                # Pydantic input/output validation schemas
│   ├── services/               # Orchestrated business logic services
│   └── utils/                  # Shared helper utilities
├── dashboard/                  # Analytical interfaces (Streamlit application)
├── data/                       # Local raw and processed datasets
├── docker/                     # Docker build and setup utilities
├── docs/                       # Architectural diagrams and Postman testing templates
├── scripts/                    # Maintenance, backup, and training utilities
├── tests/                      # Pytest automated testing suite
├── Dockerfile                  # Application multi-stage build file
├── docker-compose.yml          # Container configuration (FastAPI + PostgreSQL)
├── Makefile                    # Developer shortcut commands
├── pyproject.toml              # Tool specifications (black, isort, mypy)
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Getting Started & Installation

### Prerequisites
- Python 3.10 or higher installed.
- Docker & Docker Compose installed.

### Virtual Environment Setup
Clone the repository and initialize a local environment:

```bash
# Navigate to project root
cd Customer-Intelligence-Platform

# Create Python virtual environment
python -m venv venv

# Activate the environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip and install package dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Running Locally

### 1. Environment Configurations
Copy the example configurations file to create `.env` and adjust settings as required:
```bash
cp .env.example .env
```

### 2. Run Database Seeding
To download the customer churn dataset and seed the DB:
```bash
# Download the IBM Telco Churn CSV
python scripts/download_dataset.py

# Seed records to the database
python scripts/load_database.py
```

### 3. Launch Local REST API
Execute the local server using Uvicorn:
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```
- API Documentation (Swagger UI) is available at: [http://localhost:8000/docs](http://localhost:8000/docs)
- API Redoc is available at: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Launch Dashboard
Run the Streamlit frontend interface:
```bash
streamlit run dashboard/app.py
```
- Streamlit application is available at: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Dockerization Setup

To boot the complete application suite (FastAPI Backend + PostgreSQL Database) in isolated containers:

```bash
# Build and run containers in background
docker compose up -d

# View execution logs
docker compose logs -f
```

- PostgreSQL container listens on port `5432` internally.
- FastAPI backend binds to port `8000` (mapped to host machine).

To tear down running containers:
```bash
docker compose down -v
```

---

## 🧪 Testing and Quality Assurance

### Run Unit Tests
To run unit and integration tests (with coverage reports):
```bash
pytest
# Or using make:
make test
```

### Run Formatters & Linters
Format codebase and check syntax standards:
```bash
# Format files
black backend/ tests/ scripts/
isort backend/ tests/ scripts/

# Run static syntax checks
flake8 backend/
mypy backend/
```

---

## 🗺️ Phase Roadmap

- [x] **Phase 0: Project Setup** (Completed)
  - Folder structures, configs, and dependency tracking.
  - Logging, connection drivers, and health checks.
  - Multi-stage Docker config and API skeleton versioning.
  - Automated testing rig setup.
- [x] **Phase 1: Data Engineering & Database** (Completed)
  - Normalized database schema (customers, contracts, services, billing, predictions placeholders).
  - Robust multi-stage ETL modules (Extract, Profile, Validate, Clean, Transform, Load).
  - Pandera schema validations & Pydantic settings loading.
  - SQL reporting scripts and background REST API endpoints.
- [ ] **Phase 2: Exploratory Data Analysis & ETL Pipeline**
- [ ] **Phase 3: Feature Engineering & Data Validation**
- [ ] **Phase 4: Model Development & LTV Calculation**
- [ ] **Phase 5: API Integration & Model Explainability (SHAP)**
- [ ] **Phase 6: Dashboard Development & Analytics Superset Views**
- [ ] **Phase 7: Deployment, Monitoring & Alerting**

---

## 💾 Phase 1: Data Ingestion & Database Details

### 1. Dataset Source
We use the standard **Kaggle Telco Customer Churn dataset** (`blastchar/telco-customer-churn`).
- **Automatic Retrieval**: If Kaggle API credentials are present (`KAGGLE_USERNAME` & `KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`), the pipeline downloads it automatically.
- **Manual Placement**: Alternatively, you can download `WA_Fn-UseC_-Telco-Customer-Churn.csv` manually and place it as `data/raw/telco_customer_churn.csv`.

### 2. Database Schema (Normalized)
We have transitioned from a single flat wide table to a highly normalized relational database design:
- `contracts`: Tracks billing cycles (`contract_type`), `paperless_billing`, and `payment_method`.
- `services`: Stores details on phone lines, multiple lines, internet options, security add-ons, tech support, and streaming.
- `billings`: Contains `monthly_charges` and `total_charges`.
- `customers`: Parent table linking customer demographics (`gender`, `senior_citizen`, `partner`, `dependents`), `tenure_months`, and `churn` targets to contracts, services, and billings tables.
- `predictions`, `ltv_predictions`, `recommendations`: Placed as placeholder tables for future phases.
- `import_histories`: Tracks ETL statistics, running status, processed records, and timestamps.

### 3. Ingestion Process Flow
The ingestion pipeline runs in 6 modular stages:
1. **Extract**: Detects local file or downloads from Kaggle.
2. **Profile**: Generates missingness and numeric summaries under `reports/data_profile.json`.
3. **Validate (Raw)**: Applies Pandera schema to raw shapes.
4. **Clean**: Normalizes categories, drops duplicate IDs, coerces data types, and imputes empty `TotalCharges` strings to 0.0 for new customers. Writes logs to `reports/validation/cleaning_report.json`.
5. **Transform**: Formats cleaned tabular rows into SQLAlchemy ORM graphs.
6. **Load**: Inserts records in batches of 500. Checks for existing records to enforce idempotency and prevent duplicate records.

### 4. Running Ingestion & SQL Reports
To run the full ETL pipeline and view database summary tables:

```bash
# 1. Download and ingest data to database
python scripts/download_dataset.py
python scripts/load_database.py

# Or run it via REST API
curl -X POST http://localhost:8000/api/v1/ingest

# 2. View database distribution reports
python scripts/run_sql_reports.py
```

### 5. Troubleshooting
- **Kaggle Authentication Error**: If automated downloading fails, verify your Kaggle JSON token file exists at `~/.kaggle/kaggle.json` or paste the API CSV manually into `data/raw/telco_customer_churn.csv`.
- **Database Connection Failures**: Verify your environment configurations in `.env` match your PostgreSQL port and credentials. If running locally without Docker, you can set `USE_SQLITE_TEST=true` and `ENV=testing` to redirect operations to a local file-based database for debugging.

