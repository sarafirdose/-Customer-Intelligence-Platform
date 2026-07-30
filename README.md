# Telecom Customer Intelligence Platform (CIP) 🚀

> **Enterprise AI-Powered Telecom Subscriber Churn Prediction, LTV Forecasting & Automated Ingestion Engine**  
> Built with Python, FastAPI, Streamlit, PostgreSQL, XGBoost/LightGBM, Pandera, APScheduler, Docker & Kubernetes.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-132%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features & Capabilities](#-key-features--capabilities)
3. [Automated Data Ingestion Engine](#-automated-data-ingestion-engine)
4. [Master UI/UX Streamlit Dashboard](#-master-uiux-streamlit-dashboard)
5. [Key Architectural Pillars](#-key-architectural-pillars)
6. [Repository Directory Structure](#-repository-directory-structure)
7. [Getting Started & Installation](#-getting-started--installation)
8. [API Reference](#-api-reference)
9. [Automated Testing & MLOps](#-automated-testing--mlops)
10. [Author & Maintainer](#-author--maintainer)

---

## 🌟 Project Overview

The **Telecom Customer Intelligence Platform (CIP)** is a production-grade machine learning system designed for subscription telecommunications businesses (e.g. Jio, Airtel, Vodafone). It automatically ingests subscriber telemetry, predicts individual churn risks, calculates Customer Lifetime Value (LTV), groups accounts into K-Means clusters, and delivers explainable retention recommendations via an interactive enterprise dashboard.

---

## 🔥 Key Features & Capabilities

- 🤖 **Predictive Churn Engine**: XGBoost & LightGBM classifiers trained on telecom behavioral metrics (**ROC-AUC: 0.847**).
- 💰 **LTV Regression & Growth Forecasting**: Predicts realized subscriber lifetime value and projected future revenue horizons.
- 🎯 **Automated Action Recommendations**: Rules-based + ML retention campaign assignment with estimated ROI revenue recovery metrics.
- 🔄 **Fully Automated Data Ingestion**: Continuous CSV watch folder scanning (`data/incoming/`), incremental PostgreSQL auto-sync, and real-time REST API endpoints.
- 📊 **Master UI/UX Dashboard**: 11 multi-page Streamlit views built with Apple/Stripe-inspired glassmorphism dark theme styling and AI Copilot panels.
- ⚙️ **Production MLOps**: APScheduler background automation (6 jobs), Population Stability Index (PSI) drift monitoring, FileLock Model Registry, and Kubernetes manifests.

---

## 🔄 Automated Data Ingestion Engine

The platform eliminates manual CSV uploads through an automated background ingestion architecture:

```
+-----------------------------------------------------------------------------------+
|                               DATA SOURCES                                        |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |  PostgreSQL DB     |   |  CSV Watch Folder     |   |  Real-Time REST APIs   |  |
|  | (Incremental Sync) |   |  (data/incoming/*.csv)|   | (/ingest/record & batch|  |
|  +---------+----------+   +-----------+-----------+   +-----------+------------+  |
+------------|--------------------------|---------------------------|---------------+
             | (Every 5 mins)           | (Every 1 min)             | (On-demand HTTP)
             v                          v                           v
+-----------------------------------------------------------------------------------+
|                        AUTOMATED INGESTION ENGINE                                 |
|                     (backend/services/auto_ingestion.py)                          |
|                                                                                   |
|  1. Schema Validation -> 2. Feature Engineering -> 3. ML Inference (Churn/LTV/Seg)|
|                                                                                   |
|  * Valid Records   --> Appended to DB & reports/customer_intelligence.csv           |
|  * Processed Files --> Moved to data/processed/YYYYMMDD_filename.csv              |
|  * Failed Records  --> Logged to logs/imports.jsonl & moved to data/failed/       |
+-----------------------------------------------------------------------------------+
```

---

## 🎨 Master UI/UX Streamlit Dashboard

The Streamlit dashboard features 11 multi-page views with a sleek dark glassmorphism design system:

- 📊 **Executive Summary** (`dashboard/pages/11_Executive_Summary.py`): Strategic C-Suite overview.
- 🔍 **Subscriber 360 Explorer** (`dashboard/pages/2_Customers.py`): Live account search & score drift charts.
- 🏆 **Subscriber Segments** (`dashboard/pages/3_Segments.py`): K-Means cluster scatter & box plots.
- 💰 **LTV Analytics** (`dashboard/pages/4_LTV.py`): Revenue distribution & lifetime forecasts.
- 📉 **Churn Risk Watchlist** (`dashboard/pages/5_Churn.py`): High-risk subscriber outreach watchlist & ROC/PR curves.
- 🎯 **Recommendation Center** (`dashboard/pages/6_Recommendations.py`): Campaign priority grouping & ROI savings.
- 📥 **Batch Analysis Center** (`dashboard/pages/7_Batch_Analysis.py`): Bulk customer CSV scoring hub.
- ⚙️ **Operations Telemetry** (`dashboard/pages/9_Operations.py`): Real-time health, auto-sync state & import logs.
- 🚀 **Deployment & K8s** (`dashboard/pages/10_Deployment.py`): Blue/Green & Canary deployment controls.

---

## 📂 Repository Directory Structure

```text
Customer-Intelligence-Platform/
├── artifacts/                  # Serialized models (pkl), encoders, scalers & registry JSON
├── backend/                    # Core Python FastAPI backend & ML package
│   ├── api/                    # API Endpoints (v1 routing, health, predict, ingest, observability)
│   ├── core/                   # Shared configurations, APScheduler, audit logger, metrics
│   ├── database/               # SQLAlchemy ORM models, Session creators
│   ├── ml/                     # Machine learning pipelines (training, inference, drift, registry)
│   └── services/               # Ingestion Engine, Auto-Sync & Prediction Services
├── dashboard/                  # Streamlit enterprise application
│   ├── assets/                 # CSS Design System (styles.css)
│   ├── components/             # Reusable UI cards, headers, Plotly charts, tables
│   └── pages/                  # 11 Multi-page analytical views
├── data/
│   ├── incoming/               # Watch folder (drop incoming subscriber CSVs here)
│   ├── processed/              # Automatically processed CSVs archive
│   └── failed/                 # Validation error CSVs archive
├── docs/                       # Architecture diagrams & operational runbooks
├── k8s/                        # Kubernetes manifests & deployment configurations
├── logs/                       # Audit logs, import history, and scheduler history
├── scripts/                    # Maintenance, training & automated verification scripts
└── tests/                      # Pytest automated test suite (132/132 tests passing)
```

---

## ⚙️ Getting Started & Installation

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/sarafirdose/-Customer-Intelligence-Platform.git
cd -Customer-Intelligence-Platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Run Backend API Server

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Docs: `http://localhost:8000/docs`

### 3. Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```
- Dashboard Access: `http://localhost:8501`

---

## 🧪 Automated Testing & MLOps

Run the complete 132-test suite with coverage:

```bash
pytest tests/ -v
```

---

## 👤 Author & Maintainer

**Sara Firdose**  
- **GitHub**: [@sarafirdose](https://github.com/sarafirdose)  
- **Project Repository**: [-Customer-Intelligence-Platform](https://github.com/sarafirdose/-Customer-Intelligence-Platform)
