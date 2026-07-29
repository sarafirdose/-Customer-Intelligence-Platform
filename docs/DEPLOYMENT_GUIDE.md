# Enterprise Production Deployment Guide — CIP

## Overview
This guide documents the containerized production architecture, docker-compose orchestration, environment variables, health monitoring, and scaling for the Subscriber Intelligence Platform.

## Production Topology
- **API Server**: FastAPI with 4 Uvicorn workers (`docker/Dockerfile.backend`).
- **Dashboard**: Streamlit interactive UI (`docker/Dockerfile.dashboard`).
- **Redis Cache & Queue**: In-memory caching and task queuing.
- **PostgreSQL**: Operational data store.
- **Worker Process**: Background batch prediction execution.
- **Scheduler**: Dedicated cron worker for automated tasks.
- **Nginx**: Production reverse proxy with SSL termination support.

## Execution Commands
```bash
# Build and run production compose stack
docker compose -f docker-compose.production.yml up -d --build

# Scale API workers
docker compose -f docker-compose.production.yml scale api=3

# Inspect container logs
docker compose -f docker-compose.production.yml logs -f api
```
