# ==============================================================================
# Customer Intelligence Platform - Makefile
# ==============================================================================

.PHONY: help install test run docker-build docker-up docker-down lint format clean migrations migrate

help:
	@echo "Available commands:"
	@echo "  make install       - Set up virtual environment and install dependencies"
	@echo "  make test          - Run the pytest suite"
	@echo "  make run           - Run FastAPI local server"
	@echo "  make docker-build  - Build Docker containers"
	@echo "  make docker-up     - Start all Docker services (App + PostgreSQL)"
	@echo "  make docker-down   - Tear down all Docker services"
	@echo "  make lint          - Run linters (flake8, mypy)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make migrations    - Generate Alembic migrations"
	@echo "  make migrate       - Run Alembic migrations"
	@echo "  make clean         - Remove temporary and cached Python files"

install:
	python -m venv venv
	./venv/Scripts/pip install --upgrade pip
	./venv/Scripts/pip install -r requirements.txt
	./venv/Scripts/pip install -e .
	./venv/Scripts/pre-commit install

test:
	pytest tests/

run:
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down -v

lint:
	flake8 backend/ tests/
	mypy backend/

format:
	black backend/ tests/ scripts/
	isort backend/ tests/ scripts/

migrations:
	alembic revision --autogenerate -m "Initial schema"

migrate:
	alembic upgrade head

clean:
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
