# ==============================================================================
# Customer Intelligence Platform - Multi-Stage Dockerfile
# ==============================================================================

# --- Stage 1: Build dependency environment ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies for libraries like psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to a user-local directory to easily copy to final image
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Final lightweight runtime ---
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime PostgreSQL client library dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and binaries from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Expose local paths
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV ENV=production

EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
