"""
Customer Intelligence Platform API Application Entrypoint.

Configures FastAPI, mounts middlewares, handles startup and shutdown operations,
attaches versioned routers, and serves as the HTTP entry point.

Phase 6A Additions:
- MetricsMiddleware records per-request latency and error counts
- Security headers middleware adds X-Content-Type-Options, X-Frame-Options, etc.
- APScheduler starts/stops within the lifespan context

Phase 7 Additions:
- Scheduler start/stop via platform_scheduler
- Request ID injection via logger.request_id_var ContextVar
"""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.core import config
from backend.core.audit import log_audit_event
from backend.core.logger import logger, request_id_var
from backend.core.metrics import metrics as metrics_collector
from backend.core.scheduler import platform_scheduler
from backend.core.settings import settings
from backend.database.database import test_db_connection
from backend.api.v1.router import api_router


# ---------------------------------------------------------------------------
# Middlewares
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Records request latency and error counts for every API call.
    Also injects a unique request_id into the ContextVar for log correlation.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        import time

        # Inject correlation request ID
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            latency_ms = (time.perf_counter() - start) * 1000
            is_error = response.status_code >= 500
            metrics_collector.record_request(
                endpoint=str(request.url.path),
                latency_ms=round(latency_ms, 3),
                error=is_error,
            )
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics_collector.record_request(
                endpoint=str(request.url.path),
                latency_ms=round(latency_ms, 3),
                error=True,
            )
            log_audit_event(
                event_type="ERROR",
                endpoint=str(request.url.path),
                error=str(exc),
                latency_ms=round(latency_ms, 3),
            )
            raise
        finally:
            request_id_var.reset(token)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects security headers on every response."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle app startup and shutdown event lifecycles."""
    logger.info("Initializing Customer Intelligence Platform backend...")

    # Validate database connection on startup
    db_ok = test_db_connection()
    if db_ok:
        logger.info("Startup: database is reachable.")
    else:
        logger.error(
            "Startup: database connection FAILED. "
            "Check environmental configurations and DB container status."
        )

    # Start background scheduler
    platform_scheduler.start()

    log_audit_event(
        event_type="SYSTEM",
        endpoint="lifespan",
        result_summary={"event": "startup", "db_ok": db_ok},
    )

    yield

    # Shutdown scheduler cleanly
    platform_scheduler.shutdown()

    log_audit_event(
        event_type="SYSTEM",
        endpoint="lifespan",
        result_summary={"event": "shutdown"},
    )
    logger.info("Shutting down Customer Intelligence Platform backend...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)

# Metrics + request ID injection (must be outermost to time everything)
app.add_middleware(MetricsMiddleware)

# Route API v1 endpoints
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def redirect_to_docs() -> RedirectResponse:
    """Redirect root URL to Swagger docs."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
