"""
Customer Intelligence Platform API Application Entrypoint.

Configures FastAPI, mounts middlewares, handles startup and shutdown operations,
attaches versioned routers, and serves as the HTTP entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.core import config
from backend.core.logger import logger
from backend.core.settings import settings
from backend.database.database import test_db_connection
from backend.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handle app startup and shutdown event lifecycles.

    Ensures dependencies like databases are checked upon server boot.
    """
    logger.info("Initializing Customer Intelligence Platform backend...")

    # Validate database connection on startup
    db_ok = test_db_connection()
    if db_ok:
        logger.info("Startup validation: Database is reachable.")
    else:
        logger.error(
            "Startup validation: Database connection FAILED. "
            "Please check environmental configurations and DB container status."
        )

    yield

    logger.info("Shutting down Customer Intelligence Platform backend...")


# Initialize FastAPI App
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route API v1 endpoints
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def redirect_to_docs() -> RedirectResponse:
    """
    Redirect the root URL path to the auto-generated Swagger API documentation.

    Returns:
        RedirectResponse: Redirection to /docs.
    """
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn

    # For running the API server locally directly
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
