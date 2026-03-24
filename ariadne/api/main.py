from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from ariadne.api.dependencies.limiter import limiter
from ariadne.api.routes.analyze import router as analyze_router
from ariadne.api.routes.health import router as health_router
from ariadne.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _get_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _sentry_before_send(event: dict, hint: dict) -> dict | None:
    """Drop 4xx HTTP exceptions — only 5xx errors create Sentry events."""
    exc_info = hint.get("exc_info")
    if exc_info:
        _, exc_value, _ = exc_info
        if isinstance(exc_value, RateLimitExceeded):
            return None
        if isinstance(exc_value, StarletteHTTPException) and exc_value.status_code < 500:
            return None
    return event


def _init_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        logger.debug("SENTRY_DSN not configured — Sentry disabled")
        return

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        send_default_pii=False,
        before_send=_sentry_before_send,
    )
    logger.info("Sentry initialized (traces_sample_rate=0.1)")


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Rate limit exceeded. POST /analyze is limited to 5 requests per minute per IP. "
                "Please wait 60 seconds before trying again."
            )
        },
        headers={"Retry-After": "60"},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    _init_sentry()
    logger.info("Ariadne API starting up")
    yield
    logger.info("Ariadne API shutting down")


app = FastAPI(
    title="Ariadne — Incident Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting — limiter must be on app.state before any request is processed
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analyze_router)

# Serve the Next.js static export when built into the Docker image.
# API routes above always take priority over the catch-all static mount.
_UI_DIR = Path(__file__).parent.parent.parent / "ui" / "out"
if _UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
