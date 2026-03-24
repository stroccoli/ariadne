from __future__ import annotations

import hmac
import logging
import os

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

logger = logging.getLogger(__name__)

# Read once at import time — consistent with the pattern in core/config.py.
_API_KEY: str | None = os.getenv("API_KEY", "").strip() or None

if _API_KEY is None:
    logger.warning(
        "API_KEY is not configured — authentication is DISABLED. "
        "Set API_KEY in your environment before deploying to production."
    )


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce X-API-Key authentication on protected routes.

    Behaviour:
    - API_KEY not set in env  → auth disabled (dev mode, warning logged at startup).
    - Header missing          → 401 Unauthorized.
    - Header value wrong      → 401 Unauthorized (constant-time comparison).
    - Header value correct    → passes through (returns None).
    """
    if _API_KEY is None:
        # Dev mode: key not configured, skip auth check
        return

    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing X-API-Key header. "
                "Include your API key to access this endpoint."
            ),
        )

    # Constant-time comparison prevents timing-based key enumeration
    if not hmac.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key.")
