from __future__ import annotations

import os

from slowapi import Limiter
from starlette.requests import Request


def _get_client_ip(request: Request) -> str:
    trusted = {ip.strip() for ip in os.getenv("TRUSTED_PROXIES", "127.0.0.1").split(",") if ip.strip()}
    remote_host = request.client.host if request.client else None
    if remote_host in trusted:
        x_forwarded_for = request.headers.get("X-Forwarded-For", "")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
    return remote_host or "unknown"


limiter = Limiter(key_func=_get_client_ip, default_limits=[])
