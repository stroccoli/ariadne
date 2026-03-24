from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single Limiter instance — imported by both main.py (to attach to app.state)
# and analyze.py (for the @limiter.limit decorator).
# default_limits=[] means no global limits; per-route limits are declared explicitly.
limiter = Limiter(key_func=get_remote_address, default_limits=[])
