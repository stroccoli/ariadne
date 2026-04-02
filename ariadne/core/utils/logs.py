"""Log pre-processing utilities for the Ariadne pipeline.

The main concern is context-window overflow when raw incident logs are very
long.  Sending a prompt that exceeds the model's num_ctx causes silent
truncation by Ollama (the oldest tokens are dropped), which means the
instruction block is preserved but some evidence is lost unpredictably.

Strategy: explicit tail-first truncation before the prompt is built
-------------------------------------------------------------------
1. We truncate *lines* rather than raw characters so log structure is kept.
2. We keep the *most-recent* lines because they carry the most actionable
   signal (errors usually escalate over time).
3. A short notice is prepended to the truncated block so the model knows the
   entry was partial — this is honoured by uncertainty-aware prompts.
4. The limit is controlled by the MAX_LOG_CHARS env var (default 6000).
   Tune upward if you increase OLLAMA_NUM_CTX and have headroom.

Typical prompt overhead by mode:
   detailed classifier  ~1 500 chars
   detailed analyzer    ~2 200 chars  (includes context block)
   compact  variants    ~400–600 chars

With the default MAX_LOG_CHARS=6000 and OLLAMA_NUM_CTX=8192 (~32 000 chars
at 4 chars/token) there is ample headroom for the instruction text plus logs.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUNCATION_NOTICE = "[NOTE: logs truncated — showing most recent {kept} of {total} lines]\n"

# Default maximum characters allowed for the raw logs block in a prompt.
# Override via MAX_LOG_CHARS env var (e.g. MAX_LOG_CHARS=12000 for larger models).
_DEFAULT_MAX_CHARS = 6_000


def _get_max_chars() -> int:
    raw = os.getenv("MAX_LOG_CHARS", "")
    try:
        return int(raw) if raw.strip() else _DEFAULT_MAX_CHARS
    except ValueError:
        logger.warning("Invalid MAX_LOG_CHARS=%r, using default %d", raw, _DEFAULT_MAX_CHARS)
        return _DEFAULT_MAX_CHARS


def truncate_logs(logs: str, max_chars: int | None = None) -> str:
    """Return *logs* truncated to *max_chars*, keeping the most-recent lines.

    If *logs* already fits within the limit it is returned unchanged.
    When truncation is needed, a one-line notice is prepended so the LLM
    understands the input is partial.

    Args:
        logs:      Raw log string from the incident.
        max_chars: Character budget for the log block.  Defaults to the value
                   of the MAX_LOG_CHARS env var (default 6 000).
    """
    limit = max_chars if max_chars is not None else _get_max_chars()

    if len(logs) <= limit:
        return logs

    lines = logs.splitlines()
    total_lines = len(lines)

    # Greedily collect lines from the tail until we hit the limit.
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1  # +1 for the newline
        if used + cost > limit:
            break
        kept.append(line)
        used += cost

    kept.reverse()
    kept_count = len(kept)

    logger.warning(
        "Logs truncated: kept %d of %d lines (%d chars) — increase MAX_LOG_CHARS or OLLAMA_NUM_CTX to avoid this",
        kept_count,
        total_lines,
        used,
    )

    notice = _TRUNCATION_NOTICE.format(kept=kept_count, total=total_lines)
    return notice + "\n".join(kept)
