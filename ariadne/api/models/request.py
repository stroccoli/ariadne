from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    logs: str = Field(min_length=1, max_length=50_000, description="Raw incident log text to analyze (max 50,000 chars).")
    mode: Literal["detailed", "compact"] = Field(
        default="detailed",
        description="Prompt mode: 'detailed' for full analysis, 'compact' for brief.",
    )
