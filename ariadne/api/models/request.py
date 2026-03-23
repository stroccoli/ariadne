from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    logs: str = Field(min_length=1, description="Raw incident log text to analyze.")
    mode: Literal["detailed", "compact"] = Field(
        default="detailed",
        description="Prompt mode: 'detailed' for full analysis, 'compact' for brief.",
    )
