"""
Rising Star Trend Engine v2.0 - AI Skeptic (Module 07 Skeleton)
Adversarial LLM validator acting as devil's advocate against candidate breakout signals.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SkepticVerdict(BaseModel):
    ticker: str
    approved: bool
    risk_flags: List[str] = Field(default_factory=list)
    counter_thesis: str
    confidence_score: float


class AISkeptic:
    """Uses LLM (Gemini) to challenge bullish theses and uncover blindspots."""
    pass
