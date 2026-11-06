"""
Rising Star Trend Engine v2.0 - Earnings Scanner (Module 03 Skeleton)
Identifies earnings acceleration, revenue beats, and upcoming catalysts.
"""

from typing import Optional
from pydantic import BaseModel


class EarningsSurpriseData(BaseModel):
    ticker: str
    eps_surprise_pct: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    next_earnings_date: Optional[str] = None


class EarningsScanner:
    """Scans and scores quarterly earnings quality and catalyst timelines."""
    pass
