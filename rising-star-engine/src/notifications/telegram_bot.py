"""
Rising Star Trend Engine v2.0 - Telegram Notification Dispatcher (Module 08 Skeleton)
Broadcasts high-conviction trade setups, risk warnings, and system status updates.
"""

from typing import Optional
from pydantic import BaseModel


class TradeAlert(BaseModel):
    ticker: str
    action: str  # BUY, SELL, TRIM, STOP_OUT
    price: float
    target: float
    stop: float
    score: float
    catalyst_summary: str


class TelegramNotifier:
    """Dispatches formatted rich markdown alerts to configured Telegram channels."""
    pass
