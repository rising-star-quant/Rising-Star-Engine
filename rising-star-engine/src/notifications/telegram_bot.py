"""
Rising Star Trend Engine v2.0 - Telegram Notification Dispatcher
Backwards-compatible wrapper around TelegramAlerter.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.notifications.telegram_alerter import (
    EarningsWreckAlertPayload,
    RiskWarningAlertPayload,
    TelegramAlerter,
    TelegramMessageBuilder,
    TelegramMessageResponse,
    TradeAlertPayload,
    get_telegram_alerter,
)


class TradeAlert(BaseModel):
    """Legacy trade alert schema."""
    ticker: str
    action: str = Field(default="BUY", description="BUY, SELL, TRIM, STOP_OUT")
    price: float
    target: float
    stop: float
    score: float = 0.8
    catalyst_summary: str = ""


class TelegramNotifier(TelegramAlerter):
    """
    Subclass of TelegramAlerter providing backwards compatibility.
    """
    pass


__all__ = [
    "TradeAlert",
    "TelegramNotifier",
    "TradeAlertPayload",
    "EarningsWreckAlertPayload",
    "RiskWarningAlertPayload",
    "TelegramMessageResponse",
    "TelegramMessageBuilder",
    "TelegramAlerter",
    "get_telegram_alerter",
]
