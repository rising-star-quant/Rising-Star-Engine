"""
Notifications package for real-time trade signals, Telegram alerts, and risk dispatches.
"""

from .config import TelegramConfigError, TelegramSettings, get_telegram_settings
from .telegram_alerter import (
    EarningsWreckAlertPayload,
    RiskWarningAlertPayload,
    TelegramAlerter,
    TelegramMessageBuilder,
    TelegramMessageResponse,
    TradeAlertPayload,
    get_telegram_alerter,
)
from .telegram_bot import TelegramNotifier, TradeAlert

__all__ = [
    "TelegramSettings",
    "TelegramConfigError",
    "get_telegram_settings",
    "TelegramAlerter",
    "TelegramNotifier",
    "TelegramMessageBuilder",
    "TradeAlertPayload",
    "EarningsWreckAlertPayload",
    "RiskWarningAlertPayload",
    "TelegramMessageResponse",
    "TradeAlert",
    "get_telegram_alerter",
]
