"""
Notifications package for real-time trade signals and alert dispatches.
"""

from .telegram_bot import TelegramNotifier

__all__ = ["TelegramNotifier"]
