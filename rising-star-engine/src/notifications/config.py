"""
Rising Star Trend Engine v2.0 - Telegram Notification Configuration (Module 04)
Secure configuration loader using pydantic-settings to manage Telegram Bot credentials,
environment variables, timeout constraints, and production fail-fast validations.
"""

from functools import lru_cache
import os
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramConfigError(ValueError):
    """Raised when critical Telegram notification settings are missing or invalid."""
    pass


class TelegramSettings(BaseSettings):
    """
    Secure settings management for Telegram alerts.
    Loads credentials from environment variables (TELEGRAM_* or RSTE_TELEGRAM_*) or .env file.
    Uses SecretStr to guarantee sensitive tokens are masked and never leaked in logs.
    """

    telegram_bot_token: Optional[SecretStr] = Field(
        default=None,
        description="Telegram Bot API Token (e.g. '123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ')"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None,
        description="Target Telegram chat, group, or channel ID (e.g. '@my_channel' or '-1001234567890')"
    )
    env: str = Field(
        default="development",
        description="Runtime environment: development, testing, staging, production"
    )
    request_timeout: int = Field(
        default=15,
        ge=1,
        description="HTTP request timeout in seconds for Telegram API calls"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on transient network or rate-limit errors"
    )
    backoff_factor: float = Field(
        default=1.5,
        ge=0.1,
        description="Multiplier for exponential backoff between retries"
    )
    rate_limit_max_wait_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Maximum duration to pause when hit with HTTP 429 rate limit"
    )
    disable_notification: bool = Field(
        default=False,
        description="Sends the message silently (users receive notification with no sound)"
    )
    default_parse_mode: str = Field(
        default="MarkdownV2",
        description="Default message formatting mode: 'MarkdownV2' or 'HTML'"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def get_token_str(self) -> Optional[str]:
        """Returns the raw token string for API calls, or None if not configured."""
        if self.telegram_bot_token is None:
            return None
        return self.telegram_bot_token.get_secret_value()

    @model_validator(mode="before")
    @classmethod
    def populate_aliases(cls, values: dict) -> dict:
        """
        Supports standard env variable names (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        as well as application-prefixed names (RSTE_TELEGRAM_BOT_TOKEN, RSTE_TELEGRAM_CHAT_ID).
        """
        token = (
            values.get("telegram_bot_token")
            or os.getenv("TELEGRAM_BOT_TOKEN")
            or os.getenv("RSTE_TELEGRAM_BOT_TOKEN")
        )
        chat_id = (
            values.get("telegram_chat_id")
            or os.getenv("TELEGRAM_CHAT_ID")
            or os.getenv("RSTE_TELEGRAM_CHAT_ID")
        )
        env = values.get("env") or os.getenv("RSTE_ENV") or os.getenv("ENV") or "development"

        if token and "telegram_bot_token" not in values:
            values["telegram_bot_token"] = token
        if chat_id and "telegram_chat_id" not in values:
            values["telegram_chat_id"] = chat_id
        if "env" not in values:
            values["env"] = env

        return values

    @model_validator(mode="after")
    def validate_production_requirements(self) -> "TelegramSettings":
        """
        Fail-fast validation: ensures production deployments do not start with
        missing Telegram notification credentials.
        """
        if self.env.strip().lower() == "production":
            token_val = self.get_token_str()
            if not token_val or not token_val.strip():
                raise TelegramConfigError(
                    "Production configuration error: TELEGRAM_BOT_TOKEN is mandatory but missing. "
                    "Set the TELEGRAM_BOT_TOKEN environment variable."
                )
            if not self.telegram_chat_id or not self.telegram_chat_id.strip():
                raise TelegramConfigError(
                    "Production configuration error: TELEGRAM_CHAT_ID is mandatory but missing. "
                    "Set the TELEGRAM_CHAT_ID environment variable."
                )
        return self


@lru_cache(maxsize=1)
def get_telegram_settings() -> TelegramSettings:
    """Returns singleton cached instance of TelegramSettings."""
    return TelegramSettings()
