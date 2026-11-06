"""
Rising Star Trend Engine v2.0 - Configuration Module
Manages application configuration, risk thresholds, API credentials, and engine settings.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    from pydantic import BaseModel as BaseSettings, Field
    SettingsConfigDict = dict  # type: ignore
    PYDANTIC_SETTINGS_AVAILABLE = False


class Settings(BaseSettings):
    """
    Application settings and environment configuration.
    Reads from environment variables and optional .env file.
    """

    # Environment & Logging
    app_name: str = "Rising Star Trend Engine"
    app_version: str = "2.0.0"
    env: str = Field(default="development", description="Runtime environment: development, staging, production, testing")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    # API & Service Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host binding")
    api_port: int = Field(default=8000, description="FastAPI port binding")
    database_url: Optional[str] = Field(default=None, description="PostgreSQL or SQLite connection string")

    # Ingestion & Connectors (Module 01)
    yfinance_cache_dir: Optional[str] = Field(default=None, description="Local disk cache path for market data")
    request_timeout: int = Field(default=15, description="Timeout in seconds for external market data requests")
    max_retries: int = Field(default=4, description="Maximum retry count for exponential backoff on HTTP/API errors")
    backoff_factor: float = Field(default=1.5, description="Multiplier factor for exponential backoff delay")
    rate_limit_pause_seconds: float = Field(default=2.0, description="Pause duration when rate limited (HTTP 429)")

    # Baseline Universe Liquidity Thresholds (Per Module 01 PRD)
    min_market_cap: float = Field(default=300_000_000.0, description="Minimum market capitalization (PRD: >= $300M)")
    max_market_cap: Optional[float] = Field(default=None, description="Optional maximum market cap ceiling")
    min_adv_dollar: float = Field(default=2_000_000.0, description="Minimum 30-day Average Daily Dollar Volume (PRD: >= $2M ADV)")
    min_avg_volume_30d: int = Field(default=100_000, description="Minimum 30-day average daily share volume")
    min_share_price: float = Field(default=5.0, description="Default minimum share price to filter out penny stocks")
    max_share_price: Optional[float] = Field(default=2500.0, description="Optional maximum price ceiling")

    # Target Indices for Universe Scanner
    default_indices: List[str] = Field(
        default=["SP400", "RUSSELL1000", "RUSSELL2000"],
        description="Default benchmark index constituent lists to track"
    )

    # Notification Credentials
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot API token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram broadcast chat or channel ID")

    # AI & Skeptic Engine Credentials
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini AI API Key")

    if PYDANTIC_SETTINGS_AVAILABLE:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix="RSTE_",
            extra="ignore",
            case_sensitive=False
        )
    else:
        # Fallback initializer for environments without pydantic-settings
        def __init__(self, **kwargs):
            env_vars = {
                "env": os.getenv("RSTE_ENV", os.getenv("ENV", "development")),
                "debug": os.getenv("RSTE_DEBUG", "false").lower() in ("true", "1", "yes"),
                "log_level": os.getenv("RSTE_LOG_LEVEL", "INFO"),
                "api_host": os.getenv("RSTE_API_HOST", "0.0.0.0"),
                "api_port": int(os.getenv("RSTE_API_PORT", "8000")),
                "database_url": os.getenv("RSTE_DATABASE_URL", os.getenv("DATABASE_URL")),
                "request_timeout": int(os.getenv("RSTE_REQUEST_TIMEOUT", "15")),
                "max_retries": int(os.getenv("RSTE_MAX_RETRIES", "4")),
                "backoff_factor": float(os.getenv("RSTE_BACKOFF_FACTOR", "1.5")),
                "min_market_cap": float(os.getenv("RSTE_MIN_MARKET_CAP", "500000000.0")),
                "min_avg_volume_30d": int(os.getenv("RSTE_MIN_AVG_VOLUME_30D", "250000")),
                "min_share_price": float(os.getenv("RSTE_MIN_SHARE_PRICE", "5.0")),
                "telegram_bot_token": os.getenv("RSTE_TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN")),
                "telegram_chat_id": os.getenv("RSTE_TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID")),
                "gemini_api_key": os.getenv("RSTE_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
            }
            env_vars.update({k: v for k, v in kwargs.items() if v is not None})
            super().__init__(**env_vars)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns cached singleton instance of application settings.
    """
    return Settings()
