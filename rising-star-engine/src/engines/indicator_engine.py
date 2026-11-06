"""
Rising Star Trend Engine v2.0 - Technical Indicator Engine (Module 02)
Vectorized technical processing for ATR (14-period), 52-week high/low metrics,
moving averages, and price stabilization / reversal momentum detection.
"""

from datetime import datetime, timezone
import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("rising_star.indicators")


class TechnicalIndicators(BaseModel):
    """
    Structured payload of computed technical indicators and momentum status.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    current_price: float = Field(..., description="Latest closing or market price")
    fifty_two_week_high: float = Field(..., description="52-week high peak price")
    fifty_two_week_low: float = Field(..., description="52-week low trough price")
    distance_from_52w_high_pct: float = Field(
        ...,
        description="Percentage distance from 52-week high (e.g. -0.22 for 22% drop)"
    )
    atr_14: float = Field(..., description="14-period Average True Range in absolute currency units")
    atr_pct: float = Field(..., description="14-period ATR expressed as a fraction of current price")
    ema_5: Optional[float] = Field(default=None, description="5-day Exponential Moving Average")
    ema_20: Optional[float] = Field(default=None, description="20-day Exponential Moving Average")
    ema_50: Optional[float] = Field(default=None, description="50-day Exponential Moving Average")
    momentum_stabilizing: bool = Field(
        default=False,
        description="True if short-term downside momentum is flattening or reversing"
    )
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when indicators were computed"
    )


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculates True Range (TR) across high, low, and previous close.
    TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
    """
    if df.empty or len(df) < 2:
        return pd.Series(dtype="float64")

    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Computes Average True Range (ATR) using Wilder's smoothing / exponential moving average.
    """
    if df.empty or len(df) < period:
        return pd.Series(dtype="float64")

    tr = compute_true_range(df)
    # Wilder's smoothing alpha = 1 / period (equivalent to ewm(alpha=1/period, adjust=False))
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_distance_from_high(current_price: float, high_52w: float) -> float:
    """
    Calculates percentage drop from 52-week peak.
    Returns negative decimal: (current - high) / high (e.g., -0.20 for 20% pullback).
    """
    if high_52w <= 0:
        return 0.0
    return round((current_price - high_52w) / high_52w, 4)


def check_reversal_stabilization(df: pd.DataFrame, lookback: int = 5) -> bool:
    """
    Evaluates whether downward momentum is stabilizing or flattening.
    Criteria:
    1. Short-term 5-day EMA slope has flattened (slope >= -0.005 over the last 3 bars) OR
    2. Latest closing price has closed above the 5-day EMA or formed higher lows over 3 sessions.
    """
    if df.empty or len(df) < lookback + 5:
        return True  # Fallback to permissive if history insufficient

    close = df["Close"]
    ema_5 = close.ewm(span=5, adjust=False).mean()

    # Rate of change of EMA5 over the last 3 trading sessions
    recent_ema5 = ema_5.iloc[-3:].values
    if len(recent_ema5) >= 3:
        ema_slope = (recent_ema5[-1] - recent_ema5[0]) / recent_ema5[0]
        latest_close = close.iloc[-1]
        latest_ema = recent_ema5[-1]

        # Condition A: EMA slope is no longer steeply negative (stabilization)
        # Condition B: Price is above the 5-day EMA (curl up)
        # Condition C: Lows are stabilizing (latest low >= previous low)
        recent_lows = df["Low"].iloc[-3:].values
        lows_stabilizing = recent_lows[-1] >= recent_lows[-2] * 0.995

        if ema_slope >= -0.015 or latest_close >= latest_ema * 0.99 or lows_stabilizing:
            return True

    return False


class IndicatorEngine:
    """
    High-performance technical analysis processor for value dip and retracement evaluation.
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def calculate_indicators(
        self,
        ticker: str,
        df: pd.DataFrame,
        high_52w: Optional[float] = None,
        low_52w: Optional[float] = None,
    ) -> TechnicalIndicators:
        """
        Processes an OHLCV DataFrame into a validated TechnicalIndicators payload.
        """
        clean_ticker = ticker.strip().upper()

        if df.empty or len(df) < 5:
            raise ValueError(f"Insufficient OHLCV data for {clean_ticker}: received {len(df)} rows.")

        current_price = float(df["Close"].iloc[-1])

        # Derive 52-week boundaries (approx 252 trading days) if not provided
        lookback_window = min(len(df), 252)
        window_slice = df.iloc[-lookback_window:]

        computed_high_52w = float(window_slice["High"].max())
        computed_low_52w = float(window_slice["Low"].min())

        final_high_52w = high_52w if high_52w and high_52w > 0 else computed_high_52w
        final_low_52w = low_52w if low_52w and low_52w > 0 else computed_low_52w

        # Ensure high is at least current price to prevent positive distance distortion
        final_high_52w = max(final_high_52w, current_price)

        # Distance from 52-week high
        distance_pct = compute_distance_from_high(current_price, final_high_52w)

        # ATR calculation
        atr_series = compute_atr(df, period=self.atr_period)
        if not atr_series.empty and not pd.isna(atr_series.iloc[-1]):
            atr_val = float(atr_series.iloc[-1])
        else:
            # Fallback estimation using simple High - Low range mean
            recent_ranges = (df["High"] - df["Low"]).iloc[-14:]
            atr_val = float(recent_ranges.mean()) if not recent_ranges.empty else current_price * 0.02

        atr_pct = round(atr_val / current_price, 4) if current_price > 0 else 0.0

        # Moving Averages
        close = df["Close"]
        ema_5 = float(close.ewm(span=5, adjust=False).mean().iloc[-1]) if len(df) >= 5 else None
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(df) >= 20 else None
        ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(df) >= 50 else None

        # Reversal / stabilization check
        stabilizing = check_reversal_stabilization(df)

        return TechnicalIndicators(
            ticker=clean_ticker,
            current_price=round(current_price, 4),
            fifty_two_week_high=round(final_high_52w, 4),
            fifty_two_week_low=round(final_low_52w, 4),
            distance_from_52w_high_pct=distance_pct,
            atr_14=round(atr_val, 4),
            atr_pct=atr_pct,
            ema_5=round(ema_5, 4) if ema_5 is not None else None,
            ema_20=round(ema_20, 4) if ema_20 is not None else None,
            ema_50=round(ema_50, 4) if ema_50 is not None else None,
            momentum_stabilizing=stabilizing,
        )


def get_indicator_engine() -> IndicatorEngine:
    """Factory helper returning initialized IndicatorEngine."""
    return IndicatorEngine()
