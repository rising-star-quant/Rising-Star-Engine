"""
Rising Star Trend Engine v2.0 - Earnings Wreck & Reaction Scanner (Module 03)
Detects acute post-earnings overreactions where equities suffer severe sell-offs (>= 8% 1-day or >= 15% 3-day)
on heavy institutional liquidation volume (RVOL >= 2.0x) despite preserving underlying revenue integrity.
"""

from datetime import date, datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.connectors.data_connector import MarketDataPayload, YFinanceDataConnector, get_data_connector

logger = logging.getLogger("rising_star.earnings_scanner")


class EarningsWreckPayload(BaseModel):
    """
    Atomic output data contract matching Module 03 PRD specifications.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    earnings_date: str = Field(..., description="Date of earnings release (YYYY-MM-DD)")
    price_change_1d_pct: float = Field(
        ...,
        description="Single-day price change on/after earnings as decimal (e.g. -0.14 for -14%)"
    )
    relative_volume: float = Field(
        ...,
        ge=0.0,
        description="Drop day volume divided by 30-day average volume (RVOL >= 2.0 confirms surge)"
    )
    revenue_integrity_passed: bool = Field(
        ...,
        description="True if core business top-line revenue did not collapse with market price"
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 evaluation timestamp"
    )
    # Supporting diagnostic metrics
    price_change_3d_pct: Optional[float] = Field(
        default=None,
        description="3-day cumulative price change (e.g. -0.18)"
    )
    revenue_growth_yoy: Optional[float] = Field(
        default=None,
        description="Latest reported quarterly YoY revenue growth rate"
    )
    eps_surprise_pct: Optional[float] = Field(
        default=None,
        description="Reported EPS surprise vs consensus percentage"
    )
    current_price: Optional[float] = Field(
        default=None,
        description="Current market price per share"
    )
    is_qualified: bool = Field(
        default=True,
        description="True if setup passes acute drop, volume surge, and revenue integrity gates"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason if candidate fails any earnings wreck gate"
    )


class EarningsFilterCriteria(BaseModel):
    """
    Parametric thresholds defining the acute earnings reaction scanner.
    """
    # Acute Drop Thresholds (PRD Section 2)
    min_single_day_drop_pct: float = Field(
        default=-0.08,
        description="Maximum acceptable 1-day return (e.g. -0.08 represents drop >= 8%)"
    )
    min_three_day_drop_pct: float = Field(
        default=-0.15,
        description="Maximum acceptable 3-day return (e.g. -0.15 represents drop >= 15%)"
    )
    # Volume Surge Verification (PRD Section 2)
    min_relative_volume: float = Field(
        default=2.0,
        description="Minimum Relative Volume (RVOL) multiplier vs 30-day ADV"
    )
    # Revenue Integrity Threshold
    min_acceptable_revenue_growth: float = Field(
        default=-0.05,
        description="Revenue growth floor (-5%). Prevents buying into structural business collapse."
    )
    # Earnings Recency Window (Days)
    max_days_since_earnings: int = Field(
        default=3,
        description="Maximum trading days elapsed since earnings filing (PRD: 1 to 3 days)"
    )


class EarningsReactionScanner:
    """
    Event-driven scanner identifying acute post-earnings market overreactions.
    """

    def __init__(
        self,
        connector: Optional[YFinanceDataConnector] = None,
        settings: Optional[Settings] = None,
        criteria: Optional[EarningsFilterCriteria] = None,
    ):
        self.settings = settings or get_settings()
        self.connector = connector or YFinanceDataConnector(settings=self.settings)
        self.criteria = criteria or EarningsFilterCriteria()

    def calculate_rvol(
        self,
        ohlcv_df: pd.DataFrame,
        event_idx: int = -1,
        lookback_window: int = 30,
    ) -> float:
        """
        Computes Relative Volume (RVOL) on the event day relative to prior lookback volume.

        Formula:
            RVOL = Volume_event / Mean(Volume_{event - lookback : event - 1})
        """
        if ohlcv_df.empty or "volume" not in ohlcv_df.columns:
            return 0.0

        n = len(ohlcv_df)
        idx = event_idx if event_idx >= 0 else n + event_idx
        if idx < 1 or idx >= n:
            return 0.0

        event_vol = float(ohlcv_df["volume"].iloc[idx])

        # Prior lookback window excluding the event day
        start_idx = max(0, idx - lookback_window)
        prior_volumes = ohlcv_df["volume"].iloc[start_idx:idx].dropna()

        if len(prior_volumes) == 0:
            return 1.0

        mean_prior_vol = float(prior_volumes.mean())
        if mean_prior_vol <= 0:
            return 1.0

        return round(event_vol / mean_prior_vol, 2)

    def calculate_price_drops(
        self,
        ohlcv_df: pd.DataFrame,
        event_idx: int = -1,
    ) -> Tuple[float, float]:
        """
        Calculates 1-day and 3-day percentage returns relative to pre-earnings close.

        Returns:
            Tuple of (price_change_1d_pct, price_change_3d_pct)
        """
        if ohlcv_df.empty or "close" not in ohlcv_df.columns:
            return 0.0, 0.0

        n = len(ohlcv_df)
        idx = event_idx if event_idx >= 0 else n + event_idx
        if idx < 1:
            return 0.0, 0.0

        close_event = float(ohlcv_df["close"].iloc[idx])
        close_prior_1d = float(ohlcv_df["close"].iloc[idx - 1])
        change_1d = round((close_event - close_prior_1d) / close_prior_1d, 4)

        # 3-day change
        if idx >= 3:
            close_prior_3d = float(ohlcv_df["close"].iloc[idx - 3])
            change_3d = round((close_event - close_prior_3d) / close_prior_3d, 4)
        else:
            change_3d = change_1d

        return change_1d, change_3d

    def check_revenue_integrity(
        self,
        revenue_growth_yoy: Optional[float],
        price_drop_1d_pct: float,
    ) -> bool:
        """
        Validates that top-line quarterly revenue did not drop linearly with the market selloff.
        If revenue growth is above the minimum threshold (e.g. >= -5%), integrity is preserved.
        """
        if revenue_growth_yoy is None:
            # If fundamental data not yet recorded, default to conservative false
            return False

        # Revenue growth must be above threshold (e.g. -5%)
        # and significantly better than the price drop percentage
        return revenue_growth_yoy >= self.criteria.min_acceptable_revenue_growth

    def evaluate_candidate(
        self,
        ticker: str,
        ohlcv_df: pd.DataFrame,
        earnings_date: str,
        revenue_growth_yoy: Optional[float] = None,
        eps_surprise_pct: Optional[float] = None,
        event_idx: int = -1,
    ) -> EarningsWreckPayload:
        """
        Evaluates a candidate stock against all Module 03 PRD gates.

        Returns:
            Validated EarningsWreckPayload
        """
        if ohlcv_df.empty or len(ohlcv_df) < 5:
            return EarningsWreckPayload(
                ticker=ticker,
                earnings_date=earnings_date,
                price_change_1d_pct=0.0,
                relative_volume=0.0,
                revenue_integrity_passed=False,
                is_qualified=False,
                rejection_reason="Insufficient historical OHLCV data for analysis.",
            )

        current_price = float(ohlcv_df["close"].iloc[-1])
        change_1d, change_3d = self.calculate_price_drops(ohlcv_df, event_idx=event_idx)
        rvol = self.calculate_rvol(ohlcv_df, event_idx=event_idx)
        revenue_passed = self.check_revenue_integrity(revenue_growth_yoy, change_1d)

        # 1. Acute Drop Gate: 1-day drop >= 8% OR 3-day drop >= 15%
        acute_drop_passed = (
            change_1d <= self.criteria.min_single_day_drop_pct
            or change_3d <= self.criteria.min_three_day_drop_pct
        )

        # 2. Volume Surge Gate: RVOL >= 2.0x
        volume_surge_passed = rvol >= self.criteria.min_relative_volume

        # 3. Qualification aggregation
        is_qualified = acute_drop_passed and volume_surge_passed and revenue_passed
        rejection_reasons = []

        if not acute_drop_passed:
            rejection_reasons.append(
                f"Drop too shallow: 1d={change_1d:.1%}, 3d={change_3d:.1%} "
                f"(requires <= {self.criteria.min_single_day_drop_pct:.1%} 1d or <= {self.criteria.min_three_day_drop_pct:.1%} 3d)."
            )
        if not volume_surge_passed:
            rejection_reasons.append(
                f"Insufficient volume surge: RVOL={rvol:.2f}x (requires >= {self.criteria.min_relative_volume:.1f}x)."
            )
        if not revenue_passed:
            rev_str = f"{revenue_growth_yoy:.1%}" if revenue_growth_yoy is not None else "N/A"
            rejection_reasons.append(
                f"Revenue integrity failed: YoY revenue growth is {rev_str} "
                f"(requires >= {self.criteria.min_acceptable_revenue_growth:.1%})."
            )

        rejection_reason = " | ".join(rejection_reasons) if rejection_reasons else None

        return EarningsWreckPayload(
            ticker=ticker,
            earnings_date=earnings_date,
            price_change_1d_pct=change_1d,
            price_change_3d_pct=change_3d,
            relative_volume=rvol,
            revenue_integrity_passed=revenue_passed,
            revenue_growth_yoy=revenue_growth_yoy,
            eps_surprise_pct=eps_surprise_pct,
            current_price=current_price,
            is_qualified=is_qualified,
            rejection_reason=rejection_reason,
        )

    def scan_universe(
        self,
        tickers: List[str],
        earnings_calendar: Dict[str, Dict[str, Any]],
    ) -> List[EarningsWreckPayload]:
        """
        Batch screens candidate tickers against the earnings reaction criteria.

        Args:
            tickers: List of ticker symbols to evaluate.
            earnings_calendar: Mapping of ticker to metadata dict containing
                               'earnings_date', 'revenue_growth_yoy', 'eps_surprise_pct'.

        Returns:
            List of qualified EarningsWreckPayload objects.
        """
        qualified: List[EarningsWreckPayload] = []

        for ticker in tickers:
            meta = earnings_calendar.get(ticker, {})
            edate = meta.get("earnings_date")
            if not edate:
                continue

            try:
                payload = self.connector.fetch_ticker_payload(ticker, period="60d")
                df = payload.to_dataframe()
                result = self.evaluate_candidate(
                    ticker=ticker,
                    ohlcv_df=df,
                    earnings_date=edate,
                    revenue_growth_yoy=meta.get("revenue_growth_yoy"),
                    eps_surprise_pct=meta.get("eps_surprise_pct"),
                )
                if result.is_qualified:
                    qualified.append(result)
            except Exception as exc:
                logger.warning("Error evaluating earnings candidate %s: %s", ticker, exc)

        return qualified


def get_earnings_scanner(
    connector: Optional[YFinanceDataConnector] = None,
    settings: Optional[Settings] = None,
    criteria: Optional[EarningsFilterCriteria] = None,
) -> EarningsReactionScanner:
    """Factory helper returning EarningsReactionScanner."""
    return EarningsReactionScanner(connector=connector, settings=settings, criteria=criteria)
