"""
Rising Star Trend Engine v2.0 - Value Dip & Fundamental Disconnect Engine (Module 02)
Screens mid-to-large-cap equities for acute mean-reversion setups (15% to 30% pullback from 52W high)
with an uncompromising Fundamental Disconnect Guard (Positive Revenue Growth, Positive FCF, D/E < 1.5).
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.connectors.data_connector import (
    CompanyProfile,
    MarketDataPayload,
    YFinanceDataConnector,
    get_data_connector,
)
from src.engines.indicator_engine import (
    IndicatorEngine,
    TechnicalIndicators,
    get_indicator_engine,
)
from src.engines.universe_scanner import UniversePayload

logger = logging.getLogger("rising_star.value_dip")


class CompanyFinancials(BaseModel):
    """
    Core fundamental metrics validated by the Fundamental Disconnect Guard.
    """
    revenue_growth_qoq: Optional[float] = Field(
        default=None,
        description="Quarter-over-Quarter or Year-over-Year Revenue Growth rate (e.g. 0.08 for 8%)"
    )
    free_cash_flow: Optional[float] = Field(
        default=None,
        description="Trailing 12-month or latest quarterly Free Cash Flow in USD"
    )
    debt_to_equity: Optional[float] = Field(
        default=None,
        description="Total Debt to Shareholders' Equity ratio (e.g. 0.65)"
    )
    operating_cash_flow: Optional[float] = Field(
        default=None,
        description="Operating Cash Flow in USD"
    )


class ValueDipFilterCriteria(BaseModel):
    """
    Parametric thresholds defining the technical retracement gate and fundamental guard.
    """
    # Technical Retracement Gate (PRD: 15% to 30% below 52W high: P_current <= 0.85 * P_52w_high)
    min_drop_from_52w_high: float = Field(
        default=-0.30,
        description="Deepest acceptable drop boundary (e.g. -0.30 for 30% drop)"
    )
    max_drop_from_52w_high: float = Field(
        default=-0.15,
        description="Shallowest acceptable drop boundary (e.g. -0.15 for 15% drop)"
    )

    # Fundamental Disconnect Guard (PRD Section 2)
    min_revenue_growth: float = Field(
        default=0.0,
        description="Minimum QoQ/YoY revenue growth threshold (must be > 0%)"
    )
    require_positive_fcf: bool = Field(
        default=True,
        description="Enforces Free Cash Flow > 0"
    )
    max_debt_to_equity: float = Field(
        default=1.5,
        description="Ceiling for Debt-to-Equity ratio (must be < 1.5)"
    )

    # Reversal Momentum Check
    require_momentum_stabilization: bool = Field(
        default=True,
        description="Requires flattening of short-term moving average downward slope"
    )


class ValueDipCandidate(BaseModel):
    """
    Atomic output data contract matching Module 02 PRD specifications.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    current_price: float = Field(..., description="Latest traded market price")
    distance_from_52w_high_pct: float = Field(
        ...,
        description="Distance from 52-week peak as decimal (e.g. -0.22 for -22%)"
    )
    fundamental_guard_passed: bool = Field(
        ...,
        description="True if company passes all revenue, FCF, and debt-to-equity gates"
    )
    financials: CompanyFinancials = Field(
        ...,
        description="Audited fundamental financials payload"
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 evaluation timestamp"
    )
    # Technical & diagnostic metadata
    technical_gate_passed: bool = Field(
        default=True,
        description="True if price falls within the 15% to 30% retracement gate"
    )
    momentum_stabilized: bool = Field(
        default=True,
        description="True if downside momentum shows signs of flattening or reversal"
    )
    atr_14: Optional[float] = Field(
        default=None,
        description="14-period Average True Range for downstream position sizing"
    )
    rejection_reasons: List[str] = Field(
        default_factory=list,
        description="List of reasons if candidate was disqualified"
    )


class ValueDipPayload(BaseModel):
    """
    Collection envelope of screened value dip candidates.
    """
    execution_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of screening batch"
    )
    total_candidates_evaluated: int = Field(..., description="Total tickers inspected")
    qualified_candidates_count: int = Field(..., description="Count of qualified value dips")
    candidates: List[ValueDipCandidate] = Field(
        default_factory=list,
        description="List of qualified value dip opportunities"
    )
    rejection_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Rejection cause frequency breakdown"
    )


class ValueDisconnectEngine:
    """
    Module 02 Core Engine: Implements technical retracement gating, fundamental disconnect guards,
    and reversal momentum checks.
    """

    def __init__(
        self,
        connector: Optional[YFinanceDataConnector] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.connector = connector or get_data_connector()
        self.indicators = indicator_engine or get_indicator_engine()

    def evaluate_fundamental_guard(
        self,
        financials: CompanyFinancials,
        criteria: ValueDipFilterCriteria
    ) -> tuple[bool, List[str]]:
        """
        Validates GAAP operational soundness according to PRD:
        - Revenue Growth > 0%
        - Free Cash Flow > 0
        - Debt-to-Equity < 1.5
        Returns (is_passed, failure_reasons).
        """
        reasons: List[str] = []

        # 1. Revenue Growth Check (> 0%)
        rev_growth = financials.revenue_growth_qoq
        if rev_growth is None:
            reasons.append("missing_revenue_growth_data")
        elif rev_growth <= criteria.min_revenue_growth:
            reasons.append(
                f"negative_or_flat_revenue_growth ({rev_growth * 100:.1f}% <= {criteria.min_revenue_growth * 100:.1f}%)"
            )

        # 2. Free Cash Flow Check (> 0)
        fcf = financials.free_cash_flow
        if criteria.require_positive_fcf:
            if fcf is None:
                reasons.append("missing_free_cash_flow_data")
            elif fcf <= 0:
                reasons.append(f"negative_free_cash_flow (${fcf:,.0f} <= 0)")

        # 3. Debt to Equity Check (< 1.5)
        dte = financials.debt_to_equity
        if dte is None:
            # Conservative check: if D/E is unavailable, check if operating cash flow exists
            if financials.operating_cash_flow is None or financials.operating_cash_flow <= 0:
                reasons.append("missing_debt_to_equity_data")
        elif dte >= criteria.max_debt_to_equity:
            reasons.append(f"excessive_leverage_debt_to_equity ({dte:.2f} >= {criteria.max_debt_to_equity:.2f})")

        is_passed = len(reasons) == 0
        return is_passed, reasons

    def evaluate_technical_gate(
        self,
        distance_pct: float,
        criteria: ValueDipFilterCriteria
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluates whether current price is 15% to 30% below 52-week peak.
        Returns (is_passed, failure_reason).
        """
        # distance_pct is negative (e.g. -0.22)
        # min_drop_from_52w_high is -0.30 (lower bound, deepest drop)
        # max_drop_from_52w_high is -0.15 (upper bound, minimum drop)
        if distance_pct > criteria.max_drop_from_52w_high:
            return False, f"insufficient_dip ({distance_pct*100:.1f}% > {criteria.max_drop_from_52w_high*100:.1f}%)"

        if distance_pct < criteria.min_drop_from_52w_high:
            return False, f"excessive_decline_structural_break ({distance_pct*100:.1f}% < {criteria.min_drop_from_52w_high*100:.1f}%)"

        return True, None

    def evaluate_candidate(
        self,
        ticker: str,
        df_history: Optional[pd.DataFrame] = None,
        financials: Optional[CompanyFinancials] = None,
        high_52w: Optional[float] = None,
        criteria: Optional[ValueDipFilterCriteria] = None,
    ) -> ValueDipCandidate:
        """
        Processes a single ticker through the full Value Dip & Fundamental Disconnect pipeline.
        """
        clean_ticker = ticker.strip().upper()
        active_criteria = criteria or ValueDipFilterCriteria()
        rejection_reasons: List[str] = []

        # 1. Obtain Historical OHLCV Data
        if df_history is None or df_history.empty:
            try:
                payload = self.connector.fetch_ticker_payload(clean_ticker, period="1y")
                df_history = payload.to_dataframe()
                if high_52w is None:
                    high_52w = payload.profile.fifty_two_week_high
            except Exception as e:
                logger.error(f"Failed to fetch market data for {clean_ticker}: {e}")
                return ValueDipCandidate(
                    ticker=clean_ticker,
                    current_price=0.0,
                    distance_from_52w_high_pct=0.0,
                    fundamental_guard_passed=False,
                    financials=CompanyFinancials(),
                    technical_gate_passed=False,
                    momentum_stabilized=False,
                    rejection_reasons=[f"market_data_error: {e}"],
                )

        # 2. Compute Technical Indicators (ATR, Distance from High, Momentum)
        try:
            tech_indicators = self.indicators.calculate_indicators(
                ticker=clean_ticker,
                df=df_history,
                high_52w=high_52w,
            )
        except Exception as e:
            logger.error(f"Indicator calculation error for {clean_ticker}: {e}")
            return ValueDipCandidate(
                ticker=clean_ticker,
                current_price=float(df_history["Close"].iloc[-1]),
                distance_from_52w_high_pct=0.0,
                fundamental_guard_passed=False,
                financials=CompanyFinancials(),
                technical_gate_passed=False,
                momentum_stabilized=False,
                rejection_reasons=[f"indicator_error: {e}"],
            )

        current_price = tech_indicators.current_price
        distance_pct = tech_indicators.distance_from_52w_high_pct

        # 3. Technical Retracement Gate Evaluation
        tech_passed, tech_reason = self.evaluate_technical_gate(distance_pct, active_criteria)
        if not tech_passed and tech_reason:
            rejection_reasons.append(tech_reason)

        # 4. Momentum Stabilization Check
        momentum_passed = True
        if active_criteria.require_momentum_stabilization:
            momentum_passed = tech_indicators.momentum_stabilizing
            if not momentum_passed:
                rejection_reasons.append("unmitigated_downward_momentum")

        # 5. Fundamental Disconnect Guard Evaluation
        if financials is None:
            try:
                raw_fin = self.connector.fetch_financials(clean_ticker)
                financials = CompanyFinancials(
                    revenue_growth_qoq=raw_fin.get("revenue_growth_qoq"),
                    free_cash_flow=raw_fin.get("free_cash_flow"),
                    debt_to_equity=raw_fin.get("debt_to_equity"),
                    operating_cash_flow=raw_fin.get("operating_cash_flow"),
                )
            except Exception as e:
                logger.warning(f"Could not fetch financials for {clean_ticker}: {e}")
                financials = CompanyFinancials()

        fund_passed, fund_reasons = self.evaluate_fundamental_guard(financials, active_criteria)
        if not fund_passed:
            rejection_reasons.extend(fund_reasons)

        return ValueDipCandidate(
            ticker=clean_ticker,
            current_price=current_price,
            distance_from_52w_high_pct=distance_pct,
            fundamental_guard_passed=fund_passed,
            financials=financials,
            technical_gate_passed=tech_passed,
            momentum_stabilized=momentum_passed,
            atr_14=tech_indicators.atr_14,
            rejection_reasons=rejection_reasons,
        )

    def screen_candidates(
        self,
        tickers: List[str],
        criteria: Optional[ValueDipFilterCriteria] = None,
        preloaded_histories: Optional[Dict[str, pd.DataFrame]] = None,
        preloaded_financials: Optional[Dict[str, CompanyFinancials]] = None,
        preloaded_highs: Optional[Dict[str, float]] = None,
    ) -> ValueDipPayload:
        """
        Batch screen across candidate tickers, applying technical dip gate & fundamental guard.
        """
        active_criteria = criteria or ValueDipFilterCriteria()
        unique_tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
        qualified_candidates: List[ValueDipCandidate] = []
        rejection_breakdown: Dict[str, int] = {}

        logger.info(f"Starting Value Dip screening across {len(unique_tickers)} tickers")

        for ticker in unique_tickers:
            df = preloaded_histories.get(ticker) if preloaded_histories else None
            fin = preloaded_financials.get(ticker) if preloaded_financials else None
            high = preloaded_highs.get(ticker) if preloaded_highs else None

            candidate = self.evaluate_candidate(
                ticker=ticker,
                df_history=df,
                financials=fin,
                high_52w=high,
                criteria=active_criteria,
            )

            # A qualified candidate must pass technical gate, fundamental guard, and momentum check
            is_fully_qualified = (
                candidate.technical_gate_passed
                and candidate.fundamental_guard_passed
                and candidate.momentum_stabilized
            )

            if is_fully_qualified:
                qualified_candidates.append(candidate)
            else:
                for reason in candidate.rejection_reasons:
                    category = reason.split()[0]
                    rejection_breakdown[category] = rejection_breakdown.get(category, 0) + 1

        logger.info(
            f"Value Dip screening complete: {len(qualified_candidates)}/{len(unique_tickers)} qualified"
        )

        return ValueDipPayload(
            execution_timestamp=datetime.now(timezone.utc).isoformat(),
            total_candidates_evaluated=len(unique_tickers),
            qualified_candidates_count=len(qualified_candidates),
            candidates=qualified_candidates,
            rejection_breakdown=rejection_breakdown,
        )

    def screen_from_universe_payload(
        self,
        universe_payload: UniversePayload,
        criteria: Optional[ValueDipFilterCriteria] = None,
        preloaded_histories: Optional[Dict[str, pd.DataFrame]] = None,
        preloaded_financials: Optional[Dict[str, CompanyFinancials]] = None,
    ) -> ValueDipPayload:
        """
        Direct pipeline stage connector: Ingests output from Module 01 (UniversePayload).
        """
        tickers = [c.ticker for c in universe_payload.constituents]
        return self.screen_candidates(
            tickers=tickers,
            criteria=criteria,
            preloaded_histories=preloaded_histories,
            preloaded_financials=preloaded_financials,
        )


def get_value_disconnect_engine() -> ValueDisconnectEngine:
    """Factory helper returning initialized ValueDisconnectEngine."""
    return ValueDisconnectEngine()
