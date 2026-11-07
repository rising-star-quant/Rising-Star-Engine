"""
Rising Star Trend Engine v2.0 - Risk Guard (Module 03 / Module 06)
Calculates exact share sizing, dynamic volatility-based stops, and enforces
strict sector concentration caps and capital preservation rules.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.config import Settings, get_settings

logger = logging.getLogger("rising_star.risk_guard")


class PortfolioPosition(BaseModel):
    """
    Active position currently held in the portfolio.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    sector: str = Field(default="Unknown", description="Economic sector classification")
    shares: int = Field(..., ge=0, description="Quantity of shares held")
    entry_price: float = Field(..., gt=0, description="Execution price per share")
    current_price: float = Field(..., gt=0, description="Current market price per share")
    stop_loss: float = Field(..., gt=0, description="Active stop loss price")
    highest_price_seen: float = Field(default=0.0, description="Peak price reached while in position")

    @property
    def market_value(self) -> float:
        """Current total market value of the position."""
        return round(self.shares * self.current_price, 2)


class PortfolioState(BaseModel):
    """
    Complete snapshot of portfolio balances, cash, and open positions.
    """
    total_equity: float = Field(..., ge=0.0, description="Total account equity in USD (cash + market value)")
    cash: float = Field(..., ge=0.0, description="Unallocated settled cash balance")
    positions: Dict[str, PortfolioPosition] = Field(
        default_factory=dict,
        description="Dictionary mapping ticker symbol to active position"
    )

    def get_sector_exposure_dollars(self, sector: str) -> float:
        """Calculates total dollar value invested in a given sector."""
        total = 0.0
        for pos in self.positions.values():
            if pos.sector.strip().lower() == sector.strip().lower():
                total += pos.market_value
        return round(total, 2)

    def get_sector_exposure_pct(self, sector: str) -> float:
        """Calculates current percentage of total equity committed to a given sector."""
        if self.total_equity <= 0:
            return 0.0
        return round(self.get_sector_exposure_dollars(sector) / self.total_equity, 4)


class PositionSizingRequest(BaseModel):
    """
    Input contract for calculating position size and stop levels.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    entry_price: float = Field(..., description="Projected or current entry price")
    atr: float = Field(..., description="14-period Average True Range in price units")
    sector: str = Field(default="Unknown", description="Economic sector classification")
    account_balance: float = Field(..., description="Total account equity to size against")
    risk_pct: Optional[float] = Field(
        default=None,
        description="Fraction of account risked (e.g. 0.015 for 1.5%). Falls back to settings."
    )
    atr_multiplier: Optional[float] = Field(
        default=None,
        description="Multiplier for ATR stop distance (e.g. 2.0). Falls back to settings."
    )
    risk_reward_ratio: Optional[float] = Field(
        default=None,
        description="Reward-to-risk ratio for profit target (e.g. 2.5). Falls back to settings."
    )
    max_position_pct: Optional[float] = Field(
        default=None,
        description="Cap on capital allocated to single position (e.g. 0.15 for 15%)."
    )


class PositionSizingPlan(BaseModel):
    """
    Validated sizing plan ready for order routing and risk validation.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    sector: str = Field(default="Unknown", description="Sector classification")
    entry_price: float = Field(..., description="Entry price per share")
    stop_loss_price: float = Field(..., description="Calculated hard stop-loss price")
    target_price: float = Field(..., description="Calculated profit target price")
    stop_distance: float = Field(..., description="Dollar distance between entry and stop")
    stop_distance_pct: float = Field(..., description="Stop distance as percentage of entry price")
    risk_reward_ratio: float = Field(..., description="Reward-to-risk multiple (e.g. 2.5)")
    risk_amount: float = Field(..., description="Total dollar amount risked if stop is hit")
    recommended_shares: int = Field(..., ge=0, description="Number of whole shares to purchase")
    allocated_capital: float = Field(..., ge=0, description="Total capital required (shares * entry)")
    capital_allocation_pct: float = Field(..., ge=0, description="Percentage of account equity allocated")
    is_viable: bool = Field(default=True, description="True if position satisfies all sizing constraints")
    rejection_reason: Optional[str] = Field(default=None, description="Explanation if trade cannot be sized")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 generation timestamp"
    )


class SectorExposureValidation(BaseModel):
    """
    Validation decision regarding sector concentration limits.
    """
    is_allowed: bool = Field(..., description="True if proposed allocation complies with sector limit")
    sector: str = Field(..., description="Evaluated sector name")
    current_sector_pct: float = Field(..., description="Current sector exposure as decimal (e.g. 0.12)")
    projected_sector_pct: float = Field(..., description="Projected sector exposure after trade")
    max_sector_pct: float = Field(..., description="Hard exposure ceiling (e.g. 0.20)")
    allowable_capital: float = Field(..., description="Maximum dollar amount that can still be added to sector")
    max_allowed_shares: int = Field(default=0, ge=0, description="Maximum shares that fit under sector cap")
    reason: str = Field(..., description="Human-readable decision explanation")


class DynamicStopsResult(BaseModel):
    """
    Result of evaluating dynamic trailing stops against price movement.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    entry_price: float = Field(..., description="Initial trade entry price")
    current_price: float = Field(..., description="Current market price")
    initial_stop: float = Field(..., description="Initial structural stop-loss price")
    trailing_stop: float = Field(..., description="Ratchet trailing stop price")
    profit_target: float = Field(..., description="Fixed profit target price")
    highest_price: float = Field(..., description="Highest price recorded during the trade")
    status: Literal["ACTIVE", "STOPPED_OUT", "TARGET_REACHED"] = Field(
        default="ACTIVE",
        description="Current trade status under dynamic stops"
    )


class RiskGuard:
    """
    Principal risk engine for position sizing, trailing stop management,
    and portfolio sector concentration compliance.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def calculate_stops_and_targets(
        self,
        entry_price: float,
        atr: float,
        stop_multiplier: Optional[float] = None,
        risk_reward_ratio: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Computes the ATR-based stop-loss level and take-profit target.

        Formula:
            Stop Distance = stop_multiplier * ATR
            Stop Loss Price = max(0.01, entry_price - Stop Distance)
            Target Price = entry_price + (risk_reward_ratio * Stop Distance)

        Returns:
            Tuple of (stop_loss_price, target_price, stop_distance)
        """
        mult = stop_multiplier if stop_multiplier is not None else self.settings.default_atr_stop_multiplier
        rr = risk_reward_ratio if risk_reward_ratio is not None else self.settings.default_risk_reward_ratio

        if entry_price <= 0:
            raise ValueError(f"Entry price must be strictly positive, got {entry_price}")
        if atr <= 0:
            raise ValueError(f"ATR must be strictly positive, got {atr}")
        if mult <= 0:
            raise ValueError(f"Stop multiplier must be strictly positive, got {mult}")

        stop_distance = round(mult * atr, 4)
        stop_loss = max(0.01, round(entry_price - stop_distance, 4))
        target_price = round(entry_price + (rr * stop_distance), 4)

        return stop_loss, target_price, stop_distance

    def calculate_position_size(self, request: PositionSizingRequest) -> PositionSizingPlan:
        """
        Calculates exact share sizing based on:
        1. Account balance
        2. Fixed risk percentage per trade (e.g. 1% - 2%)
        3. ATR-based stop distance
        4. Maximum capital allocation cap per position (e.g. 15% of account balance)

        Returns:
            Fully validated PositionSizingPlan
        """
        account_balance = request.account_balance
        entry_price = request.entry_price
        atr = request.atr
        sector = request.sector
        ticker = request.ticker

        risk_pct = (
            request.risk_pct
            if request.risk_pct is not None
            else self.settings.risk_per_trade_pct
        )
        atr_multiplier = (
            request.atr_multiplier
            if request.atr_multiplier is not None
            else self.settings.default_atr_stop_multiplier
        )
        risk_reward_ratio = (
            request.risk_reward_ratio
            if request.risk_reward_ratio is not None
            else self.settings.default_risk_reward_ratio
        )
        max_pos_pct = (
            request.max_position_pct
            if request.max_position_pct is not None
            else self.settings.max_position_capital_pct
        )

        # Baseline guard on negative or zero equity / price / atr
        if account_balance <= 0:
            return PositionSizingPlan(
                ticker=ticker,
                sector=sector,
                entry_price=max(0.01, entry_price),
                stop_loss_price=0.0,
                target_price=0.0,
                stop_distance=0.0,
                stop_distance_pct=0.0,
                risk_reward_ratio=risk_reward_ratio,
                risk_amount=0.0,
                recommended_shares=0,
                allocated_capital=0.0,
                capital_allocation_pct=0.0,
                is_viable=False,
                rejection_reason="Account balance must be greater than zero.",
            )

        if entry_price <= 0:
            return PositionSizingPlan(
                ticker=ticker,
                sector=sector,
                entry_price=entry_price,
                stop_loss_price=0.0,
                target_price=0.0,
                stop_distance=0.0,
                stop_distance_pct=0.0,
                risk_reward_ratio=risk_reward_ratio,
                risk_amount=0.0,
                recommended_shares=0,
                allocated_capital=0.0,
                capital_allocation_pct=0.0,
                is_viable=False,
                rejection_reason="Entry price must be strictly greater than zero.",
            )

        if atr <= 0:
            return PositionSizingPlan(
                ticker=ticker,
                sector=sector,
                entry_price=entry_price,
                stop_loss_price=0.0,
                target_price=0.0,
                stop_distance=0.0,
                stop_distance_pct=0.0,
                risk_reward_ratio=risk_reward_ratio,
                risk_amount=0.0,
                recommended_shares=0,
                allocated_capital=0.0,
                capital_allocation_pct=0.0,
                is_viable=False,
                rejection_reason="ATR must be strictly greater than zero to compute stop distance.",
            )

        # Compute dynamic stops
        stop_loss, target_price, stop_distance = self.calculate_stops_and_targets(
            entry_price=entry_price,
            atr=atr,
            stop_multiplier=atr_multiplier,
            risk_reward_ratio=risk_reward_ratio,
        )

        stop_distance_pct = round(stop_distance / entry_price, 4)

        # Account dollar risk limit
        dollar_risk_budget = round(account_balance * risk_pct, 2)

        # 1. Shares bounded by risk budget: Risk / Stop Distance
        if stop_distance <= 0:
            shares_by_risk = 0
        else:
            shares_by_risk = math.floor(dollar_risk_budget / stop_distance)

        # 2. Shares bounded by maximum capital allocation per position
        max_capital_budget = round(account_balance * max_pos_pct, 2)
        shares_by_capital = math.floor(max_capital_budget / entry_price)

        # Final whole share sizing is the minimum of both bounds
        recommended_shares = max(0, min(shares_by_risk, shares_by_capital))
        allocated_capital = round(recommended_shares * entry_price, 2)
        capital_allocation_pct = round(allocated_capital / account_balance, 4)

        is_viable = recommended_shares > 0
        rejection_reason = None
        if not is_viable:
            if shares_by_risk == 0:
                rejection_reason = (
                    f"Stop distance (${stop_distance:.2f}) exceeds total trade risk budget "
                    f"(${dollar_risk_budget:.2f}). Account balance insufficient for 1 share."
                )
            elif shares_by_capital == 0:
                rejection_reason = (
                    f"Single share price (${entry_price:.2f}) exceeds maximum allowable position "
                    f"capital cap (${max_capital_budget:.2f})."
                )

        return PositionSizingPlan(
            ticker=ticker,
            sector=sector,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            target_price=target_price,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            risk_reward_ratio=risk_reward_ratio,
            risk_amount=dollar_risk_budget,
            recommended_shares=recommended_shares,
            allocated_capital=allocated_capital,
            capital_allocation_pct=capital_allocation_pct,
            is_viable=is_viable,
            rejection_reason=rejection_reason,
        )

    def validate_sector_exposure(
        self,
        portfolio: PortfolioState,
        ticker: str,
        sector: str,
        planned_capital: float,
        max_sector_pct: Optional[float] = None,
        share_price: Optional[float] = None,
    ) -> SectorExposureValidation:
        """
        Enforces hard limits preventing sector concentration limits (e.g. max 20% exposure per sector).

        Args:
            portfolio: Current portfolio state with equity and active positions.
            ticker: Ticker symbol being considered.
            sector: Sector of the candidate.
            planned_capital: Total capital proposed for allocation.
            max_sector_pct: Optional custom threshold (defaults to settings.max_sector_exposure_pct).
            share_price: Optional share price for calculating max allowed shares.

        Returns:
            SectorExposureValidation containing approval, projected percentage, and allowable capacity.
        """
        limit_pct = (
            max_sector_pct
            if max_sector_pct is not None
            else self.settings.max_sector_exposure_pct
        )

        total_equity = portfolio.total_equity
        if total_equity <= 0:
            return SectorExposureValidation(
                is_allowed=False,
                sector=sector,
                current_sector_pct=0.0,
                projected_sector_pct=0.0,
                max_sector_pct=limit_pct,
                allowable_capital=0.0,
                max_allowed_shares=0,
                reason="Portfolio total equity is zero or negative.",
            )

        current_sector_dollars = portfolio.get_sector_exposure_dollars(sector)
        current_sector_pct = round(current_sector_dollars / total_equity, 4)

        max_allowed_sector_capital = round(total_equity * limit_pct, 2)
        allowable_capital = max(0.0, round(max_allowed_sector_capital - current_sector_dollars, 2))

        projected_sector_dollars = current_sector_dollars + planned_capital
        projected_sector_pct = round(projected_sector_dollars / total_equity, 4)

        max_allowed_shares = 0
        if share_price and share_price > 0:
            max_allowed_shares = math.floor(allowable_capital / share_price)

        is_allowed = projected_sector_pct <= limit_pct

        if is_allowed:
            reason = (
                f"Sector '{sector}' exposure remains compliant: "
                f"projected {projected_sector_pct:.1%} <= ceiling {limit_pct:.1%}."
            )
        else:
            reason = (
                f"Sector '{sector}' concentration breach: projected {projected_sector_pct:.1%} "
                f"exceeds max ceiling of {limit_pct:.1%}. "
                f"Available headroom in sector is ${allowable_capital:.2f}."
            )

        return SectorExposureValidation(
            is_allowed=is_allowed,
            sector=sector,
            current_sector_pct=current_sector_pct,
            projected_sector_pct=projected_sector_pct,
            max_sector_pct=limit_pct,
            allowable_capital=allowable_capital,
            max_allowed_shares=max_allowed_shares,
            reason=reason,
        )

    def update_trailing_stop(
        self,
        current_price: float,
        highest_price: float,
        initial_stop: float,
        atr: float,
        trailing_multiplier: Optional[float] = None,
    ) -> float:
        """
        Ratchets the trailing stop loss upward as the trade progresses.
        The stop never moves downward.

        Formula:
            new_stop = max(initial_stop, highest_price - (trailing_multiplier * atr))
        """
        mult = (
            trailing_multiplier
            if trailing_multiplier is not None
            else self.settings.default_atr_stop_multiplier
        )
        if atr <= 0:
            return initial_stop

        calculated_stop = round(highest_price - (mult * atr), 4)
        return max(initial_stop, calculated_stop)

    def evaluate_dynamic_stops(
        self,
        position: PortfolioPosition,
        current_price: float,
        atr: float,
        profit_target: float,
        trailing_multiplier: Optional[float] = None,
    ) -> DynamicStopsResult:
        """
        Evaluates active position status against dynamic trailing stops and profit targets.
        """
        highest_seen = max(position.highest_price_seen, current_price, position.entry_price)
        updated_stop = self.update_trailing_stop(
            current_price=current_price,
            highest_price=highest_seen,
            initial_stop=position.stop_loss,
            atr=atr,
            trailing_multiplier=trailing_multiplier,
        )

        status: Literal["ACTIVE", "STOPPED_OUT", "TARGET_REACHED"] = "ACTIVE"
        if current_price <= updated_stop:
            status = "STOPPED_OUT"
        elif current_price >= profit_target:
            status = "TARGET_REACHED"

        return DynamicStopsResult(
            ticker=position.ticker,
            entry_price=position.entry_price,
            current_price=current_price,
            initial_stop=position.stop_loss,
            trailing_stop=updated_stop,
            profit_target=profit_target,
            highest_price=highest_seen,
            status=status,
        )


def get_risk_guard(settings: Optional[Settings] = None) -> RiskGuard:
    """Factory helper returning RiskGuard instance."""
    return RiskGuard(settings=settings)
