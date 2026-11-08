"""
Pytest suite for Module 03 (RiskGuard, AI Skeptic, and Earnings Reaction Scanner).
Comprehensive unit tests covering position sizing, dynamic ATR stops, sector exposure guards,
Gemini-powered adversarial AI validation with deterministic offline fallbacks, and earnings crash screening.
"""

from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest

from src.ai.ai_skeptic import (
    AISkeptic,
    SkepticAnalysisRequest,
    SkepticVerdict,
    get_ai_skeptic,
)
from src.config import Settings
from src.engines.earnings_scanner import (
    EarningsFilterCriteria,
    EarningsReactionScanner,
    EarningsWreckPayload,
    get_earnings_scanner,
)
from src.engines.value_dip_engine import CompanyFinancials, ValueDipCandidate
from src.risk.risk_guard import (
    DynamicStopsResult,
    PortfolioPosition,
    PortfolioState,
    PositionSizingPlan,
    PositionSizingRequest,
    RiskGuard,
    SectorExposureValidation,
    get_risk_guard,
)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def risk_settings():
    """Provides isolated testing settings for risk management."""
    return Settings(
        env="testing",
        risk_per_trade_pct=0.015,       # 1.5%
        max_sector_exposure_pct=0.20,   # 20.0%
        max_position_capital_pct=0.15,  # 15.0%
        default_atr_stop_multiplier=2.0,
        default_risk_reward_ratio=2.5,
        ai_skeptic_rejection_threshold=0.60,
    )


@pytest.fixture
def risk_guard(risk_settings):
    """Provides a fresh RiskGuard instance."""
    return RiskGuard(settings=risk_settings)


@pytest.fixture
def ai_skeptic(risk_settings):
    """Provides an AISkeptic instance with forced offline deterministic mode."""
    return AISkeptic(settings=risk_settings, force_offline=True)


@pytest.fixture
def sample_portfolio():
    """Provides a sample multi-position portfolio state."""
    positions = {
        "NVDA": PortfolioPosition(
            ticker="NVDA",
            sector="Technology",
            shares=50,
            entry_price=120.0,
            current_price=130.0,
            stop_loss=110.0,
            highest_price_seen=132.0,
        ),
        "AMD": PortfolioPosition(
            ticker="AMD",
            sector="Technology",
            shares=40,
            entry_price=150.0,
            current_price=155.0,
            stop_loss=140.0,
            highest_price_seen=160.0,
        ),
        "PFE": PortfolioPosition(
            ticker="PFE",
            sector="Healthcare",
            shares=200,
            entry_price=28.0,
            current_price=29.0,
            stop_loss=26.5,
            highest_price_seen=30.0,
        ),
    }
    # NVDA market value: 50 * 130 = $6,500
    # AMD market value: 40 * 155 = $6,200
    # Tech total: $12,700
    # PFE market value: 200 * 29 = $5,800
    # Total positions value: $18,500
    # Total equity: $100,000, Cash: $81,500
    return PortfolioState(
        total_equity=100_000.0,
        cash=81_500.0,
        positions=positions,
    )


# =====================================================================
# 1. POSITION SIZING & DYNAMIC STOPS TESTS
# =====================================================================

class TestRiskGuardPositionSizing:
    """Tests the core position sizing engine and capital allocation bounds."""

    def test_standard_position_sizing_calculation(self, risk_guard):
        """
        Verify:
        Account = $100,000, Risk = 1.5% ($1,500)
        Entry = $100.00, ATR = $2.50, Multiplier = 2.0x -> Stop Distance = $5.00
        Stop Loss = $95.00, Target = $100 + (2.5 * 5.0) = $112.50
        Shares by risk = floor(1500 / 5) = 300 shares
        Cap by position (15% = $15,000) -> floor(15000 / 100) = 150 shares
        Recommended shares = min(300, 150) = 150 shares
        Allocated capital = 150 * $100 = $15,000 (15%)
        """
        req = PositionSizingRequest(
            ticker="ALPHA",
            entry_price=100.0,
            atr=2.5,
            sector="Technology",
            account_balance=100_000.0,
            risk_pct=0.015,
            atr_multiplier=2.0,
            risk_reward_ratio=2.5,
            max_position_pct=0.15,
        )

        plan = risk_guard.calculate_position_size(req)

        assert plan.ticker == "ALPHA"
        assert plan.entry_price == 100.0
        assert plan.stop_loss_price == 95.0
        assert plan.target_price == 112.50
        assert plan.stop_distance == 5.0
        assert plan.stop_distance_pct == 0.05
        assert plan.recommended_shares == 150
        assert plan.allocated_capital == 15_000.0
        assert plan.capital_allocation_pct == 0.15
        assert plan.is_viable is True
        assert plan.rejection_reason is None

    def test_shares_bounded_purely_by_risk_budget(self, risk_guard):
        """
        When position capital limit is generous (e.g. 50%), risk budget determines shares.
        Account = $100k, Risk = 1% ($1,000)
        Entry = $50, ATR = $2.0, Mult = 2.0x -> Stop Distance = $4.00
        Shares = floor(1000 / 4) = 250 shares
        Allocated capital = 250 * 50 = $12,500 (12.5% of account, under 50% cap)
        """
        req = PositionSizingRequest(
            ticker="BETA",
            entry_price=50.0,
            atr=2.0,
            sector="Industrials",
            account_balance=100_000.0,
            risk_pct=0.01,
            atr_multiplier=2.0,
            risk_reward_ratio=2.0,
            max_position_pct=0.50,
        )

        plan = risk_guard.calculate_position_size(req)

        assert plan.stop_distance == 4.0
        assert plan.recommended_shares == 250
        assert plan.allocated_capital == 12_500.0
        assert plan.stop_loss_price == 46.0
        assert plan.target_price == 58.0
        assert plan.is_viable is True

    def test_zero_or_negative_account_balance(self, risk_guard):
        """Zero or negative account balance safely yields non-viable plan with 0 shares."""
        req = PositionSizingRequest(
            ticker="ZERO",
            entry_price=50.0,
            atr=1.5,
            account_balance=0.0,
        )

        plan = risk_guard.calculate_position_size(req)

        assert plan.recommended_shares == 0
        assert plan.allocated_capital == 0.0
        assert plan.is_viable is False
        assert "Account balance" in plan.rejection_reason

    def test_invalid_entry_price_or_atr(self, risk_guard):
        """Non-positive entry price or ATR yields non-viable sizing plan."""
        req_bad_price = PositionSizingRequest(
            ticker="BADP",
            entry_price=-10.0,
            atr=2.0,
            account_balance=50_000.0,
        )
        plan_price = risk_guard.calculate_position_size(req_bad_price)
        assert plan_price.is_viable is False
        assert "Entry price" in plan_price.rejection_reason

        req_bad_atr = PositionSizingRequest(
            ticker="BADATR",
            entry_price=20.0,
            atr=0.0,
            account_balance=50_000.0,
        )
        plan_atr = risk_guard.calculate_position_size(req_bad_atr)
        assert plan_atr.is_viable is False
        assert "ATR" in plan_atr.rejection_reason

    def test_single_share_exceeds_risk_budget(self, risk_guard):
        """If risk budget is too small to even afford 1 share's stop distance, plan is rejected."""
        # Account $1,000, Risk 1% = $10.00
        # Entry $500, ATR $20, Stop Dist $40 -> 10 / 40 = 0.25 -> 0 shares
        req = PositionSizingRequest(
            ticker="EXPENSIVE",
            entry_price=500.0,
            atr=20.0,
            account_balance=1_000.0,
            risk_pct=0.01,
        )

        plan = risk_guard.calculate_position_size(req)

        assert plan.recommended_shares == 0
        assert plan.is_viable is False
        assert "insufficient for 1 share" in plan.rejection_reason


# =====================================================================
# 2. SECTOR EXPOSURE GUARD TESTS
# =====================================================================

class TestSectorExposureGuard:
    """Tests portfolio sector concentration monitoring and limit enforcement."""

    def test_compliant_sector_allocation(self, risk_guard, sample_portfolio):
        """
        Tech sector currently: $12,700 (12.7% of $100k).
        Ceiling is 20.0% ($20,000).
        Adding $5,000 planned capital raises Tech to $17,700 (17.7%) <= 20.0%.
        Should be APPROVED.
        """
        validation = risk_guard.validate_sector_exposure(
            portfolio=sample_portfolio,
            ticker="INTC",
            sector="Technology",
            planned_capital=5_000.0,
            share_price=25.0,
        )

        assert validation.is_allowed is True
        assert validation.current_sector_pct == 0.127
        assert validation.projected_sector_pct == 0.177
        assert validation.max_sector_pct == 0.20
        assert validation.allowable_capital == 7_300.0  # $20k - $12.7k
        assert validation.max_allowed_shares == 292     # floor(7300 / 25)
        assert "remains compliant" in validation.reason

    def test_sector_concentration_breach_rejected(self, risk_guard, sample_portfolio):
        """
        Tech sector currently: $12,700 (12.7%).
        Adding $10,000 planned capital raises Tech to $22,700 (22.7%) > 20.0%.
        Should be REJECTED.
        """
        validation = risk_guard.validate_sector_exposure(
            portfolio=sample_portfolio,
            ticker="QCOM",
            sector="Technology",
            planned_capital=10_000.0,
            share_price=150.0,
        )

        assert validation.is_allowed is False
        assert validation.projected_sector_pct == 0.227
        assert "concentration breach" in validation.reason
        assert validation.allowable_capital == 7_300.0

    def test_new_unrepresented_sector(self, risk_guard, sample_portfolio):
        """A new sector (e.g. Financials) starts at 0% and is accepted if <= 20%."""
        validation = risk_guard.validate_sector_exposure(
            portfolio=sample_portfolio,
            ticker="JPM",
            sector="Financial Services",
            planned_capital=14_000.0,
            share_price=200.0,
        )

        assert validation.is_allowed is True
        assert validation.current_sector_pct == 0.0
        assert validation.projected_sector_pct == 0.14
        assert validation.allowable_capital == 20_000.0
        assert validation.max_allowed_shares == 100

    def test_zero_total_equity_portfolio(self, risk_guard):
        """Empty or zero equity portfolio safely rejects trades."""
        empty_portfolio = PortfolioState(total_equity=0.0, cash=0.0, positions={})
        validation = risk_guard.validate_sector_exposure(
            portfolio=empty_portfolio,
            ticker="TEST",
            sector="Energy",
            planned_capital=1_000.0,
        )
        assert validation.is_allowed is False
        assert "zero or negative" in validation.reason


# =====================================================================
# 3. DYNAMIC ATR TRAILING STOPS TESTS
# =====================================================================

class TestDynamicStops:
    """Tests trailing stop ratchets and exit signals."""

    def test_trailing_stop_ratchets_up_only(self, risk_guard):
        """
        Entry = $100, Initial Stop = $90, ATR = $5, Multiplier = 2.0x (offset $10).
        Price rises to $115 -> Trailing stop = 115 - 10 = $105 (ratcheted up).
        Price pulls back to $108 -> Trailing stop stays at $105 (never drops).
        """
        initial_stop = 90.0
        atr = 5.0
        mult = 2.0

        # At $115 new high
        stop_1 = risk_guard.update_trailing_stop(
            current_price=115.0,
            highest_price=115.0,
            initial_stop=initial_stop,
            atr=atr,
            trailing_multiplier=mult,
        )
        assert stop_1 == 105.0

        # When price drops back to $108, highest remains 115 -> stop stays 105
        stop_2 = risk_guard.update_trailing_stop(
            current_price=108.0,
            highest_price=115.0,
            initial_stop=initial_stop,
            atr=atr,
            trailing_multiplier=mult,
        )
        assert stop_2 == 105.0

    def test_evaluate_dynamic_stops_stopped_out(self, risk_guard):
        """Position triggered STOPPED_OUT when current price drops to or below stop."""
        pos = PortfolioPosition(
            ticker="TSLA",
            shares=10,
            entry_price=200.0,
            current_price=175.0,
            stop_loss=180.0,
            highest_price_seen=210.0,
        )
        result = risk_guard.evaluate_dynamic_stops(
            position=pos,
            current_price=175.0,
            atr=10.0,
            profit_target=250.0,
        )

        assert result.status == "STOPPED_OUT"
        assert result.trailing_stop >= 180.0

    def test_evaluate_dynamic_stops_target_reached(self, risk_guard):
        """Position triggered TARGET_REACHED when price hits or exceeds target."""
        pos = PortfolioPosition(
            ticker="AAPL",
            shares=50,
            entry_price=150.0,
            current_price=190.0,
            stop_loss=140.0,
            highest_price_seen=190.0,
        )
        result = risk_guard.evaluate_dynamic_stops(
            position=pos,
            current_price=190.0,
            atr=5.0,
            profit_target=180.0,
        )

        assert result.status == "TARGET_REACHED"


# =====================================================================
# 4. AI SKEPTIC TESTS (DETERMINISTIC HEURISTIC & GEMINI MOCKING)
# =====================================================================

class TestAISkeptic:
    """Tests the adversarial AI Skeptic module across offline and online modes."""

    @pytest.mark.asyncio
    async def test_offline_pristine_candidate_approved(self, ai_skeptic):
        """
        Pristine candidate with strong positive revenue (+12%), positive FCF ($850M),
        and low debt-to-equity (0.45) receives low skepticism score and is APPROVED.
        """
        candidate = ValueDipCandidate(
            ticker="PRISTINE",
            current_price=85.0,
            distance_from_52w_high_pct=-0.18,
            fundamental_guard_passed=True,
            financials=CompanyFinancials(
                revenue_growth_qoq=0.12,
                free_cash_flow=850_000_000.0,
                debt_to_equity=0.45,
            ),
        )

        verdict = await ai_skeptic.evaluate_candidate(candidate)

        assert verdict.ticker == "PRISTINE"
        assert verdict.verdict == "APPROVED"
        assert verdict.skepticism_score < 0.60
        assert len(verdict.red_flags) == 0
        assert verdict.is_mock is True
        assert len(verdict.justification) > 0

    @pytest.mark.asyncio
    async def test_offline_troubled_candidate_rejected(self, ai_skeptic):
        """
        Troubled candidate with high debt-to-equity (2.1), contracting revenue (-8%),
        and negative free cash flow (-$50M) triggers multiple red flags and is REJECTED.
        """
        req = SkepticAnalysisRequest(
            ticker="TRAP",
            current_price=22.0,
            distance_from_52w_high_pct=-0.29,
            sector="Consumer Discretionary",
            financials={
                "revenue_growth_qoq": -0.08,
                "free_cash_flow": -50_000_000.0,
                "debt_to_equity": 2.1,
            },
            news_headlines=["SEC probe opened into accounting practices", "CFO resigns unexpectedly"],
        )

        verdict = await ai_skeptic.evaluate_candidate(req)

        assert verdict.ticker == "TRAP"
        assert verdict.verdict == "REJECTED"
        assert verdict.skepticism_score >= 0.60
        assert any("leverage" in f.lower() or "d/e" in f.lower() for f in verdict.red_flags)
        assert any("contracting revenue" in f.lower() for f in verdict.red_flags)
        assert any("sec probe" in f.lower() or "cfo resigns" in f.lower() for f in verdict.red_flags)
        assert verdict.counter_thesis is not None

    @pytest.mark.asyncio
    async def test_mocked_gemini_api_success(self, risk_settings):
        """
        Simulate real Gemini API returning validated JSON schema.
        Ensures proper deserialization and confidence scoring.
        """
        mock_response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "ticker": "TECH1",
                                    "verdict": "APPROVED",
                                    "skepticism_score": 0.32,
                                    "red_flags": ["Minor supply chain delay noted in 10-Q"],
                                    "justification": [
                                        "Cyclical semiconductor inventory correction nearing trough.",
                                        "Gross margins expanded 180bps year-over-year.",
                                    ],
                                    "counter_thesis": "Potential near-term capex slowdown, but secular moat remains robust.",
                                    "confidence_score": 0.92,
                                })
                            }
                        ]
                    }
                }
            ]
        }

        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_payload
        mock_http_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_http_response

        online_settings = Settings(
            env="production",
            gemini_api_key="TEST_API_KEY_12345",
            gemini_model="gemini-3.8-flash",
        )
        skeptic = AISkeptic(settings=online_settings, force_offline=False, client=mock_client)

        req = SkepticAnalysisRequest(
            ticker="TECH1",
            current_price=140.0,
            distance_from_52w_high_pct=-0.21,
            sector="Technology",
        )

        verdict = await skeptic.evaluate_candidate(req)

        assert verdict.ticker == "TECH1"
        assert verdict.verdict == "APPROVED"
        assert verdict.skepticism_score == 0.32
        assert verdict.is_mock is False
        assert "semiconductor" in verdict.justification[0].lower()
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gemini_api_failure_engages_deterministic_fallback(self, risk_settings):
        """
        When Gemini API throws a network or HTTP error, engine gracefully falls back
        to deterministic heuristic and completes the assessment without throwing.
        """
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectTimeout("Connection to Gemini timed out")

        online_settings = Settings(
            env="production",
            gemini_api_key="TEST_API_KEY_12345",
        )
        skeptic = AISkeptic(settings=online_settings, force_offline=False, client=mock_client)

        req = SkepticAnalysisRequest(
            ticker="FALLBACK",
            current_price=55.0,
            distance_from_52w_high_pct=-0.16,
            financials={"debt_to_equity": 0.8, "revenue_growth_qoq": 0.05, "free_cash_flow": 100_000_000.0},
        )

        verdict = await skeptic.evaluate_candidate(req)

        assert verdict.ticker == "FALLBACK"
        assert verdict.verdict == "APPROVED"
        assert verdict.is_mock is True
        assert any("offline fallback engaged" in j.lower() for j in verdict.justification)

    def test_sync_evaluation_helpers(self, ai_skeptic):
        """Synchronous wrapper correctly evaluates candidate and batch."""
        req = SkepticAnalysisRequest(
            ticker="SYNC1",
            current_price=45.0,
            distance_from_52w_high_pct=-0.17,
            financials={"debt_to_equity": 0.5, "revenue_growth_qoq": 0.10, "free_cash_flow": 200_000_000.0},
        )
        verdict = ai_skeptic.evaluate_candidate_sync(req)
        assert verdict.ticker == "SYNC1"
        assert verdict.verdict == "APPROVED"

        batch_verdicts = ai_skeptic.evaluate_batch_sync([req])
        assert len(batch_verdicts) == 1
        assert batch_verdicts[0].ticker == "SYNC1"

    def test_hebrew_prompt_and_analysis_contract(self, ai_skeptic):
        """Verifies adversarial prompt strictly demands Hebrew output and offline evaluator produces Hebrew critique."""
        req = SkepticAnalysisRequest(
            ticker="HEB1",
            current_price=120.0,
            distance_from_52w_high_pct=-0.22,
            sector="טכנולוגיה",
            industry="תוכנה ארגונית",
            financials={"debt_to_equity": 0.9, "revenue_growth_qoq": 0.08, "free_cash_flow": 50_000_000.0},
        )
        prompt = ai_skeptic._build_adversarial_prompt(req)

        # Prompt explicitly instructs Hebrew output for critique and risks
        assert "Provide your critique, risk analysis, and recovery catalysts strictly in professional Hebrew" in prompt
        assert "עברית מקצועית ופיננסית" in prompt
        assert "פרקליט השטן" in prompt

        # Offline heuristic produces valid Hebrew justification and counter thesis
        verdict = ai_skeptic._evaluate_deterministic_heuristic(req, rejection_threshold=0.60)
        assert any("אושר" in j or "מבנה הון" in j or "תזרים" in j for j in verdict.justification)
        assert "המניה" in verdict.counter_thesis or "תיקון" in verdict.counter_thesis



# =====================================================================
# 5. EARNINGS WRECK REACTION SCANNER TESTS (MODULE 03 PRD)
# =====================================================================

class TestEarningsReactionScanner:
    """Tests the EarningsReactionScanner against the Module 03 Micro-PRD specifications."""

    @pytest.fixture
    def earnings_scanner(self):
        """Returns EarningsReactionScanner instance."""
        return get_earnings_scanner()

    @pytest.fixture
    def sample_wreck_df(self):
        """
        Creates a synthetic OHLCV DataFrame with:
        30 normal trading days: close ~$100, volume ~100,000.
        Drop day (event): close drops to $86 (-14%), volume surges to 320,000 (RVOL = 3.2x).
        """
        n_days = 35
        dates = pd.date_range("2026-10-01", periods=n_days, freq="B")
        closes = [100.0] * (n_days - 1) + [86.0]  # Drop from 100 to 86 = -14%
        volumes = [100_000] * (n_days - 1) + [320_000]  # RVOL = 3.2x
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]
        opens = closes.copy()

        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }, index=dates)

    def test_qualified_earnings_wreck_matches_prd_contract(self, earnings_scanner, sample_wreck_df):
        """
        PRD Target Contract:
        {
          "ticker": "WRECK",
          "earnings_date": "2026-11-04",
          "price_change_1d_pct": -0.14,
          "relative_volume": 3.2,
          "revenue_integrity_passed": true,
          "evaluated_at": "..."
        }
        """
        result = earnings_scanner.evaluate_candidate(
            ticker="WRECK",
            ohlcv_df=sample_wreck_df,
            earnings_date="2026-11-04",
            revenue_growth_yoy=0.08,  # Healthy 8% top line
            eps_surprise_pct=-0.02,   # Slight 2c miss causing panic
        )

        assert isinstance(result, EarningsWreckPayload)
        assert result.ticker == "WRECK"
        assert result.earnings_date == "2026-11-04"
        assert result.price_change_1d_pct == -0.14
        assert result.relative_volume == 3.2
        assert result.revenue_integrity_passed is True
        assert result.is_qualified is True
        assert result.rejection_reason is None

    def test_three_day_cumulative_drop_qualification(self, earnings_scanner):
        """
        Tests multi-day bleed scenario:
        Day -3: $100, Day -2: $95, Day -1: $90, Event Day: $84 (Total 3-day drop = -16% >= 15%).
        RVOL = 2.5x.
        """
        n_days = 35
        dates = pd.date_range("2026-10-01", periods=n_days, freq="B")
        closes = [100.0] * (n_days - 4) + [100.0, 95.0, 90.0, 84.0]
        volumes = [100_000] * (n_days - 1) + [250_000]

        df = pd.DataFrame({
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=dates)

        result = earnings_scanner.evaluate_candidate(
            ticker="BLEED",
            ohlcv_df=df,
            earnings_date="2026-11-05",
            revenue_growth_yoy=0.02,
        )

        assert result.price_change_3d_pct == -0.16
        assert result.relative_volume == 2.5
        assert result.is_qualified is True

    def test_rejected_when_drop_too_shallow(self, earnings_scanner):
        """A drop of only -4% does not meet the >= 8% acute drop threshold and is rejected."""
        n_days = 35
        closes = [100.0] * (n_days - 1) + [96.0]  # -4% drop
        volumes = [100_000] * (n_days - 1) + [300_000]  # Strong volume

        df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})

        result = earnings_scanner.evaluate_candidate(
            ticker="SHALLOW",
            ohlcv_df=df,
            earnings_date="2026-11-04",
            revenue_growth_yoy=0.10,
        )

        assert result.is_qualified is False
        assert "Drop too shallow" in result.rejection_reason

    def test_rejected_when_volume_surge_insufficient(self, earnings_scanner):
        """Acute drop (-12%) without institutional volume (RVOL = 1.1x < 2.0x) is rejected."""
        n_days = 35
        closes = [100.0] * (n_days - 1) + [88.0]  # -12% drop
        volumes = [100_000] * (n_days - 1) + [110_000]  # RVOL = 1.1x

        df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})

        result = earnings_scanner.evaluate_candidate(
            ticker="LOWVOL",
            ohlcv_df=df,
            earnings_date="2026-11-04",
            revenue_growth_yoy=0.05,
        )

        assert result.is_qualified is False
        assert "Insufficient volume surge" in result.rejection_reason

    def test_rejected_when_revenue_integrity_fails(self, earnings_scanner, sample_wreck_df):
        """Acute drop with RVOL surge but collapsing revenues (-25%) is rejected as fundamental decay."""
        result = earnings_scanner.evaluate_candidate(
            ticker="COLLAPSE",
            ohlcv_df=sample_wreck_df,
            earnings_date="2026-11-04",
            revenue_growth_yoy=-0.25,  # Top-line cratered 25%
        )

        assert result.revenue_integrity_passed is False
        assert result.is_qualified is False
        assert "Revenue integrity failed" in result.rejection_reason
