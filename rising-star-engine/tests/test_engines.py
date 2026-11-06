"""
Pytest suite for universe_scanner.py (Module 01)
Tests index constituent extraction, liquidity filtering, sector whitelists/blacklists, and edge cases.
"""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.connectors.data_connector import CompanyProfile
from src.engines.universe_scanner import (
    UniverseFilterCriteria,
    UniverseScanResult,
    UniverseScanner,
)


@pytest.fixture
def mock_scanner():
    """Provides UniverseScanner with a mocked data connector."""
    mock_connector = MagicMock()
    settings = Settings(env="testing", min_market_cap=500_000_000.0, min_avg_volume_30d=250_000)
    return UniverseScanner(connector=mock_connector, settings=settings)


@pytest.fixture
def sample_profiles():
    """Generates synthetic candidate profiles with diverse characteristics."""
    return {
        "QUALIFIED_TECH": CompanyProfile(
            ticker="QUALIFIED_TECH",
            name="Alpha Tech Inc",
            sector="Technology",
            industry="Semiconductors",
            market_cap=2_500_000_000.0,  # $2.5B
            current_price=65.0,
            avg_volume_30d=850_000,
        ),
        "LOW_VOLUME": CompanyProfile(
            ticker="LOW_VOLUME",
            name="Illiquid Corp",
            sector="Technology",
            market_cap=1_000_000_000.0,
            current_price=25.0,
            avg_volume_30d=50_000,  # Below 250k
        ),
        "MICRO_CAP": CompanyProfile(
            ticker="MICRO_CAP",
            name="Tiny Bio",
            sector="Healthcare",
            market_cap=50_000_000.0,  # $50M (Below $500M)
            current_price=8.5,
            avg_volume_30d=600_000,
        ),
        "PENNY_STOCK": CompanyProfile(
            ticker="PENNY_STOCK",
            name="Penny Inc",
            sector="Consumer Cyclical",
            market_cap=800_000_000.0,
            current_price=1.20,  # Below $5
            avg_volume_30d=2_000_000,
        ),
        "FINANCIAL_SECTOR": CompanyProfile(
            ticker="FINANCIAL_SECTOR",
            name="First National",
            sector="Financial Services",
            market_cap=5_000_000_000.0,
            current_price=50.0,
            avg_volume_30d=1_500_000,
        ),
    }


class TestUniverseScanner:
    """Verifies constituent resolution and filter logic."""

    def test_get_index_constituents(self, mock_scanner):
        sp400 = mock_scanner.get_index_constituents("SP400")
        assert len(sp400) > 0
        assert "DECK" in sp400 or "MPWR" in sp400

        r2000 = mock_scanner.get_index_constituents("RUSSELL2000")
        assert len(r2000) > 0
        assert "CELH" in r2000 or "APP" in r2000

    def test_composite_universe_deduplication(self, mock_scanner):
        mock_scanner.register_index_constituents("INDEX_A", ["AAPL", "MSFT", "NVDA"])
        mock_scanner.register_index_constituents("INDEX_B", ["NVDA", "AMZN", "AAPL"])

        composite = mock_scanner.get_composite_universe(["INDEX_A", "INDEX_B"])
        assert len(composite) == 4
        assert composite == ["AAPL", "AMZN", "MSFT", "NVDA"]

    def test_filter_universe_criteria_pipeline(self, mock_scanner, sample_profiles):
        criteria = UniverseFilterCriteria(
            min_market_cap=500_000_000.0,
            min_avg_volume=250_000,
            min_price=5.0,
            excluded_sectors=["Financial Services"],
        )

        tickers = list(sample_profiles.keys())
        result = mock_scanner.filter_universe(
            tickers=tickers,
            criteria=criteria,
            preloaded_profiles=sample_profiles
        )

        assert isinstance(result, UniverseScanResult)
        assert result.total_candidates == 5
        assert result.passed_tickers == ["QUALIFIED_TECH"]
        assert len(result.profiles) == 1
        assert "QUALIFIED_TECH" in result.profiles

        # Rejection breakdown checks
        assert result.rejection_reasons.get("volume_below_min") == 1
        assert result.rejection_reasons.get("market_cap_below_min") == 1
        assert result.rejection_reasons.get("price_below_min") == 1
        assert result.rejection_reasons.get("sector_excluded") == 1

    def test_allowed_sectors_whitelist(self, mock_scanner, sample_profiles):
        # Whitelist ONLY Healthcare
        criteria = UniverseFilterCriteria(
            min_market_cap=10_000_000.0,  # Lower to include Tiny Bio
            min_avg_volume=100_000,
            min_price=1.0,
            allowed_sectors=["Healthcare"],
        )

        result = mock_scanner.filter_universe(
            tickers=list(sample_profiles.keys()),
            criteria=criteria,
            preloaded_profiles=sample_profiles
        )

        assert result.passed_tickers == ["MICRO_CAP"]
        assert result.profiles["MICRO_CAP"].sector == "Healthcare"

    def test_scan_and_generate_payload_prd_contract(self, mock_scanner, sample_profiles):
        mock_scanner.register_index_constituents("S&P_400", ["QUALIFIED_TECH", "LOW_VOLUME"])
        mock_scanner.register_index_constituents("RUSSELL_1000", ["QUALIFIED_TECH", "FINANCIAL_SECTOR"])

        payload = mock_scanner.scan_and_generate_payload(
            indices=["S&P_400", "RUSSELL_1000"],
            criteria=UniverseFilterCriteria(
                min_market_cap=300_000_000.0,
                min_adv_dollar=2_000_000.0,
                excluded_sectors=["Financial Services"]
            ),
            preloaded_profiles=sample_profiles
        )

        assert payload.total_symbols_scanned == 4
        assert payload.qualified_symbols_count == 1
        assert len(payload.constituents) == 1
        assert payload.constituents[0].ticker == "QUALIFIED_TECH"
        assert payload.constituents[0].sector == "Technology"
        assert payload.constituents[0].market_cap == 2_500_000_000.0
        assert payload.constituents[0].avg_daily_volume == 850_000
        assert payload.execution_timestamp is not None


# =====================================================================
# Tests for Module 02: IndicatorEngine & ValueDisconnectEngine
# =====================================================================

from src.engines.indicator_engine import (
    IndicatorEngine,
    compute_atr,
    compute_distance_from_high,
    check_reversal_stabilization,
)
from src.engines.value_dip_engine import (
    CompanyFinancials,
    ValueDipCandidate,
    ValueDipFilterCriteria,
    ValueDipPayload,
    ValueDisconnectEngine,
)


@pytest.fixture
def sample_ohlcv_history():
    """Generates a 30-day synthetic OHLCV DataFrame with high of 100.0 and recent pullback to 78.0 (-22%)."""
    import numpy as np
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D", tz="UTC")
    # Simulate run up to 100 on day 10, then pullback to 78 over days 11-28, then stabilization
    prices = [80.0 + i * 2 for i in range(11)]  # 80 to 100
    prices += [100.0 - (i + 1) * 1.5 for i in range(15)]  # pull back towards 77.5
    prices += [78.0, 78.2, 78.5, 78.6]  # stabilize near 78.6 (-21.4% from 100)

    # Trim or match length to 30
    prices = prices[:30]
    highs = [p + 1.5 for p in prices]
    lows = [p - 1.5 for p in prices]
    opens = [p - 0.5 for p in prices]
    closes = prices
    volumes = [1_000_000] * 30

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


class TestIndicatorEngine:
    """Verifies technical processing, ATR math, distance from highs, and momentum."""

    def test_distance_from_high_calculation(self):
        # Current: 42.50, 52W High: 54.49 -> (42.50 - 54.49) / 54.49 = -0.22
        dist = compute_distance_from_high(42.50, 54.49)
        assert dist == -0.22

        # At the high
        assert compute_distance_from_high(100.0, 100.0) == 0.0

        # Edge case: zero or negative high
        assert compute_distance_from_high(50.0, 0.0) == 0.0

    def test_compute_atr_accuracy(self, sample_ohlcv_history):
        atr_series = compute_atr(sample_ohlcv_history, period=14)
        assert isinstance(atr_series, pd.Series)
        assert len(atr_series) == 30

        # Each bar has High - Low = 3.0, with modest inter-bar gaps
        latest_atr = atr_series.iloc[-1]
        assert latest_atr > 0
        assert 2.0 <= latest_atr <= 4.0

    def test_check_reversal_stabilization(self, sample_ohlcv_history):
        # sample_ohlcv_history stabilizes over the last 4 days (78.0, 78.2, 78.5, 78.6)
        stabilized = check_reversal_stabilization(sample_ohlcv_history)
        assert stabilized is True

    def test_calculate_indicators_payload(self, sample_ohlcv_history):
        engine = IndicatorEngine(atr_period=14)
        indicators = engine.calculate_indicators(
            ticker="MSFT",
            df=sample_ohlcv_history,
            high_52w=100.0,
        )

        assert indicators.ticker == "MSFT"
        assert indicators.fifty_two_week_high == 100.0
        assert indicators.current_price == round(sample_ohlcv_history["Close"].iloc[-1], 4)
        assert -0.25 <= indicators.distance_from_52w_high_pct <= -0.15
        assert indicators.atr_14 > 0
        assert indicators.momentum_stabilizing is True


class TestValueDisconnectEngine:
    """
    Verifies Module 02 Value Dip & Fundamental Disconnect Guard.
    Tests boundary conditions where technical dips are met but fundamental guards fail.
    """

    @pytest.fixture
    def engine(self):
        mock_connector = MagicMock()
        return ValueDisconnectEngine(connector=mock_connector)

    @pytest.fixture
    def qualified_financials(self):
        """Passes all operational criteria: Revenue Growth > 0%, FCF > 0, D/E < 1.5."""
        return CompanyFinancials(
            revenue_growth_qoq=0.08,  # +8%
            free_cash_flow=12_500_000.0,  # +$12.5M
            debt_to_equity=0.65,  # 0.65 < 1.5
            operating_cash_flow=18_000_000.0,
        )

    def test_technical_retracement_gate_pass(self, engine):
        criteria = ValueDipFilterCriteria(
            min_drop_from_52w_high=-0.30,
            max_drop_from_52w_high=-0.15,
        )
        # -22% drop is squarely between -30% and -15%
        passed, reason = engine.evaluate_technical_gate(-0.22, criteria)
        assert passed is True
        assert reason is None

    def test_technical_retracement_gate_too_shallow(self, engine):
        criteria = ValueDipFilterCriteria(
            min_drop_from_52w_high=-0.30,
            max_drop_from_52w_high=-0.15,
        )
        # -8% drop is too shallow (has not retraced 15%)
        passed, reason = engine.evaluate_technical_gate(-0.08, criteria)
        assert passed is False
        assert "insufficient_dip" in reason

    def test_technical_retracement_gate_too_deep_structural_break(self, engine):
        criteria = ValueDipFilterCriteria(
            min_drop_from_52w_high=-0.30,
            max_drop_from_52w_high=-0.15,
        )
        # -45% collapse exceeds 30% ceiling (structural impairment)
        passed, reason = engine.evaluate_technical_gate(-0.45, criteria)
        assert passed is False
        assert "excessive_decline" in reason

    def test_fundamental_guard_qualified_pass(self, engine, qualified_financials):
        criteria = ValueDipFilterCriteria()
        passed, reasons = engine.evaluate_fundamental_guard(qualified_financials, criteria)
        assert passed is True
        assert len(reasons) == 0

    def test_fundamental_guard_fails_on_negative_free_cash_flow(self, engine):
        """Boundary condition: Technical dip met, but Free Cash Flow <= 0 -> REJECTED."""
        insolvent_financials = CompanyFinancials(
            revenue_growth_qoq=0.15,  # Strong revenue growth
            free_cash_flow=-5_000_000.0,  # Negative FCF (Burning cash)
            debt_to_equity=0.50,
        )
        criteria = ValueDipFilterCriteria()
        passed, reasons = engine.evaluate_fundamental_guard(insolvent_financials, criteria)

        assert passed is False
        assert any("negative_free_cash_flow" in r for r in reasons)

    def test_fundamental_guard_fails_on_negative_revenue_growth(self, engine):
        """Boundary condition: Positive FCF, but Revenue Growth <= 0% -> REJECTED."""
        contracting_financials = CompanyFinancials(
            revenue_growth_qoq=-0.04,  # -4% revenue decline
            free_cash_flow=10_000_000.0,
            debt_to_equity=0.80,
        )
        criteria = ValueDipFilterCriteria()
        passed, reasons = engine.evaluate_fundamental_guard(contracting_financials, criteria)

        assert passed is False
        assert any("negative_or_flat_revenue_growth" in r for r in reasons)

    def test_fundamental_guard_fails_on_excessive_debt_to_equity(self, engine):
        """Boundary condition: Overleveraged entity D/E >= 1.5 -> REJECTED."""
        leveraged_financials = CompanyFinancials(
            revenue_growth_qoq=0.10,
            free_cash_flow=20_000_000.0,
            debt_to_equity=2.40,  # 2.40 >= 1.5 limit
        )
        criteria = ValueDipFilterCriteria()
        passed, reasons = engine.evaluate_fundamental_guard(leveraged_financials, criteria)

        assert passed is False
        assert any("excessive_leverage_debt_to_equity" in r for r in reasons)

    def test_evaluate_candidate_end_to_end_prd_contract(self, engine, sample_ohlcv_history, qualified_financials):
        """
        Tests full single-ticker candidate evaluation and verifies PRD output contract schema.
        """
        candidate = engine.evaluate_candidate(
            ticker="EXAMPLE",
            df_history=sample_ohlcv_history,
            financials=qualified_financials,
            high_52w=100.0,
        )

        assert isinstance(candidate, ValueDipCandidate)
        assert candidate.ticker == "EXAMPLE"
        assert candidate.current_price > 0
        assert -0.25 <= candidate.distance_from_52w_high_pct <= -0.15
        assert candidate.fundamental_guard_passed is True
        assert candidate.technical_gate_passed is True
        assert candidate.financials.revenue_growth_qoq == 0.08
        assert candidate.financials.free_cash_flow == 12_500_000.0
        assert candidate.financials.debt_to_equity == 0.65
        assert candidate.evaluated_at is not None

        # Verify JSON serialization compliance with PRD Output Interface
        json_dict = candidate.model_dump()
        assert "ticker" in json_dict
        assert "current_price" in json_dict
        assert "distance_from_52w_high_pct" in json_dict
        assert "fundamental_guard_passed" in json_dict
        assert "financials" in json_dict
        assert "evaluated_at" in json_dict

    def test_screen_candidates_batch_filtration(self, engine, sample_ohlcv_history, qualified_financials):
        """
        Tests screening a multi-ticker universe where only the qualified candidate passes.
        """
        # Failing candidate with negative FCF
        failing_financials = CompanyFinancials(
            revenue_growth_qoq=0.05,
            free_cash_flow=-2_000_000.0,
            debt_to_equity=0.40,
        )

        preloaded_histories = {
            "GOOD_DIP": sample_ohlcv_history,
            "BAD_FUNDAMENTALS": sample_ohlcv_history,
        }
        preloaded_financials = {
            "GOOD_DIP": qualified_financials,
            "BAD_FUNDAMENTALS": failing_financials,
        }
        preloaded_highs = {
            "GOOD_DIP": 100.0,
            "BAD_FUNDAMENTALS": 100.0,
        }

        payload = engine.screen_candidates(
            tickers=["GOOD_DIP", "BAD_FUNDAMENTALS"],
            preloaded_histories=preloaded_histories,
            preloaded_financials=preloaded_financials,
            preloaded_highs=preloaded_highs,
        )

        assert isinstance(payload, ValueDipPayload)
        assert payload.total_candidates_evaluated == 2
        assert payload.qualified_candidates_count == 1
        assert len(payload.candidates) == 1
        assert payload.candidates[0].ticker == "GOOD_DIP"
        assert "negative_free_cash_flow" in payload.rejection_breakdown

