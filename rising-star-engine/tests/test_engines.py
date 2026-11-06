"""
Pytest suite for universe_scanner.py (Module 01)
Tests index constituent extraction, liquidity filtering, sector whitelists/blacklists, and edge cases.
"""

from unittest.mock import MagicMock
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
