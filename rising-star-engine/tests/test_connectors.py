"""
Pytest suite for data_connector.py (Module 01)
Tests data ingestion, Pydantic validation, exponential backoff, rate limits, and failure cases.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Settings
from src.connectors.data_connector import (
    CompanyProfile,
    DataConnectorError,
    MarketDataPayload,
    OHLCVBar,
    RateLimitError,
    TickerNotFoundError,
    YFinanceDataConnector,
)


@pytest.fixture
def mock_settings():
    """Provides test-specific configuration with fast backoff."""
    return Settings(
        env="testing",
        debug=True,
        max_retries=3,
        backoff_factor=0.01,  # Sub-second delay for swift test runs
        rate_limit_pause_seconds=0.02,
        request_timeout=5,
    )


@pytest.fixture
def connector(mock_settings):
    """Provides an initialized YFinanceDataConnector with test settings."""
    return YFinanceDataConnector(settings=mock_settings)


@pytest.fixture
def sample_ohlcv_dataframe():
    """Generates synthetic 5-day OHLCV pandas DataFrame matching yfinance output."""
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D", tz="UTC")
    data = {
        "Open": [150.0, 152.0, 151.5, 155.0, 158.0],
        "High": [153.5, 154.0, 156.0, 159.0, 161.0],
        "Low": [149.0, 151.0, 150.5, 154.0, 157.5],
        "Close": [152.5, 151.8, 155.2, 158.1, 160.5],
        "Volume": [1_200_000, 950_000, 1_500_000, 1_800_000, 2_100_000],
        "Adj Close": [152.5, 151.8, 155.2, 158.1, 160.5],
    }
    return pd.DataFrame(data, index=dates)


class TestPydanticModels:
    """Verifies strict Pydantic model validation and helper methods."""

    def test_ohlcv_bar_valid(self):
        now = datetime.now(timezone.utc)
        bar = OHLCVBar(
            timestamp=now,
            open=100.5555,
            high=105.10,
            low=99.20,
            close=104.8889,
            volume=50000,
            adj_close=104.89,
        )
        assert bar.open == 100.5555
        assert bar.close == 104.8889
        assert bar.volume == 50000

    def test_ohlcv_bar_coercion_and_none_volume(self):
        now = datetime.now(timezone.utc)
        bar = OHLCVBar(
            timestamp=now,
            open=10.0,
            high=12.0,
            low=9.0,
            close=11.5,
            volume="12345",  # string should coerce
            adj_close=None,
        )
        assert bar.volume == 12345
        assert bar.adj_close is None

    def test_company_profile_sanitization(self):
        profile = CompanyProfile(
            ticker="  aapl  ",
            name="Apple Inc.",
            sector="Technology",
            market_cap=2_800_000_000_000.0,
            current_price=185.50,
        )
        assert profile.ticker == "AAPL"
        assert profile.name == "Apple Inc."
        assert profile.market_cap == 2.8e12

    def test_market_data_payload_to_dataframe(self, sample_ohlcv_dataframe):
        now = datetime.now(timezone.utc)
        profile = CompanyProfile(ticker="NVDA", name="NVIDIA Corp", current_price=120.0)
        bars = [
            OHLCVBar(
                timestamp=now,
                open=100.0,
                high=105.0,
                low=98.0,
                close=102.0,
                volume=1_000_000,
                adj_close=102.0,
            )
        ]
        payload = MarketDataPayload(ticker="NVDA", profile=profile, bars=bars)
        df = payload.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Close" in df.columns
        assert df.iloc[0]["Close"] == 102.0
        assert payload.latest_close == 102.0


class TestYFinanceDataConnector:
    """Tests ingestion workflows, network mocks, retries, and errors."""

    @patch("yfinance.Ticker")
    def test_fetch_profile_success(self, mock_ticker_cls, connector):
        # Configure mock yf.Ticker
        mock_instance = MagicMock()
        mock_instance.fast_info.market_cap = 1_500_000_000.0
        mock_instance.fast_info.last_price = 45.20
        mock_instance.fast_info.year_high = 52.00
        mock_instance.fast_info.year_low = 30.00
        mock_instance.fast_info.shares = 33_000_000
        mock_instance.info = {
            "shortName": "Super Micro Computer",
            "sector": "Technology",
            "industry": "Computer Hardware",
            "floatShares": 28_000_000,
            "averageVolume": 4_500_000,
            "beta": 1.45,
        }
        mock_ticker_cls.return_value = mock_instance

        profile = connector.fetch_profile("SMCI")

        assert profile.ticker == "SMCI"
        assert profile.name == "Super Micro Computer"
        assert profile.sector == "Technology"
        assert profile.market_cap == 1_500_000_000.0
        assert profile.current_price == 45.20
        assert profile.beta == 1.45

    @patch("yfinance.Ticker")
    def test_fetch_historical_ohlcv_success(self, mock_ticker_cls, connector, sample_ohlcv_dataframe):
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_ohlcv_dataframe
        mock_ticker_cls.return_value = mock_instance

        bars = connector.fetch_historical_ohlcv("AMD", period="1y", interval="1d")

        assert len(bars) == 5
        assert bars[0].close == 152.5
        assert bars[-1].close == 160.5
        assert bars[-1].volume == 2_100_000

    @patch("yfinance.Ticker")
    def test_fetch_ticker_not_found_empty_dataframe(self, mock_ticker_cls, connector):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()  # Empty data
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(TickerNotFoundError) as exc_info:
            connector.fetch_historical_ohlcv("INVALID_TICKER_XYZ")
        assert "No OHLCV history returned" in str(exc_info.value)

    @patch("yfinance.Ticker")
    def test_rate_limit_backoff_and_failure(self, mock_ticker_cls, connector):
        mock_instance = MagicMock()
        # Simulate 429 Too Many Requests exception
        mock_instance.history.side_effect = Exception("429 Client Error: Too Many Requests")
        mock_ticker_cls.return_value = mock_instance

        with pytest.raises(RateLimitError):
            connector.fetch_historical_ohlcv("RATE_LIMITED_CO")

    @patch("yfinance.Ticker")
    def test_transient_error_retry_success(self, mock_ticker_cls, connector, sample_ohlcv_dataframe):
        mock_instance = MagicMock()
        # Fail first time with connection error, then succeed
        mock_instance.history.side_effect = [
            ConnectionResetError("Connection dropped by peer"),
            sample_ohlcv_dataframe,
        ]
        mock_ticker_cls.return_value = mock_instance

        bars = connector.fetch_historical_ohlcv("RECOVERING_CO")
        assert len(bars) == 5
        assert mock_instance.history.call_count == 2

    @patch("yfinance.Ticker")
    def test_fetch_ticker_payload_full_pipeline(self, mock_ticker_cls, connector, sample_ohlcv_dataframe):
        mock_instance = MagicMock()
        mock_instance.fast_info.market_cap = 50_000_000_000.0
        mock_instance.fast_info.last_price = 220.0
        mock_instance.info = {
            "shortName": "Tesla Inc",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
        }
        mock_instance.history.return_value = sample_ohlcv_dataframe
        mock_ticker_cls.return_value = mock_instance

        payload = connector.fetch_ticker_payload("TSLA")

        assert payload.is_valid is True
        assert payload.ticker == "TSLA"
        assert payload.profile.sector == "Consumer Cyclical"
        assert len(payload.bars) == 5
        assert payload.latest_close == 160.5

    @patch("yfinance.Ticker")
    def test_batch_payload_resilience(self, mock_ticker_cls, connector, sample_ohlcv_dataframe):
        def side_effect_ticker(symbol, **kwargs):
            mock = MagicMock()
            if symbol == "BAD":
                mock.history.side_effect = Exception("Delisted or invalid")
                mock.info = {}
            else:
                mock.fast_info.market_cap = 10_000_000_000.0
                mock.fast_info.last_price = 100.0
                mock.info = {"shortName": f"Company {symbol}", "sector": "Technology"}
                mock.history.return_value = sample_ohlcv_dataframe
            return mock

        mock_ticker_cls.side_effect = side_effect_ticker

        results = connector.fetch_batch_payloads(["GOOD1", "BAD", "GOOD2"], max_workers=2)

        assert len(results) == 3
        assert results["GOOD1"].is_valid is True
        assert results["GOOD2"].is_valid is True
        assert results["BAD"].is_valid is False
        assert results["BAD"].error_message is not None
