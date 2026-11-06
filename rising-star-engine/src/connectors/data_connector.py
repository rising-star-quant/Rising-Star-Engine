"""
Rising Star Trend Engine v2.0 - Data Connector
Robust market data ingestion layer using yfinance and requests with exponential backoff.
Validates payloads using Pydantic models.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import logging
import random
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

from src.config import Settings, get_settings

logger = logging.getLogger("rising_star.connector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- Domain Exceptions ---
class DataConnectorError(Exception):
    """Base exception for data connector operations."""
    pass


class RateLimitError(DataConnectorError):
    """Raised when external market data API returns HTTP 429 / rate limit."""
    pass


class TickerNotFoundError(DataConnectorError):
    """Raised when ticker cannot be found or has no trading data."""
    pass


# --- Pydantic Data Models ---

class OHLCVBar(BaseModel):
    """Single trading bar representing daily/intraday price and volume."""
    timestamp: datetime = Field(..., description="Date or timestamp of the trading bar")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Session highest price")
    low: float = Field(..., description="Session lowest price")
    close: float = Field(..., description="Session closing price")
    volume: int = Field(..., description="Traded share volume")
    adj_close: Optional[float] = Field(default=None, description="Dividend and split adjusted close")

    @field_validator("volume", mode="before")
    @classmethod
    def parse_volume(cls, v: Any) -> int:
        if v is None or pd.isna(v):
            return 0
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0

    @field_validator("open", "high", "low", "close", "adj_close", mode="before")
    @classmethod
    def parse_floats(cls, v: Any) -> Optional[float]:
        if v is None or pd.isna(v):
            return None
        return round(float(v), 4)


class CompanyProfile(BaseModel):
    """Company fundamentals, float, capitalization, and classification."""
    ticker: str = Field(..., description="Uppercase equity symbol (e.g., AAPL)")
    name: str = Field(default="Unknown", description="Legal or registered company name")
    sector: Optional[str] = Field(default="Unknown", description="GICS sector or broad classification")
    industry: Optional[str] = Field(default="Unknown", description="Specific industry classification")
    country: Optional[str] = Field(default="US", description="Country of incorporation/headquarters")
    market_cap: Optional[float] = Field(default=None, description="Total market capitalization in USD")
    float_shares: Optional[float] = Field(default=None, description="Publicly floating shares")
    shares_outstanding: Optional[float] = Field(default=None, description="Total issued and outstanding shares")
    current_price: Optional[float] = Field(default=None, description="Most recent traded market price")
    fifty_two_week_high: Optional[float] = Field(default=None, description="52-week trailing price peak")
    fifty_two_week_low: Optional[float] = Field(default=None, description="52-week trailing price trough")
    beta: Optional[float] = Field(default=None, description="5-year monthly beta relative to market index")
    avg_volume_10d: Optional[int] = Field(default=None, description="10-day average daily share volume")
    avg_volume_30d: Optional[int] = Field(default=None, description="30-day average daily share volume")

    @field_validator("ticker")
    @classmethod
    def sanitize_ticker(cls, v: str) -> str:
        return v.strip().upper()


class MarketDataPayload(BaseModel):
    """
    Complete ingestion packet for a ticker containing verified profile and OHLCV history.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    profile: CompanyProfile = Field(..., description="Company fundamental profile")
    bars: List[OHLCVBar] = Field(default_factory=list, description="Ordered chronological OHLCV bars")
    period: str = Field(default="1y", description="Requested history duration")
    interval: str = Field(default="1d", description="Granularity of bars")
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC ingestion timestamp"
    )
    is_valid: bool = Field(default=True, description="True if payload successfully ingested and parsed")
    error_message: Optional[str] = Field(default=None, description="Error diagnostics if ingestion failed")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the list of OHLCVBar objects into a standardized pandas DataFrame
        indexed by timestamp, suitable for quantitative indicators and technical analysis.
        """
        if not self.bars:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Adj Close"])

        data = [
            {
                "Date": bar.timestamp,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
                "Adj Close": bar.adj_close if bar.adj_close is not None else bar.close,
            }
            for bar in self.bars
        ]
        df = pd.DataFrame(data)
        df = df.set_index("Date").sort_index()
        return df

    @property
    def latest_close(self) -> Optional[float]:
        """Returns the most recent closing price from bars or profile."""
        if self.bars:
            return self.bars[-1].close
        return self.profile.current_price

    @property
    def average_volume_recent(self) -> int:
        """Computes average volume from the latest 20 trading sessions."""
        if not self.bars:
            return self.profile.avg_volume_30d or 0
        recent = self.bars[-20:]
        volumes = [b.volume for b in recent if b.volume > 0]
        return int(sum(volumes) / len(volumes)) if volumes else 0


# --- Data Ingestion Connector ---

class YFinanceDataConnector:
    """
    Production-grade market data connector.
    Wraps yfinance with resilient session configuration, custom retry adapters,
    exponential backoff with jitter for HTTP 429 rate limits, and Pydantic validation.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.session = self._build_resilient_session()

    def _build_resilient_session(self) -> requests.Session:
        """
        Configures a requests.Session with connection pooling, timeouts, and standard retries.
        """
        session = requests.Session()
        retries = Retry(
            total=self.settings.max_retries,
            backoff_factor=self.settings.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=20,
            pool_maxsize=50
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        })
        return session

    def _execute_with_backoff(self, operation_name: str, func, *args, **kwargs) -> Any:
        """
        Executes a callable with custom exponential backoff, rate limit (429) detection,
        and random jitter.
        """
        attempt = 0
        max_retries = self.settings.max_retries

        while attempt < max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                error_str = str(exc).lower()
                is_rate_limit = "429" in error_str or "too many requests" in error_str or "rate limit" in error_str

                if attempt >= max_retries:
                    logger.error(
                        f"Failed {operation_name} after {attempt} attempts: {exc}"
                    )
                    if is_rate_limit:
                        raise RateLimitError(f"Rate limit exceeded during {operation_name}: {exc}") from exc
                    raise DataConnectorError(f"{operation_name} failed: {exc}") from exc

                # Calculate exponential delay with randomized jitter to prevent thundering herd
                base_delay = (self.settings.backoff_factor ** attempt) + random.uniform(0.1, 0.6)
                if is_rate_limit:
                    base_delay += self.settings.rate_limit_pause_seconds * attempt
                    logger.warning(
                        f"Rate limit encountered on {operation_name}. Backing off {base_delay:.2f}s (Attempt {attempt}/{max_retries})"
                    )
                else:
                    logger.warning(
                        f"Error in {operation_name}: {exc}. Retrying in {base_delay:.2f}s (Attempt {attempt}/{max_retries})"
                    )

                time.sleep(base_delay)

    def fetch_profile(self, ticker: str, ticker_obj: Optional[yf.Ticker] = None) -> CompanyProfile:
        """
        Fetches fundamental data, market capitalization, share float, and sector info for a ticker.
        """
        clean_ticker = ticker.strip().upper()
        t = ticker_obj or yf.Ticker(clean_ticker, session=self.session)

        def _get_info():
            info = {}
            try:
                # yfinance fast_info provides low-latency key metrics
                fast_info = getattr(t, "fast_info", None)
                if fast_info:
                    info["market_cap"] = getattr(fast_info, "market_cap", None)
                    info["current_price"] = getattr(fast_info, "last_price", None)
                    info["fifty_two_week_high"] = getattr(fast_info, "year_high", None)
                    info["fifty_two_week_low"] = getattr(fast_info, "year_low", None)
                    info["shares_outstanding"] = getattr(fast_info, "shares", None)
            except Exception as e:
                logger.debug(f"Could not load fast_info for {clean_ticker}: {e}")

            try:
                # Full info dict provides sector, industry, floatShares, beta
                full_info = t.info
                if isinstance(full_info, dict) and full_info:
                    info.setdefault("market_cap", full_info.get("marketCap"))
                    info.setdefault("current_price", full_info.get("currentPrice") or full_info.get("regularMarketPrice"))
                    info.setdefault("fifty_two_week_high", full_info.get("fiftyTwoWeekHigh"))
                    info.setdefault("fifty_two_week_low", full_info.get("fiftyTwoWeekLow"))
                    info.setdefault("shares_outstanding", full_info.get("sharesOutstanding"))
                    info["name"] = full_info.get("shortName") or full_info.get("longName") or clean_ticker
                    info["sector"] = full_info.get("sector") or "Unknown"
                    info["industry"] = full_info.get("industry") or "Unknown"
                    info["country"] = full_info.get("country") or "US"
                    info["float_shares"] = full_info.get("floatShares")
                    info["beta"] = full_info.get("beta")
                    info["avg_volume_10d"] = full_info.get("averageVolume10days")
                    info["avg_volume_30d"] = full_info.get("averageVolume")
            except Exception as e:
                logger.debug(f"Full info lookup failed for {clean_ticker}: {e}")

            return info

        raw_info = self._execute_with_backoff(f"fetch_profile({clean_ticker})", _get_info)

        return CompanyProfile(
            ticker=clean_ticker,
            name=raw_info.get("name") or clean_ticker,
            sector=raw_info.get("sector") or "Unknown",
            industry=raw_info.get("industry") or "Unknown",
            country=raw_info.get("country") or "US",
            market_cap=raw_info.get("market_cap"),
            float_shares=raw_info.get("float_shares"),
            shares_outstanding=raw_info.get("shares_outstanding"),
            current_price=raw_info.get("current_price"),
            fifty_two_week_high=raw_info.get("fifty_two_week_high"),
            fifty_two_week_low=raw_info.get("fifty_two_week_low"),
            beta=raw_info.get("beta"),
            avg_volume_10d=raw_info.get("avg_volume_10d"),
            avg_volume_30d=raw_info.get("avg_volume_30d"),
        )

    def fetch_historical_ohlcv(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        ticker_obj: Optional[yf.Ticker] = None
    ) -> List[OHLCVBar]:
        """
        Fetches chronological OHLCV bars for the specified period and interval.
        """
        clean_ticker = ticker.strip().upper()
        t = ticker_obj or yf.Ticker(clean_ticker, session=self.session)

        def _get_history() -> pd.DataFrame:
            df = t.history(period=period, interval=interval, auto_adjust=False)
            return df

        df = self._execute_with_backoff(f"fetch_history({clean_ticker})", _get_history)

        if df is None or df.empty:
            raise TickerNotFoundError(f"No OHLCV history returned for ticker '{clean_ticker}' (period={period})")

        bars: List[OHLCVBar] = []
        for idx, row in df.iterrows():
            # Index can be Timestamp or DatetimeIndex
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            try:
                bar = OHLCVBar(
                    timestamp=ts,
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                    adj_close=row.get("Adj Close", row["Close"]),
                )
                bars.append(bar)
            except Exception as e:
                logger.warning(f"Error parsing bar for {clean_ticker} at {ts}: {e}")
                continue

        if not bars:
            raise TickerNotFoundError(f"All history bars failed validation for ticker '{clean_ticker}'")

        return bars

    def fetch_ticker_payload(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> MarketDataPayload:
        """
        Fetches both company profile and chronological OHLCV history, returning
        a single validated MarketDataPayload.
        """
        clean_ticker = ticker.strip().upper()
        t = yf.Ticker(clean_ticker, session=self.session)

        try:
            profile = self.fetch_profile(clean_ticker, ticker_obj=t)
            bars = self.fetch_historical_ohlcv(clean_ticker, period=period, interval=interval, ticker_obj=t)

            # Reconcile latest price if missing from profile
            if profile.current_price is None and bars:
                profile.current_price = bars[-1].close

            return MarketDataPayload(
                ticker=clean_ticker,
                profile=profile,
                bars=bars,
                period=period,
                interval=interval,
                is_valid=True
            )
        except Exception as exc:
            logger.error(f"Failed to generate MarketDataPayload for {clean_ticker}: {exc}")
            # Return invalid payload packet with error diagnostics for non-fatal batch operations
            return MarketDataPayload(
                ticker=clean_ticker,
                profile=CompanyProfile(ticker=clean_ticker, name=clean_ticker),
                bars=[],
                period=period,
                interval=interval,
                is_valid=False,
                error_message=str(exc)
            )

    def fetch_batch_payloads(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 4
    ) -> Dict[str, MarketDataPayload]:
        """
        Concurrently fetches market data payloads for a list of tickers.
        Uses a worker threadpool with error containment per ticker.
        """
        results: Dict[str, MarketDataPayload] = {}
        unique_tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))

        logger.info(f"Starting batch fetch for {len(unique_tickers)} tickers (workers={max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_ticker_payload, ticker, period, interval): ticker
                for ticker in unique_tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    payload = future.result()
                    results[ticker] = payload
                except Exception as exc:
                    logger.error(f"Unhandled threadpool exception for ticker {ticker}: {exc}")
                    results[ticker] = MarketDataPayload(
                        ticker=ticker,
                        profile=CompanyProfile(ticker=ticker, name=ticker),
                        bars=[],
                        is_valid=False,
                        error_message=str(exc)
                    )

        valid_count = sum(1 for p in results.values() if p.is_valid)
        logger.info(f"Batch fetch completed: {valid_count}/{len(unique_tickers)} tickers valid")
        return results


def get_data_connector() -> YFinanceDataConnector:
    """Factory helper returning initialized data connector."""
    return YFinanceDataConnector()
