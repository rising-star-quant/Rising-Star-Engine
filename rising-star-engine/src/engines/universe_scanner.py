"""
Rising Star Trend Engine v2.0 - Universe & Sector Scanner (Module 01)
Responsible for index constituent extraction (S&P 400, Russell 1000, Russell 2000)
and baseline liquidity/sector filtering.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.connectors.data_connector import CompanyProfile, YFinanceDataConnector, get_data_connector

logger = logging.getLogger("rising_star.universe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class UniverseFilterCriteria(BaseModel):
    """
    Parametric filter thresholds for baseline liquidity, price, and sector allocation.
    Adheres to Module 01 PRD: Market Cap >= $300M, ADV >= $2M.
    """
    min_market_cap: float = Field(default=300_000_000.0, description="Minimum market cap in USD (PRD: >= $300M)")
    max_market_cap: Optional[float] = Field(default=None, description="Optional maximum market cap ceiling")
    min_adv_dollar: float = Field(default=2_000_000.0, description="Minimum 30-day Average Daily Dollar Volume (PRD: >= $2M)")
    min_avg_volume: int = Field(default=100_000, description="Minimum 30-day average daily share volume")
    min_price: float = Field(default=5.0, description="Minimum share price to avoid penny stocks")
    max_price: Optional[float] = Field(default=2500.0, description="Optional maximum share price")
    allowed_sectors: Optional[List[str]] = Field(
        default=None,
        description="Whitelist of allowed GICS sectors. If None, all non-excluded sectors are accepted"
    )
    excluded_sectors: Optional[List[str]] = Field(
        default_factory=list,
        description="Blacklist of sectors to filter out (e.g., ['Financial Services', 'Utilities'])"
    )
    allowed_industries: Optional[List[str]] = Field(
        default=None,
        description="Whitelist of allowed industries"
    )
    excluded_industries: Optional[List[str]] = Field(
        default_factory=list,
        description="Blacklist of industries to filter out"
    )


class ConstituentCandidate(BaseModel):
    """
    Atomic constituent entry defined by Module 01 PRD Output Interface.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    index_source: str = Field(..., description="Index origin: S&P_400, RUSSELL_1000, RUSSELL_2000")
    sector: str = Field(default="Unknown", description="GICS sector classification")
    market_cap: float = Field(..., description="Market capitalization in USD")
    avg_daily_volume: int = Field(..., description="30-day Average Daily Traded Volume")
    price: Optional[float] = Field(default=None, description="Latest market price")
    industry: Optional[str] = Field(default=None, description="Detailed industry")


class UniversePayload(BaseModel):
    """
    Exact output data contract specified by Module 01 PRD.
    """
    execution_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC execution timestamp"
    )
    total_symbols_scanned: int = Field(..., description="Total symbols ingested across indices before filtering")
    qualified_symbols_count: int = Field(..., description="Count of constituents passing baseline gates")
    constituents: List[ConstituentCandidate] = Field(
        default_factory=list,
        description="List of qualified constituent candidates"
    )


class UniverseScanResult(BaseModel):
    """
    Diagnostic report of a universe scan operation including passed tickers and rejection breakdown.
    """
    total_candidates: int = Field(..., description="Number of tickers evaluated")
    passed_tickers: List[str] = Field(default_factory=list, description="Tickers meeting all criteria")
    rejected_count: int = Field(default=0, description="Total count of discarded tickers")
    rejection_reasons: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of rejection causes (market_cap, volume, adv_dollar, price, sector, data_error)"
    )
    profiles: Dict[str, CompanyProfile] = Field(
        default_factory=dict,
        description="Map of passed ticker symbols to their CompanyProfile"
    )
    scanned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC scan execution timestamp"
    )

    def summary_table(self) -> pd.DataFrame:
        """Returns a formatted pandas summary of all passed candidate companies."""
        if not self.profiles:
            return pd.DataFrame()
        records = [
            {
                "Ticker": p.ticker,
                "Name": p.name,
                "Sector": p.sector,
                "Industry": p.industry,
                "Price": p.current_price,
                "MarketCap ($M)": round(p.market_cap / 1e6, 2) if p.market_cap else None,
                "Float ($M)": round(p.float_shares / 1e6, 2) if p.float_shares else None,
                "AvgVol30d": p.avg_volume_30d,
                "Beta": p.beta,
            }
            for p in self.profiles.values()
        ]
        return pd.DataFrame(records)


# --- Curated Constituent Seed Lists ---
# High-confidence verified universe seeds representing key components of
# S&P 400 Mid-Cap, Russell 1000, and Russell 2000 indices.
BENCHMARK_CONSTITUENTS: Dict[str, List[str]] = {
    # S&P MidCap 400 (Select high-velocity mid-caps)
    "SP400": [
        "DECK", "SAIA", "LSCC", "MANH", "MPWR", "ON", "MEDP", "EXP", "SSD",
        "WSO", "TREX", "LII", "BLD", "CHE", "WING", "POWI", "HUBB", "FND",
        "GWRE", "NEU", "RBC", "ENSG", "CHDN", "TEX", "ELF", "SNX", "ALGM"
    ],
    # Russell 1000 (Select high-growth & quality large/mid-cap components)
    "RUSSELL1000": [
        "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD",
        "CRM", "PANW", "NOW", "CRWD", "ANET", "UBER", "ABNB", "SNOW",
        "PLTR", "FTNT", "DDOG", "NET", "ZS", "AXON", "SMCI", "ARM", "DECK"
    ],
    # Russell 2000 (Select liquid small-cap growth candidates)
    "RUSSELL2000": [
        "APP", "DUOL", "CELH", "SMCI", "SNDX", "KRYS", "INSP", "HALO",
        "ACLX", "AMPH", "KYMR", "IMVT", "VRT", "MOD", "POWL", "FIX",
        "CRDO", "IOT", "MNDY", "CFLT", "SYM", "ACMR", "HIMS", "S"
    ],
    # S&P 500 benchmark
    "SP500": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
        "AVGO", "JPM", "TSLA", "V", "UNH", "MA", "COST", "HD", "PG", "NFLX"
    ]
}


class UniverseScanner:
    """
    Scans index constituents, manages universe registries, and filters candidates
    against baseline liquidity, price, and sector specifications.
    """

    def __init__(
        self,
        connector: Optional[YFinanceDataConnector] = None,
        settings: Optional[Settings] = None
    ):
        self.settings = settings or get_settings()
        self.connector = connector or get_data_connector()
        self._indices_cache: Dict[str, List[str]] = dict(BENCHMARK_CONSTITUENTS)

    def register_index_constituents(self, index_name: str, tickers: List[str]) -> None:
        """
        Registers or updates the constituent list for an index in memory.
        """
        key = index_name.strip().upper()
        clean_tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
        self._indices_cache[key] = clean_tickers
        logger.info(f"Registered {len(clean_tickers)} constituents for index {key}")

    def get_index_constituents(self, index_name: str) -> List[str]:
        """
        Returns list of tickers for a given index symbol (e.g., 'SP400', 'RUSSELL1000', 'RUSSELL2000').
        """
        key = index_name.strip().upper()
        if key in self._indices_cache:
            return list(self._indices_cache[key])

        # Aliases
        alias_map = {
            "S&P400": "SP400",
            "SPMID": "SP400",
            "IJH": "SP400",
            "R1000": "RUSSELL1000",
            "IWB": "RUSSELL1000",
            "R2000": "RUSSELL2000",
            "IWM": "RUSSELL2000",
            "SPY": "SP500",
        }
        if key in alias_map and alias_map[key] in self._indices_cache:
            return list(self._indices_cache[alias_map[key]])

        logger.warning(f"Index '{index_name}' not pre-cached. Returning empty list.")
        return []

    def get_composite_universe(self, indices: Optional[List[str]] = None) -> List[str]:
        """
        Gathers unique de-duplicated union of constituents across requested index names.
        """
        target_indices = indices or self.settings.default_indices
        union_tickers: Set[str] = set()

        for idx in target_indices:
            constituents = self.get_index_constituents(idx)
            union_tickers.update(constituents)
            logger.info(f"Loaded {len(constituents)} tickers from {idx}")

        sorted_universe = sorted(list(union_tickers))
        logger.info(f"Constructed composite universe of {len(sorted_universe)} unique tickers")
        return sorted_universe

    def evaluate_candidate(
        self,
        profile: CompanyProfile,
        criteria: UniverseFilterCriteria
    ) -> tuple[bool, Optional[str]]:
        """
        Evaluates a single CompanyProfile against liquidity and sector criteria.
        Returns (is_passed, rejection_reason).
        """
        # Price Check
        price = profile.current_price
        if price is None:
            return False, "missing_price_data"
        if price < criteria.min_price:
            return False, f"price_below_min ({price:.2f} < {criteria.min_price})"
        if criteria.max_price is not None and price > criteria.max_price:
            return False, f"price_above_max ({price:.2f} > {criteria.max_price})"

        # Market Cap Check
        market_cap = profile.market_cap
        if market_cap is None:
            # Fallback estimation if shares_outstanding and price present
            if profile.shares_outstanding and price:
                market_cap = profile.shares_outstanding * price
        if market_cap is None:
            return False, "missing_market_cap"
        if market_cap < criteria.min_market_cap:
            return False, f"market_cap_below_min (${market_cap/1e6:.1f}M < ${criteria.min_market_cap/1e6:.1f}M)"
        if criteria.max_market_cap and market_cap > criteria.max_market_cap:
            return False, f"market_cap_above_max (${market_cap/1e6:.1f}M > ${criteria.max_market_cap/1e6:.1f}M)"

        # Liquidity / Average Volume Check (PRD requires ADV >= $2M)
        vol = profile.avg_volume_30d or profile.avg_volume_10d
        if vol is not None and vol < criteria.min_avg_volume:
            return False, f"volume_below_min ({vol:,} < {criteria.min_avg_volume:,})"

        # Average Daily Dollar Volume (ADV = Price * Volume >= $2M)
        if vol is not None and price is not None:
            adv_dollar = vol * price
            if adv_dollar < criteria.min_adv_dollar:
                return False, f"adv_dollar_below_min (${adv_dollar/1e6:.2f}M < ${criteria.min_adv_dollar/1e6:.2f}M)"

        # Sector Blacklist Check
        sector = (profile.sector or "Unknown").strip()
        if criteria.excluded_sectors and sector in criteria.excluded_sectors:
            return False, f"sector_excluded ({sector})"

        # Sector Whitelist Check
        if criteria.allowed_sectors is not None:
            allowed_clean = [s.strip().lower() for s in criteria.allowed_sectors]
            if sector.lower() not in allowed_clean:
                return False, f"sector_not_allowed ({sector})"

        # Industry Checks
        industry = (profile.industry or "Unknown").strip()
        if criteria.excluded_industries and industry in criteria.excluded_industries:
            return False, f"industry_excluded ({industry})"
        if criteria.allowed_industries is not None:
            allowed_ind_clean = [i.strip().lower() for i in criteria.allowed_industries]
            if industry.lower() not in allowed_ind_clean:
                return False, f"industry_not_allowed ({industry})"

        return True, None

    def filter_universe(
        self,
        tickers: List[str],
        criteria: Optional[UniverseFilterCriteria] = None,
        preloaded_profiles: Optional[Dict[str, CompanyProfile]] = None
    ) -> UniverseScanResult:
        """
        Filters a universe of tickers against the provided criteria.
        Uses preloaded profiles if provided, or fetches them via the data connector.
        """
        filter_criteria = criteria or UniverseFilterCriteria(
            min_market_cap=self.settings.min_market_cap,
            max_market_cap=self.settings.max_market_cap,
            min_adv_dollar=self.settings.min_adv_dollar,
            min_avg_volume=self.settings.min_avg_volume_30d,
            min_price=self.settings.min_share_price,
            max_price=self.settings.max_share_price
        )

        unique_tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
        passed_tickers: List[str] = []
        profiles_map: Dict[str, CompanyProfile] = {}
        rejection_reasons: Dict[str, int] = {}

        logger.info(f"Filtering {len(unique_tickers)} tickers against baseline criteria")

        for ticker in unique_tickers:
            # Retrieve or fetch company profile
            if preloaded_profiles and ticker in preloaded_profiles:
                profile = preloaded_profiles[ticker]
            else:
                try:
                    profile = self.connector.fetch_profile(ticker)
                except Exception as e:
                    logger.warning(f"Could not load profile for {ticker}: {e}")
                    rejection_reasons["profile_fetch_error"] = rejection_reasons.get("profile_fetch_error", 0) + 1
                    continue

            is_passed, reason = self.evaluate_candidate(profile, filter_criteria)

            if is_passed:
                passed_tickers.append(ticker)
                profiles_map[ticker] = profile
            else:
                category = reason.split()[0] if reason else "rejected"
                rejection_reasons[category] = rejection_reasons.get(category, 0) + 1

        result = UniverseScanResult(
            total_candidates=len(unique_tickers),
            passed_tickers=passed_tickers,
            rejected_count=len(unique_tickers) - len(passed_tickers),
            rejection_reasons=rejection_reasons,
            profiles=profiles_map
        )

        logger.info(
            f"Universe scan complete: {len(passed_tickers)}/{len(unique_tickers)} candidates passed."
        )
        return result

    def scan_and_generate_payload(
        self,
        indices: Optional[List[str]] = None,
        criteria: Optional[UniverseFilterCriteria] = None,
        preloaded_profiles: Optional[Dict[str, CompanyProfile]] = None
    ) -> UniversePayload:
        """
        Executes multi-index ingestion, deduplication, and baseline screening.
        Emits the exact UniversePayload output data contract defined in Module 01 PRD.
        """
        target_indices = indices or self.settings.default_indices
        ticker_index_map: Dict[str, str] = {}
        total_scanned_count = 0

        for idx in target_indices:
            constituents = self.get_index_constituents(idx)
            total_scanned_count += len(constituents)
            for t in constituents:
                clean_t = t.strip().upper()
                if clean_t not in ticker_index_map:
                    ticker_index_map[clean_t] = idx

        composite_tickers = list(ticker_index_map.keys())
        scan_result = self.filter_universe(
            tickers=composite_tickers,
            criteria=criteria,
            preloaded_profiles=preloaded_profiles
        )

        qualified_constituents: List[ConstituentCandidate] = []
        for ticker in scan_result.passed_tickers:
            profile = scan_result.profiles[ticker]
            qualified_constituents.append(
                ConstituentCandidate(
                    ticker=ticker,
                    index_source=ticker_index_map.get(ticker, "UNKNOWN"),
                    sector=profile.sector or "Unknown",
                    industry=profile.industry,
                    market_cap=profile.market_cap or 0.0,
                    avg_daily_volume=profile.avg_volume_30d or profile.avg_volume_10d or 0,
                    price=profile.current_price
                )
            )

        return UniversePayload(
            execution_timestamp=datetime.now(timezone.utc).isoformat(),
            total_symbols_scanned=total_scanned_count,
            qualified_symbols_count=len(qualified_constituents),
            constituents=qualified_constituents
        )


def get_universe_scanner() -> UniverseScanner:
    """Factory helper returning initialized UniverseScanner."""
    return UniverseScanner()
