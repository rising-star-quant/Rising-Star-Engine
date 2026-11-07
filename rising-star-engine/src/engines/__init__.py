"""
Engines package for screening, scanning, metrics, scoring, and analysis.
"""

from .universe_scanner import (
    UniverseFilterCriteria,
    ConstituentCandidate,
    UniversePayload,
    UniverseScanResult,
    UniverseScanner,
    get_universe_scanner,
)
from .indicator_engine import (
    TechnicalIndicators,
    IndicatorEngine,
    compute_atr,
    compute_distance_from_high,
    check_reversal_stabilization,
    get_indicator_engine,
)
from .value_dip_engine import (
    CompanyFinancials,
    ValueDipFilterCriteria,
    ValueDipCandidate,
    ValueDipPayload,
    ValueDisconnectEngine,
    get_value_disconnect_engine,
)
from .earnings_scanner import (
    EarningsWreckPayload,
    EarningsFilterCriteria,
    EarningsReactionScanner,
    get_earnings_scanner,
)

__all__ = [
    # Universe Scanner (Module 01)
    "UniverseFilterCriteria",
    "ConstituentCandidate",
    "UniversePayload",
    "UniverseScanResult",
    "UniverseScanner",
    "get_universe_scanner",
    # Indicator Engine (Module 02)
    "TechnicalIndicators",
    "IndicatorEngine",
    "compute_atr",
    "compute_distance_from_high",
    "check_reversal_stabilization",
    "get_indicator_engine",
    # Value Dip Engine (Module 02)
    "CompanyFinancials",
    "ValueDipFilterCriteria",
    "ValueDipCandidate",
    "ValueDipPayload",
    "ValueDisconnectEngine",
    "get_value_disconnect_engine",
    # Earnings Scanner (Module 03)
    "EarningsWreckPayload",
    "EarningsFilterCriteria",
    "EarningsReactionScanner",
    "get_earnings_scanner",
]
