"""
Rising Star Trend Engine v2.0 - Headless Pipeline Orchestrator (CLI / Cloud Job)
Executes automated backend ingestion, universe filtering, and downstream dispatch.
"""

import argparse
import json
import logging
import sys
from typing import List, Optional

from src.config import Settings, get_settings
from src.connectors.data_connector import YFinanceDataConnector, get_data_connector
from src.engines.universe_scanner import (
    UniverseFilterCriteria,
    UniversePayload,
    UniverseScanner,
    get_universe_scanner,
)

logger = logging.getLogger("rising_star.main")


def run_universe_pipeline(
    indices: Optional[List[str]] = None,
    min_market_cap: Optional[float] = None,
    min_adv_dollar: Optional[float] = None,
    min_avg_volume: Optional[int] = None,
    excluded_sectors: Optional[List[str]] = None,
    output_json_path: Optional[str] = None,
) -> UniversePayload:
    """
    Headless execution entrypoint for Module 01 (Universe & Sector Ingestion).
    Ingests constituents across indices, executes baseline liquidity & sector gates,
    and returns the validated UniversePayload contract.
    """
    settings = get_settings()
    scanner = get_universe_scanner()

    criteria = UniverseFilterCriteria(
        min_market_cap=min_market_cap if min_market_cap is not None else settings.min_market_cap,
        min_adv_dollar=min_adv_dollar if min_adv_dollar is not None else settings.min_adv_dollar,
        min_avg_volume=min_avg_volume if min_avg_volume is not None else settings.min_avg_volume_30d,
        excluded_sectors=excluded_sectors if excluded_sectors is not None else [],
    )

    target_indices = indices or settings.default_indices
    logger.info(
        f"[PIPELINE START] Running Headless Universe Ingestion for indices: {target_indices}"
    )

    payload = scanner.scan_and_generate_payload(
        indices=target_indices,
        criteria=criteria,
    )

    logger.info(
        f"[PIPELINE COMPLETE] Scanned {payload.total_symbols_scanned} symbols | "
        f"Qualified: {payload.qualified_symbols_count} constituents"
    )

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(payload.model_dump_json(indent=2))
        logger.info(f"Persisted UniversePayload to: {output_json_path}")

    return payload


def main():
    """CLI Entrypoint for automated cron, background task, or container execution."""
    parser = argparse.ArgumentParser(
        description="Rising Star Trend Engine v2.0 - Headless Quantitative Ingestion Engine"
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=["SP400", "RUSSELL1000", "RUSSELL2000"],
        help="Target index symbols (e.g. SP400, RUSSELL1000, RUSSELL2000)",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=None,
        help="Minimum market capitalization in USD (default: 300,000,000)",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=None,
        help="Minimum 30-day average daily dollar volume in USD (default: 2,000,000)",
    )
    parser.add_argument(
        "--exclude-sectors",
        nargs="+",
        default=[],
        help="GICS sector names to exclude (e.g. 'Financial Services' 'Utilities')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path to write JSON UniversePayload",
    )

    args = parser.parse_args()

    payload = run_universe_pipeline(
        indices=args.indices,
        min_market_cap=args.min_market_cap,
        min_adv_dollar=args.min_adv,
        excluded_sectors=args.exclude_sectors,
        output_json_path=args.output,
    )

    # Print summary JSON to stdout for piping/composability
    print(
        json.dumps(
            {
                "execution_timestamp": payload.execution_timestamp,
                "total_symbols_scanned": payload.total_symbols_scanned,
                "qualified_symbols_count": payload.qualified_symbols_count,
                "qualified_tickers": [c.ticker for c in payload.constituents],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
