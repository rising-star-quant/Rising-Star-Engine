"""
Rising Star Trend Engine v2.0 - Headless Pipeline Orchestrator (CLI / Cloud Job)
Executes automated backend ingestion (Module 01) and value dip screening (Module 02).
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
from src.engines.value_dip_engine import (
    ValueDipCandidate,
    ValueDipFilterCriteria,
    ValueDipPayload,
    ValueDisconnectEngine,
    get_value_disconnect_engine,
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
        f"[STAGE 01] Running Headless Universe Ingestion for indices: {target_indices}"
    )

    payload = scanner.scan_and_generate_payload(
        indices=target_indices,
        criteria=criteria,
    )

    logger.info(
        f"[STAGE 01 COMPLETE] Scanned {payload.total_symbols_scanned} symbols | "
        f"Qualified: {payload.qualified_symbols_count} constituents"
    )

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(payload.model_dump_json(indent=2))
        logger.info(f"Persisted UniversePayload to: {output_json_path}")

    return payload


def run_value_dip_pipeline(
    universe_payload: Optional[UniversePayload] = None,
    tickers: Optional[List[str]] = None,
    min_drop: float = -0.30,
    max_drop: float = -0.15,
    output_json_path: Optional[str] = None,
) -> ValueDipPayload:
    """
    Headless execution entrypoint for Module 02 (Value Dip & Fundamental Disconnect Engine).
    """
    engine = get_value_disconnect_engine()
    criteria = ValueDipFilterCriteria(
        min_drop_from_52w_high=min_drop,
        max_drop_from_52w_high=max_drop,
    )

    if universe_payload:
        logger.info(
            f"[STAGE 02] Running Value Dip screening over {universe_payload.qualified_symbols_count} universe candidates"
        )
        dip_payload = engine.screen_from_universe_payload(
            universe_payload=universe_payload,
            criteria=criteria,
        )
    elif tickers:
        logger.info(f"[STAGE 02] Running Value Dip screening over {len(tickers)} tickers")
        dip_payload = engine.screen_candidates(tickers=tickers, criteria=criteria)
    else:
        # Default to benchmark seeds
        default_universe = run_universe_pipeline()
        return run_value_dip_pipeline(universe_payload=default_universe, min_drop=min_drop, max_drop=max_drop, output_json_path=output_json_path)

    logger.info(
        f"[STAGE 02 COMPLETE] Screened {dip_payload.total_candidates_evaluated} candidates | "
        f"Qualified Value Dips: {dip_payload.qualified_candidates_count}"
    )

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(dip_payload.model_dump_json(indent=2))
        logger.info(f"Persisted ValueDipPayload to: {output_json_path}")

    return dip_payload


def main():
    """CLI Entrypoint for automated cron, background task, or CI/CD container execution."""
    parser = argparse.ArgumentParser(
        description="Rising Star Trend Engine v2.0 - Headless Quantitative Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["universe", "value-dip", "full"],
        default="full",
        help="Pipeline execution stage (default: full)",
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
        help="Optional destination path to write output JSON",
    )

    args = parser.parse_args()

    if args.stage == "universe":
        universe = run_universe_pipeline(
            indices=args.indices,
            min_market_cap=args.min_market_cap,
            min_adv_dollar=args.min_adv,
            excluded_sectors=args.exclude_sectors,
            output_json_path=args.output,
        )
        print(
            json.dumps(
                {
                    "stage": "universe_ingestion",
                    "execution_timestamp": universe.execution_timestamp,
                    "total_symbols_scanned": universe.total_symbols_scanned,
                    "qualified_symbols_count": universe.qualified_symbols_count,
                    "qualified_tickers": [c.ticker for c in universe.constituents],
                },
                indent=2,
            )
        )
    elif args.stage == "value-dip":
        dip_payload = run_value_dip_pipeline(output_json_path=args.output)
        print(
            json.dumps(
                {
                    "stage": "value_dip_screening",
                    "execution_timestamp": dip_payload.execution_timestamp,
                    "total_candidates_evaluated": dip_payload.total_candidates_evaluated,
                    "qualified_candidates_count": dip_payload.qualified_candidates_count,
                    "qualified_candidates": [c.model_dump() for c in dip_payload.candidates],
                },
                indent=2,
            )
        )
    else:  # full pipeline
        universe = run_universe_pipeline(
            indices=args.indices,
            min_market_cap=args.min_market_cap,
            min_adv_dollar=args.min_adv,
            excluded_sectors=args.exclude_sectors,
        )
        dip_payload = run_value_dip_pipeline(
            universe_payload=universe,
            output_json_path=args.output,
        )
        print(
            json.dumps(
                {
                    "stage": "full_pipeline",
                    "universe_scanned": universe.total_symbols_scanned,
                    "universe_qualified": universe.qualified_symbols_count,
                    "value_dips_qualified": dip_payload.qualified_candidates_count,
                    "qualified_candidates": [c.model_dump() for c in dip_payload.candidates],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
