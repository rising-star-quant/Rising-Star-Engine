"""
Rising Star Trend Engine v2.0 - Headless Pipeline Orchestrator (CLI / Cloud Job)
Executes automated backend ingestion (Module 01) and value dip screening (Module 02).
"""

import argparse
import json
import logging
import sys
from typing import List, Optional

from src.ai.ai_skeptic import AISkeptic, SkepticVerdict, get_ai_skeptic
from src.config import Settings, get_settings
from src.connectors.data_connector import YFinanceDataConnector, get_data_connector
from src.engines.earnings_scanner import (
    EarningsFilterCriteria,
    EarningsReactionScanner,
    EarningsWreckPayload,
    get_earnings_scanner,
)
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
from src.notifications.telegram_alerter import (
    EarningsWreckAlertPayload,
    RiskWarningAlertPayload,
    TelegramAlerter,
    TradeAlertPayload,
    get_telegram_alerter,
)
from src.risk.risk_guard import (
    PortfolioPosition,
    PortfolioState,
    PositionSizingPlan,
    PositionSizingRequest,
    RiskGuard,
    get_risk_guard,
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


def run_risk_and_skeptic_pipeline(
    candidates: Optional[List[ValueDipCandidate]] = None,
    account_balance: float = 100_000.0,
    force_offline: bool = False,
    output_json_path: Optional[str] = None,
    notify: bool = False,
    chat_id: Optional[str] = None,
) -> List[dict]:
    """
    Headless execution entrypoint for Module 03/04 (RiskGuard, AI Skeptic, and Telegram Alerting).
    """
    if candidates is None:
        # Generate candidates from value dip stage
        dip_payload = run_value_dip_pipeline()
        candidates = dip_payload.candidates

    logger.info(f"[STAGE 03: RISK & AI] Evaluating {len(candidates)} candidates through AI Skeptic and RiskGuard")
    skeptic = get_ai_skeptic(force_offline=force_offline)
    risk_guard = get_risk_guard()
    alerter = get_telegram_alerter(chat_id=chat_id, force_offline=force_offline) if notify else None

    portfolio = PortfolioState(total_equity=account_balance, cash=account_balance, positions={})
    results = []

    for cand in candidates:
        # 1. Adversarial Skeptic Audit
        verdict = skeptic.evaluate_candidate_sync(cand)

        # 2. Position Sizing & Stops Calculation
        atr = cand.atr_14 or max(1.0, cand.current_price * 0.02)
        sizing_req = PositionSizingRequest(
            ticker=cand.ticker,
            entry_price=cand.current_price,
            atr=atr,
            account_balance=account_balance,
        )
        plan = risk_guard.calculate_position_size(sizing_req)

        # 3. Sector Concentration Validation
        sector_val = risk_guard.validate_sector_exposure(
            portfolio=portfolio,
            ticker=cand.ticker,
            sector="Technology",  # Default or extracted
            planned_capital=plan.allocated_capital,
            share_price=cand.current_price,
        )

        is_final_approved = (verdict.verdict == "APPROVED" and plan.is_viable and sector_val.is_allowed)

        # 4. Optional Telegram Dispatch (Module 04)
        if notify and is_final_approved and alerter:
            trade_payload = TradeAlertPayload(
                ticker=cand.ticker,
                current_price=cand.current_price,
                stop_loss=plan.stop_loss_price,
                target_price=plan.target_price,
                stop_distance=plan.stop_distance,
                stop_distance_pct=plan.stop_distance_pct,
                risk_reward_ratio=plan.risk_reward_ratio,
                recommended_shares=plan.recommended_shares,
                allocated_capital=plan.allocated_capital,
                capital_allocation_pct=plan.capital_allocation_pct,
                risk_amount=plan.risk_amount,
                distance_from_52w_high_pct=cand.distance_from_52w_high_pct,
                atr_14=atr,
                ai_skeptic_verdict=verdict.verdict,
                skepticism_score=verdict.skepticism_score,
                devils_advocate_argument=verdict.counter_thesis,
                justifications=verdict.justification,
            )
            alerter.dispatch_trade_alert_sync(trade_payload)

        entry = {
            "ticker": cand.ticker,
            "current_price": cand.current_price,
            "ai_skeptic_verdict": verdict.model_dump(),
            "position_sizing_plan": plan.model_dump(),
            "sector_exposure": sector_val.model_dump(),
            "final_approved": is_final_approved,
        }
        results.append(entry)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Persisted Risk & AI Skeptic results to: {output_json_path}")

    return results


def run_earnings_pipeline(
    tickers: Optional[List[str]] = None,
    output_json_path: Optional[str] = None,
) -> List[EarningsWreckPayload]:
    """
    Headless execution entrypoint for Module 03 (Earnings Reaction Scanner).
    """
    scanner = get_earnings_scanner()
    target_tickers = tickers or ["WRECK", "PANIC", "MISS"]
    # Simulated calendar for demonstration / CLI execution
    calendar = {
        t: {
            "earnings_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "revenue_growth_yoy": 0.06,
            "eps_surprise_pct": -0.01,
        }
        for t in target_tickers
    }

    logger.info(f"[STAGE 03: EARNINGS] Scanning {len(target_tickers)} tickers for acute reaction overreactions")
    results = scanner.scan_universe(target_tickers, calendar)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)
        logger.info(f"Persisted EarningsWreck results to: {output_json_path}")

    return results


def main():
    """CLI Entrypoint for automated cron, background task, or CI/CD container execution."""
    parser = argparse.ArgumentParser(
        description="Rising Star Trend Engine v2.0 - Headless Quantitative Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["universe", "value-dip", "risk-ai", "earnings", "full"],
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
        "--account-balance",
        type=float,
        default=100_000.0,
        help="Total account equity in USD for position sizing (default: 100,000)",
    )
    parser.add_argument(
        "--force-offline",
        action="store_true",
        help="Force deterministic offline heuristic fallback for AI Skeptic",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Dispatch real-time alerts to Telegram for qualified trades",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=None,
        help="Optional destination Telegram chat ID override",
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
    elif args.stage == "risk-ai":
        risk_results = run_risk_and_skeptic_pipeline(
            account_balance=args.account_balance,
            force_offline=args.force_offline,
            output_json_path=args.output,
            notify=args.notify,
            chat_id=args.telegram_chat_id,
        )
        print(
            json.dumps(
                {
                    "stage": "risk_and_ai_skeptic",
                    "account_balance": args.account_balance,
                    "total_evaluated": len(risk_results),
                    "approved_trades": [r for r in risk_results if r["final_approved"]],
                },
                indent=2,
            )
        )
    elif args.stage == "earnings":
        earnings_results = run_earnings_pipeline(output_json_path=args.output)
        print(
            json.dumps(
                {
                    "stage": "earnings_reaction_scanner",
                    "qualified_wrecks": [r.model_dump() for r in earnings_results],
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
        )
        risk_results = run_risk_and_skeptic_pipeline(
            candidates=dip_payload.candidates,
            account_balance=args.account_balance,
            force_offline=args.force_offline,
            output_json_path=args.output,
            notify=args.notify,
            chat_id=args.telegram_chat_id,
        )
        approved = [r for r in risk_results if r["final_approved"]]
        print(
            json.dumps(
                {
                    "stage": "full_pipeline",
                    "universe_scanned": universe.total_symbols_scanned,
                    "universe_qualified": universe.qualified_symbols_count,
                    "value_dips_qualified": dip_payload.qualified_candidates_count,
                    "risk_and_ai_approved_count": len(approved),
                    "actionable_orders": approved,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
