"""
Rising Star Trend Engine v2.0 - Central Pipeline Orchestrator (Module 05)
Executes end-to-end quantitative trend discovery, fundamental guard filtering,
adversarial AI verification, dynamic volatility sizing, and Telegram alerting.
Includes structured telemetry logging, fault-tolerant execution, CLI entrypoints,
and decoupled FastAPI backend endpoints for the modular dashboard.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import sys
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid

from fastapi import BackgroundTasks, FastAPI, Query
from pydantic import BaseModel, Field

from src.ai.ai_skeptic import (
    AISkeptic,
    AISkepticEvaluation,
    SkepticAnalysisRequest,
    SkepticVerdict,
    get_ai_skeptic,
)
from src.config import Settings, get_settings
from src.connectors.data_connector import YFinanceDataConnector, get_data_connector
from src.engines.earnings_scanner import (
    EarningsFilterCriteria,
    EarningsReactionScanner,
    EarningsWreckPayload,
    get_earnings_scanner,
)
from src.engines.universe_scanner import (
    ConstituentCandidate,
    UniverseFilterCriteria,
    UniversePayload,
    UniverseScanner,
    get_universe_scanner,
)
from src.engines.value_dip_engine import (
    CompanyFinancials,
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
    TelegramMessageResponse,
    TradeAlertPayload,
    get_telegram_alerter,
)
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rising_star.pipeline")


# =====================================================================
# MODULE 05 DATA CONTRACTS & SCHEMAS
# =====================================================================

class DashboardRiskMetrics(BaseModel):
    """Risk parameters for dashboard display matching Module 05 PRD."""
    stop_loss: float = Field(..., description="ATR-based stop-loss level")
    take_profit: float = Field(..., description="Target profit level")
    atr: float = Field(..., description="14-period Average True Range")


class DashboardOpportunity(BaseModel):
    """Active validated trade setup matching Module 05 Micro-PRD specifications."""
    ticker: str = Field(..., description="Equity ticker symbol")
    score: int = Field(..., ge=0, le=100, description="Normalized conviction score 0-100")
    ai_verdict: Literal["GOLDEN_OPPORTUNITY", "FOOLS_TRAP", "HIGH_RISK_DIP"] = Field(
        ...,
        description="Qualitative adversarial classification"
    )
    risk_metrics: DashboardRiskMetrics = Field(..., description="Volatility stop levels")


class DashboardPayload(BaseModel):
    """
    Decoupled backend API response contract matching Module 05 Micro-PRD.
    Serves active opportunities and risk telemetry to modular UI consumers.
    """
    active_opportunities: List[DashboardOpportunity] = Field(
        default_factory=list,
        description="Catalog of verified, executable setups"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 generation timestamp"
    )


class PipelineConfig(BaseModel):
    """Configuration parameters for pipeline execution."""
    indices: List[str] = Field(
        default_factory=lambda: ["SP400", "RUSSELL1000", "RUSSELL2000"],
        description="Target index universe to ingest"
    )
    account_balance: float = Field(default=100_000.0, gt=0, description="Trading equity for position sizing")
    min_market_cap: Optional[float] = Field(default=None, description="Market cap floor")
    min_adv_dollar: Optional[float] = Field(default=None, description="Average daily dollar volume floor")
    excluded_sectors: List[str] = Field(default_factory=list, description="Sectors to ignore")
    min_drop: float = Field(default=-0.30, description="Deepest acceptable pullback from 52W high")
    max_drop: float = Field(default=-0.15, description="Shallowest acceptable pullback from 52W high")
    force_offline: bool = Field(default=False, description="Enforce deterministic mock mode for testing/CI")
    notify: bool = Field(default=False, description="Dispatch verified trades to Telegram")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat destination override")
    output_path: Optional[str] = Field(default=None, description="Destination path for execution JSON")
    stage: str = Field(default="full", description="Pipeline stage: universe, value-dip, risk-ai, earnings, full")


class PipelineStepTelemetry(BaseModel):
    """Telemetry metrics recorded for each pipeline step."""
    step_name: str
    status: Literal["SUCCESS", "FAILED", "SKIPPED"]
    items_in: int
    items_out: int
    elapsed_seconds: float
    error_message: Optional[str] = None


class PipelineResult(BaseModel):
    """Comprehensive pipeline execution payload and diagnostic report."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["SUCCESS", "PARTIAL_SUCCESS", "FAILED"]
    started_at: str
    completed_at: str
    total_elapsed_seconds: float
    universe_scanned: int = 0
    universe_qualified: int = 0
    value_dips_screened: int = 0
    value_dips_qualified: int = 0
    risk_evaluated: int = 0
    orders_approved: int = 0
    alerts_dispatched: int = 0
    telemetry: List[PipelineStepTelemetry] = Field(default_factory=list)
    dashboard: DashboardPayload = Field(default_factory=DashboardPayload)
    approved_orders: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# In-memory storage of latest execution state for FastAPI endpoints
_LATEST_DASHBOARD_PAYLOAD = DashboardPayload()
_LATEST_PIPELINE_RESULT: Optional[PipelineResult] = None


# =====================================================================
# CORE PIPELINE ORCHESTRATION ENGINE
# =====================================================================

class PipelineOrchestrator:
    """
    Coordinates asynchronous execution of all Rising Star quantitative sub-modules.
    Enforces atomic isolation, graceful partial-failure recovery, and metric collection.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        universe_scanner: Optional[UniverseScanner] = None,
        value_dip_engine: Optional[ValueDisconnectEngine] = None,
        risk_guard: Optional[RiskGuard] = None,
        ai_skeptic: Optional[AISkeptic] = None,
        telegram_alerter: Optional[TelegramAlerter] = None,
        earnings_scanner: Optional[EarningsReactionScanner] = None,
    ):
        self.settings = settings or get_settings()
        self.universe_scanner = universe_scanner or get_universe_scanner()
        self.value_dip_engine = value_dip_engine or get_value_disconnect_engine()
        self.risk_guard = risk_guard or get_risk_guard(settings=self.settings)
        self.ai_skeptic = ai_skeptic
        self.telegram_alerter = telegram_alerter
        self.earnings_scanner = earnings_scanner or get_earnings_scanner(settings=self.settings)

    def _get_ai_skeptic(self, force_offline: bool) -> AISkeptic:
        if self.ai_skeptic:
            return self.ai_skeptic
        return get_ai_skeptic(settings=self.settings, force_offline=force_offline)

    def _get_telegram_alerter(self, chat_id: Optional[str], force_offline: bool) -> TelegramAlerter:
        if self.telegram_alerter:
            return self.telegram_alerter
        return get_telegram_alerter(
            chat_id=chat_id,
            settings=self.settings,
            force_offline=force_offline,
        )

    async def execute_universe_step(
        self,
        config: PipelineConfig,
    ) -> Tuple[UniversePayload, PipelineStepTelemetry]:
        """Step 1: Ingest Universe Tickers & Sector Data (Module 01)."""
        t0 = time.perf_counter()
        step_name = "Universe Ingestion (Module 01)"
        logger.info(f"[{step_name}] Starting constituent extraction for indices: {config.indices}")

        criteria = UniverseFilterCriteria(
            min_market_cap=config.min_market_cap if config.min_market_cap is not None else self.settings.min_market_cap,
            min_adv_dollar=config.min_adv_dollar if config.min_adv_dollar is not None else self.settings.min_adv_dollar,
            min_avg_volume=self.settings.min_avg_volume_30d,
            excluded_sectors=config.excluded_sectors,
        )

        try:
            payload = self.universe_scanner.scan_and_generate_payload(
                indices=config.indices,
                criteria=criteria,
            )
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info(
                f"[{step_name}] Completed in {elapsed}s: "
                f"Scanned {payload.total_symbols_scanned} symbols -> "
                f"Qualified {payload.qualified_symbols_count} constituents."
            )
            telemetry = PipelineStepTelemetry(
                step_name=step_name,
                status="SUCCESS",
                items_in=payload.total_symbols_scanned,
                items_out=payload.qualified_symbols_count,
                elapsed_seconds=elapsed,
            )
            return payload, telemetry
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.error(f"[{step_name}] Failed after {elapsed}s: {exc}")
            empty_payload = UniversePayload(
                indices_scanned=config.indices,
                total_symbols_scanned=0,
                qualified_symbols_count=0,
                constituents=[],
            )
            telemetry = PipelineStepTelemetry(
                step_name=step_name,
                status="FAILED",
                items_in=0,
                items_out=0,
                elapsed_seconds=elapsed,
                error_message=str(exc),
            )
            return empty_payload, telemetry

    async def execute_value_dip_step(
        self,
        universe: UniversePayload,
        config: PipelineConfig,
    ) -> Tuple[ValueDipPayload, PipelineStepTelemetry]:
        """Step 2: Run Value Dip & Fundamental Disconnect Screening (Module 02)."""
        t0 = time.perf_counter()
        step_name = "Value Dip Screening (Module 02)"
        count_in = universe.qualified_symbols_count
        logger.info(f"[{step_name}] Screening {count_in} constituents against retracement and fundamental gates")

        criteria = ValueDipFilterCriteria(
            min_drop_from_52w_high=config.min_drop,
            max_drop_from_52w_high=config.max_drop,
        )

        try:
            if count_in == 0:
                logger.info(f"[{step_name}] No universe candidates available to screen. Skipping.")
                elapsed = round(time.perf_counter() - t0, 3)
                empty_dip = ValueDipPayload(
                    total_candidates_evaluated=0,
                    qualified_candidates_count=0,
                    candidates=[],
                )
                telemetry = PipelineStepTelemetry(
                    step_name=step_name,
                    status="SKIPPED",
                    items_in=0,
                    items_out=0,
                    elapsed_seconds=elapsed,
                )
                return empty_dip, telemetry

            dip_payload = self.value_dip_engine.screen_from_universe_payload(
                universe_payload=universe,
                criteria=criteria,
            )
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info(
                f"[{step_name}] Completed in {elapsed}s: "
                f"Evaluated {dip_payload.total_candidates_evaluated} -> "
                f"Qualified {dip_payload.qualified_candidates_count} value dips."
            )
            telemetry = PipelineStepTelemetry(
                step_name=step_name,
                status="SUCCESS",
                items_in=dip_payload.total_candidates_evaluated,
                items_out=dip_payload.qualified_candidates_count,
                elapsed_seconds=elapsed,
            )
            return dip_payload, telemetry
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.error(f"[{step_name}] Failed after {elapsed}s: {exc}")
            empty_dip = ValueDipPayload(
                total_candidates_evaluated=count_in,
                qualified_candidates_count=0,
                candidates=[],
            )
            telemetry = PipelineStepTelemetry(
                step_name=step_name,
                status="FAILED",
                items_in=count_in,
                items_out=0,
                elapsed_seconds=elapsed,
                error_message=str(exc),
            )
            return empty_dip, telemetry

    async def execute_risk_and_ai_step(
        self,
        candidates: List[ValueDipCandidate],
        config: PipelineConfig,
    ) -> Tuple[List[Dict[str, Any]], PipelineStepTelemetry]:
        """Step 3: Filter candidates through RiskGuard & AI Skeptic Engine (Module 03)."""
        t0 = time.perf_counter()
        step_name = "RiskGuard & AI Skeptic (Module 03)"
        count_in = len(candidates)
        logger.info(f"[{step_name}] Auditing {count_in} candidates through adversarial AI and volatility sizing")

        if count_in == 0:
            elapsed = round(time.perf_counter() - t0, 3)
            return [], PipelineStepTelemetry(
                step_name=step_name,
                status="SKIPPED",
                items_in=0,
                items_out=0,
                elapsed_seconds=elapsed,
            )

        skeptic = self._get_ai_skeptic(config.force_offline)
        portfolio = PortfolioState(total_equity=config.account_balance, cash=config.account_balance, positions={})
        approved_orders: List[Dict[str, Any]] = []

        for cand in candidates:
            try:
                # 1. Adversarial Skeptic Audit
                verdict = await skeptic.evaluate_candidate(cand)
                hub_eval = await skeptic.evaluate_hub(cand)

                # 2. Position Sizing & Stops Calculation
                atr = cand.atr_14 or max(1.0, cand.current_price * 0.02)
                sizing_req = PositionSizingRequest(
                    ticker=cand.ticker,
                    entry_price=cand.current_price,
                    atr=atr,
                    account_balance=config.account_balance,
                )
                plan = self.risk_guard.calculate_position_size(sizing_req)

                # 3. Sector Concentration Validation
                sector_val = self.risk_guard.validate_sector_exposure(
                    portfolio=portfolio,
                    ticker=cand.ticker,
                    sector="Technology",  # Default or extracted
                    planned_capital=plan.allocated_capital,
                    share_price=cand.current_price,
                )

                is_final_approved = (
                    verdict.verdict == "APPROVED"
                    and plan.is_viable
                    and sector_val.is_allowed
                )

                order_entry = {
                    "ticker": cand.ticker,
                    "current_price": cand.current_price,
                    "atr": atr,
                    "ai_skeptic_verdict": verdict.model_dump(),
                    "ai_hub_evaluation": hub_eval.model_dump(),
                    "position_sizing_plan": plan.model_dump(),
                    "sector_exposure": sector_val.model_dump(),
                    "final_approved": is_final_approved,
                }

                if is_final_approved:
                    approved_orders.append(order_entry)
                    logger.info(
                        f"[{step_name}] APPROVED {cand.ticker}: Shares={plan.recommended_shares}, "
                        f"Allocated=${plan.allocated_capital:,.2f}, Stop=${plan.stop_loss_price:.2f}"
                    )
                else:
                    reasons = []
                    if verdict.verdict != "APPROVED":
                        reasons.append(f"AI Skeptic {verdict.verdict} (score {verdict.skepticism_score})")
                    if not plan.is_viable:
                        reasons.append(f"Sizing: {plan.rejection_reason}")
                    if not sector_val.is_allowed:
                        reasons.append(f"Sector: {sector_val.reason}")
                    logger.info(f"[{step_name}] REJECTED {cand.ticker}: {' | '.join(reasons)}")

            except Exception as exc:
                # Fault tolerance: Single candidate failure must never crash the pipeline
                logger.warning(f"[{step_name}] Error auditing ticker {cand.ticker}: {exc}. Skipping ticker.")

        elapsed = round(time.perf_counter() - t0, 3)
        telemetry = PipelineStepTelemetry(
            step_name=step_name,
            status="SUCCESS",
            items_in=count_in,
            items_out=len(approved_orders),
            elapsed_seconds=elapsed,
        )
        return approved_orders, telemetry

    async def execute_alert_step(
        self,
        approved_orders: List[Dict[str, Any]],
        config: PipelineConfig,
    ) -> Tuple[int, PipelineStepTelemetry]:
        """Step 4: Dispatch Validated Signals via Telegram Alerter (Module 04)."""
        t0 = time.perf_counter()
        step_name = "Telegram Alerting (Module 04)"
        count_in = len(approved_orders)

        if not config.notify or count_in == 0:
            elapsed = round(time.perf_counter() - t0, 3)
            return 0, PipelineStepTelemetry(
                step_name=step_name,
                status="SKIPPED",
                items_in=count_in,
                items_out=0,
                elapsed_seconds=elapsed,
            )

        alerter = self._get_telegram_alerter(config.telegram_chat_id, config.force_offline)
        dispatched_count = 0

        for order in approved_orders:
            try:
                plan = order["position_sizing_plan"]
                verdict = order["ai_skeptic_verdict"]
                hub = order.get("ai_hub_evaluation", {})

                trade_payload = TradeAlertPayload(
                    ticker=order["ticker"],
                    current_price=order["current_price"],
                    stop_loss=plan["stop_loss_price"],
                    target_price=plan["target_price"],
                    stop_distance=plan["stop_distance"],
                    stop_distance_pct=plan["stop_distance_pct"],
                    risk_reward_ratio=plan["risk_reward_ratio"],
                    recommended_shares=plan["recommended_shares"],
                    allocated_capital=plan["allocated_capital"],
                    capital_allocation_pct=plan["capital_allocation_pct"],
                    risk_amount=plan["risk_amount"],
                    atr_14=order.get("atr"),
                    ai_skeptic_verdict=verdict["verdict"],
                    skepticism_score=verdict["skepticism_score"],
                    skepticism_index=hub.get("skepticism_index"),
                    devils_advocate_argument=verdict.get("counter_thesis"),
                    recovery_catalyst=hub.get("recovery_catalyst"),
                    justifications=verdict.get("justification", []),
                )
                response = await alerter.dispatch_trade_alert(trade_payload)
                if response.success:
                    dispatched_count += 1
            except Exception as exc:
                logger.error(f"[{step_name}] Failed to dispatch alert for {order['ticker']}: {exc}")

        elapsed = round(time.perf_counter() - t0, 3)
        telemetry = PipelineStepTelemetry(
            step_name=step_name,
            status="SUCCESS",
            items_in=count_in,
            items_out=dispatched_count,
            elapsed_seconds=elapsed,
        )
        return dispatched_count, telemetry

    def generate_dashboard_payload(
        self,
        approved_orders: List[Dict[str, Any]],
    ) -> DashboardPayload:
        """
        Transforms approved pipeline orders into the exact Module 05 DashboardPayload contract.
        """
        opportunities: List[DashboardOpportunity] = []
        for order in approved_orders:
            hub = order.get("ai_hub_evaluation", {})
            verdict = order.get("ai_skeptic_verdict", {})
            plan = order.get("position_sizing_plan", {})

            # Map raw skepticism to normalized conviction score 0-100
            skepticism = verdict.get("skepticism_score", 0.5)
            conviction_score = max(0, min(100, int(round((1.0 - skepticism) * 100))))

            # Normalize qualitative verdict
            raw_hub_verdict = hub.get("verdict", "GOLDEN_OPPORTUNITY")
            if raw_hub_verdict not in ("GOLDEN_OPPORTUNITY", "FOOLS_TRAP", "HIGH_RISK_DIP"):
                raw_hub_verdict = "GOLDEN_OPPORTUNITY"

            opp = DashboardOpportunity(
                ticker=order["ticker"],
                score=conviction_score,
                ai_verdict=raw_hub_verdict,
                risk_metrics=DashboardRiskMetrics(
                    stop_loss=plan.get("stop_loss_price", 0.0),
                    take_profit=plan.get("target_price", 0.0),
                    atr=order.get("atr", 1.0),
                ),
            )
            opportunities.append(opp)

        return DashboardPayload(
            active_opportunities=opportunities,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    async def run_pipeline(self, config: Optional[PipelineConfig] = None) -> PipelineResult:
        """
        Executes the complete headless quantitative pipeline.
        """
        cfg = config or PipelineConfig()
        started_at = datetime.now(timezone.utc).isoformat()
        t_global_start = time.perf_counter()

        logger.info(f"=== Rising Star Trend Engine v2.0 Pipeline Initialized [Stage: {cfg.stage}] ===")
        telemetry_steps: List[PipelineStepTelemetry] = []
        errors: List[str] = []

        # Step 1: Universe Ingestion
        universe_payload, step1_tel = await self.execute_universe_step(cfg)
        telemetry_steps.append(step1_tel)
        if step1_tel.status == "FAILED":
            errors.append(f"Step 1 failed: {step1_tel.error_message}")

        if cfg.stage == "universe":
            t_total = round(time.perf_counter() - t_global_start, 3)
            return PipelineResult(
                status="SUCCESS" if not errors else "FAILED",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_elapsed_seconds=t_total,
                universe_scanned=universe_payload.total_symbols_scanned,
                universe_qualified=universe_payload.qualified_symbols_count,
                telemetry=telemetry_steps,
                errors=errors,
            )

        # Step 2: Value Dip Screening
        dip_payload, step2_tel = await self.execute_value_dip_step(universe_payload, cfg)
        telemetry_steps.append(step2_tel)
        if step2_tel.status == "FAILED":
            errors.append(f"Step 2 failed: {step2_tel.error_message}")

        if cfg.stage == "value-dip":
            t_total = round(time.perf_counter() - t_global_start, 3)
            return PipelineResult(
                status="SUCCESS" if not errors else "FAILED",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_elapsed_seconds=t_total,
                universe_scanned=universe_payload.total_symbols_scanned,
                universe_qualified=universe_payload.qualified_symbols_count,
                value_dips_screened=dip_payload.total_candidates_evaluated,
                value_dips_qualified=dip_payload.qualified_candidates_count,
                telemetry=telemetry_steps,
                errors=errors,
            )

        # Step 3: RiskGuard & AI Skeptic Verification
        approved_orders, step3_tel = await self.execute_risk_and_ai_step(dip_payload.candidates, cfg)
        telemetry_steps.append(step3_tel)
        if step3_tel.status == "FAILED":
            errors.append(f"Step 3 failed: {step3_tel.error_message}")

        # Step 4: Dispatch Telegram Alerts
        alerts_sent, step4_tel = await self.execute_alert_step(approved_orders, cfg)
        telemetry_steps.append(step4_tel)

        # Step 5: Dashboard Payload Generation (Module 05 PRD)
        dashboard_payload = self.generate_dashboard_payload(approved_orders)

        global _LATEST_DASHBOARD_PAYLOAD, _LATEST_PIPELINE_RESULT
        _LATEST_DASHBOARD_PAYLOAD = dashboard_payload

        t_total = round(time.perf_counter() - t_global_start, 3)
        overall_status: Literal["SUCCESS", "PARTIAL_SUCCESS", "FAILED"] = "SUCCESS"
        if errors and approved_orders:
            overall_status = "PARTIAL_SUCCESS"
        elif errors and not approved_orders:
            overall_status = "FAILED"

        result = PipelineResult(
            status=overall_status,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_elapsed_seconds=t_total,
            universe_scanned=universe_payload.total_symbols_scanned,
            universe_qualified=universe_payload.qualified_symbols_count,
            value_dips_screened=dip_payload.total_candidates_evaluated,
            value_dips_qualified=dip_payload.qualified_candidates_count,
            risk_evaluated=len(dip_payload.candidates),
            orders_approved=len(approved_orders),
            alerts_dispatched=alerts_sent,
            telemetry=telemetry_steps,
            dashboard=dashboard_payload,
            approved_orders=approved_orders,
            errors=errors,
        )
        _LATEST_PIPELINE_RESULT = result

        logger.info(
            f"=== Pipeline Finished in {t_total}s | Status: {overall_status} | "
            f"Orders Approved: {len(approved_orders)} | Alerts Dispatched: {alerts_sent} ==="
        )

        if cfg.output_path:
            with open(cfg.output_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"Persisted PipelineResult to: {cfg.output_path}")

        return result


def get_pipeline_orchestrator(settings: Optional[Settings] = None) -> PipelineOrchestrator:
    """Factory helper returning initialized PipelineOrchestrator."""
    return PipelineOrchestrator(settings=settings)


# =====================================================================
# DECOUPLED FASTAPI BACKEND FOR MODULAR DASHBOARD (MODULE 05)
# =====================================================================

app = FastAPI(
    title="Rising Star Trend Engine v2.0 - Headless API",
    description="Decoupled backend API serving active trend opportunities, risk metrics, and pipeline controls.",
    version="2.0.0",
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for container probes and uptime monitors."""
    return {
        "status": "healthy",
        "service": "Rising Star Trend Engine",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/dashboard", response_model=DashboardPayload)
async def get_dashboard() -> DashboardPayload:
    """
    Module 05 PRD Data Contract endpoint:
    Returns the latest active opportunities with qualitative AI verdicts and volatility risk metrics.
    """
    global _LATEST_DASHBOARD_PAYLOAD
    return _LATEST_DASHBOARD_PAYLOAD


@app.get("/api/pipeline/latest", response_model=Optional[PipelineResult])
async def get_latest_pipeline_result() -> Optional[PipelineResult]:
    """Returns telemetry and results of the most recent pipeline run."""
    global _LATEST_PIPELINE_RESULT
    return _LATEST_PIPELINE_RESULT


@app.post("/api/pipeline/run", response_model=PipelineResult)
async def trigger_pipeline_run(config: Optional[PipelineConfig] = None) -> PipelineResult:
    """Triggers an on-demand asynchronous run of the complete quantitative pipeline."""
    orchestrator = get_pipeline_orchestrator()
    return await orchestrator.run_pipeline(config)


# =====================================================================
# CLI ENTRYPOINTS
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for headless CLI execution."""
    parser = argparse.ArgumentParser(
        description="Rising Star Trend Engine v2.0 - Headless Quantitative Pipeline Orchestrator"
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
        "--account-balance",
        type=float,
        default=100_000.0,
        help="Total account equity in USD for position sizing (default: 100,000)",
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
        "--force-offline",
        action="store_true",
        help="Force deterministic offline heuristic fallback for AI Skeptic and Telegram",
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
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the decoupled FastAPI backend server on configured host and port",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port binding when --serve is passed (default: 8000)",
    )
    return parser.parse_args()


async def async_main():
    """Asynchronous entrypoint for CLI execution."""
    args = parse_cli_args()

    if args.serve:
        import uvicorn
        logger.info(f"Starting Rising Star Trend Engine Headless API on 0.0.0.0:{args.port}...")
        uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="info")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        return

    config = PipelineConfig(
        indices=args.indices,
        account_balance=args.account_balance,
        min_market_cap=args.min_market_cap,
        min_adv_dollar=args.min_adv,
        excluded_sectors=args.exclude_sectors,
        force_offline=args.force_offline,
        notify=args.notify,
        telegram_chat_id=args.telegram_chat_id,
        output_path=args.output,
        stage=args.stage,
    )

    orchestrator = get_pipeline_orchestrator()
    result = await orchestrator.run_pipeline(config)

    # Output structured JSON result summary to stdout
    print(
        json.dumps(
            {
                "execution_id": result.execution_id,
                "status": result.status,
                "total_elapsed_seconds": result.total_elapsed_seconds,
                "universe_scanned": result.universe_scanned,
                "universe_qualified": result.universe_qualified,
                "value_dips_screened": result.value_dips_screened,
                "value_dips_qualified": result.value_dips_qualified,
                "orders_approved": result.orders_approved,
                "alerts_dispatched": result.alerts_dispatched,
                "dashboard_opportunities_count": len(result.dashboard.active_opportunities),
                "errors": result.errors,
            },
            indent=2,
        )
    )


def main():
    """Main CLI entrypoint."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Pipeline execution cancelled by user. Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
