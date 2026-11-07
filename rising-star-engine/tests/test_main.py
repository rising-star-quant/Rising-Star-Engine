"""
Pytest suite for Module 05: Main Pipeline Orchestrator and Headless API.
Comprehensive unit and integration tests covering end-to-end execution,
candidate filtering, adversarial AI evaluation, dynamic risk sizing,
Telegram notification toggles, partial failure isolation, CLI argument handling,
and decoupled FastAPI endpoints matching the Module 05 PRD contract.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from src.ai.ai_skeptic import (
    AISkeptic,
    AISkepticEvaluation,
    SkepticVerdict,
)
from src.config import Settings
from src.engines.universe_scanner import (
    ConstituentCandidate,
    UniverseFilterCriteria,
    UniversePayload,
    UniverseScanner,
)
from src.engines.value_dip_engine import (
    CompanyFinancials,
    ValueDipCandidate,
    ValueDipFilterCriteria,
    ValueDipPayload,
    ValueDisconnectEngine,
)
from src.main import (
    DashboardOpportunity,
    DashboardPayload,
    DashboardRiskMetrics,
    PipelineConfig,
    PipelineOrchestrator,
    PipelineResult,
    PipelineStepTelemetry,
    app,
    async_main,
    get_pipeline_orchestrator,
    parse_cli_args,
)
from src.notifications.telegram_alerter import (
    TelegramAlerter,
    TelegramMessageResponse,
    TradeAlertPayload,
)
from src.risk.risk_guard import (
    DynamicStopsResult,
    PortfolioState,
    PositionSizingPlan,
    PositionSizingRequest,
    RiskGuard,
    SectorExposureValidation,
)


# =====================================================================
# FIXTURES & MOCK FACTORIES
# =====================================================================

@pytest.fixture
def mock_settings() -> Settings:
    """Isolated test settings for orchestrator testing."""
    return Settings(
        env="testing",
        min_market_cap=300_000_000.0,
        min_adv_dollar=2_000_000.0,
        min_avg_volume_30d=200_000.0,
        risk_per_trade_pct=0.015,
        max_sector_exposure_pct=0.20,
        max_position_capital_pct=0.15,
        default_atr_stop_multiplier=2.0,
        default_risk_reward_ratio=2.5,
        ai_skeptic_rejection_threshold=0.60,
    )


@pytest.fixture
def sample_universe_payload() -> UniversePayload:
    """Provides a realistic UniversePayload with 2 qualified stocks."""
    return UniversePayload(
        indices_scanned=["SP400", "RUSSELL1000"],
        total_symbols_scanned=2,
        qualified_symbols_count=2,
        constituents=[
            ConstituentCandidate(
                ticker="ALSN",
                index_source="SP400",
                sector="Industrials",
                market_cap=7_500_000_000.0,
                avg_daily_volume=350_000,
                price=78.50,
            ),
            ConstituentCandidate(
                ticker="MTD",
                index_source="RUSSELL1000",
                sector="Healthcare",
                market_cap=28_000_000_000.0,
                avg_daily_volume=120_000,
                price=1240.00,
            ),
        ],
    )


@pytest.fixture
def sample_value_dip_payload() -> ValueDipPayload:
    """Provides a realistic ValueDipPayload matching sample constituents."""
    financials_alsn = CompanyFinancials(
        operating_margin=0.22,
        return_on_equity=0.25,
        debt_to_equity=1.2,
        free_cash_flow_yield=0.07,
        current_ratio=1.6,
        revenue_growth_yoy=0.08,
    )
    financials_mtd = CompanyFinancials(
        operating_margin=0.28,
        return_on_equity=0.35,
        debt_to_equity=0.8,
        free_cash_flow_yield=0.05,
        current_ratio=1.4,
        revenue_growth_yoy=0.06,
    )

    candidates = [
        ValueDipCandidate(
            ticker="ALSN",
            current_price=78.50,
            distance_from_52w_high_pct=-0.2304,
            fundamental_guard_passed=True,
            financials=financials_alsn,
            sector="Industrials",
            atr_14=2.45,
        ),
        ValueDipCandidate(
            ticker="MTD",
            current_price=1240.00,
            distance_from_52w_high_pct=-0.2152,
            fundamental_guard_passed=True,
            financials=financials_mtd,
            sector="Healthcare",
            atr_14=32.10,
        ),
    ]

    return ValueDipPayload(
        total_candidates_evaluated=2,
        qualified_candidates_count=2,
        candidates=candidates,
    )


# =====================================================================
# MODULE 05 SCHEMA CONTRACT TESTS
# =====================================================================

def test_dashboard_payload_contract_validation():
    """Validates the exact Module 05 Micro-PRD DashboardPayload contract."""
    metrics = DashboardRiskMetrics(
        stop_loss=73.60,
        take_profit=90.75,
        atr=2.45,
    )
    opp = DashboardOpportunity(
        ticker="ALSN",
        score=82,
        ai_verdict="GOLDEN_OPPORTUNITY",
        risk_metrics=metrics,
    )
    payload = DashboardPayload(active_opportunities=[opp])

    assert payload.active_opportunities[0].ticker == "ALSN"
    assert payload.active_opportunities[0].score == 82
    assert payload.active_opportunities[0].ai_verdict == "GOLDEN_OPPORTUNITY"
    assert payload.active_opportunities[0].risk_metrics.stop_loss == 73.60
    assert payload.active_opportunities[0].risk_metrics.take_profit == 90.75
    assert payload.active_opportunities[0].risk_metrics.atr == 2.45
    assert payload.generated_at is not None


def test_dashboard_opportunity_score_validation():
    """Score must be an integer between 0 and 100."""
    metrics = DashboardRiskMetrics(stop_loss=10.0, take_profit=20.0, atr=1.0)

    # Valid bounds
    DashboardOpportunity(ticker="X", score=0, ai_verdict="HIGH_RISK_DIP", risk_metrics=metrics)
    DashboardOpportunity(ticker="X", score=100, ai_verdict="FOOLS_TRAP", risk_metrics=metrics)

    # Invalid score < 0
    with pytest.raises(ValidationError):
        DashboardOpportunity(ticker="X", score=-1, ai_verdict="GOLDEN_OPPORTUNITY", risk_metrics=metrics)

    # Invalid score > 100
    with pytest.raises(ValidationError):
        DashboardOpportunity(ticker="X", score=101, ai_verdict="GOLDEN_OPPORTUNITY", risk_metrics=metrics)


def test_dashboard_opportunity_ai_verdict_literal():
    """ai_verdict must be one of GOLDEN_OPPORTUNITY, FOOLS_TRAP, HIGH_RISK_DIP."""
    metrics = DashboardRiskMetrics(stop_loss=10.0, take_profit=20.0, atr=1.0)

    # Invalid literal value
    with pytest.raises(ValidationError):
        DashboardOpportunity(ticker="X", score=50, ai_verdict="INVALID_VERDICT", risk_metrics=metrics)  # type: ignore


def test_generate_dashboard_payload_mapping(mock_settings):
    """Tests orchestrator mapping of internal approved orders to Module 05 DashboardPayload."""
    orchestrator = PipelineOrchestrator(settings=mock_settings)

    mock_approved_orders = [
        {
            "ticker": "ALSN",
            "current_price": 78.50,
            "atr": 2.45,
            "ai_skeptic_verdict": {
                "verdict": "APPROVED",
                "skepticism_score": 0.18,
                "justification": ["Strong free cash flow yield"],
                "counter_thesis": "Industrial slowdown risk",
            },
            "ai_hub_evaluation": {
                "ticker": "ALSN",
                "verdict": "GOLDEN_OPPORTUNITY",
                "skepticism_index": 0.18,
                "devils_advocate_thesis": "Cyclical peak",
                "recovery_catalyst": "Infrastructure bill",
            },
            "position_sizing_plan": {
                "is_viable": True,
                "recommended_shares": 191,
                "allocated_capital": 14_993.50,
                "stop_loss_price": 73.60,
                "target_price": 90.75,
                "stop_distance": 4.90,
                "stop_distance_pct": 0.0624,
                "risk_reward_ratio": 2.5,
                "capital_allocation_pct": 0.1499,
                "risk_amount": 935.90,
            },
            "sector_exposure": {"is_allowed": True},
            "final_approved": True,
        }
    ]

    dashboard = orchestrator.generate_dashboard_payload(mock_approved_orders)

    assert len(dashboard.active_opportunities) == 1
    opp = dashboard.active_opportunities[0]
    assert opp.ticker == "ALSN"
    # Conviction score = (1.0 - 0.18) * 100 = 82
    assert opp.score == 82
    assert opp.ai_verdict == "GOLDEN_OPPORTUNITY"
    assert opp.risk_metrics.stop_loss == 73.60
    assert opp.risk_metrics.take_profit == 90.75
    assert opp.risk_metrics.atr == 2.45


# =====================================================================
# FULL PIPELINE ORCHESTRATION TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_run_pipeline_full_happy_path(
    mock_settings,
    sample_universe_payload,
    sample_value_dip_payload,
):
    """
    End-to-end happy path test:
    Universe Ingestion -> Value Dip Screen -> Risk & AI Verification -> Telegram Alerting.
    Verifies all steps record SUCCESS telemetry and dashboard payload is populated.
    """
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    mock_ai_skeptic = AsyncMock(spec=AISkeptic)
    mock_ai_skeptic.evaluate_candidate.return_value = SkepticVerdict(
        ticker="ALSN",
        verdict="APPROVED",
        skepticism_score=0.20,
        justification=["Sound fundamentals", "Attractive valuation"],
        counter_thesis="Minor economic headwinds",
    )
    mock_ai_skeptic.evaluate_hub.return_value = AISkepticEvaluation(
        ticker="ALSN",
        verdict="GOLDEN_OPPORTUNITY",
        skepticism_index=2,
        devils_advocate_argument="Minor economic headwinds",
        recovery_catalyst="Margin expansion and secular electrification backlog",
    )

    mock_alerter = AsyncMock(spec=TelegramAlerter)
    mock_alerter.dispatch_trade_alert.return_value = TelegramMessageResponse(
        success=True,
        message_id=999,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        ai_skeptic=mock_ai_skeptic,
        telegram_alerter=mock_alerter,
    )

    config = PipelineConfig(
        indices=["SP400", "RUSSELL1000"],
        account_balance=100_000.0,
        notify=True,
        force_offline=True,
    )

    result = await orchestrator.run_pipeline(config)

    assert result.status == "SUCCESS"
    assert result.universe_scanned == 2
    assert result.universe_qualified == 2
    assert result.value_dips_screened == 2
    assert result.value_dips_qualified == 2
    assert result.risk_evaluated == 2
    assert result.orders_approved == 2
    assert result.alerts_dispatched == 2
    assert len(result.errors) == 0

    # Validate step telemetry
    assert len(result.telemetry) == 4
    for step in result.telemetry:
        assert step.status == "SUCCESS"
        assert step.elapsed_seconds >= 0.0

    # Validate dashboard payload contract
    assert len(result.dashboard.active_opportunities) == 2
    opp = result.dashboard.active_opportunities[0]
    assert opp.score == 80  # (1 - 0.20) * 100
    assert opp.ai_verdict == "GOLDEN_OPPORTUNITY"
    assert opp.risk_metrics.stop_loss > 0.0
    assert opp.risk_metrics.take_profit > opp.risk_metrics.stop_loss

    # Assert alerter dispatched twice
    assert mock_alerter.dispatch_trade_alert.await_count == 2


@pytest.mark.asyncio
async def test_run_pipeline_zero_candidates_graceful(mock_settings):
    """
    When universe scanner finds 0 qualified symbols,
    subsequent steps skip gracefully and do not fail.
    """
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = UniversePayload(
        indices_scanned=["SP400"],
        total_symbols_scanned=50,
        qualified_symbols_count=0,
        constituents=[],
    )

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_alerter = AsyncMock(spec=TelegramAlerter)

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        telegram_alerter=mock_alerter,
    )

    result = await orchestrator.run_pipeline(PipelineConfig(notify=True))

    assert result.status == "SUCCESS"
    assert result.universe_qualified == 0
    assert result.value_dips_qualified == 0
    assert result.orders_approved == 0
    assert result.alerts_dispatched == 0
    assert len(result.dashboard.active_opportunities) == 0

    # Steps 2, 3, 4 should be SKIPPED
    assert result.telemetry[0].status == "SUCCESS"
    assert result.telemetry[1].status == "SKIPPED"
    assert result.telemetry[2].status == "SKIPPED"
    assert result.telemetry[3].status == "SKIPPED"

    # Alerter was never called
    mock_alerter.dispatch_trade_alert.assert_not_called()


@pytest.mark.asyncio
async def test_run_pipeline_partial_failure_isolation(
    mock_settings,
    sample_universe_payload,
    sample_value_dip_payload,
):
    """
    If one ticker encounters an exception during AI Skeptic evaluation,
    the pipeline catches and logs it, continues processing the remaining candidate,
    and returns a successful execution for valid setups.
    """
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    # First candidate (ALSN) raises unexpected exception; second (MTD) succeeds
    mock_ai_skeptic = AsyncMock(spec=AISkeptic)
    mock_ai_skeptic.evaluate_candidate.side_effect = [
        RuntimeError("Transient Gemini API timeout"),
        SkepticVerdict(
            ticker="MTD",
            verdict="APPROVED",
            skepticism_score=0.15,
            justification=["Pristine balance sheet"],
            counter_thesis="Valuation premium",
        ),
    ]
    mock_ai_skeptic.evaluate_hub.return_value = AISkepticEvaluation(
        ticker="MTD",
        verdict="GOLDEN_OPPORTUNITY",
        skepticism_index=2,
        devils_advocate_argument="Valuation premium",
        recovery_catalyst="Industrial laboratory automation cycle",
    )

    mock_alerter = AsyncMock(spec=TelegramAlerter)
    mock_alerter.dispatch_trade_alert.return_value = TelegramMessageResponse(success=True)

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        ai_skeptic=mock_ai_skeptic,
        telegram_alerter=mock_alerter,
    )

    result = await orchestrator.run_pipeline(PipelineConfig(notify=True))

    assert result.status == "SUCCESS"
    assert result.orders_approved == 1
    assert result.approved_orders[0]["ticker"] == "MTD"
    assert len(result.dashboard.active_opportunities) == 1
    assert result.dashboard.active_opportunities[0].ticker == "MTD"
    assert result.alerts_dispatched == 1


@pytest.mark.asyncio
async def test_run_pipeline_rejection_by_ai_skeptic(
    mock_settings,
    sample_universe_payload,
    sample_value_dip_payload,
):
    """Candidates rejected by AI Skeptic are filtered out and not sent as trade alerts."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    # AI Skeptic rejects both candidates as FOOLS_TRAP
    mock_ai_skeptic = AsyncMock(spec=AISkeptic)
    mock_ai_skeptic.evaluate_candidate.return_value = SkepticVerdict(
        ticker="ALSN",
        verdict="REJECTED",
        skepticism_score=0.85,
        justification=["Secular margin compression", "Unserviceable debt load"],
        counter_thesis="Fatal structural debt trap",
    )
    mock_ai_skeptic.evaluate_hub.return_value = AISkepticEvaluation(
        ticker="ALSN",
        verdict="FOOLS_TRAP",
        skepticism_index=9,
        devils_advocate_argument="Fatal structural debt trap",
        recovery_catalyst="Unlikely without capital raise",
    )

    mock_alerter = AsyncMock(spec=TelegramAlerter)

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        ai_skeptic=mock_ai_skeptic,
        telegram_alerter=mock_alerter,
    )

    result = await orchestrator.run_pipeline(PipelineConfig(notify=True))

    assert result.status == "SUCCESS"
    assert result.orders_approved == 0
    assert result.alerts_dispatched == 0
    assert len(result.dashboard.active_opportunities) == 0
    mock_alerter.dispatch_trade_alert.assert_not_called()


@pytest.mark.asyncio
async def test_run_pipeline_stages(mock_settings, sample_universe_payload, sample_value_dip_payload):
    """Validates early exit when stage is set to 'universe' or 'value-dip'."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
    )

    # 1. Test stage="universe"
    result_u = await orchestrator.run_pipeline(PipelineConfig(stage="universe"))
    assert result_u.status == "SUCCESS"
    assert len(result_u.telemetry) == 1
    assert result_u.telemetry[0].step_name.startswith("Universe Ingestion")
    assert result_u.universe_qualified == 2

    # 2. Test stage="value-dip"
    result_v = await orchestrator.run_pipeline(PipelineConfig(stage="value-dip"))
    assert result_v.status == "SUCCESS"
    assert len(result_v.telemetry) == 2
    assert result_v.value_dips_qualified == 2


@pytest.mark.asyncio
async def test_run_pipeline_output_file_persistence(
    tmp_path,
    mock_settings,
    sample_universe_payload,
    sample_value_dip_payload,
):
    """Validates that passing output_path persists formatted JSON result to disk."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    mock_ai_skeptic = AsyncMock(spec=AISkeptic)
    mock_ai_skeptic.evaluate_candidate.return_value = SkepticVerdict(
        ticker="TEST",
        verdict="APPROVED",
        skepticism_score=0.10,
        justification=["Solid"],
        counter_thesis=None,
    )
    mock_ai_skeptic.evaluate_hub.return_value = AISkepticEvaluation(
        ticker="TEST",
        verdict="GOLDEN_OPPORTUNITY",
        skepticism_index=1,
        devils_advocate_argument="None identified",
        recovery_catalyst="Growth",
    )

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        ai_skeptic=mock_ai_skeptic,
    )

    out_file = tmp_path / "execution_result.json"
    config = PipelineConfig(output_path=str(out_file), notify=False)

    result = await orchestrator.run_pipeline(config)

    assert out_file.exists()
    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["execution_id"] == result.execution_id
    assert content["status"] == "SUCCESS"
    assert "dashboard" in content
    assert "telemetry" in content


@pytest.mark.asyncio
async def test_run_pipeline_step1_failure(mock_settings):
    """When Step 1 throws an unhandled exception, it logs failure and returns FAILED status."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.side_effect = RuntimeError("Constituent scrape failed")

    orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
    )

    result = await orchestrator.run_pipeline(PipelineConfig(stage="universe"))
    assert result.status == "FAILED"
    assert len(result.errors) > 0
    assert "Constituent scrape failed" in result.errors[0]


# =====================================================================
# FASTAPI BACKEND ENDPOINT TESTS (MODULE 05 DECOUPLED API)
# =====================================================================

@pytest.mark.asyncio
async def test_fastapi_health_endpoint():
    """Validates /health endpoint returns HTTP 200 with service metadata."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["service"] == "Rising Star Trend Engine"
        assert body["version"] == "2.0.0"
        assert "timestamp" in body


@pytest.mark.asyncio
async def test_fastapi_dashboard_endpoint():
    """Validates /api/dashboard returns the latest DashboardPayload matching PRD contract."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "active_opportunities" in data
        assert "generated_at" in data
        assert isinstance(data["active_opportunities"], list)


@pytest.mark.asyncio
async def test_fastapi_trigger_pipeline_run_mocked(
    mock_settings,
    sample_universe_payload,
    sample_value_dip_payload,
):
    """Validates /api/pipeline/run triggers execution and returns valid PipelineResult."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_value_engine = MagicMock(spec=ValueDisconnectEngine)
    mock_value_engine.screen_from_universe_payload.return_value = sample_value_dip_payload

    mock_ai_skeptic = AsyncMock(spec=AISkeptic)
    mock_ai_skeptic.evaluate_candidate.return_value = SkepticVerdict(
        ticker="ALSN",
        verdict="APPROVED",
        skepticism_score=0.15,
        justification=["Strong fundamentals"],
        counter_thesis=None,
    )
    mock_ai_skeptic.evaluate_hub.return_value = AISkepticEvaluation(
        ticker="ALSN",
        verdict="GOLDEN_OPPORTUNITY",
        skepticism_index=2,
        devils_advocate_argument="Minor cyclical slowing",
        recovery_catalyst="Backlog growth",
    )

    mock_orchestrator = PipelineOrchestrator(
        settings=mock_settings,
        universe_scanner=mock_universe_scanner,
        value_dip_engine=mock_value_engine,
        ai_skeptic=mock_ai_skeptic,
    )

    transport = httpx.ASGITransport(app=app)
    with patch("src.main.get_pipeline_orchestrator", return_value=mock_orchestrator):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/pipeline/run",
                json={"account_balance": 50_000.0, "notify": False, "stage": "full"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert data["orders_approved"] == 2
            assert "dashboard" in data
            assert len(data["dashboard"]["active_opportunities"]) == 2

            # Now verify GET /api/pipeline/latest reflects this run
            latest_resp = await client.get("/api/pipeline/latest")
            assert latest_resp.status_code == 200
            latest_data = latest_resp.json()
            assert latest_data["execution_id"] == data["execution_id"]


# =====================================================================
# CLI ARGUMENT PARSER TESTS
# =====================================================================

def test_parse_cli_args_defaults():
    """Validates CLI default arguments."""
    with patch("sys.argv", ["main.py"]):
        args = parse_cli_args()
        assert args.stage == "full"
        assert args.indices == ["SP400", "RUSSELL1000", "RUSSELL2000"]
        assert args.account_balance == 100_000.0
        assert args.notify is False
        assert args.force_offline is False
        assert args.serve is False
        assert args.port == 8000


def test_parse_cli_args_custom():
    """Validates custom flags passed via CLI."""
    cli_flags = [
        "main.py",
        "--stage", "value-dip",
        "--indices", "SP400",
        "--account-balance", "50000.0",
        "--min-market-cap", "500000000",
        "--min-adv", "3000000",
        "--exclude-sectors", "Utilities", "Energy",
        "--force-offline",
        "--notify",
        "--telegram-chat-id", "-100123456789",
        "--output", "/tmp/results.json",
    ]
    with patch("sys.argv", cli_flags):
        args = parse_cli_args()
        assert args.stage == "value-dip"
        assert args.indices == ["SP400"]
        assert args.account_balance == 50000.0
        assert args.min_market_cap == 500_000_000.0
        assert args.min_adv == 3_000_000.0
        assert args.exclude_sectors == ["Utilities", "Energy"]
        assert args.force_offline is True
        assert args.notify is True
        assert args.telegram_chat_id == "-100123456789"
        assert args.output == "/tmp/results.json"


@pytest.mark.asyncio
async def test_async_main_cli_execution(capsys, sample_universe_payload):
    """Validates CLI async_main runs pipeline and outputs structured JSON summary to stdout."""
    mock_universe_scanner = MagicMock(spec=UniverseScanner)
    mock_universe_scanner.scan_and_generate_payload.return_value = sample_universe_payload

    mock_orchestrator = PipelineOrchestrator(
        universe_scanner=mock_universe_scanner,
    )

    with patch("sys.argv", ["main.py", "--stage", "universe"]), \
         patch("src.main.get_pipeline_orchestrator", return_value=mock_orchestrator):
        await async_main()

    captured = capsys.readouterr()
    assert "execution_id" in captured.out
    assert "status" in captured.out
    parsed = json.loads(captured.out)
    assert parsed["status"] == "SUCCESS"
    assert parsed["universe_scanned"] == 2
