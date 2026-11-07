"""
Pytest suite for Module 04: Telegram Alert & Notification Engine.
Tests secure configuration management, fail-fast production validations, SecretStr masking,
mobile MarkdownV2 & HTML formatting, asynchronous dispatching, HTTP 429 rate limit backoff,
HTTP 400/401 client error handling, network timeouts, and batch dispatching.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import SecretStr
import pytest
import respx

from src.config import Settings
from src.notifications.config import (
    TelegramConfigError,
    TelegramSettings,
    get_telegram_settings,
)
from src.notifications.telegram_alerter import (
    EarningsWreckAlertPayload,
    RiskWarningAlertPayload,
    TelegramAlerter,
    TelegramMessageBuilder,
    TelegramMessageResponse,
    TradeAlertPayload,
    get_telegram_alerter,
)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def mock_telegram_settings():
    """Provides secure test settings for notifications."""
    return TelegramSettings(
        env="testing",
        telegram_bot_token=SecretStr("TEST_BOT_TOKEN_12345"),
        telegram_chat_id="-1001234567890",
        request_timeout=5,
        max_retries=2,
        backoff_factor=0.1,
        rate_limit_max_wait_seconds=1.0,
    )


@pytest.fixture
def sample_trade_payload():
    """Provides a realistic Value Dip trade alert payload."""
    return TradeAlertPayload(
        title="🎯 VALUE DIP DETECTED",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        sector="Technology",
        current_price=115.50,
        stop_loss=107.00,
        target_price=136.75,
        stop_distance=8.50,
        stop_distance_pct=0.0736,
        risk_reward_ratio=2.50,
        recommended_shares=150,
        allocated_capital=17_325.00,
        capital_allocation_pct=0.173,
        risk_amount=1_275.00,
        distance_from_52w_high_pct=-0.224,
        atr_14=4.25,
        ai_skeptic_verdict="APPROVED",
        skepticism_score=0.28,
        skepticism_index=3,
        justifications=[
            "Data center accelerator demand remains robust.",
            "Trailing free cash flow generation exceeds $15B.",
        ],
        devils_advocate_argument="Near-term hyperscaler capex pause could cause multiple compression.",
        recovery_catalyst="Next quarter margin expansion confirmation.",
    )


@pytest.fixture
def sample_earnings_wreck_payload():
    """Provides an acute post-earnings overreaction alert payload."""
    return EarningsWreckAlertPayload(
        title="⚡ ACUTE EARNINGS OVERREACTION",
        ticker="WRECK",
        earnings_date="2026-11-04",
        current_price=86.00,
        price_change_1d_pct=-0.14,
        price_change_3d_pct=-0.18,
        relative_volume=3.20,
        revenue_integrity_passed=True,
        revenue_growth_yoy=0.08,
        eps_surprise_pct=-0.02,
        ai_skeptic_commentary="Panic selloff triggered by minor 2-cent EPS miss while revenue expanded 8% YoY.",
    )


@pytest.fixture
def sample_risk_warning_payload():
    """Provides a portfolio risk limit warning payload."""
    return RiskWarningAlertPayload(
        title="⚠️ PORTFOLIO RISK WARNING",
        warning_type="SECTOR_CAP_BREACH",
        ticker="INTC",
        sector="Technology",
        details="Adding $15,000 to Technology pushes sector exposure to 24.5%, breaching 20.0% ceiling.",
        action_taken="REJECTED_NEW_ALLOCATION",
        current_metrics={
            "Current Tech Capital": "$19,500.00",
            "Max Allowed Tech Capital": "$20,000.00",
            "Headroom Capital": "$500.00",
        },
    )


# =====================================================================
# 1. SECURE CONFIGURATION MANAGEMENT TESTS
# =====================================================================

class TestTelegramSettingsSecurity:
    """Tests secure credentials loading, SecretStr masking, and production fail-fast rules."""

    def test_default_development_settings(self):
        """In development/testing, missing credentials do not raise errors."""
        settings = TelegramSettings(env="development", telegram_bot_token=None, telegram_chat_id=None)
        assert settings.get_token_str() is None
        assert settings.telegram_chat_id is None
        assert settings.env == "development"

    def test_secret_str_masks_token_in_logs_and_repr(self):
        """Tokens are stored as SecretStr and never leaked via str() or repr()."""
        raw_token = "987654321:SECRET_TOKEN_ABCXYZ"
        settings = TelegramSettings(
            env="development",
            telegram_bot_token=SecretStr(raw_token),
            telegram_chat_id="123456",
        )

        assert raw_token not in repr(settings)
        assert raw_token not in str(settings)
        assert "**********" in repr(settings.telegram_bot_token)
        assert settings.get_token_str() == raw_token

    def test_environment_variable_alias_loading(self, monkeypatch):
        """Loads from TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_BOT_TOKEN_123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100999888777")
        monkeypatch.setenv("RSTE_ENV", "testing")

        settings = TelegramSettings()

        assert settings.get_token_str() == "ENV_BOT_TOKEN_123"
        assert settings.telegram_chat_id == "-100999888777"
        assert settings.env == "testing"

    def test_rste_prefixed_environment_variable_fallback(self, monkeypatch):
        """Supports RSTE_ prefixed environment variables."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        monkeypatch.setenv("RSTE_TELEGRAM_BOT_TOKEN", "PREFIXED_TOKEN_555")
        monkeypatch.setenv("RSTE_TELEGRAM_CHAT_ID", "@prefixed_channel")

        settings = TelegramSettings()

        assert settings.get_token_str() == "PREFIXED_TOKEN_555"
        assert settings.telegram_chat_id == "@prefixed_channel"

    def test_production_fail_fast_on_missing_token(self):
        """Production environment raises TelegramConfigError/ValueError if bot token is missing."""
        with pytest.raises((TelegramConfigError, ValueError)) as exc_info:
            TelegramSettings(
                env="production",
                telegram_bot_token=None,
                telegram_chat_id="-1001234567890",
            )
        assert "TELEGRAM_BOT_TOKEN is mandatory but missing" in str(exc_info.value)

    def test_production_fail_fast_on_missing_chat_id(self):
        """Production environment raises TelegramConfigError/ValueError if chat ID is missing."""
        with pytest.raises((TelegramConfigError, ValueError)) as exc_info:
            TelegramSettings(
                env="production",
                telegram_bot_token=SecretStr("PROD_TOKEN_VALID"),
                telegram_chat_id="",
            )
        assert "TELEGRAM_CHAT_ID is mandatory but missing" in str(exc_info.value)

    def test_production_passes_when_all_secrets_present(self):
        """Production initializes cleanly when all mandatory credentials are provided."""
        settings = TelegramSettings(
            env="production",
            telegram_bot_token=SecretStr("PROD_VALID_TOKEN"),
            telegram_chat_id="-1001234567890",
        )
        assert settings.env == "production"
        assert settings.get_token_str() == "PROD_VALID_TOKEN"


# =====================================================================
# 2. MESSAGE BUILDER & FORMATTING TESTS (MARKDOWN V2 & HTML)
# =====================================================================

class TestTelegramMessageBuilder:
    """Tests MarkdownV2 and HTML mobile message formatting across all alert types."""

    def test_build_trade_alert_markdown_v2_escaping(self, sample_trade_payload):
        """MarkdownV2 format safely escapes reserved characters while maintaining markdown markup."""
        msg = TelegramMessageBuilder.build_trade_alert_markdown(sample_trade_payload)

        # Markdown bold and monospace indicators
        assert "*🎯 VALUE DIP DETECTED: NVDA*" in msg
        assert "`$115.50`" in msg
        assert "`$107.00`" in msg
        assert "`$136.75`" in msg
        assert "`2.50:1`" in msg
        # Escaped periods and hyphens in plain text
        assert "Stop\\-Loss" in msg
        assert "Take\\-Profit" in msg
        assert "52W Pullback" in msg
        assert "APPROVED" in msg
        assert "3/10" in msg

    def test_build_trade_alert_html_full(self, sample_trade_payload):
        """Verify HTML format contains bold tags and monetary amounts."""
        msg = TelegramMessageBuilder.build_trade_alert_html(sample_trade_payload)

        assert "<b>🎯 VALUE DIP DETECTED: NVDA</b>" in msg
        assert "NVIDIA Corporation" in msg
        assert "Technology" in msg
        assert "-22.4%" in msg
        assert "$115.50" in msg
        assert "$107.00" in msg
        assert "$136.75" in msg
        assert "2.50:1" in msg
        assert "$17,325.00" in msg
        assert "APPROVED" in msg

    def test_build_trade_alert_minimal(self):
        """Minimal required fields format cleanly in both modes without exceptions."""
        minimal = TradeAlertPayload(
            ticker="MINI",
            current_price=50.0,
            stop_loss=46.0,
            target_price=60.0,
        )
        msg_md = TelegramMessageBuilder.build_trade_alert(minimal, parse_mode="MarkdownV2")
        msg_html = TelegramMessageBuilder.build_trade_alert(minimal, parse_mode="HTML")

        assert "MINI" in msg_md
        assert "$50.00" in msg_md
        assert "MINI" in msg_html
        assert "$50.00" in msg_html

    def test_build_earnings_wreck_alert(self, sample_earnings_wreck_payload):
        """Formats acute post-earnings overreaction alert."""
        msg = TelegramMessageBuilder.build_earnings_wreck_alert(sample_earnings_wreck_payload)

        assert "ACUTE EARNINGS OVERREACTION: WRECK" in msg
        assert "2026-11-04" in msg
        assert "$86.00" in msg
        assert "-14.0%" in msg
        assert "-18.0%" in msg
        assert "3.20x" in msg
        assert "PASS (Intact)" in msg
        assert "+8.0%" in msg
        assert "-2.0%" in msg
        assert "Panic selloff triggered by minor 2-cent EPS miss" in msg

    def test_build_risk_warning_alert(self, sample_risk_warning_payload):
        """Formats portfolio risk warning alert."""
        msg = TelegramMessageBuilder.build_risk_warning_alert(sample_risk_warning_payload)

        assert "PORTFOLIO RISK WARNING" in msg
        assert "SECTOR_CAP_BREACH" in msg
        assert "INTC" in msg
        assert "Technology" in msg
        assert "breaching 20.0% ceiling" in msg
        assert "REJECTED_NEW_ALLOCATION" in msg
        assert "$19,500.00" in msg
        assert "$500.00" in msg

    def test_html_escaping_prevents_markup_injection(self):
        """Special HTML characters (<, >, &, ") in dynamic fields are safely escaped."""
        payload = TradeAlertPayload(
            ticker="EVIL",
            company_name="Dangerous <script>alert(1)</script> & Co",
            current_price=10.0,
            stop_loss=8.0,
            target_price=15.0,
            devils_advocate_argument="Growth < 0% & Debt > 5.0x",
        )
        msg = TelegramMessageBuilder.build_trade_alert_html(payload)

        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg
        assert "&amp; Co" in msg
        assert "Growth &lt; 0% &amp; Debt &gt; 5.0x" in msg


# =====================================================================
# 3. ASYNC TELEGRAM ALERTER DISPATCH TESTS
# =====================================================================

class TestTelegramAlerterDispatch:
    """Tests non-blocking HTTP dispatch to Telegram Bot API with mock responses."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_dispatch_200(self, mock_telegram_settings, sample_trade_payload):
        """HTTP 200 response confirms delivery and extracts message_id."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        mock_route = respx.post(bot_url).respond(
            status_code=200,
            json={
                "ok": True,
                "result": {
                    "message_id": 98765,
                    "chat": {"id": -1001234567890, "title": "Quantitative Alerts"},
                    "date": 1731000000,
                    "text": "Alert Body",
                },
            },
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.dispatch_trade_alert(sample_trade_payload)

        assert response.success is True
        assert response.message_id == 98765
        assert response.chat_id == mock_telegram_settings.telegram_chat_id
        assert "Message delivered successfully" in response.description
        assert mock_route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_rate_limit_429_with_retry_and_recovery(self, mock_telegram_settings, sample_trade_payload):
        """HTTP 429 triggers retry-after wait and succeeds on second attempt."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"

        mock_route = respx.post(bot_url)
        mock_route.side_effect = [
            httpx.Response(
                status_code=429,
                json={
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests: retry after 1",
                    "parameters": {"retry_after": 1},
                },
            ),
            httpx.Response(
                status_code=200,
                json={"ok": True, "result": {"message_id": 99999}},
            ),
        ]

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.dispatch_trade_alert(sample_trade_payload)

        assert response.success is True
        assert response.message_id == 99999
        assert mock_route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_rate_limit_429_exhausted(self, mock_telegram_settings):
        """Persistent HTTP 429 returns graceful error response without raising exception."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(
            status_code=429,
            json={
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests: retry after 2",
                "parameters": {"retry_after": 2},
            },
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.send_message(text="Test", max_retries=1)

        assert response.success is False
        assert response.error_code == 429
        assert response.retry_after == 2
        assert "rate limit exceeded" in response.description.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_bad_request_400_chat_not_found(self, mock_telegram_settings, sample_trade_payload):
        """HTTP 400 (e.g. Chat not found / Invalid group) fails gracefully without crashing."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(
            status_code=400,
            json={
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            },
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.dispatch_trade_alert(sample_trade_payload)

        assert response.success is False
        assert response.error_code == 400
        assert "chat not found" in response.description.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unauthorized_401_invalid_token(self, mock_telegram_settings, sample_trade_payload):
        """HTTP 401 Unauthorized returns diagnostic error description and masks token in logs."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(
            status_code=401,
            json={
                "ok": False,
                "error_code": 401,
                "description": "Unauthorized: invalid bot token",
            },
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.dispatch_trade_alert(sample_trade_payload)

        assert response.success is False
        assert response.error_code == 401
        assert "invalid bot token" in response.description.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_server_error_500_retries_and_fails_gracefully(self, mock_telegram_settings):
        """HTTP 500 server error retries up to max_retries and returns failure response."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(status_code=500, text="Internal Server Error")

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.send_message(text="Server Error Test", max_retries=1)

        assert response.success is False
        assert response.error_code == 500

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_timeout_handling(self, mock_telegram_settings):
        """Network timeout (httpx.ConnectTimeout) is caught and logged safely without crashing."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).mock(side_effect=httpx.ConnectTimeout("Connection timed out"))

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = await alerter.send_message(text="Timeout Test", max_retries=0)

        assert response.success is False
        assert response.error_code == 408
        assert "timeout" in response.description.lower()

    @pytest.mark.asyncio
    async def test_unconfigured_offline_mode_mock_success(self):
        """When bot is unconfigured (empty token/chat_id), dispatch succeeds locally as mock."""
        settings = TelegramSettings(
            env="testing",
            telegram_bot_token=None,
            telegram_chat_id=None,
        )
        alerter = TelegramAlerter(settings=settings, force_offline=False)

        assert alerter.is_configured is False
        response = await alerter.send_message("Offline message")

        assert response.success is True
        assert response.message_id == 999999
        assert "offline mode" in response.description.lower()

    @pytest.mark.asyncio
    async def test_missing_chat_id_validation(self):
        """If chat_id is missing, request is rejected with HTTP 400 error description."""
        settings = TelegramSettings(
            env="testing",
            telegram_bot_token=SecretStr("VALID_TOKEN"),
            telegram_chat_id=None,
        )
        alerter = TelegramAlerter(
            token="VALID_TOKEN",
            chat_id=None,
            settings=settings,
            client=httpx.AsyncClient(),
            force_offline=False,
        )
        response = await alerter.send_message("Missing Chat ID", chat_id=None)

        assert response.success is False
        assert response.error_code == 400
        assert "missing destination chat_id" in response.description.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_batch_alert_dispatch(
        self,
        mock_telegram_settings,
        sample_trade_payload,
        sample_earnings_wreck_payload,
        sample_risk_warning_payload,
    ):
        """Dispatches multiple diverse alerts concurrently using asyncio.gather."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(
            status_code=200,
            json={"ok": True, "result": {"message_id": 10001}},
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        payloads = [
            sample_trade_payload,
            sample_earnings_wreck_payload,
            sample_risk_warning_payload,
        ]

        responses = await alerter.dispatch_batch(payloads)

        assert len(responses) == 3
        assert all(r.success is True for r in responses)
        assert all(r.message_id == 10001 for r in responses)

    @respx.mock
    def test_sync_wrappers(self, mock_telegram_settings, sample_trade_payload):
        """Synchronous wrapper methods execute properly in non-async contexts."""
        raw_token = mock_telegram_settings.get_token_str()
        bot_url = f"https://api.telegram.org/bot{raw_token}/sendMessage"
        respx.post(bot_url).respond(
            status_code=200,
            json={"ok": True, "result": {"message_id": 77777}},
        )

        alerter = TelegramAlerter(settings=mock_telegram_settings, force_offline=False)
        response = alerter.dispatch_trade_alert_sync(sample_trade_payload)

        assert response.success is True
        assert response.message_id == 77777
