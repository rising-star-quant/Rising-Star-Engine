"""
Rising Star Trend Engine v2.0 - Telegram Alert & Notification Engine (Module 04)
Asynchronous, hardened alerting pipeline dispatching quantitative signals, risk parameters,
and AI Skeptic adversarial audits to Telegram channels via the Telegram Bot API.
Features robust MarkdownV2 & HTML formatting, SecretStr token masking, and resilient error handling.
"""

import asyncio
from datetime import datetime, timezone
import html
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.notifications.config import TelegramSettings, get_telegram_settings

logger = logging.getLogger("rising_star.notifications")


# =====================================================================
# DATA CONTRACTS & PAYLOAD SCHEMAS
# =====================================================================

class TradeAlertPayload(BaseModel):
    """
    Validated payload for Value Dip and high-conviction trade setups.
    """
    title: str = Field(default="🎯 VALUE DIP DETECTED", description="Alert header title")
    ticker: str = Field(..., description="Equity ticker symbol")
    company_name: Optional[str] = Field(default=None, description="Company name")
    sector: str = Field(default="Unknown", description="Economic sector classification")
    current_price: float = Field(..., gt=0, description="Entry or latest market price")
    stop_loss: float = Field(..., gt=0, description="Calculated ATR stop-loss price")
    target_price: float = Field(..., gt=0, description="Profit target price")
    stop_distance: Optional[float] = Field(default=None, description="Dollar distance to stop")
    stop_distance_pct: Optional[float] = Field(default=None, description="Stop distance as percentage")
    risk_reward_ratio: float = Field(default=2.5, description="Reward-to-risk ratio")
    recommended_shares: Optional[int] = Field(default=None, ge=0, description="Sized share count")
    allocated_capital: Optional[float] = Field(default=None, ge=0, description="Total allocated capital in USD")
    capital_allocation_pct: Optional[float] = Field(default=None, description="Fraction of account equity allocated")
    risk_amount: Optional[float] = Field(default=None, description="Dollar risk budget")
    distance_from_52w_high_pct: Optional[float] = Field(default=None, description="Retracement from 52W high")
    atr_14: Optional[float] = Field(default=None, description="14-period Average True Range")
    # AI Skeptic Verification Fields
    ai_skeptic_verdict: Optional[str] = Field(
        default=None,
        description="Verdict: APPROVED, REJECTED, GOLDEN_OPPORTUNITY, etc."
    )
    skepticism_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Score 0.0 to 1.0")
    skepticism_index: Optional[int] = Field(default=None, ge=1, le=10, description="Index 1 to 10")
    justifications: List[str] = Field(default_factory=list, description="Bullet justifications")
    red_flags: List[str] = Field(default_factory=list, description="Detected red flags")
    devils_advocate_argument: Optional[str] = Field(default=None, description="Counter-thesis or primary risk")
    recovery_catalyst: Optional[str] = Field(default=None, description="Expected turnaround driver")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 generation timestamp"
    )


class EarningsWreckAlertPayload(BaseModel):
    """
    Validated payload for post-earnings overreaction alerts (Module 03).
    """
    title: str = Field(default="⚡ ACUTE EARNINGS OVERREACTION", description="Alert header title")
    ticker: str = Field(..., description="Equity ticker symbol")
    earnings_date: str = Field(..., description="Filing date (YYYY-MM-DD)")
    current_price: Optional[float] = Field(default=None, description="Post-crash share price")
    price_change_1d_pct: float = Field(..., description="Single-day price change (e.g. -0.14 for -14%)")
    price_change_3d_pct: Optional[float] = Field(default=None, description="3-day cumulative price change")
    relative_volume: float = Field(..., description="Relative volume (RVOL) multiplier")
    revenue_integrity_passed: bool = Field(..., description="True if top-line revenue held steady")
    revenue_growth_yoy: Optional[float] = Field(default=None, description="Quarterly revenue growth rate")
    eps_surprise_pct: Optional[float] = Field(default=None, description="EPS surprise vs consensus")
    ai_skeptic_commentary: Optional[str] = Field(default=None, description="Adversarial sanity check")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 generation timestamp"
    )


class RiskWarningAlertPayload(BaseModel):
    """
    Validated payload for portfolio risk warnings, stop-outs, or sector concentration limits.
    """
    title: str = Field(default="⚠️ PORTFOLIO RISK WARNING", description="Alert header title")
    warning_type: Literal[
        "SECTOR_CAP_BREACH",
        "STOP_LOSS_TRIGGERED",
        "TRAILING_STOP_UPDATED",
        "DRAWDOWN_LIMIT",
        "EXECUTION_ERROR"
    ] = Field(..., description="Classification of risk event")
    ticker: Optional[str] = Field(default=None, description="Impacted ticker symbol if applicable")
    sector: Optional[str] = Field(default=None, description="Impacted sector if applicable")
    details: str = Field(..., description="Detailed description of risk condition")
    action_taken: str = Field(..., description="System action taken (e.g. REJECTED, LIQUIDATED, RATCHETED)")
    current_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Supporting numeric metrics")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 generation timestamp"
    )


class TelegramMessageResponse(BaseModel):
    """
    Delivery confirmation and diagnostic telemetry from Telegram Bot API.
    """
    success: bool = Field(..., description="True if message was accepted by Telegram")
    message_id: Optional[int] = Field(default=None, description="Telegram assigned message ID")
    chat_id: Optional[Union[str, int]] = Field(default=None, description="Destination chat ID")
    error_code: Optional[int] = Field(default=None, description="HTTP status or Telegram error code")
    description: Optional[str] = Field(default=None, description="Telegram response description or error detail")
    retry_after: Optional[int] = Field(default=None, description="Rate limit backoff seconds if HTTP 429")
    delivered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 dispatch timestamp"
    )


# =====================================================================
# MESSAGE FORMATTERS (MARKDOWN V2 & HTML)
# =====================================================================

class TelegramMessageBuilder:
    """
    Transforms structured quantitative schemas into formatted MarkdownV2 or HTML messages
    optimized for mobile Telegram reading.
    """

    # Characters that MUST be escaped in Telegram MarkdownV2 plain text:
    # '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'
    _MARKDOWN_V2_ESCAPE_PATTERN = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")

    @classmethod
    def escape_markdown_v2(cls, text: Optional[Any]) -> str:
        """Escapes reserved characters for Telegram MarkdownV2."""
        if text is None:
            return ""
        s = str(text)
        return cls._MARKDOWN_V2_ESCAPE_PATTERN.sub(r"\\\1", s)

    @classmethod
    def escape_html(cls, text: Optional[Any]) -> str:
        """HTML escapes dynamic strings to prevent parse failures."""
        if text is None:
            return ""
        return html.escape(str(text))

    # --- MARKDOWN V2 FORMATTERS ---

    @classmethod
    def build_trade_alert_markdown(cls, payload: TradeAlertPayload) -> str:
        """Formats a high-conviction trade alert into Telegram MarkdownV2."""
        esc = cls.escape_markdown_v2

        lines = [
            f"*{esc(payload.title)}: {esc(payload.ticker)}*",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if payload.company_name:
            lines.append(f"🏢 *Company:* {esc(payload.company_name)}")

        retrace_str = (
            f"{payload.distance_from_52w_high_pct:.1%}"
            if payload.distance_from_52w_high_pct is not None
            else "N/A"
        )
        lines.append(f"🌐 *Sector:* {esc(payload.sector)} \\| *52W Pullback:* {esc(retrace_str)}")
        lines.append("")

        stop_dist_pct = (
            f"(-{payload.stop_distance_pct:.1%})"
            if payload.stop_distance_pct is not None
            else f"(-{((payload.current_price - payload.stop_loss) / payload.current_price):.1%})"
        )
        target_dist_pct = f"(+{((payload.target_price - payload.current_price) / payload.current_price):.1%})"

        lines.append(f"💵 *Entry Price:* `${payload.current_price:.2f}`")
        lines.append(f"🛑 *Stop\\-Loss:* `${payload.stop_loss:.2f}` {esc(stop_dist_pct)}")
        lines.append(f"🎯 *Take\\-Profit:* `${payload.target_price:.2f}` {esc(target_dist_pct)}")
        lines.append(f"⚖️ *Risk/Reward:* `{payload.risk_reward_ratio:.2f}:1`")

        if payload.atr_14:
            lines.append(f"📐 *ATR \\(14\\):* `${payload.atr_14:.2f}`")

        # Position Sizing
        if payload.recommended_shares is not None and payload.recommended_shares > 0:
            lines.append("")
            lines.append("💼 *Position Sizing Plan:*")
            lines.append(f"• *Recommended Shares:* `{payload.recommended_shares:,}`")
            if payload.allocated_capital is not None:
                cap_pct_str = (
                    f" \\({esc(f'{payload.capital_allocation_pct:.1%}')} equity\\)"
                    if payload.capital_allocation_pct is not None
                    else ""
                )
                lines.append(f"• *Capital Allocated:* `${payload.allocated_capital:,.2f}`{cap_pct_str}")
            if payload.risk_amount is not None:
                lines.append(f"• *Max Risk Budget:* `${payload.risk_amount:,.2f}`")

        # AI Skeptic Verification
        if payload.ai_skeptic_verdict or payload.devils_advocate_argument:
            lines.append("")
            verdict = payload.ai_skeptic_verdict or "FLAGGED"
            verdict_icon = "✅" if verdict in ("APPROVED", "GOLDEN_OPPORTUNITY") else "⚠️"
            score_str = f"{payload.skepticism_score:.2f}" if payload.skepticism_score is not None else "N/A"
            index_str = f" \\({payload.skepticism_index}/10\\)" if payload.skepticism_index is not None else ""
            lines.append(
                f"{verdict_icon} *AI Skeptic Audit:* `{esc(verdict)}` "
                f"\\(Doubt: {esc(score_str)}{index_str}\\)"
            )

            if payload.devils_advocate_argument:
                lines.append(f"• *Devil's Advocate:* _{esc(payload.devils_advocate_argument)}_")
            if payload.recovery_catalyst:
                lines.append(f"• *Recovery Catalyst:* _{esc(payload.recovery_catalyst)}_")
            for just in payload.justifications[:2]:
                lines.append(f"• {esc(just)}")

        lines.append("")
        lines.append(f"⏰ _{esc(payload.timestamp[:19])} UTC_")
        return "\n".join(lines)

    # --- HTML FORMATTERS ---

    @classmethod
    def build_trade_alert_html(cls, payload: TradeAlertPayload) -> str:
        """Formats a high-conviction trade alert into Telegram HTML."""
        esc = cls.escape_html

        lines = [
            f"<b>{esc(payload.title)}: {esc(payload.ticker)}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        if payload.company_name:
            lines.append(f"🏢 <b>Company:</b> {esc(payload.company_name)}")

        retrace_str = (
            f"{payload.distance_from_52w_high_pct:.1%}"
            if payload.distance_from_52w_high_pct is not None
            else "N/A"
        )
        lines.append(f"🌐 <b>Sector:</b> {esc(payload.sector)} | <b>52W Pullback:</b> {retrace_str}")
        lines.append("")

        stop_dist_pct = (
            f"(-{payload.stop_distance_pct:.1%})"
            if payload.stop_distance_pct is not None
            else f"(-{((payload.current_price - payload.stop_loss) / payload.current_price):.1%})"
        )
        target_dist_pct = f"(+{((payload.target_price - payload.current_price) / payload.current_price):.1%})"

        lines.append(f"💵 <b>Entry Price:</b> <code>${payload.current_price:.2f}</code>")
        lines.append(f"🛑 <b>Stop-Loss:</b> <code>${payload.stop_loss:.2f}</code> {stop_dist_pct}")
        lines.append(f"🎯 <b>Take-Profit:</b> <code>${payload.target_price:.2f}</code> {target_dist_pct}")
        lines.append(f"⚖️ <b>Risk/Reward:</b> <code>{payload.risk_reward_ratio:.2f}:1</code>")

        if payload.atr_14:
            lines.append(f"📐 <b>ATR (14):</b> <code>${payload.atr_14:.2f}</code>")

        if payload.recommended_shares is not None and payload.recommended_shares > 0:
            lines.append("")
            lines.append("💼 <b>Position Sizing Plan:</b>")
            lines.append(f"• <b>Recommended Shares:</b> <code>{payload.recommended_shares:,}</code>")
            if payload.allocated_capital is not None:
                cap_pct_str = (
                    f" ({payload.capital_allocation_pct:.1%} equity)"
                    if payload.capital_allocation_pct is not None
                    else ""
                )
                lines.append(f"• <b>Capital Allocated:</b> <code>${payload.allocated_capital:,.2f}</code>{cap_pct_str}")
            if payload.risk_amount is not None:
                lines.append(f"• <b>Max Risk Budget:</b> <code>${payload.risk_amount:,.2f}</code>")

        if payload.ai_skeptic_verdict or payload.devils_advocate_argument:
            lines.append("")
            verdict = payload.ai_skeptic_verdict or "FLAGGED"
            verdict_icon = "✅" if verdict in ("APPROVED", "GOLDEN_OPPORTUNITY") else "⚠️"
            score_str = f"{payload.skepticism_score:.2f}" if payload.skepticism_score is not None else "N/A"
            index_str = f" ({payload.skepticism_index}/10)" if payload.skepticism_index is not None else ""
            lines.append(
                f"{verdict_icon} <b>AI Skeptic Audit:</b> <code>{esc(verdict)}</code> "
                f"(Doubt: {score_str}{index_str})"
            )

            if payload.devils_advocate_argument:
                lines.append(f"• <b>Devil's Advocate:</b> <i>{esc(payload.devils_advocate_argument)}</i>")
            if payload.recovery_catalyst:
                lines.append(f"• <b>Recovery Catalyst:</b> <i>{esc(payload.recovery_catalyst)}</i>")
            for just in payload.justifications[:2]:
                lines.append(f"• {esc(just)}")

        lines.append("")
        lines.append(f"⏰ <i>{esc(payload.timestamp[:19])} UTC</i>")
        return "\n".join(lines)

    @classmethod
    def build_trade_alert(cls, payload: TradeAlertPayload, parse_mode: str = "HTML") -> str:
        """Dispatches to either MarkdownV2 or HTML builder based on parse_mode."""
        if parse_mode.lower() in ("markdown", "markdownv2"):
            return cls.build_trade_alert_markdown(payload)
        return cls.build_trade_alert_html(payload)

    @classmethod
    def build_earnings_wreck_alert(cls, payload: EarningsWreckAlertPayload) -> str:
        """Formats an earnings overreaction alert into Telegram HTML."""
        esc = cls.escape_html
        lines = [
            f"<b>{esc(payload.title)}: {esc(payload.ticker)}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 <b>Earnings Date:</b> {esc(payload.earnings_date)}",
        ]

        if payload.current_price:
            lines.append(f"💵 <b>Current Price:</b> <code>${payload.current_price:.2f}</code>")

        lines.append(f"📉 <b>1-Day Drop:</b> <code>{payload.price_change_1d_pct:.1%}</code>")
        if payload.price_change_3d_pct is not None:
            lines.append(f"📉 <b>3-Day Drop:</b> <code>{payload.price_change_3d_pct:.1%}</code>")

        lines.append(f"📊 <b>Relative Volume:</b> <code>{payload.relative_volume:.2f}x</code> 30d ADV")

        rev_status = "✅ PASS (Intact)" if payload.revenue_integrity_passed else "❌ FAIL (Deterioration)"
        lines.append(f"🛡️ <b>Revenue Integrity:</b> {rev_status}")

        if payload.revenue_growth_yoy is not None:
            lines.append(f"📈 <b>YoY Revenue Growth:</b> <code>{payload.revenue_growth_yoy:+.1%}</code>")
        if payload.eps_surprise_pct is not None:
            lines.append(f"🎯 <b>EPS Surprise:</b> <code>{payload.eps_surprise_pct:.1%}</code>")

        if payload.ai_skeptic_commentary:
            lines.append("")
            lines.append(f"🧠 <b>AI Forensic Sanity Check:</b>\n<i>{esc(payload.ai_skeptic_commentary)}</i>")

        lines.append("")
        lines.append(f"⏰ <i>{esc(payload.timestamp[:19])} UTC</i>")
        return "\n".join(lines)

    @classmethod
    def build_risk_warning_alert(cls, payload: RiskWarningAlertPayload) -> str:
        """Formats a portfolio risk limit warning into Telegram HTML."""
        esc = cls.escape_html
        lines = [
            f"<b>{esc(payload.title)}</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"🚨 <b>Warning Type:</b> <code>{esc(payload.warning_type)}</code>",
        ]

        if payload.ticker:
            lines.append(f"🏷️ <b>Ticker:</b> <code>{esc(payload.ticker)}</code>")
        if payload.sector:
            lines.append(f"🌐 <b>Sector:</b> <code>{esc(payload.sector)}</code>")

        lines.append("")
        lines.append(f"📝 <b>Condition:</b> {esc(payload.details)}")
        lines.append(f"🛡️ <b>Enforced Action:</b> <b>{esc(payload.action_taken)}</b>")

        if payload.current_metrics:
            lines.append("")
            lines.append("📊 <b>Metrics Snapshot:</b>")
            for k, v in payload.current_metrics.items():
                lines.append(f"• {esc(k)}: <code>{esc(v)}</code>")

        lines.append("")
        lines.append(f"⏰ <i>{esc(payload.timestamp[:19])} UTC</i>")
        return "\n".join(lines)


# =====================================================================
# ASYNC TELEGRAM ALERTER ENGINE
# =====================================================================

class TelegramAlerter:
    """
    Production-grade asynchronous Telegram Bot API dispatcher.
    Features:
    - HTML and MarkdownV2 formatted payload parsing
    - Non-blocking async HTTP client via httpx
    - Sensitive token masking in all log output
    - Resilient error handling (rate limits HTTP 429, invalid token, network timeouts)
    - Safe offline/mock fallback during testing or missing configuration
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        settings: Optional[Union[TelegramSettings, Settings]] = None,
        client: Optional[httpx.AsyncClient] = None,
        force_offline: bool = False,
    ):
        # Support either TelegramSettings or general application Settings
        if isinstance(settings, TelegramSettings):
            self.telegram_settings = settings
        else:
            self.telegram_settings = get_telegram_settings()

        # Resolve token securely: explicit arg > telegram_settings > general settings
        resolved_token = token
        if not resolved_token:
            resolved_token = self.telegram_settings.get_token_str()
        if not resolved_token and settings and hasattr(settings, "telegram_bot_token"):
            resolved_token = getattr(settings, "telegram_bot_token")

        self.token: Optional[str] = resolved_token

        # Resolve chat ID
        resolved_chat = chat_id
        if not resolved_chat:
            resolved_chat = self.telegram_settings.telegram_chat_id
        if not resolved_chat and settings and hasattr(settings, "telegram_chat_id"):
            resolved_chat = getattr(settings, "telegram_chat_id")

        self.chat_id: Optional[str] = resolved_chat

        self._custom_client = client
        self.force_offline = force_offline
        self.timeout = self.telegram_settings.request_timeout

    @property
    def is_configured(self) -> bool:
        """True if both token and chat_id are present and not forced offline."""
        return bool(self.token and self.chat_id and not self.force_offline)

    @property
    def masked_token(self) -> str:
        """Returns safe masked token representation for logs."""
        if not self.token:
            return "<UNSET>"
        if len(self.token) <= 8:
            return "***"
        return f"{self.token[:4]}...{self.token[-4:]}"

    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        max_retries: int = 2,
    ) -> TelegramMessageResponse:
        """
        Asynchronously sends a formatted text message to Telegram.

        Args:
            text: Formatted message body.
            chat_id: Optional chat ID override.
            parse_mode: Telegram parse mode ('HTML' or 'MarkdownV2').
            disable_web_page_preview: Suppress link previews.
            max_retries: Retry attempts on transient errors.

        Returns:
            TelegramMessageResponse with delivery status.
        """
        target_chat = chat_id if chat_id is not None else self.chat_id

        # 1. Safe offline / unconfigured mock handler
        if self.force_offline or not self.token:
            logger.info("Telegram notification skipped: Bot unconfigured or running offline.")
            return TelegramMessageResponse(
                success=True,
                message_id=999999,
                chat_id=target_chat or "OFFLINE_CHAT",
                description="Notification logged locally (bot unconfigured or offline mode).",
            )

        # 2. Configured bot with missing destination chat ID
        if not target_chat:
            logger.warning("Telegram dispatch rejected: No chat_id specified.")
            return TelegramMessageResponse(
                success=False,
                error_code=400,
                description="Missing destination chat_id.",
            )

        endpoint = self.BASE_URL.format(token=self.token, method="sendMessage")
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        timeout = httpx.Timeout(self.timeout)
        client = self._custom_client or httpx.AsyncClient(timeout=timeout)
        close_client = self._custom_client is None

        try:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(endpoint, json=payload)
                    status_code = response.status_code

                    # HTTP 200 OK
                    if status_code == 200:
                        data = response.json()
                        result = data.get("result", {})
                        msg_id = result.get("message_id")
                        logger.info("Telegram message successfully sent. message_id=%s", msg_id)
                        return TelegramMessageResponse(
                            success=True,
                            message_id=msg_id,
                            chat_id=target_chat,
                            description="Message delivered successfully.",
                        )

                    # HTTP 429 Too Many Requests (Rate limit)
                    if status_code == 429:
                        data = response.json()
                        retry_after = data.get("parameters", {}).get("retry_after", 3)
                        logger.warning(
                            "Telegram rate limit hit (HTTP 429). retry_after=%s seconds. Attempt %s/%s",
                            retry_after,
                            attempt + 1,
                            max_retries + 1,
                        )
                        if attempt < max_retries:
                            wait_time = min(float(retry_after), self.telegram_settings.rate_limit_max_wait_seconds)
                            await asyncio.sleep(wait_time)
                            continue
                        return TelegramMessageResponse(
                            success=False,
                            error_code=429,
                            retry_after=retry_after,
                            chat_id=target_chat,
                            description=f"Telegram rate limit exceeded (retry after {retry_after}s).",
                        )

                    # HTTP 400 Bad Request / 401 Unauthorized / 403 Forbidden
                    if 400 <= status_code < 500:
                        data = response.json() if "application/json" in response.headers.get("Content-Type", "") else {}
                        err_desc = data.get("description", response.text)
                        logger.error(
                            "Telegram API client error %s (token: %s): %s",
                            status_code,
                            self.masked_token,
                            err_desc,
                        )
                        return TelegramMessageResponse(
                            success=False,
                            error_code=status_code,
                            chat_id=target_chat,
                            description=f"Client error ({status_code}): {err_desc}",
                        )

                    # HTTP 5xx Server Error: retry if attempts remain
                    if status_code >= 500:
                        logger.warning(
                            "Telegram API server error %s. Attempt %s/%s",
                            status_code,
                            attempt + 1,
                            max_retries + 1,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(self.telegram_settings.backoff_factor * (attempt + 1))
                            continue
                        return TelegramMessageResponse(
                            success=False,
                            error_code=status_code,
                            chat_id=target_chat,
                            description=f"Telegram server error ({status_code}).",
                        )

                except httpx.TimeoutException as exc:
                    logger.warning("Telegram API request timed out on attempt %s: %s", attempt + 1, exc)
                    if attempt < max_retries:
                        await asyncio.sleep(1.0)
                        continue
                    return TelegramMessageResponse(
                        success=False,
                        error_code=408,
                        chat_id=target_chat,
                        description=f"Request timeout connecting to Telegram: {exc}",
                    )
                except httpx.RequestError as exc:
                    logger.error("Network error communicating with Telegram API: %s", exc)
                    return TelegramMessageResponse(
                        success=False,
                        error_code=503,
                        chat_id=target_chat,
                        description=f"Network error: {exc}",
                    )

            return TelegramMessageResponse(
                success=False,
                error_code=500,
                chat_id=target_chat,
                description="Max retries exhausted without delivery.",
            )

        finally:
            if close_client:
                await client.aclose()

    async def dispatch_trade_alert(
        self,
        payload: TradeAlertPayload,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> TelegramMessageResponse:
        """Formats and dispatches a high-conviction trade setup alert."""
        mode = parse_mode or self.telegram_settings.default_parse_mode
        text = TelegramMessageBuilder.build_trade_alert(payload, parse_mode=mode)
        return await self.send_message(text=text, chat_id=chat_id, parse_mode=mode)

    async def dispatch_earnings_wreck_alert(
        self,
        payload: EarningsWreckAlertPayload,
        chat_id: Optional[str] = None,
    ) -> TelegramMessageResponse:
        """Formats and dispatches a post-earnings overreaction alert."""
        text = TelegramMessageBuilder.build_earnings_wreck_alert(payload)
        return await self.send_message(text=text, chat_id=chat_id, parse_mode="HTML")

    async def dispatch_risk_warning(
        self,
        payload: RiskWarningAlertPayload,
        chat_id: Optional[str] = None,
    ) -> TelegramMessageResponse:
        """Formats and dispatches a portfolio risk limit warning."""
        text = TelegramMessageBuilder.build_risk_warning_alert(payload)
        return await self.send_message(text=text, chat_id=chat_id, parse_mode="HTML")

    async def dispatch_batch(
        self,
        payloads: List[Union[TradeAlertPayload, EarningsWreckAlertPayload, RiskWarningAlertPayload]],
        chat_id: Optional[str] = None,
    ) -> List[TelegramMessageResponse]:
        """Dispatches multiple alerts asynchronously in parallel."""
        tasks = []
        for p in payloads:
            if isinstance(p, TradeAlertPayload):
                tasks.append(self.dispatch_trade_alert(p, chat_id=chat_id))
            elif isinstance(p, EarningsWreckAlertPayload):
                tasks.append(self.dispatch_earnings_wreck_alert(p, chat_id=chat_id))
            elif isinstance(p, RiskWarningAlertPayload):
                tasks.append(self.dispatch_risk_warning(p, chat_id=chat_id))
            else:
                logger.warning("Unsupported payload type: %s", type(p))
        return await asyncio.gather(*tasks)

    # Synchronous execution convenience helpers
    def send_message_sync(self, text: str, chat_id: Optional[str] = None) -> TelegramMessageResponse:
        """Synchronous wrapper for send_message."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(asyncio.run, self.send_message(text, chat_id)).result()
            return loop.run_until_complete(self.send_message(text, chat_id))
        except RuntimeError:
            return asyncio.run(self.send_message(text, chat_id))

    def dispatch_trade_alert_sync(
        self,
        payload: TradeAlertPayload,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> TelegramMessageResponse:
        """Synchronous wrapper for dispatch_trade_alert."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(
                        asyncio.run,
                        self.dispatch_trade_alert(payload, chat_id, parse_mode)
                    ).result()
            return loop.run_until_complete(self.dispatch_trade_alert(payload, chat_id, parse_mode))
        except RuntimeError:
            return asyncio.run(self.dispatch_trade_alert(payload, chat_id, parse_mode))


def get_telegram_alerter(
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    settings: Optional[Union[TelegramSettings, Settings]] = None,
    force_offline: bool = False,
) -> TelegramAlerter:
    """Factory helper returning TelegramAlerter instance."""
    return TelegramAlerter(token=token, chat_id=chat_id, settings=settings, force_offline=force_offline)
