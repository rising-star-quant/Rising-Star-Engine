"""
Rising Star Trend Engine v2.0 - AI Skeptic Engine (Module 03 / Module 07)
Asynchronous adversarial verification module acting as Devil's Advocate against candidate breakout/dip signals.
Evaluates candidates for accounting irregularities, structural secular decay, litigation, and leverage traps.
Powered by Google Gemini 3.8 Flash with deterministic offline fallback for offline CI/CD resilience.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.engines.value_dip_engine import CompanyFinancials, ValueDipCandidate

logger = logging.getLogger("rising_star.ai_skeptic")


class SkepticAnalysisRequest(BaseModel):
    """
    Input schema for adversarial candidate audit.
    """
    ticker: str = Field(..., description="Equity ticker symbol")
    current_price: float = Field(..., description="Current market share price")
    distance_from_52w_high_pct: float = Field(
        ...,
        description="Distance from 52-week peak as decimal (e.g. -0.22)"
    )
    sector: Optional[str] = Field(default="Unknown", description="Economic sector")
    industry: Optional[str] = Field(default="Unknown", description="Industry segment")
    financials: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of fundamental ratios (revenue growth, FCF, D/E)"
    )
    news_headlines: Optional[List[str]] = Field(
        default_factory=list,
        description="Recent news headlines or SEC filing catalysts"
    )
    additional_notes: Optional[str] = Field(
        default=None,
        description="Optional qualitative context or earnings commentary"
    )

    @classmethod
    def from_value_dip_candidate(
        cls,
        candidate: ValueDipCandidate,
        sector: str = "Unknown",
        industry: str = "Unknown",
        news_headlines: Optional[List[str]] = None,
    ) -> "SkepticAnalysisRequest":
        """Factory constructor converting ValueDipCandidate to SkepticAnalysisRequest."""
        return cls(
            ticker=candidate.ticker,
            current_price=candidate.current_price,
            distance_from_52w_high_pct=candidate.distance_from_52w_high_pct,
            sector=sector,
            industry=industry,
            financials=candidate.financials.model_dump(),
            news_headlines=news_headlines or [],
        )


class SkepticVerdict(BaseModel):
    """
    Structured adversarial assessment emitted by AI Skeptic.
    """
    ticker: str = Field(..., description="Audited equity ticker symbol")
    verdict: Literal["APPROVED", "REJECTED"] = Field(
        ...,
        description="Binary decision: APPROVED permits capital allocation; REJECTED blocks trade"
    )
    skepticism_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Adversarial risk score (0.0=zero red flags, 1.0=unmitigated trap)"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="Catalog of detected vulnerabilities, governance risks, or financial distress"
    )
    justification: List[str] = Field(
        default_factory=list,
        description="Point-by-point rationales supporting the final verdict"
    )
    counter_thesis: Optional[str] = Field(
        default=None,
        description="Bear thesis articulating why this apparent dip is a value trap"
    )
    confidence_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Model confidence level in verdict"
    )
    is_mock: bool = Field(
        default=False,
        description="True if evaluated via offline deterministic heuristic fallback"
    )
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO 8601 audit timestamp"
    )


class AISkeptic:
    """
    Asynchronous AI Skeptic engine.
    Uses Google Gemini 3.8 Flash (via Google GenAI REST) to actively challenge bullish setups.
    Includes a deterministic offline heuristic validator for test runners and zero-network environments.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        force_offline: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings or get_settings()
        self.force_offline = force_offline or (self.settings.env == "testing")
        self._custom_client = client

    def _normalize_request(
        self,
        candidate: Union[SkepticAnalysisRequest, ValueDipCandidate, Dict[str, Any]],
    ) -> SkepticAnalysisRequest:
        """Converts diverse input types into a standardized SkepticAnalysisRequest."""
        if isinstance(candidate, SkepticAnalysisRequest):
            return candidate
        if isinstance(candidate, ValueDipCandidate):
            return SkepticAnalysisRequest.from_value_dip_candidate(candidate)
        if isinstance(candidate, dict):
            # Parse dict
            return SkepticAnalysisRequest(**candidate)
        raise TypeError(f"Unsupported candidate input type: {type(candidate)}")

    def _evaluate_deterministic_heuristic(
        self,
        request: SkepticAnalysisRequest,
        rejection_threshold: float,
    ) -> SkepticVerdict:
        """
        Deterministic, offline rule-based forensic evaluator.
        Provides 100% reproducible adversarial audit during unit tests, CI/CD, or API outages.
        """
        red_flags: List[str] = []
        justification: List[str] = []
        score = 0.15  # Baseline benign skepticism

        # 1. Analyze Debt-to-Equity & Balance Sheet
        financials = request.financials or {}
        de = financials.get("debt_to_equity")
        if de is not None:
            if de > 1.8:
                red_flags.append(f"Severe balance sheet leverage: D/E ratio at {de:.2f} (ceiling 1.5)")
                score += 0.45
            elif de > 1.3:
                red_flags.append(f"Elevated debt load: D/E ratio at {de:.2f}")
                score += 0.20
            else:
                justification.append(f"Prudent capital structure: D/E of {de:.2f} well within safety bounds.")

        # 2. Analyze Free Cash Flow
        fcf = financials.get("free_cash_flow")
        if fcf is not None:
            if fcf < 0:
                red_flags.append(f"Negative Free Cash Flow (${fcf:,.0f}): indicates structural cash burn.")
                score += 0.35
            else:
                justification.append(f"Positive Free Cash Flow (${fcf:,.0f}) supports solvency through downturn.")

        # 3. Analyze Revenue Growth (Top-line integrity)
        rev_growth = financials.get("revenue_growth_qoq")
        if rev_growth is not None:
            if rev_growth < -0.05:
                red_flags.append(f"Contracting revenue growth ({rev_growth:.1%}): suggests secular market share loss.")
                score += 0.30
            elif rev_growth <= 0.0:
                red_flags.append(f"Flat/stagnant top-line growth ({rev_growth:.1%}).")
                score += 0.15
            else:
                justification.append(f"Top-line expansion intact: Revenue growth at +{rev_growth:.1%}.")

        # 4. Severe Drawdown Guard (Deeper than -28% approaches structural breakdown)
        if request.distance_from_52w_high_pct <= -0.28:
            red_flags.append(
                f"Severe price collapse of {request.distance_from_52w_high_pct:.1%} "
                "approaches structural trend invalidation."
            )
            score += 0.25

        # 5. Adverse Headline Keyword Scanning
        adverse_keywords = [
            "lawsuit", "investigation", "sec probe", "subpoena", "restatement",
            "fraud", "accounting error", "default", "bankruptcy", "covenant breach",
            "executive resignation", "cfo departure", "ceo ousted", "downgrade"
        ]
        detected_adverse = []
        for headline in request.news_headlines or []:
            lower_h = headline.lower()
            for kw in adverse_keywords:
                if kw in lower_h:
                    detected_adverse.append(f"Adverse disclosure detected: '{kw}' in headline: {headline}")
                    score += 0.35
                    break
        red_flags.extend(detected_adverse)

        # Normalize score
        final_score = min(1.0, round(score, 2))
        verdict: Literal["APPROVED", "REJECTED"] = "REJECTED" if final_score >= rejection_threshold else "APPROVED"

        if verdict == "REJECTED":
            justification.insert(
                0,
                f"REJECTED: Skepticism score {final_score:.2f} meets or exceeds risk threshold {rejection_threshold:.2f}."
            )
            counter_thesis = (
                f"Candidate {request.ticker} exhibits notable vulnerabilities ({', '.join(red_flags[:2])}). "
                "Market selloff appears justified rather than an overreaction. Capital protection warrants rejection."
            )
        else:
            justification.insert(
                0,
                f"APPROVED: Skepticism score {final_score:.2f} is well below threshold {rejection_threshold:.2f}. "
                "Fundamentals demonstrate resilience."
            )
            counter_thesis = (
                f"While {request.ticker} has corrected {request.distance_from_52w_high_pct:.1%}, core revenue "
                "and cash flows remain defensible. Downside appears to be a cyclical dip rather than insolvency."
            )

        return SkepticVerdict(
            ticker=request.ticker,
            verdict=verdict,
            skepticism_score=final_score,
            red_flags=red_flags,
            justification=justification,
            counter_thesis=counter_thesis,
            confidence_score=0.88,
            is_mock=True,
        )

    def _build_adversarial_prompt(self, request: SkepticAnalysisRequest) -> str:
        """Constructs detailed adversarial evaluation prompt for Gemini LLM."""
        return (
            f"You are the Adversarial Skeptic and Lead Forensic Analyst for a quantitative trend fund.\n"
            f"Analyze the following candidate stock that has triggered a Value Dip setup:\n\n"
            f"Ticker: {request.ticker}\n"
            f"Current Price: ${request.current_price:.2f}\n"
            f"Retracement from 52-Week Peak: {request.distance_from_52w_high_pct:.1%}\n"
            f"Sector: {request.sector}\n"
            f"Industry: {request.industry}\n"
            f"Fundamental Indicators: {json.dumps(request.financials or {}, indent=2)}\n"
            f"News / Catalysts: {json.dumps(request.news_headlines or [], indent=2)}\n\n"
            f"TASK:\n"
            f"1. Play Devil's Advocate. Uncover reasons why this dip is NOT an overreaction but a value trap.\n"
            f"2. Check for accounting irregularities, debt maturity cliffs, litigation, technological displacement, or margin decay.\n"
            f"3. Assign a skepticism_score between 0.0 (pristine/safe) and 1.0 (deadly value trap).\n"
            f"4. If skepticism_score >= 0.60, set verdict to 'REJECTED'. Otherwise 'APPROVED'.\n"
            f"5. Provide concrete red_flags and concise bullet justifications.\n"
        )

    async def _call_gemini_api(
        self,
        request: SkepticAnalysisRequest,
        api_key: str,
    ) -> SkepticVerdict:
        """
        Calls Gemini 3.8 Flash via Google GenAI REST endpoint with JSON Schema enforcement.
        """
        model = self.settings.gemini_model or "gemini-3.8-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        prompt = self._build_adversarial_prompt(request)

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "ticker": {"type": "STRING"},
                        "verdict": {
                            "type": "STRING",
                            "enum": ["APPROVED", "REJECTED"]
                        },
                        "skepticism_score": {"type": "NUMBER"},
                        "red_flags": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "justification": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"}
                        },
                        "counter_thesis": {"type": "STRING"},
                        "confidence_score": {"type": "NUMBER"}
                    },
                    "required": [
                        "ticker",
                        "verdict",
                        "skepticism_score",
                        "red_flags",
                        "justification"
                    ]
                }
            }
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "aistudio-build",
        }

        timeout = httpx.Timeout(self.settings.request_timeout)
        client = self._custom_client or httpx.AsyncClient(timeout=timeout)

        close_client_after = self._custom_client is None
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini API returned no candidates in response payload")

            content_parts = candidates[0].get("content", {}).get("parts", [])
            if not content_parts or "text" not in content_parts[0]:
                raise ValueError("Gemini response missing text in content part")

            raw_text = content_parts[0]["text"]
            parsed = json.loads(raw_text)

            # Ensure ticker aligns
            parsed["ticker"] = request.ticker
            # Ensure score within bounds
            score = float(parsed.get("skepticism_score", 0.5))
            parsed["skepticism_score"] = max(0.0, min(1.0, score))
            parsed["is_mock"] = False

            return SkepticVerdict(**parsed)
        finally:
            if close_client_after:
                await client.aclose()

    async def evaluate_candidate(
        self,
        candidate: Union[SkepticAnalysisRequest, ValueDipCandidate, Dict[str, Any]],
        rejection_threshold: Optional[float] = None,
    ) -> SkepticVerdict:
        """
        Asynchronously evaluates a candidate setup using Gemini LLM with offline fallback.

        Args:
            candidate: Candidate object, request schema, or dictionary.
            rejection_threshold: Threshold above which verdict is REJECTED (defaults to settings).

        Returns:
            Structured SkepticVerdict.
        """
        request = self._normalize_request(candidate)
        threshold = (
            rejection_threshold
            if rejection_threshold is not None
            else self.settings.ai_skeptic_rejection_threshold
        )

        api_key = self.settings.gemini_api_key

        # If offline forced, in testing mode, or API key missing, run deterministic heuristic
        if self.force_offline or not api_key:
            logger.info("Evaluating candidate %s via deterministic offline heuristic.", request.ticker)
            return self._evaluate_deterministic_heuristic(request, threshold)

        try:
            logger.info("Calling Gemini API (%s) for %s adversarial verification.", self.settings.gemini_model, request.ticker)
            return await self._call_gemini_api(request, api_key)
        except Exception as exc:
            logger.warning(
                "Gemini API request failed for %s (%s). Falling back to deterministic heuristic.",
                request.ticker,
                exc,
            )
            verdict = self._evaluate_deterministic_heuristic(request, threshold)
            verdict.justification.append(f"Offline fallback engaged due to API error: {exc}")
            return verdict

    async def evaluate_batch(
        self,
        candidates: List[Union[SkepticAnalysisRequest, ValueDipCandidate, Dict[str, Any]]],
        rejection_threshold: Optional[float] = None,
    ) -> List[SkepticVerdict]:
        """
        Evaluates a batch of candidates concurrently.
        """
        tasks = [
            self.evaluate_candidate(c, rejection_threshold=rejection_threshold)
            for c in candidates
        ]
        return await asyncio.gather(*tasks)

    def evaluate_candidate_sync(
        self,
        candidate: Union[SkepticAnalysisRequest, ValueDipCandidate, Dict[str, Any]],
        rejection_threshold: Optional[float] = None,
    ) -> SkepticVerdict:
        """Synchronous helper for CLI scripts or non-async pipelines."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already running loop: run via new event loop or background runner
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(
                        asyncio.run,
                        self.evaluate_candidate(candidate, rejection_threshold)
                    ).result()
            return loop.run_until_complete(
                self.evaluate_candidate(candidate, rejection_threshold)
            )
        except RuntimeError:
            return asyncio.run(self.evaluate_candidate(candidate, rejection_threshold))

    def evaluate_batch_sync(
        self,
        candidates: List[Union[SkepticAnalysisRequest, ValueDipCandidate, Dict[str, Any]]],
        rejection_threshold: Optional[float] = None,
    ) -> List[SkepticVerdict]:
        """Synchronous batch evaluation helper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(
                        asyncio.run,
                        self.evaluate_batch(candidates, rejection_threshold)
                    ).result()
            return loop.run_until_complete(
                self.evaluate_batch(candidates, rejection_threshold)
            )
        except RuntimeError:
            return asyncio.run(self.evaluate_batch(candidates, rejection_threshold))


def get_ai_skeptic(
    settings: Optional[Settings] = None,
    force_offline: bool = False,
) -> AISkeptic:
    """Factory helper returning AISkeptic instance."""
    return AISkeptic(settings=settings, force_offline=force_offline)
