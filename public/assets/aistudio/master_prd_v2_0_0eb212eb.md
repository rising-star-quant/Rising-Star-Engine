# Master Product Requirements Document (Master PRD v2.0)
# Value Dip & Fundamental Disconnect Engine

## 1. Executive Summary & Strategic Pivot
The **Value Disconnect Engine (v2.0)** is an advanced algorithmic screening and verification system designed to identify high-quality mid-to-large-cap equities experiencing acute market overreactions ("Value Dips" and "Earnings Wrecks"). Pivoting from legacy momentum-breakout models, v2.0 combines rigorous fundamental viability guards with adversarial AI verification to exploit mean-reversion anomalies across the S&P 400, Russell 1000, and Russell 2000 indices.

## 2. System Architecture & Modular Data Flow
The system operates on an event-driven, decoupled micro-module architecture where each feature is fully atomic and independently testable:

```
[External Data Providers: yfinance, FMP, SEC EDGAR]
                        |
                        v
        +-------------------------------+
        |  1. Universe & Sector Engine  |  <-- S&P 400, Russell 1000/2000, ADV Filters
        +-------------------------------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
    +---------------+       +------------------+
    | 2. Value Dip  |       | 3. Earnings Wreck|  <-- Parallel Screening Pipelines
    |    Engine     |       |     Scanner      |
    +---------------+       +------------------+
            |                       |
            +-----------+-----------+
                        |
                        v
        +-------------------------------+
        |   4. AI Skeptic Agent Hub     |  <-- Adversarial LLM Verification & Scoring
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  5. RiskGuard & Modular UI    |  <-- ATR Stop Loss, Take Profit & Dashboard API
        +-------------------------------+
                        |
                        v
        +-------------------------------+
        |  6. Notification & Dispatch   |  <-- Telegram Alerts & Idempotency Cache
        +-------------------------------+
```

## 3. Core Modules Index & Atomic Boundaries
- **Module 01 (`prd_universe_scanner.md`):** Dynamic ingestion and sector-based filtering across S&P 400, Russell 1000, and Russell 2000.
- **Module 02 (`prd_value_dip_engine.md`):** Technical retracement gate (15-30% below 52W high) coupled with Fundamental Disconnect Guard (Positive Revenue Growth & Free Cash Flow).
- **Module 03 (`prd_earnings_scanner.md`):** Event-driven post-earnings overreaction scanner capturing acute price drops ($\ge 8\%$ 1-day or $\ge 15\%$ 3-day) with high relative volume.
- **Module 04 (`prd_ai_skeptic.md`):** LLM-powered adversarial verification agent returning structured verdicts (`GOLDEN_OPPORTUNITY` vs `FOOLS_TRAP`) and skepticism indexes.
- **Module 05 (`prd_riskguard_dashboard.md`):** ATR-based volatility risk management, dynamic stop-loss/take-profit sizing, and decoupled backend API supporting the modular frontend dashboard.