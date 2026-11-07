# Micro-Module PRD: RiskGuard & Modular UI Dashboard (Module 05)

## 1. Overview & Objective
This module combines volatility-based risk management (`RiskGuard`) with a decoupled backend API architecture designed to feed a modular frontend dashboard (ready for local deployment or future mobile/web app scaling).

## 2. Functional Requirements
- **ATR Volatility Sizing:** Compute 14-period Average True Range ($\text{ATR}(14)$) to size dynamic stops:
  - Stop Loss: $\text{Current Price} - (2.5 \times \text{ATR})$.
  - Take Profit: $\text{Current Price} + (5.0 \times \text{ATR})$ (1:2 Risk/Reward ratio).
- **Exit Guardrails:** Trailing stop-losses (10% from peak) and MA50 volume breakdowns.
- **Modular Dashboard API:** FastAPI endpoints serving JSON payloads of active scan results, AI verdicts, and risk metrics to frontend consumers.

## 3. Data Contracts & Interfaces
### Input Interface:
Consumes qualified candidates post-AI verification.
### Output Interface (`DashboardPayload`):
```json
{
  "active_opportunities": [
    {
      "ticker": "EXAMPLE",
      "score": 88,
      "ai_verdict": "GOLDEN_OPPORTUNITY",
      "risk_metrics": {
        "stop_loss": 38.50,
        "take_profit": 52.50,
        "atr": 1.60
      }
    }
  ]
}
```

## 4. Testing & Atomic Isolation
- **Unit Test Coverage:** Verify ATR calculation precision, stop-loss threshold triggers, and FastAPI endpoint response schemas.