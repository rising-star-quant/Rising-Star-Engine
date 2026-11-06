# Micro-Module PRD: Value Dip & Fundamental Disconnect Engine (Module 02)

## 1. Overview & Objective
The `ValueDisconnectEngine` identifies mid-to-large-cap equities undergoing significant price retracements (mean-reversion setups) while ensuring core business fundamentals remain robust. It enforces a strict "Fundamental Guard" to filter out structurally insolvent companies.

## 2. Functional Requirements
- **Technical Retracement Gate:** Current price must be 15% to 30% below its 52-week high ($P_{\text{current}} \le 0.85 \times P_{52\text{w\_high}}$).
- **Fundamental Disconnect Guard:** Candidates MUST pass all operational criteria:
  - Quarter-over-Quarter or Year-over-Year Revenue Growth $> 0\%$.
  - Trailing 12-month or quarterly Free Cash Flow ($\text{FCF}) > 0$.
  - Debt-to-Equity ratio $< 1.5$.
- **Reversal Momentum Check:** Identification of price stabilization or slowing downside momentum via flattening short-term moving averages.

## 3. Data Contracts & Interfaces
### Input Interface:
Consumes `UniversePayload` records and OHLCV price series.
### Output Interface (`ValueDipCandidate`):
```json
{
  "ticker": "EXAMPLE",
  "current_price": 42.50,
  "distance_from_52w_high_pct": -0.22,
  "fundamental_guard_passed": true,
  "financials": {
    "revenue_growth_qoq": 0.08,
    "free_cash_flow": 12500000,
    "debt_to_equity": 0.65
  },
  "evaluated_at": "2026-11-06T02:15:00Z"
}
```

## 4. Testing & Atomic Isolation
- **Unit Test Coverage:** Test boundary conditions where technical dips are met but fundamental guards fail (e.g., negative FCF), verifying correct rejection.