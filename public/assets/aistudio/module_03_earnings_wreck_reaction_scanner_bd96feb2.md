# Micro-Module PRD: Earnings Wreck & Reaction Scanner (Module 03)

## 1. Overview & Objective
The `EarningsReactionScanner` is an event-driven parallel pipeline designed to detect acute post-earnings market overreactions. It isolates companies that experienced severe single-day or multi-day sell-offs due to minor analyst consensus misses despite maintaining healthy underlying revenues.

## 2. Functional Requirements
- **Earnings Window Tracking:** Monitor companies filing quarterly SEC reports (10-Q/10-K) within the last 1 to 3 trading days.
- **Acute Drop Threshold:** Single-day price drop $\ge 8\%$ or 3-day cumulative drop $\ge 15\%$.
- **Volume Surge Verification:** Relative Volume ($\text{RVOL}$) on the drop day $\ge 2.0 \times$ 30-day average volume, confirming institutional liquidation or panic selling.
- **Revenue Integrity Check:** Verify that core quarterly revenue did not drop linearly with the stock price.

## 3. Data Contracts & Interfaces
### Input Interface:
Consumes daily earnings calendar feeds and OHLCV market series.
### Output Interface (`EarningsWreckPayload`):
```json
{
  "ticker": "WRECK",
  "earnings_date": "2026-11-04",
  "price_change_1d_pct": -0.14,
  "relative_volume": 3.2,
  "revenue_integrity_passed": true,
  "evaluated_at": "2026-11-06T02:30:00Z"
}
```

## 4. Testing & Atomic Isolation
- **Unit Test Coverage:** Test historical earnings drop scenarios with simulated RVOL surges and revenue metrics.