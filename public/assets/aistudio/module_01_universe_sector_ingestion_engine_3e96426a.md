# Micro-Module PRD: Universe & Sector Ingestion Engine (Module 01)

## 1. Overview & Objective
The Universe Ingestion Engine dynamically fetches, merges, and filters constituent tickers from broad institutional indices (**S&P 400 Mid-Cap**, **Russell 1000**, and **Russell 2000 Small-Cap**). It establishes strict liquidity and market capitalization baseline gates before passing clean candidate arrays to downstream screening pipelines.

## 2. Functional Requirements
- **Multi-Index Ingestion:** Automated retrieval of ticker symbols and sector classifications from S&P 400, Russell 1000, and Russell 2000.
- **Deduplication Layer:** Automatic removal of duplicate tickers across overlapping index definitions.
- **Liquidity & Market Cap Gates:**
  - Minimum Average Daily Volume ($\text{ADV} \ge \$2\text{M}$ over 30 days).
  - Minimum Market Capitalization ($\ge \$300\text{M}$).
- **Sector Filtering:** Tagging and filtering equities by GICS sectors (e.g., Technology, Healthcare, Industrials) to enable sector-wide drag analysis.

## 3. Data Contracts & Interfaces
### Input Interface:
Configuration payload defining active indices and sector filters.
### Output Interface (`UniversePayload`):
```json
{
  "execution_timestamp": "2026-11-06T02:00:00Z",
  "total_symbols_scanned": 3250,
  "qualified_symbols_count": 1420,
  "constituents": [
    {
      "ticker": "EXAMPLE",
      "index_source": "RUSSELL_1000",
      "sector": "Technology",
      "market_cap": 2500000000,
      "avg_daily_volume": 4500000
    }
  ]
}
```

## 4. Testing & Atomic Isolation
- **Unit Test Coverage:** Verify deduplication logic, liquidity boundary filters, and sector tag assignment using mock index arrays without external API calls.