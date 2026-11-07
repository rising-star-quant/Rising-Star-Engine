# Micro-Module PRD: AI Skeptic Agent Verification Hub (Module 04)

## 1. Overview & Objective
The AI Skeptic Agent acts as an adversarial qualitative sanity-checker ("The Wall Street Skeptic"). It evaluates structured quantitative outputs from the `ValueDipEngine` and `EarningsWreckScanner`, along with recent earnings call summaries, to determine whether a dip represents a genuine overreaction or a structural value trap.

## 2. Functional Requirements
- **Adversarial Persona:** Enforce a skeptical, risk-averse analytical lens to uncover hidden structural headwinds.
- **Structured LLM Prompting:** Construct strict JSON prompt payloads containing quantitative metrics, fundamental snapshots, and recent news snippets.
- **Normalized Scoring & Verdicts:**
  - `verdict`: `[GOLDEN_OPPORTUNITY | FOOLS_TRAP | HIGH_RISK_DIP]`
  - `skepticism_index`: Integer score from 1 (low doubt) to 10 (extreme doubt).
  - `devils_advocate_argument`: Concise textual explanation of primary risks.

## 3. Data Contracts & Interfaces
### Input Interface:
Consumes `ValueDipCandidate` or `EarningsWreckPayload`.
### Output Interface (`AISkepticEvaluation`):
```json
{
  "ticker": "EXAMPLE",
  "verdict": "GOLDEN_OPPORTUNITY",
  "skepticism_index": 3,
  "devils_advocate_argument": "Market overreacted to supply chain guidance cut which is already resolved.",
  "recovery_catalyst": "Next quarter margin expansion confirmation.",
  "evaluated_at": "2026-11-06T02:45:00Z"
}
```

## 4. Testing & Atomic Isolation
- **Unit Test Coverage:** Mock LLM API responses and test schema validation against malformed or out-of-bounds outputs.