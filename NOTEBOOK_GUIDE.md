# SAAM Project 2026 — Notebook Guide & Compliance Review

**Group BR · North America + Europe · Scope 1 + 2**

This document has three parts:
1. **What the notebook does** — a section-by-section walkthrough of `SAAM_Project_2026.ipynb`.
2. **Compliance check** — the project verified against the official brief and our group assignment.
3. **Ideas** — optional extensions that would strengthen the submission.

---

## 1. What the notebook does

The notebook builds **five long-only portfolios** and back-tests them month by
month over **Jan 2014 – Dec 2024 (T = 132 months)**. Allocations are decided
each December from 2013 to 2023 using the previous 10 years (120 months) of
returns, then held for the next year with weights drifting on realised returns.

| Tag | Name | Objective | Carbon constraint |
|---|---|---|---|
| `P_vw` | Value-weighted benchmark | Hold every firm by market cap | none |
| `P_mv` | Minimum variance | Minimise portfolio variance | none |
| `P_mv(0.5)` | Decarbonised min-variance | Minimise variance | CF ≤ 50% of `P_mv` |
| `P_vw(0.5)` | Decarbonised tracker | Minimise tracking error vs `P_vw` | CF ≤ 50% of `P_vw` |
| `P_vw(NZ)` | Net-zero | Minimise tracking error vs `P_vw` | CF cut 10%/year vs 2013 |

### Section-by-section

- **§0 Setup** — imports; constants: `YEARS` = 2013…2023 (allocation years),
  `THETA` = 10% (net-zero rate), plot colours/labels, output folder
  (`report/figures/`).
- **§1 Data loading** — reads the 7 Excel files from `data/`. Monthly series
  (prices, market cap) become `(date × ISIN)` matrices; annual series (CO₂,
  revenue, year-end cap) become `(ISIN × year)` matrices. Scope 1 + Scope 2
  emissions are **summed** (our assigned scope); carbon and revenue are
  **forward-filled** across years. Risk-free rate converted to a monthly decimal.
- **§2 Cleaning & returns** — drops fully-empty ISINs (unmatched in Datastream);
  sets any price below 0.5 to missing; computes simple monthly returns; books a
  −100% return the month after a firm's last valid price (**112 delistings**).
- **§3 Investment set** — `get_eligible(Y)` keeps firms that are in-region, have
  ≥ 36 monthly returns over the 10-year window, ≤ 50% stale (zero) returns, a
  valid year-end price and cap, **and** carbon + revenue data. `year_data(Y)`
  then builds the sample covariance matrix (+ a `1e-8` ridge so the QP is
  stable), caps, emissions, carbon intensity, carbon-per-cap, and VW weights.
- **§4 Optimisation building blocks** — `_solve` runs a long-only quadratic
  program (CLARABEL, OSQP fallback); `min_variance` and `min_tracking_error`
  add the optional linear carbon-footprint cap; `cf()` and `waci()` compute the
  portfolio carbon metrics.
- **§5 Part I** — `backtest_vw` (cap-weighted, rebalanced on the previous
  month's caps) and `backtest_optimized` (annual rebalance + monthly weight
  drift); builds `P_vw` and `P_mv`, reports summary statistics and the
  cumulative-performance figure.
- **§6 Part II §3.1** — carbon profile (WACI and CF) of the two base
  portfolios, plus the **top-10 firms** driving the benchmark's intensity.
- **§7 Part II §3.2** — `P_mv(0.5)`: min-variance with CF ≤ 50% of `P_mv`.
- **§8 Part II §3.3** — `P_vw(0.5)`: tracking-error min with CF ≤ 50% of `P_vw`.
- **§9 Part II §4** — `P_vw(NZ)`: tracking-error min on a 10%-per-year glide path.
- **§10 Consolidated results** — performance table for all five portfolios,
  average carbon metrics, effective number of holdings, and the export of
  `Part_II_Results.xlsx`.
- **§11 Extension — cost-of-decarbonisation frontier** — sweeps the footprint
  cap on the otherwise-passive tracking portfolio and plots Sharpe ratio and
  tracking error against the achieved footprint reduction (beyond the brief).

Every table and figure in the report is produced here; figures are written to
`report/figures/` (the 10 report figures plus the extension figure).

---

## 2. Compliance check — brief & group

### Group assignment ✓

`Groups_Strategy_2026` confirms **Group BR = North America + Europe / Scope 1+2**.
The notebook filters `Region ∈ {AMER, EUR}` and sums Scope 1 + Scope 2 — correct.

### Requirements

| Brief requirement | What the project does | Status |
|---|---|---|
| Region & scope assigned to the group | AMER + EUR, Scope 1+2 | ✓ |
| Sample: first alloc. Dec 2013, last Dec 2023, plots Jan 2014–Dec 2024 (per instructor's correction note) | `YEARS` 2013–2023, 132 months — correction applied | ✓ |
| Drop unmatched ISINs; price < 0.5 → missing; simple returns; −100% on delisting | All implemented (§2) | ✓ |
| Carbon/revenue: forward-fill gaps; no data → not investable | `.ffill`; eligibility requires carbon + revenue | ✓ |
| Investment set: region, ≥ 3 yrs returns, stale-price filter, carbon data; same set for both parts | ≥ 36 obs, ≤ 50% stale, single `get_eligible` | ✓ |
| Estimators: sample mean & covariance over τ = 120 months | Implemented (+ ridge for the QP) | ✓ |
| Min-variance: long-only QP, weights sum to 1, α ≥ 0 | `min_variance` via CLARABEL/OSQP | ✓ |
| VW benchmark on previous-month caps | `backtest_vw` uses prior-month caps | ✓ |
| Annual rebalancing + monthly weight drift | `backtest_optimized` | ✓ |
| Carbon intensity (revenue ÷ 1000), WACI, CF; top-10 emitters w/ ISIN | §6, top-10 table with names + ISIN | ✓ |
| `P_mv(0.5)`, `P_vw(0.5)`, `P_vw(NZ)` exactly as specified | §7–§9 | ✓ |
| Report: implementation, tables/figures, interpretation, improvements, **5 limitation bullets**, ≤ 30 pages | `report.tex` has all sections incl. 5 bullets | ✓ (verify page count) |
| Single notebook, runs top to bottom, relative paths | Relative paths only; reproduces all outputs | ✓ (see open items) |
| Sales pitch (1 page) | `sales_pitch.tex` | ✓ |
| Video (10 min) | Script ready, **not yet recorded** | ⚠ |

### Notes on judgement calls (defensible, worth knowing)

- **Net-zero objective.** The brief's §4 intro mentions "minimum variance", but
  §4.1 explicitly says *"the same as Section 3.3"* (tracking-error / otherwise-
  passive investor). The notebook follows §4.1 — the explicit instruction.
- **Year-end cap.** The brief supplies an annual cap file; the notebook uses the
  December value of the *monthly* cap series instead. Numerically equivalent.
- **Covariance normalisation.** pandas `.cov()` divides by (τ−1); the brief
  writes 1/τ. Irrelevant — a constant scaling does not change the optimiser.

### Open items before 29 May

- [ ] **Author names** — `report.tex` and `sales_pitch.tex` still show only
      "Group BR". Add the three members' names (the project is graded per group
      with peer evaluation).
- [ ] **Record the video** — script is in `report/video_script.md`.
- [ ] **Re-run the notebook end to end** — recent cell edits cleared two cells'
      outputs; run all cells once more so every figure/table is embedded.
- [ ] **Add `jupyter` to `requirements.txt`** — the brief asks for "no missing
      dependencies"; currently only the analysis packages are listed.
- [ ] **Verify the compiled report PDF is ≤ 30 pages.**
- [ ] Quick sanity check: confirm `AMER` in `Static_2025.xlsx` means North
      America (developed), not all of the Americas.

---

## 3. Ideas to strengthen the project (optional)

The project already meets the brief — these are polish, not gaps. Ranked by
value-for-effort with eight days left:

**Already implemented** — the **carbon-cost frontier** (notebook §11): the
footprint cap on the tracking portfolio is swept and Sharpe ratio / tracking
error are plotted against the achieved reduction. To use the figure in the
report, add `carbon_cost_frontier.png` on Overleaf.

**High value, low effort**
- **Turnover & transaction costs.** Report annual turnover per portfolio and
  apply a simple cost (e.g. 10–20 bps); the carbon-constrained portfolios may
  trade more as the high-intensity name list rotates.

**Medium value, medium effort**
- **Covariance shrinkage.** Replace the raw sample covariance with Ledoit–Wolf
  shrinkage as a robustness appendix — it most affects `P_mv`/`P_mv(0.5)`, whose
  concentration is an estimation-error artefact.
- **Parameter sensitivity grid.** Re-run with stale thresholds 30/70%, minimum
  observations 24/60, and 5/7-year windows; tabulate how Sharpe and CF move.

**Nice to have**
- **Net-zero design variants.** Anchor the glide path to a rolling benchmark, or
  add a "50% of current" floor, to make `P_vw(NZ)` genuinely demanding.
- **Sector view.** The dataset has no sector IDs; mapping ISINs to GICS sectors
  would turn the country-level commentary into a proper sector-tilt analysis.
- **Scope 3 discussion.** A short note on how excluding Scope 3 understates
  footprints, especially for oil majors.

> Suggestion: with the carbon-cost frontier already in place, add at most
> **one** more item (turnover & transaction costs is the most natural next
> step) rather than spreading effort thin this close to the deadline.
