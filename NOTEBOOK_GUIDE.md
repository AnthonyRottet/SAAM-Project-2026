# SAAM Project 2026 — Notebook Guide & Compliance Review

**Group BR · North America + Europe · Scope 1 + 2**

This document has three parts:
1. **Cell-by-cell walkthrough** — what every cell of `SAAM_Project_2026.ipynb`
   does and *why* we do it that way.
2. **Compliance check** — the project verified against the official brief and
   our group assignment.
3. **Ideas** — optional extensions that would strengthen the submission.

It is meant as a working reference for the group (also useful for the video and
for answering questions on methodology).

---

## 1. Cell-by-cell walkthrough

The notebook builds **five long-only portfolios** and back-tests them month by
month over **Jan 2014 – Dec 2024 (T = 132 months)**:

| Tag | Name | Objective | Carbon constraint |
|---|---|---|---|
| `P_vw` | Value-weighted benchmark | Hold every firm by market cap | none |
| `P_mv` | Minimum variance | Minimise portfolio variance | none |
| `P_mv(0.5)` | Decarbonised min-variance | Minimise variance | CF ≤ 50% of `P_mv` |
| `P_vw(0.5)` | Decarbonised tracker | Minimise tracking error vs `P_vw` | CF ≤ 50% of `P_vw` |
| `P_vw(NZ)` | Net-zero | Minimise tracking error vs `P_vw` | CF cut 10%/year vs 2013 |

The core logic: each December (2013–2023) we decide weights using the previous
10 years (120 months) of returns, then hold them for the next year while they
drift with realised prices. Performance is measured monthly.

### §0 — Setup

**Code cell — imports & global constants.**
- *What.* Imports `numpy`, `pandas`, `matplotlib`, and `cvxpy` (the convex
  optimiser, aliased `cp`). Sets a clean plot style and silences warnings so the
  notebook output stays readable. Defines `DATA_DIR = "data/"` and
  `PLOTS_DIR = "report/figures/"`, and creates the figures folder if missing.
  Defines `YEARS = 2013…2023`, `THETA = 0.10`, and the plotting dictionaries
  `COL` (colours) and `LABEL` (portfolio names).
- *Why.* Paths are **relative**, never absolute — the brief requires the grader
  to run the notebook anywhere with no path edits. `YEARS` is the list of 11
  **allocation years**: the first allocation is December 2013 and the last is
  December 2023, exactly as fixed by the instructor's correction note. `THETA`
  is the 10%-per-year decarbonisation rate used by the net-zero portfolio.

### §1 — Data loading and reshaping

**Code cell — load and reshape the Datastream files.**
- *What.* Reads `Static_2025.xlsx` (the firm reference table) and builds
  `name_map` / `country_map` (ISIN → name / country). `our_isins` keeps only
  firms whose `Region` is `AMER` or `EUR`. Two helper functions then load the
  time-series files: `load_and_reshape` for monthly files and `load_yearly` for
  annual files. It loads monthly prices (`prices`, the total-return index) and
  market caps (`mktcap`), annual Scope 1 and Scope 2 emissions, revenue, and the
  risk-free rate.
- *Why.*
  - `drop_duplicates("ISIN")` — Datastream occasionally repeats an ISIN; we keep
    one row per firm.
  - **Region filter** — Group BR is assigned North America + Europe, so only
    `AMER` and `EUR` firms enter the universe (1,302 firms).
  - **Transpose** — Datastream delivers firms as rows; we transpose to a
    `(date × ISIN)` matrix so every column is one firm's time series, which is
    the natural layout for returns and covariance.
  - **`co2` = Scope 1 + Scope 2**, summed — our group's assigned perimeter.
  - **`.ffill(axis=1)`** on carbon and revenue — the brief says a missing annual
    value in the middle or at the end of the sample should be replaced by the
    most recent available figure (carry forward).
  - The risk-free rate is divided by `100 / 12` to turn an annualised percentage
    into a **monthly decimal**, so it is on the same scale as monthly returns.

### §2 — Data cleaning and returns

**Code cell — cleaning rules and return construction.**
- *What.* Drops ISIN columns that are entirely empty; sets any price below 0.5
  to missing; computes simple monthly returns; then loops over firms and books a
  −100% return the month after a firm's last valid price.
- *Why.*
  - **Drop fully-empty columns** — these are ISINs Datastream could not match to
    a security; the brief says delete them.
  - **Price < 0.5 → missing** — near-zero prices make returns explode to
    nonsensical values; the brief recommends treating them as missing.
  - **Simple returns** (`pct_change`) — the brief specifies simple, not log,
    returns.
  - **−100% on delisting** — when a firm's price series ends before the sample
    end, that is a default/delisting: the investor loses everything. Booking
    −100% (rather than silently dropping the firm) avoids survivorship bias.
    The cell reports **112 delistings**.

### §3 — Investment set

**Code cell — `dec`, `get_eligible`, `year_data`.**
- *What.* `dec(year, index)` returns the December month-end date for a year.
  `get_eligible(year)` returns the list of firms investable over the *next*
  year. `year_data(year)` packages everything the optimiser needs for that
  allocation. `yd` precomputes this for all years; the cell prints the
  investment-set size per year (835 → 1,150).
- *Why — `get_eligible` criteria.* A firm enters the set only if it:
  1. is in the assigned region;
  2. has **≥ 36** monthly returns over the 10-year window — enough data to
     estimate moments (brief: at least 3 years);
  3. has **≤ 50%** stale (exactly-zero) returns — a stale/illiquid stock has
     artificially low volatility, and an unconstrained min-variance optimiser
     would pile weight into it; this filter removes that trap;
  4. has a valid year-end price and a positive market cap;
  5. has **both carbon and revenue data**. The brief insists the *same*
     investment set is used in Part I and Part II, so carbon intensity and
     footprint must be defined for every firm.
- *Why — `year_data`.*
  - `cov` — the sample covariance of eligible firms' returns over the 120-month
    window. Missing returns are filled with 0 first, then a tiny ridge
    `1e-8 · I` is added. Filling gaps gives a consistent, positive-semidefinite
    matrix; the ridge makes it strictly **positive-definite**. This matters
    because there are ~1,000 firms but only 120 observations, so the raw
    covariance is rank-deficient (singular) — the ridge keeps the quadratic
    program numerically stable.
  - `ci = emis / rev_m` — carbon intensity, with revenue divided by 1,000 first
    (the brief: revenue is in *thousands*, intensity needs *millions*).
  - `carb = emis / cap` — emissions per dollar of market cap; this is the
    per-firm building block of the portfolio carbon footprint.
  - `w_vw = cap / cap.sum()` — value-weighted benchmark weights.

### §4 — Optimisation building blocks

**Code cell — `_solve`, `min_variance`, `min_tracking_error`, `cf`, `waci`.**
- *What.* `_solve` is a generic long-only quadratic-program (QP) solver:
  variable `w`, constraints `sum(w) = 1` and `w ≥ 0`, plus any extra constraint.
  `min_variance` minimises `wᵀΣw`; `min_tracking_error` minimises
  `(w − w_vw)ᵀ Σ (w − w_vw)`. Both accept an optional carbon-footprint cap.
  `cf` and `waci` compute a portfolio's footprint and weighted-average carbon
  intensity.
- *Why.*
  - A **QP, not a closed-form solution** — because Σ is singular and we impose
    `w ≥ 0`, the textbook inverse-covariance formula does not apply; we solve
    numerically.
  - **Long-only** (`w ≥ 0`) — the brief keeps weights non-negative so the carbon
    footprint is unambiguous (no negative "carbon credits" from short
    positions).
  - **Two solvers** — it tries CLARABEL first, then OSQP as a fallback, for
    numerical robustness; tiny negative weights are clipped to 0 and the vector
    is renormalised.
  - The **carbon cap is linear** in the weights (`carb · w ≤ limit`), so adding
    it keeps the problem a convex QP — fast and reliable.

### §5 — Part I: standard portfolio allocation

**Code cell — `backtest_optimized`, `backtest_vw`, build `P_mv`.**
- *What.* `backtest_optimized` simulates an annually-rebalanced portfolio:
  start each year at the optimal weights, record the monthly return, then let
  the weights drift with realised returns. `backtest_vw` simulates the
  value-weighted benchmark. `w_mv` solves the min-variance problem for each
  year; `ret_series` collects the monthly returns of `P_vw` and `P_mv`.
- *Why.*
  - **Weight drift** (`w = w*(1+r); w = w/w.sum()`) — the portfolio is
    rebalanced only once a year; between rebalances the weights move with prices
    on their own. This reproduces a realistic buy-and-hold investor.
  - **Benchmark uses previous-month caps** — the instructor's note says to weight
    each month by the *prior* month's market caps, so there is no look-ahead.

**Code cell — `summary` and the Part I statistics table.**
- *What.* `summary` turns a monthly return series into annualised statistics:
  average return, volatility, geometric (cumulative) annualised return, Sharpe
  ratio, min/max month, and total cumulative return. The cell displays the
  `P_vw` vs `P_mv` comparison.
- *Why.* The **Sharpe ratio uses excess returns** (return minus the risk-free
  rate): the instructor stressed that `R_f` must follow the same convention as
  the return used in the numerator.

**Code cell — figure helpers and the Part I cumulative plot.**
- *What.* `growth_index` builds the growth of $1, `_titles` formats titles, and
  `fig_cumulative` plots cumulative performance and saves the PNG. The cell
  produces `cum_part1.png`.
- *Why.* The growth series is **anchored at 1.0** the month before the first
  return — the instructor asked for cumulative plots that start at 1.

### §6 — Part II §3.1: carbon profile of the base portfolios

**Code cell — build the `carbon` dictionary.**
- *What.* Computes the carbon footprint (CF) and weighted-average carbon
  intensity (WACI) of `P_vw` and `P_mv` for every year, and stores
  `cf_vw_2013` — the 2013 benchmark footprint.
- *Why.* CF is `Σ wᵢ · Eᵢ/Capᵢ` — the emissions "owned" per million USD
  invested; the portfolio value cancels out, so it is a pure intensity.
  `cf_vw_2013` is saved because it is the **baseline** for the net-zero glide
  path in §9.

**Code cell — `fig_carbon` and the WACI/CF figures.**
- *What.* A reusable carbon-time-series plotter; produces `waci_base.png` and
  `cf_base.png` comparing the benchmark and the min-variance portfolio.

**Code cell — top-10 carbon contributors.**
- *What.* Accumulates each firm's average contribution to the benchmark's WACI
  (`w_vw · ci`), ranks them, and builds a table with ISIN, name, country, 2023
  intensity and 2023 weight, plus the `top10_emitters.png` bar chart.
- *Why.* The brief explicitly asks which firms drive the carbon intensity up
  (top 10, with names **and** ISIN codes).

### §7 — Part II §3.2: minimum variance with a 50% cut — `P_mv(0.5)`

**Code cell — solve `P_mv(0.5)`.**
- *What.* For each year, sets the footprint cap at 50% of `P_mv`'s footprint and
  re-solves the min-variance QP with that cap; back-tests the result.
- *Why.* The cap is `max(target, min_cf · 1.0001)` — if 50% would fall below the
  lowest footprint any long-only portfolio can reach (`min_cf`), the problem
  would be infeasible, so we floor the cap at the feasible minimum plus a tiny
  slack. (`feas_mv05` records that all targets are in fact feasible.)

**Code cell — `P_mv(0.5)` figures** (`cum_mv05.png`, `cf_mv05.png`).

### §8 — Part II §3.3: tracking-error minimisation with a 50% cut — `P_vw(0.5)`

**Code cell — solve `P_vw(0.5)`.**
- *What.* Same structure as §7, but the objective is **tracking error vs the
  benchmark** and the cap is 50% of the *benchmark's* footprint.
- *Why.* This is the "otherwise-passive" investor: stay as close as possible to
  the market portfolio while halving its carbon footprint.

**Code cell — `P_vw(0.5)` figures** (`cum_vw05.png`, `cf_vw05.png`).

### §9 — Part II §4: net-zero portfolio — `P_vw(NZ)`

**Code cell — solve `P_vw(NZ)`.**
- *What.* Same tracking-error objective, but the cap now follows a glide path:
  `CF ≤ (1 − θ)^(Y − 2013 + 1) × CF_vw_2013`, tightening 10% per year.
  `nz_target` stores the ceiling for plotting.
- *Why.* This is a cumulative decarbonisation commitment anchored to the 2013
  benchmark footprint — by the last allocation the ceiling is about 31% of the
  2013 level.

**Code cell — `P_vw(NZ)` figures** (`cum_netzero.png`, `cf_netzero.png`, the
latter with the dashed glide-path line).

### §10 — Consolidated results

**Code cell — performance table.** Builds the five-portfolio performance table
and adds an **annualised tracking-error** row (standard deviation of the
portfolio-minus-benchmark return series, annualised).

**Code cell — average carbon table.** Average CF and WACI per portfolio and each
one's footprint relative to `P_vw`.

**Code cell — concentration.** The effective number of holdings,
`N_eff = 1 / Σ wᵢ²`. *Why:* it quantifies diversification — it exposes that
`P_mv` concentrates in ≈ 17 effective names while `P_vw` holds ≈ 159.

**Code cell — export.** Writes `Part_II_Results.xlsx` with four sheets
(Performance, Carbon, Monthly returns, Top-10 emitters) — a deliverable file.

### §11 — Extension: the cost-of-decarbonisation frontier

**Markdown cell.** Frames the extension: instead of one fixed target, trace the
whole trade-off curve.

**Code cell — sweep.** Re-solves the tracking-error portfolio for footprint cuts
of 0%, 10%, …, 80%, back-tests each, and records Sharpe ratio, tracking error
and the realised footprint reduction into `frontier_df`.

**Code cell — plot.** A dual-axis figure — Sharpe ratio and tracking error
against the achieved footprint reduction — saved as `carbon_cost_frontier.png`.
*Why:* it generalises the single 50% result into a full "cost of
decarbonisation" curve, beyond what the brief requires.

> Note: the §11 cells are currently un-executed — run the notebook top to bottom
> to produce the table and figure. The sweep solves 9 × 11 extra QPs, adding
> roughly 3–5 minutes of runtime.

---

## 2. Compliance check — brief & group

### Group assignment ✓

`Groups_Strategy_2026` confirms **Group BR = North America + Europe / Scope 1+2**.
The notebook filters `Region ∈ {AMER, EUR}` and sums Scope 1 + Scope 2 — correct.

### Requirements

| Brief requirement | What the project does | Status |
|---|---|---|
| Region & scope assigned to the group | AMER + EUR, Scope 1+2 | ✓ |
| Sample: first alloc. Dec 2013, last Dec 2023, plots Jan 2014–Dec 2024 (instructor's correction) | `YEARS` 2013–2023, 132 months | ✓ |
| Drop unmatched ISINs; price < 0.5 → missing; simple returns; −100% on delisting | All implemented (§2) | ✓ |
| Carbon/revenue: forward-fill gaps; no data → not investable | `.ffill`; eligibility requires carbon + revenue | ✓ |
| Investment set: region, ≥ 3 yrs returns, stale-price filter, carbon data; same set both parts | ≥ 36 obs, ≤ 50% stale, single `get_eligible` | ✓ |
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
      "Group BR". Add the three members' names.
- [ ] **Record the video** — script is in `report/video_script.md`.
- [ ] **Re-run the notebook end to end** — recent cell edits cleared some
      outputs and §11 is new; run all cells so every figure/table is embedded.
- [ ] **Add `jupyter` to `requirements.txt`** — the brief asks for "no missing
      dependencies"; currently only the analysis packages are listed.
- [ ] **Verify the compiled report PDF is ≤ 30 pages.**
- [ ] Quick sanity check: confirm `AMER` in `Static_2025.xlsx` means North
      America (developed), not all of the Americas.

---

## 3. Ideas to strengthen the project (optional)

The project already meets the brief — these are polish, not gaps.

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
