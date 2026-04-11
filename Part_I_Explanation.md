# Part I - Step-by-Step Explanation

## Overview

This notebook builds two portfolios from a universe of North American and European firms, backtested from 2014 to 2025:

1. **Value-Weighted (VW)** - benchmark portfolio weighted by market capitalization
2. **Minimum-Variance (MV)** - long-only portfolio that minimizes variance

---

## Step 1: Load and Reshape Data

We load 5 data files and reshape them for analysis.

**What we load:**

| File | Content | Format after reshaping |
|---|---|---|
| `Static_2025.xlsx` | ISIN, name, country, region for 2545 firms | Flat table |
| `DS_RI_T_USD_M_2025.xlsx` | Monthly total return index (prices with dividends) | Matrix: dates x firms |
| `DS_MV_T_USD_M_2025.xlsx` | Monthly market capitalization (million USD) | Matrix: dates x firms |
| `DS_CO2_SCOPE_1_Y_2025.xlsx` | Yearly Scope 1 CO2 emissions (tonnes) | Matrix: firms x years |
| `DS_CO2_SCOPE_2_Y_2025.xlsx` | Yearly Scope 2 CO2 emissions (tonnes) | Matrix: firms x years |
| `Risk_Free_Rate_2025.xlsx` | Monthly risk-free rate (annualized %) | Time series |

**Region filter:** We keep only firms with `Region = AMER` (North America, 669 firms) or `Region = EUR` (Europe, 633 firms) = **1302 firms total**.

**CO2 processing:** We sum Scope 1 + Scope 2 for each firm and forward-fill missing years (if a firm reported in 2012 but not 2013, we carry the 2012 value forward). CO2 data is only used here for filtering — we need it later in Part II.

**Risk-free rate:** Converted from annualized percentage to monthly decimal (divide by 100, then by 12).

---

## Step 2: Data Cleaning and Returns

We clean the price data following the project instructions, then compute monthly returns.

### 2a. Remove empty firms
Some ISINs have no price data at all (Datastream couldn't match them). We drop these entirely.

### 2b. Low prices
Prices below 0.5 are treated as missing (`NaN`). This avoids extreme/infinite returns from near-zero prices (e.g., a price going from 0.01 to 0.02 would be a +100% return, which is noise).

### 2c. Compute returns
Simple returns: `R_t = P_t / P_{t-1} - 1`

For example, if a stock's total return index goes from 100 to 105, the monthly return is 5%.

### 2d. Handle delistings
When a firm is delisted (price exists, then becomes `NaN` permanently), we set the return to **-100%** at the delisting date. This reflects the reality that an investor holding the stock loses everything.

**Output:** A return matrix of ~313 months x ~1302 firms, plus a histogram showing the distribution of all returns.

---

## Step 3: Investment Set + Optimization Helpers

We define two functions used in the backtest.

### `get_eligible(year)` — Which firms can we invest in?

At the end of each year Y, we check 4 criteria for each firm:

1. **Enough data:** At least 36 months (3 years) of valid returns in the 10-year estimation window (Y-10 to Y). If a firm was listed only recently, we don't have enough history to estimate its risk.

2. **No stale prices:** Fewer than 50% of returns are exactly zero. A firm with many zero-returns likely has no trading activity (illiquid). Including it would artificially lower the estimated variance and trick the optimizer into overweighting it.

3. **Price available at year-end:** We can only invest if we know the current price. If the price is `NaN` at end of year Y, the firm is excluded.

4. **CO2 data available:** The firm must have reported CO2 emissions (Scope 1 or Scope 2) at some point up to year Y. This ensures the same investment set works for Part II (carbon analysis).

**Result:** ~842 eligible firms in 2013, growing to ~1176 by 2021 as more firms report CO2 data.

### `min_variance_weights(year, eligible)` — How much to invest in each firm?

This solves the **long-only minimum-variance optimization**:

```
minimize    w' * Sigma * w       (portfolio variance)
subject to  sum(w) = 1           (fully invested)
            w >= 0               (no short selling)
```

Where:
- `w` = vector of portfolio weights (one per firm)
- `Sigma` = covariance matrix estimated from 10 years of monthly returns

We use `cvxpy` with the OSQP solver, which is a specialized quadratic programming solver (much faster than general-purpose optimizers for this type of problem).

**Note:** With ~800-1000 firms but only 120 monthly observations, the covariance matrix is singular (more variables than observations). The optimizer concentrates weight on ~30 low-volatility stocks. A small regularization term (`1e-8 * I`) is added for numerical stability.

---

## Step 4: Backtest (2014-2025)

We loop over each rebalancing year (2013 to 2024) and compute portfolio returns for the following year.

### For each year Y:

1. **Get eligible firms** at end of year Y
2. **Compute min-variance weights** using returns from Y-10 to Y
3. **Compute monthly returns** for year Y+1 using these weights

### Min-variance portfolio returns

For each month in year Y+1:
- Portfolio return = `sum(w_i * R_i)` where `w_i` is the weight and `R_i` is the stock return
- **Drift adjustment:** After each month, weights change because stocks with positive returns now represent a larger share. We update: `w_i_new = w_i * (1 + R_i) / sum(w_j * (1 + R_j))`. This reflects buy-and-hold behavior between annual rebalancings.

### Value-weighted portfolio returns

For each month:
- Weights = market cap of each firm / total market cap of all eligible firms
- We use the **previous month's** market cap as weights (information available at the time)
- Portfolio return = `sum(w_vw_i * R_i)`

**Output:** 144 monthly returns (12 years x 12 months) for each portfolio.

---

## Step 5: Results

We compute summary statistics for both portfolios:

| Metric | Formula |
|---|---|
| Annualized avg return | `mean(monthly returns) * 12` |
| Annualized volatility | `std(monthly returns) * sqrt(12)` |
| Annualized cumulative return | `(product(1 + R_t))^(12/T) - 1` (geometric) |
| Sharpe ratio | `mean(excess returns) / std(excess returns) * sqrt(12)` |
| Min / Max | Smallest and largest monthly return |

**Excess returns** = portfolio return minus the risk-free rate for that month.

We also plot the **cumulative return** (growth of $1 invested at the start) for both portfolios on the same chart, allowing visual comparison.

---

## Step 6: Export to Template

We fill in the provided Excel template (`Template for Part I-SAAM.xlsx`) with:
- **Left side:** The 6 summary statistics for both portfolios
- **Right side:** All 144 monthly returns with their dates

The result is saved as `Part_I_Results.xlsx`, ready for submission.
