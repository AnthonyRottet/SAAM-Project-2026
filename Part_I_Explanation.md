# Part I - Line-by-Line Explanation

## Overview

This notebook builds two portfolios from a universe of North American and European firms, backtested from 2014 to 2025:

1. **Value-Weighted (VW)** - benchmark portfolio weighted by market capitalization
2. **Minimum-Variance (MV)** - long-only portfolio that minimizes variance

---

## Imports

```python
import numpy as np          # numerical operations (arrays, matrix math)
import pandas as pd         # data manipulation (DataFrames, Excel reading)
import cvxpy as cp          # convex optimization (for portfolio optimization)
import matplotlib.pyplot as plt  # plotting
import warnings
warnings.filterwarnings('ignore')  # suppress pandas/numpy warnings

DATA_DIR = 'data/'          # folder where all Excel files are stored
```

---

## Step 1: Load and Reshape Data

```python
static = pd.read_excel(DATA_DIR + 'Static_2025.xlsx')
```
Loads the firm metadata file (ISIN, name, country, region) for all 2545 firms.

```python
our_isins = static[static['Region'].isin(['AMER', 'EUR'])]['ISIN'].tolist()
```
Filters to keep only firms in North America (`AMER`) or Europe (`EUR`). Extracts their ISIN codes into a list. Result: 1302 ISINs.

```python
def load_and_reshape(filename, isins):
    df = pd.read_excel(DATA_DIR + filename)       # load the Excel file
    df = df[df['ISIN'].isin(isins)]               # keep only our 1302 firms
    date_cols = [c for c in df.columns if c not in ['NAME', 'ISIN']]  # all columns except NAME and ISIN are dates
    matrix = df.set_index('ISIN')[date_cols].T     # pivot: rows=dates, columns=ISINs
    matrix.index = pd.to_datetime(matrix.index)    # convert index to datetime
    return matrix.apply(pd.to_numeric, errors='coerce')  # force all values to numbers (non-numeric -> NaN)
```
Helper function that loads any of our data files and reshapes them from "firms as rows, dates as columns" into "dates as rows, firms as columns". The `.T` transposes the matrix. `errors='coerce'` means if a cell contains text (like an error code from Datastream), it becomes `NaN`.

```python
prices = load_and_reshape('DS_RI_T_USD_M_2025.xlsx', our_isins)
```
Loads the monthly Total Return Index. This is like a price that also accounts for dividends. Result: 314 dates x 1302 firms.

```python
mktcap = load_and_reshape('DS_MV_T_USD_M_2025.xlsx', our_isins)
```
Loads the monthly market capitalization (in million USD). Same shape.

```python
def load_co2(filename):
    df = pd.read_excel(DATA_DIR + filename)
    year_cols = [c for c in df.columns if isinstance(c, (int, float))]  # columns like 1999, 2000, ..., 2025
    return df.set_index('ISIN')[year_cols].apply(pd.to_numeric, errors='coerce')
```
Helper to load a CO2 file. Keeps only year columns (integers), drops NAME. Returns a matrix: firms x years.

```python
co2 = load_co2('DS_CO2_SCOPE_1_Y_2025.xlsx').add(
    load_co2('DS_CO2_SCOPE_2_Y_2025.xlsx'), fill_value=0
)
```
Loads Scope 1 and Scope 2 separately, then **adds them together**. `fill_value=0` means: if a firm has Scope 1 data but no Scope 2 (or vice versa), treat the missing one as 0 instead of making the sum NaN.

```python
.ffill(axis=1)
```
Forward-fill along years: if a firm has data for 2012 but not 2013, the 2012 value is carried forward to 2013. This is done left-to-right across columns (years).

```python
.loc[lambda df: df.index.isin(our_isins)]
```
Keep only our 1302 AMER+EUR firms.

```python
rf = pd.read_excel(DATA_DIR + 'Risk_Free_Rate_2025.xlsx')
rf.columns = ['Date', 'RF']
```
Loads the risk-free rate file. Renames the columns (original names are generic).

```python
rf['Date'] = pd.to_datetime(rf['Date'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
```
The date column contains values like `200001` (= January 2000). We convert to datetime, then snap to the end of the month (e.g., 2000-01-31).

```python
rf = rf.set_index('Date')
rf['RF'] = rf['RF'] / 100 / 12
```
The raw RF is an annualized percentage (e.g., 4.92 means 4.92% per year). We convert to a monthly decimal: `4.92 / 100 / 12 = 0.0041`.

---

## Step 2: Data Cleaning and Returns

```python
prices = prices.dropna(axis=1, how='all')
```
Remove firms where **every single price** is NaN. These are ISINs that Datastream couldn't match.

```python
mktcap = mktcap[prices.columns.intersection(mktcap.columns)]
```
Keep only the same firms in the market cap matrix (sync with prices).

```python
prices[prices < 0.5] = np.nan
```
Any price below 0.5 is replaced with NaN. Reason: very low prices produce extreme returns when they fluctuate (e.g., 0.01 to 0.02 = +100%). The project instructions say to treat these as missing.

```python
returns = prices.pct_change().iloc[1:]
```
`pct_change()` computes `(P_t - P_{t-1}) / P_{t-1}` for each cell = simple monthly return. `iloc[1:]` drops the first row (Dec 1999) which has no prior price to compute a return from.

```python
for col in returns.columns:
    last_valid = prices[col].last_valid_index()
```
For each firm, find the last date where the price is not NaN.

```python
    if last_valid is not None and last_valid < prices.index[-1]:
```
If the last valid price is **before** the end of the dataset, the firm was delisted.

```python
        next_pos = prices.index.get_loc(last_valid) + 1
        next_date = prices.index[next_pos]
        if next_date in returns.index:
            returns.loc[next_date, col] = -1.0
```
Set the return to -100% on the month after the last valid price. This means: if you held this stock, you lost everything when it was delisted.

```python
returns.stack().hist(bins=100, range=(-0.5, 0.5), figsize=(10, 3))
```
Sanity check: `.stack()` converts the 2D matrix into a single long series of all returns. Then we plot a histogram to visually verify the distribution looks normal-ish and centered near 0.

---

## Step 3: Investment Set + Optimization Helpers

### `dec(year, index)` - Find December date

```python
def dec(year, index):
    dates = index[(index.month == 12) & (index.year == year)]
    return dates[0] if len(dates) > 0 else None
```
Finds the December date for a given year in a DatetimeIndex. For example, `dec(2013, returns.index)` returns `2013-12-31`. Used throughout to locate year-end dates.

### `get_eligible(year)` - Filter the investment set

```python
window = returns.loc[dec(year-10, returns.index):dec(year, returns.index)]
```
Slice the returns matrix to the 10-year estimation window. For year=2013, this is Dec 2003 to Dec 2013 (~120 months).

```python
valid_count = window.notna().sum()
```
For each firm (column), count how many months have a non-NaN return. Result: a Series with one count per firm.

```python
zero_pct = (window == 0).sum() / valid_count
```
For each firm, compute what fraction of its returns are exactly 0. High zero-return percentage = stale/illiquid stock.

```python
has_price = prices.loc[dec(year, prices.index)].notna()
```
Check if each firm has a valid price at the end of year Y. Returns a True/False Series.

```python
has_co2 = pd.Series({
    isin: co2.loc[isin, :year].notna().any() if isin in co2.index else False
    for isin in window.columns
})
```
For each firm, check if it has **any** CO2 data reported up to year Y. `co2.loc[isin, :year]` selects all years up to Y for that firm. `.notna().any()` returns True if at least one year has data.

```python
mask = (valid_count >= 36) & (zero_pct <= 0.5) & has_price & has_co2
return mask[mask].index.tolist()
```
Combine all 4 criteria with AND. Keep only firms where all are True. Return their ISINs as a list.

### `min_variance_weights(year, eligible)` - Optimize portfolio

```python
window = returns.loc[dec(year-10, returns.index):dec(year, returns.index)]
ret = window[eligible].fillna(0)
```
Get the 10-year return window, keep only eligible firms. Replace remaining NaN with 0 (a firm with a missing return in the middle of the window is assumed to have 0 return that month).

```python
cov = ret.cov().values.copy()
```
Compute the sample covariance matrix from the returns. `.values` converts to a numpy array. `.copy()` makes it writable.

```python
cov += np.eye(len(eligible)) * 1e-8
```
Add a tiny value (0.00000001) to the diagonal. This is **regularization** — it ensures the matrix is invertible and the solver doesn't crash on numerical edge cases.

```python
w = cp.Variable(len(eligible))
```
Create a cvxpy optimization variable: a vector of N weights (one per firm).

```python
prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)),
                  [cp.sum(w) == 1, w >= 0])
```
Define the optimization problem:
- **Objective:** minimize `w' * cov * w` (portfolio variance)
- **Constraint 1:** weights sum to 1 (fully invested)
- **Constraint 2:** all weights >= 0 (long-only, no short selling)

```python
prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-9, eps_rel=1e-9)
```
Solve using OSQP (a fast quadratic programming solver). `max_iter=20000` allows more iterations for convergence. `eps_abs` and `eps_rel` are tight tolerance settings for accuracy.

```python
return pd.Series(w.value, index=eligible)
```
Return the optimal weights as a pandas Series indexed by ISIN.

---

## Step 4: Backtest (2014-2025)

```python
r_vw_list, r_mv_list = [], []
```
Two empty lists to collect monthly returns for each portfolio.

```python
for year in range(2013, 2025):
```
Loop over rebalancing years. At the end of each year, we decide weights for the next year.

```python
    eligible = get_eligible(year)
    w_mv = min_variance_weights(year, eligible)
```
Get the investment set and compute optimal min-variance weights.

```python
    future = returns.loc[returns.index.year == year + 1, eligible].fillna(0)
```
Get all monthly returns for year Y+1, only for eligible firms. Fill NaN with 0 (if a firm has a missing return mid-year, treat as 0).

### Min-variance returns with drift adjustment

```python
    w = w_mv.copy()
    for date, row in future.iterrows():
```
Start with the optimal weights. Loop over each month of the next year. `row` contains the return of each firm for that month.

```python
        r_mv_list.append({'Date': date, 'Return': (w * row).sum()})
```
Portfolio return = sum of (weight x return) for each firm. Store it.

```python
        w = w * (1 + row)
        w = w / w.sum()
```
**Drift adjustment:** If a stock went up 5% this month, its weight increases because it's now worth more. Multiply each weight by (1 + return), then renormalize to sum to 1. This is what happens in reality between rebalancing dates — you don't trade, so weights drift with performance.

### Value-weighted returns

```python
    for date in future.index:
        prev = mktcap.index[mktcap.index.get_loc(date) - 1]
```
For each month, find the **previous month's** date in the market cap data. We use previous month because that's the information available when the month starts.

```python
        caps = mktcap.loc[prev, eligible].fillna(0)
        w_vw = caps / caps.sum()
```
Get the market cap of each eligible firm at end of previous month. Divide by total to get weights. A firm worth $100B out of $1T total gets weight 0.10 (10%).

```python
        r_vw_list.append({'Date': date, 'Return': (w_vw * future.loc[date]).sum()})
```
Portfolio return = sum of (market cap weight x stock return).

```python
df_vw = pd.DataFrame(r_vw_list).set_index('Date')
df_mv = pd.DataFrame(r_mv_list).set_index('Date')
```
Convert the lists of dictionaries into DataFrames with Date as index. Each has 144 rows (12 years x 12 months) and one column `Return`.

---

## Step 5: Results

```python
def summary(r, name):
    rf_matched = rf['RF'].reindex(r.index, method='nearest')
```
Match the risk-free rate to our return dates. `method='nearest'` handles the case where dates don't align exactly (e.g., our return is on Jan 30 but RF is on Jan 31 — it picks the closest).

```python
    excess = r - rf_matched
```
Excess return = portfolio return minus risk-free rate, for each month.

```python
    'Ann. avg return': r.mean() * 12,
```
Average monthly return, multiplied by 12 to annualize. If the average month is +0.8%, annualized = 9.6%.

```python
    'Ann. volatility': r.std() * np.sqrt(12),
```
Monthly standard deviation, multiplied by sqrt(12) to annualize. This is the standard way to scale volatility from monthly to annual.

```python
    'Ann. cum return': (1 + r).prod() ** (12 / len(r)) - 1,
```
Geometric annualized return. `(1 + r).prod()` compounds all monthly returns into a total growth factor. Then we take the (12/T)-th power to annualize, and subtract 1. This accounts for compounding (unlike the arithmetic average above).

```python
    'Sharpe ratio': excess.mean() / excess.std() * np.sqrt(12),
```
Sharpe ratio = reward per unit of risk. Mean excess return divided by its standard deviation, annualized with sqrt(12). Higher = better risk-adjusted performance.

```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot((1 + df_vw['Return']).cumprod(), label='Value-Weighted')
ax.plot((1 + df_mv['Return']).cumprod(), label='Minimum-Variance')
```
Plot cumulative returns. `(1 + R).cumprod()` gives the growth of $1: if month 1 is +2% and month 2 is -1%, the cumulative values are 1.02, then 1.02 * 0.99 = 1.0098.

---

## Step 6: Export to Template

```python
wb = load_workbook('informations/Template for Part I-SAAM.xlsx')
ws = wb.active
```
Open the provided Excel template. `wb.active` gets the first (and only) sheet.

```python
for i, key in enumerate(['Ann. avg return', 'Ann. volatility', 'Ann. cum return',
                          'Sharpe ratio', 'Min', 'Max']):
    ws.cell(row=3+i, column=2, value=stats_vw[key])   # column B = Value-Weighted
    ws.cell(row=3+i, column=3, value=stats_mv[key])    # column C = Min-Variance
```
Write the 6 summary statistics into rows 3-8 of the template. Column B = VW portfolio, Column C = MV portfolio.

```python
for i, date in enumerate(df_vw.index):
    ws.cell(row=3+i, column=5, value=date)                         # column E = date
    ws.cell(row=3+i, column=6, value=df_vw.loc[date, 'Return'])    # column F = VW return
    ws.cell(row=3+i, column=7, value=df_mv.loc[date, 'Return'])    # column G = MV return
```
Write all 144 monthly returns with their dates into the right side of the template.

```python
wb.save('Part_I_Results.xlsx')
```
Save as a new file (does not overwrite the original template).
