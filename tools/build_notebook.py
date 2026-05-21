"""Assemble the final, self-contained SAAM Project 2026 notebook.

Builds SAAM_Project_2026.ipynb from cell definitions and (optionally) executes
it top-to-bottom so all tables and figures are embedded.
"""
import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell, new_notebook

cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# SAAM Project 2026 — Portfolio Allocation with a Carbon Objective
**Group BR — North America + Europe — Scope 1 + Scope 2**

This notebook reproduces, end to end, every table and figure of the report.
It runs top to bottom with no manual intervention and only relative paths.

**Structure**
- Part I — standard allocation: value-weighted benchmark `P_vw` and global
  minimum-variance portfolio `P_mv`.
- Part II — carbon-aware allocation: `P_mv(0.5)`, `P_vw(0.5)` and the
  net-zero portfolio `P_vw(NZ)`.

**Sample.** Following the instructor's consistency note, the first allocation
is decided in December 2013 and the last in December 2023; performance is
measured monthly from January 2014 to December 2024 (T = 132 months).
""")

# ---------------------------------------------------------------- imports
md("## 0. Setup")
code(r"""
import json
import os
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

DATA_DIR = "data/"
PLOTS_DIR = "report/figures/"
os.makedirs(PLOTS_DIR, exist_ok=True)

YEARS = list(range(2013, 2024))   # allocation years: Dec 2013 ... Dec 2023
THETA = 0.10                      # net-zero annual decarbonisation rate
SUB = "Group BR  |  North America + Europe  |  Scope 1+2"
COL = {"vw": "#1f3a93", "mv": "#c0392b", "mv05": "#e67e22",
       "vw05": "#16a085", "nz": "#27ae60"}
LABEL = {"vw": "P_vw", "mv": "P_mv", "mv05": "P_mv(0.5)",
         "vw05": "P_vw(0.5)", "nz": "P_vw(NZ)"}
""")

# ---------------------------------------------------------------- 1. load
md(r"""
## 1. Data loading and reshaping

All Datastream files are reshaped to a common layout. Monthly series become
`(date x ISIN)` matrices; annual series become `(ISIN x year)` matrices.
Scope 1 and Scope 2 emissions are summed (the group's assigned scope) and
emissions and revenues are forward-filled across years, as instructed.
""")
code(r"""
static = pd.read_excel(DATA_DIR + "Static_2025.xlsx").drop_duplicates("ISIN")
name_map = static.set_index("ISIN")["NAME"]
country_map = static.set_index("ISIN")["Country"]
our_isins = static[static["Region"].isin(["AMER", "EUR"])]["ISIN"].tolist()


def load_and_reshape(filename, isins):
    '''Monthly Excel file -> (date x ISIN) numeric matrix.'''
    df = pd.read_excel(DATA_DIR + filename).drop_duplicates("ISIN")
    df = df[df["ISIN"].isin(isins)]
    date_cols = [c for c in df.columns if c not in ("NAME", "ISIN")]
    matrix = df.set_index("ISIN")[date_cols].T
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_yearly(filename, isins):
    '''Annual Excel file -> (ISIN x year) numeric matrix.'''
    df = pd.read_excel(DATA_DIR + filename).drop_duplicates("ISIN")
    df = df[df["ISIN"].isin(isins)]
    year_cols = [c for c in df.columns if isinstance(c, (int, float))]
    return df.set_index("ISIN")[year_cols].apply(pd.to_numeric, errors="coerce")


prices = load_and_reshape("DS_RI_T_USD_M_2025.xlsx", our_isins)
mktcap = load_and_reshape("DS_MV_T_USD_M_2025.xlsx", our_isins)

co2 = (load_yearly("DS_CO2_SCOPE_1_Y_2025.xlsx", our_isins)
       .add(load_yearly("DS_CO2_SCOPE_2_Y_2025.xlsx", our_isins), fill_value=0)
       .ffill(axis=1))
revenue = load_yearly("DS_REV_Y_2025.xlsx", our_isins).ffill(axis=1)

rf = pd.read_excel(DATA_DIR + "Risk_Free_Rate_2025.xlsx")
rf.columns = ["Date", "RF"]
rf["Date"] = pd.to_datetime(rf["Date"].astype(str), format="%Y%m") \
    + pd.offsets.MonthEnd(0)
rf = rf.set_index("Date")
rf["RF"] = rf["RF"] / 100 / 12     # annualised % -> monthly decimal

print(f"{len(our_isins)} AMER+EUR firms | {prices.shape[0]} monthly dates")
""")

# ---------------------------------------------------------------- 2. clean
md(r"""
## 2. Data cleaning and returns

Cleaning rules from the project statement: drop fully-empty ISINs; treat any
total-return-index value below 0.5 as missing; compute simple monthly returns;
and book a −100% return in the month after a firm's last valid price
(delisting / default).
""")
code(r"""
prices = prices.dropna(axis=1, how="all")          # drop unmatched ISINs
mktcap = mktcap.reindex(columns=prices.columns)
prices = prices.where(prices >= 0.5)               # low prices -> missing
returns = prices.pct_change().iloc[1:]             # simple monthly returns

n_delist = 0
for col in returns.columns:
    last_valid = prices[col].last_valid_index()
    if last_valid is not None and last_valid < prices.index[-1]:
        nxt = prices.index[prices.index.get_loc(last_valid) + 1]
        if nxt in returns.index:
            returns.loc[nxt, col] = -1.0
            n_delist += 1

print(f"Returns: {returns.shape[0]} months x {returns.shape[1]} firms "
      f"| {n_delist} delistings booked at -100%")
""")

# ---------------------------------------------------------------- 3. set
md(r"""
## 3. Investment set

At the end of each year `Y` the investment set keeps firms that: belong to
the assigned region; have at least 36 monthly returns over the 10-year window;
show at most 50% stale (zero) returns; have a valid year-end price and market
cap; and have carbon and revenue data so the carbon footprint and intensity
are defined for **every** firm. The same set is used by all five portfolios.
""")
code(r"""
def dec(year, index):
    '''December year-end date of `year` present in a DatetimeIndex.'''
    d = index[(index.month == 12) & (index.year == year)]
    return d[0] if len(d) else None


_elig_cache = {}


def get_eligible(year):
    '''Investment set at the end of year Y (identical for all portfolios).'''
    if year in _elig_cache:
        return _elig_cache[year]
    win = returns.loc[dec(year - 10, returns.index):dec(year, returns.index)]
    valid = win.notna().sum()
    zero_pct = (win == 0).sum() / valid.replace(0, np.nan)
    d_year = dec(year, prices.index)
    has_price = prices.loc[d_year].notna()
    cap = mktcap.loc[d_year]
    has_cap = cap.notna() & (cap > 0)
    has_co2 = pd.Series({i: (co2.loc[i, :year].notna().any()
                             if i in co2.index else False)
                         for i in win.columns})
    has_rev = pd.Series({i: (i in revenue.index and year in revenue.columns
                             and pd.notna(revenue.loc[i, year]))
                         for i in win.columns})
    mask = (valid >= 36) & (zero_pct <= 0.5) & has_price & has_cap \
        & has_co2 & has_rev
    _elig_cache[year] = mask[mask].index.tolist()
    return _elig_cache[year]


def year_data(year):
    '''Everything needed for the allocation decided at the end of year Y.'''
    elig = get_eligible(year)
    win = returns.loc[dec(year - 10, returns.index):dec(year, returns.index)]
    cov = win[elig].fillna(0.0).cov().values
    cov = cov + np.eye(len(elig)) * 1e-8           # regularise -> pos. definite
    d_year = dec(year, prices.index)
    cap = mktcap.loc[d_year, elig].astype(float)   # year-end cap (USD million)
    emis = co2.loc[elig, year].astype(float)       # Scope 1+2 (tonnes CO2)
    rev_m = revenue.loc[elig, year].astype(float) / 1000.0   # -> USD million
    return {"year": year, "elig": elig, "cov": cov, "cap": cap, "emis": emis,
            "ci": emis / rev_m,        # carbon intensity   (tCO2 / $M revenue)
            "carb": emis / cap,        # carbon per cap     (tCO2 / $M invested)
            "w_vw": cap / cap.sum()}   # value-weighted benchmark weights


yd = {Y: year_data(Y) for Y in YEARS}
pd.Series({Y: len(yd[Y]["elig"]) for Y in YEARS}, name="Investment set size")
""")

# ---------------------------------------------------------------- 4. opt
md(r"""
## 4. Optimisation building blocks

Every portfolio is a long-only, fully-invested quadratic program. The carbon
footprint is linear in the weights — $CF_Y(\alpha)=\sum_i \alpha_i E_{i,Y}/Cap_{i,Y}$
— so a footprint ceiling is a single linear constraint. Problems are solved
with CLARABEL (OSQP as fallback).
""")
code(r"""
def _solve(objective, n, extra_cons):
    '''Solve a long-only QP; return cleaned weights and solver status.'''
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0] + extra_cons(w)
    prob = cp.Problem(cp.Minimize(objective(w)), cons)
    for solver in (cp.CLARABEL, cp.OSQP):
        try:
            if solver == cp.OSQP:
                prob.solve(solver=solver, max_iter=60000,
                           eps_abs=1e-9, eps_rel=1e-9, polish=True)
            else:
                prob.solve(solver=solver)
            if w.value is not None and prob.status in (
                    "optimal", "optimal_inaccurate"):
                val = np.clip(np.asarray(w.value), 0, None)
                return val / val.sum(), prob.status
        except Exception:
            continue
    return None, "failed"


def min_variance(d, cf_limit=None):
    '''Long-only minimum-variance weights, optional carbon-footprint cap.'''
    P, carb = cp.psd_wrap(d["cov"]), d["carb"].values
    extra = (lambda w: [carb @ w <= cf_limit]) if cf_limit is not None \
        else (lambda w: [])
    w, st = _solve(lambda w: cp.quad_form(w, P), len(d["elig"]), extra)
    return pd.Series(w, index=d["elig"]), st


def min_tracking_error(d, cf_limit=None):
    '''Long-only tracking-error minimisation vs the VW benchmark.'''
    P, wvw, carb = cp.psd_wrap(d["cov"]), d["w_vw"].values, d["carb"].values
    extra = (lambda w: [carb @ w <= cf_limit]) if cf_limit is not None \
        else (lambda w: [])
    w, st = _solve(lambda w: cp.quad_form(w - wvw, P), len(d["elig"]), extra)
    return pd.Series(w, index=d["elig"]), st


def cf(d, w):      # portfolio carbon footprint  (tCO2 / $M invested)
    return float(d["carb"].values @ np.asarray(w))


def waci(d, w):    # weighted-average carbon intensity (tCO2 / $M revenue)
    return float(d["ci"].values @ np.asarray(w))
""")

# ---------------------------------------------------------------- 5. partI
md(r"""
## 5. Part I — Standard portfolio allocation

`P_mv` minimises portfolio variance each December for the next year, long-only
and fully invested. `P_vw` is the value-weighted benchmark, rebalanced monthly
on the previous month's market caps. Weights drift with realised returns
between annual rebalancing dates.
""")
code(r"""
def backtest_optimized(weights_by_year):
    '''Monthly returns of an annually-rebalanced portfolio with weight drift.'''
    rows = []
    for Y, w0 in weights_by_year.items():
        elig = list(w0.index)
        future = returns.loc[returns.index.year == Y + 1, elig].fillna(0.0)
        w = w0.copy()
        for date, row in future.iterrows():
            rows.append({"Date": date, "Return": float((w * row).sum())})
            w = w * (1 + row)
            w = w / w.sum()
    return pd.DataFrame(rows).set_index("Date")["Return"]


def backtest_vw():
    '''Value-weighted benchmark: monthly rebalanced on previous-month caps.'''
    rows = []
    for Y in YEARS:
        elig = get_eligible(Y)
        future = returns.loc[returns.index.year == Y + 1, elig].fillna(0.0)
        for date in future.index:
            prev = mktcap.index[mktcap.index.get_loc(date) - 1]
            w = (mktcap.loc[prev, elig].fillna(0.0)).pipe(lambda c: c / c.sum())
            rows.append({"Date": date,
                         "Return": float((w * future.loc[date]).sum())})
    return pd.DataFrame(rows).set_index("Date")["Return"]


w_mv = {}
for Y in YEARS:
    w_mv[Y], _ = min_variance(yd[Y])

ret_series = {"vw": backtest_vw(), "mv": backtest_optimized(w_mv)}
print(f"Backtest: {len(ret_series['vw'])} monthly returns "
      f"({ret_series['vw'].index[0]:%Y-%m} to {ret_series['vw'].index[-1]:%Y-%m})")
""")
code(r"""
def summary(r):
    '''Annualised performance statistics for a monthly return series.'''
    excess = r - rf["RF"].reindex(r.index, method="nearest")
    T = len(r)
    return {"Ann. avg return": r.mean() * 12,
            "Ann. volatility": r.std() * np.sqrt(12),
            "Ann. cum return": (1 + r).prod() ** (12 / T) - 1,
            "Sharpe ratio": excess.mean() / excess.std() * np.sqrt(12),
            "Min": r.min(), "Max": r.max(),
            "Cumulative return": (1 + r).prod() - 1}


stats = {p: summary(r) for p, r in ret_series.items()}
pd.DataFrame({LABEL[p]: stats[p] for p in ("vw", "mv")}).round(4)
""")
code(r"""
def growth_index(r):
    '''Growth of 1 USD, anchored at 1.0 the month before the first return.'''
    cum = (1 + r).cumprod()
    start = r.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[start]), cum])


def _titles(ax, title):
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=30)
    ax.text(0.0, 1.045, SUB, transform=ax.transAxes, fontsize=8.5,
            alpha=0.65, style="italic")


def _stagger(values, gap):
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if pos[i] - pos[j] < gap:
            pos[i] = pos[j] + gap
    return pos


def fig_cumulative(keys, labels, title, fname):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ends = []
    for key, lab in zip(keys, labels):
        g = growth_index(ret_series[key])
        ax.plot(g.index, g.values, color=COL[key], linewidth=2.1, label=lab)
        ends.append(g.iloc[-1])
    last_x = growth_index(ret_series[keys[0]]).index[-1]
    for key, ev in zip(keys, ends):
        ax.annotate(f"{ev:.2f}", xy=(last_x, ev), xytext=(8, 0),
                    textcoords="offset points", color=COL[key],
                    fontweight="bold", va="center", fontsize=9,
                    annotation_clip=False)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Cumulative growth of 1 USD")
    ax.set_xlabel("Date")
    _titles(ax, title)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + fname, dpi=150, bbox_inches="tight")
    plt.show()


fig_cumulative(["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
               "Part I - Cumulative performance (2014-2024)", "cum_part1.png")
""")

# ---------------------------------------------------------------- 6. 3.1
md(r"""
## 6. Part II §3.1 — Carbon emissions of the base portfolios

Carbon intensity is $CI_{i,Y}=E_{i,Y}/(Rev_{i,Y}/1000)$ (revenue converted from
thousands to millions of USD). For a portfolio with weights $\alpha$,
$$WACI_Y=\sum_i\alpha_{i,Y}\,CI_{i,Y},\qquad
  CF_Y=\sum_i\alpha_{i,Y}\,E_{i,Y}/Cap_{i,Y}.$$
For the value-weighted portfolio this collapses to
$CF_Y=\sum_iE_{i,Y}/\sum_iCap_{i,Y}$.
""")
code(r"""
carbon = {p: {"cf": {}, "waci": {}} for p in
          ("vw", "mv", "mv05", "vw05", "nz")}
for Y in YEARS:
    d = yd[Y]
    for tag, w in (("vw", d["w_vw"]), ("mv", w_mv[Y])):
        carbon[tag]["cf"][Y] = cf(d, w)
        carbon[tag]["waci"][Y] = waci(d, w)

cf_vw_2013 = carbon["vw"]["cf"][2013]      # net-zero baseline
carbon_tbl = pd.DataFrame({
    "P_vw CF": carbon["vw"]["cf"], "P_mv CF": carbon["mv"]["cf"],
    "P_vw WACI": carbon["vw"]["waci"], "P_mv WACI": carbon["mv"]["waci"]})
carbon_tbl.round(1)
""")
code(r"""
def fig_carbon(metric, keys, labels, title, ylabel, fname, target=None):
    fig, ax = plt.subplots(figsize=(11, 5.3))
    for key, lab in zip(keys, labels):
        ax.plot(YEARS, [carbon[key][metric][Y] for Y in YEARS], "o-",
                color=COL[key], linewidth=2.1, markersize=5, label=lab)
    if target is not None:
        ax.plot(YEARS, target, "--", color="black", linewidth=1.5, alpha=0.7,
                label="Net-zero target ceiling")
    ax.set_xlabel("Allocation year Y")
    ax.set_ylabel(ylabel)
    ax.set_xticks(YEARS)
    ax.set_ylim(bottom=0)
    _titles(ax, title)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + fname, dpi=150, bbox_inches="tight")
    plt.show()


fig_carbon("waci", ["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
           "3.1 - WACI: value-weighted vs minimum-variance",
           "WACI (tCO2 / $M revenue)", "waci_base.png")
fig_carbon("cf", ["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
           "3.1 - Carbon footprint: value-weighted vs minimum-variance",
           "Carbon footprint (tCO2 / $M invested)", "cf_base.png")
""")
code(r"""
# Top-10 contributors to the value-weighted portfolio's carbon intensity
contrib = {}
for Y in YEARS:
    d = yd[Y]
    for isin, val in (d["w_vw"] * d["ci"]).items():
        contrib[isin] = contrib.get(isin, 0.0) + val / len(YEARS)

top10 = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:10]
last = yd[2023]
top10_rows = [{"ISIN": i, "Name": str(name_map.get(i, "")),
               "Country": str(country_map.get(i, "")),
               "Avg WACI contribution": c,
               "CI 2023": float(last["ci"].get(i, np.nan)),
               "VW weight 2023 %": float(last["w_vw"].get(i, np.nan)) * 100}
              for i, c in top10]
top10_df = pd.DataFrame(top10_rows)

fig, ax = plt.subplots(figsize=(11, 5.3))
ax.barh([r["Name"][:34] for r in top10_rows[::-1]],
        [r["Avg WACI contribution"] for r in top10_rows[::-1]],
        color=COL["vw"], alpha=0.85)
ax.set_xlabel("Average contribution to VW portfolio WACI (tCO2 / $M revenue)")
_titles(ax, "3.1 - Top-10 carbon-intensity contributors (P_vw)")
plt.tight_layout()
plt.savefig(PLOTS_DIR + "top10_emitters.png", dpi=150, bbox_inches="tight")
plt.show()
top10_df.round(2)
""")

# ---------------------------------------------------------------- 7. 3.2
md(r"""
## 7. Part II §3.2 — Minimum-variance with a 50% footprint cut: `P_mv(0.5)`

Same minimum-variance program as Part I, plus the linear constraint
$CF_Y(\alpha)\le 0.5\,CF_Y(P^{(mv)}_{oos})$.
""")
code(r"""
w_mv05, feas_mv05 = {}, {}
for Y in YEARS:
    d = yd[Y]
    target = 0.5 * carbon["mv"]["cf"][Y]
    min_cf = float(d["carb"].min())            # lowest reachable long-only CF
    feas_mv05[Y] = target >= min_cf
    w_mv05[Y], _ = min_variance(d, cf_limit=max(target, min_cf * 1.0001))
    carbon["mv05"]["cf"][Y] = cf(d, w_mv05[Y])
    carbon["mv05"]["waci"][Y] = waci(d, w_mv05[Y])

ret_series["mv05"] = backtest_optimized(w_mv05)
stats["mv05"] = summary(ret_series["mv05"])
print("All 50% targets feasible:", all(feas_mv05.values()))
pd.DataFrame({LABEL[p]: stats[p] for p in ("mv", "mv05")}).round(4)
""")
code(r"""
fig_cumulative(["mv", "mv05"], ["P_mv", "P_mv(0.5)"],
               "3.2 - Minimum-variance: standard vs 50% decarbonised",
               "cum_mv05.png")
fig_carbon("cf", ["mv", "mv05"], ["P_mv", "P_mv(0.5)"],
           "3.2 - Carbon footprint: P_mv vs P_mv(0.5)",
           "Carbon footprint (tCO2 / $M invested)", "cf_mv05.png")
""")

# ---------------------------------------------------------------- 8. 3.3
md(r"""
## 8. Part II §3.3 — Tracking-error minimisation with a 50% cut: `P_vw(0.5)`

The "otherwise passive" investor stays as close as possible to the benchmark.
We minimise $(\alpha-\alpha^{vw})'\Sigma_Y(\alpha-\alpha^{vw})$ subject to
$CF_Y(\alpha)\le 0.5\,CF_Y(P^{(vw)})$, long-only and fully invested.
""")
code(r"""
w_vw05, feas_vw05 = {}, {}
for Y in YEARS:
    d = yd[Y]
    target = 0.5 * carbon["vw"]["cf"][Y]
    min_cf = float(d["carb"].min())
    feas_vw05[Y] = target >= min_cf
    w_vw05[Y], _ = min_tracking_error(d, cf_limit=max(target, min_cf * 1.0001))
    carbon["vw05"]["cf"][Y] = cf(d, w_vw05[Y])
    carbon["vw05"]["waci"][Y] = waci(d, w_vw05[Y])

ret_series["vw05"] = backtest_optimized(w_vw05)
stats["vw05"] = summary(ret_series["vw05"])
print("All 50% targets feasible:", all(feas_vw05.values()))
pd.DataFrame({LABEL[p]: stats[p] for p in ("vw", "vw05")}).round(4)
""")
code(r"""
fig_cumulative(["vw", "vw05"], ["P_vw", "P_vw(0.5)"],
               "3.3 - Value-weighted vs 50% decarbonised tracking portfolio",
               "cum_vw05.png")
fig_carbon("cf", ["vw", "vw05"], ["P_vw", "P_vw(0.5)"],
           "3.3 - Carbon footprint: P_vw vs P_vw(0.5)",
           "Carbon footprint (tCO2 / $M invested)", "cf_vw05.png")
""")

# ---------------------------------------------------------------- 9. NZ
md(r"""
## 9. Part II §4 — Net-zero portfolio `P_vw(NZ)`

Same tracking-error program, but the footprint ceiling tightens by
$\theta=10\%$ every year relative to the 2013 value-weighted footprint:
$CF_Y(\alpha)\le(1-\theta)^{\,Y-Y_0+1}\,CF_{Y_0}(P^{(vw)})$ with $Y_0=2013$.
""")
code(r"""
w_nz, feas_nz, nz_target = {}, {}, []
for Y in YEARS:
    d = yd[Y]
    target = (1 - THETA) ** (Y - 2013 + 1) * cf_vw_2013
    nz_target.append(target)
    min_cf = float(d["carb"].min())
    feas_nz[Y] = target >= min_cf
    w_nz[Y], _ = min_tracking_error(d, cf_limit=max(target, min_cf * 1.0001))
    carbon["nz"]["cf"][Y] = cf(d, w_nz[Y])
    carbon["nz"]["waci"][Y] = waci(d, w_nz[Y])

ret_series["nz"] = backtest_optimized(w_nz)
stats["nz"] = summary(ret_series["nz"])
print("All net-zero targets feasible:", all(feas_nz.values()))
pd.DataFrame({LABEL[p]: stats[p] for p in ("vw", "vw05", "nz")}).round(4)
""")
code(r"""
fig_cumulative(["vw", "vw05", "nz"], ["P_vw", "P_vw(0.5)", "P_vw(NZ)"],
               "4.2 - Value-weighted family: benchmark, 50% cut, net-zero",
               "cum_netzero.png")
fig_carbon("cf", ["vw", "vw05", "nz"], ["P_vw", "P_vw(0.5)", "P_vw(NZ)"],
           "4.1 - Carbon footprint vs the net-zero glide path",
           "Carbon footprint (tCO2 / $M invested)", "cf_netzero.png",
           target=nz_target)
""")

# ---------------------------------------------------------------- 10. results
md(r"""
## 10. Consolidated results

Performance and carbon metrics for all five portfolios, plus annualised
ex-post tracking error against the value-weighted benchmark.
""")
code(r"""
def tracking_error(p, bench="vw"):
    return (ret_series[p] - ret_series[bench]).std() * np.sqrt(12)


perf = pd.DataFrame({LABEL[p]: stats[p] for p in
                     ("vw", "mv", "mv05", "vw05", "nz")})
perf.loc["Ann. tracking error"] = [np.nan] + [
    tracking_error(p) for p in ("mv", "mv05", "vw05", "nz")]
perf.round(4)
""")
code(r"""
carbon_avg = pd.DataFrame({
    "Avg CF (tCO2/$M)": {LABEL[p]: np.mean(list(carbon[p]["cf"].values()))
                         for p in carbon},
    "Avg WACI (tCO2/$M rev)": {LABEL[p]: np.mean(list(carbon[p]["waci"].values()))
                               for p in carbon}})
carbon_avg["CF vs P_vw"] = (carbon_avg["Avg CF (tCO2/$M)"]
                            / carbon_avg.loc["P_vw", "Avg CF (tCO2/$M)"] - 1)
carbon_avg.round(3)
""")
code(r"""
# Concentration: effective number of names  N_eff = 1 / sum(w^2)
def eff_n(w):
    w = np.asarray(w)
    return 1.0 / np.sum(w ** 2)


concentration = pd.DataFrame({
    "Avg effective N": {
        "P_vw": np.mean([eff_n(yd[Y]["w_vw"]) for Y in YEARS]),
        "P_mv": np.mean([eff_n(w_mv[Y]) for Y in YEARS]),
        "P_mv(0.5)": np.mean([eff_n(w_mv05[Y]) for Y in YEARS]),
        "P_vw(0.5)": np.mean([eff_n(w_vw05[Y]) for Y in YEARS]),
        "P_vw(NZ)": np.mean([eff_n(w_nz[Y]) for Y in YEARS])}})
concentration.round(1)
""")
code(r"""
# Export deliverable tables
with pd.ExcelWriter("Part_II_Results.xlsx") as xl:
    perf.to_excel(xl, sheet_name="Performance")
    carbon_tbl_full = pd.DataFrame(
        {f"{LABEL[p]} CF": carbon[p]["cf"] for p in carbon}
        | {f"{LABEL[p]} WACI": carbon[p]["waci"] for p in carbon})
    carbon_tbl_full.to_excel(xl, sheet_name="Carbon")
    pd.DataFrame({LABEL[p]: ret_series[p] for p in ret_series}).to_excel(
        xl, sheet_name="Monthly returns")
    top10_df.to_excel(xl, sheet_name="Top10 emitters", index=False)

print("Saved Part_II_Results.xlsx and all figures in report/figures/.")
print("Notebook reproduced every report table and figure.")
""")

# ---------------------------------------------------------------- write
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}})

with open("SAAM_Project_2026.ipynb", "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"Wrote SAAM_Project_2026.ipynb ({len(cells)} cells)")
