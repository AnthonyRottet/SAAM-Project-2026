"""
SAAM Project 2026 - Group BR (North America + Europe / Scope 1+2)
=================================================================
Full analysis engine: Part I (standard allocation) + Part II (carbon-aware
allocation). Builds five portfolios, computes financial and carbon metrics,
saves all report figures to report/figures/, and dumps results to tools/results.json
and Part_II_Results.xlsx.

Portfolios
----------
  P_vw         Value-weighted benchmark (monthly cap-weighted)
  P_mv         Long-only global minimum-variance ("active investor")
  P_mv(0.5)    Min-variance with CF <= 50% of P_mv carbon footprint
  P_vw(0.5)    Tracking-error min. with CF <= 50% of P_vw carbon footprint
  P_vw(NZ)     Net-zero: CF cut 10%/year cumulatively vs P_vw 2013 footprint

Sample (per instructor's MSF_SAAM_Comments.pdf correction)
----------------------------------------------------------
  First allocation: Dec 2013   Last allocation: Dec 2023
  Performance window: Jan 2014 -> Dec 2024  (T = 132 months)
"""

import json
import os
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "data/"
PLOTS_DIR = "report/figures/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Allocation years: Dec 2013 ... Dec 2023 -> performance Jan 2014 ... Dec 2024
YEARS = list(range(2013, 2024))
THETA = 0.10  # net-zero annual decarbonisation rate

# Consistent colour palette across all figures
COL = {
    "vw": "#1f3a93",    # value-weighted benchmark   - navy
    "mv": "#c0392b",    # minimum-variance           - dark red
    "mv05": "#e67e22",  # min-variance 50% decarb.   - orange
    "vw05": "#16a085",  # tracking-error 50% decarb. - teal
    "nz": "#27ae60",    # net-zero                   - green
}

# ===========================================================================
# 1. LOAD AND RESHAPE DATA
# ===========================================================================
static = pd.read_excel(DATA_DIR + "Static_2025.xlsx").drop_duplicates("ISIN")
name_map = static.set_index("ISIN")["NAME"]
country_map = static.set_index("ISIN")["Country"]
region_map = static.set_index("ISIN")["Region"]
our_isins = static[static["Region"].isin(["AMER", "EUR"])]["ISIN"].tolist()


def load_and_reshape(filename, isins):
    """Load a monthly Excel file and pivot to a (dates x ISIN) numeric matrix."""
    df = pd.read_excel(DATA_DIR + filename).drop_duplicates("ISIN")
    df = df[df["ISIN"].isin(isins)]
    date_cols = [c for c in df.columns if c not in ("NAME", "ISIN")]
    matrix = df.set_index("ISIN")[date_cols].T
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_yearly(filename, isins):
    """Load an annual Excel file -> (ISIN x year) numeric matrix."""
    df = pd.read_excel(DATA_DIR + filename).drop_duplicates("ISIN")
    df = df[df["ISIN"].isin(isins)]
    year_cols = [c for c in df.columns if isinstance(c, (int, float))]
    return df.set_index("ISIN")[year_cols].apply(pd.to_numeric, errors="coerce")


prices = load_and_reshape("DS_RI_T_USD_M_2025.xlsx", our_isins)
mktcap = load_and_reshape("DS_MV_T_USD_M_2025.xlsx", our_isins)

# Scope 1 + Scope 2 emissions (group assignment), forward-filled across years
co2_s1 = load_yearly("DS_CO2_SCOPE_1_Y_2025.xlsx", our_isins)
co2_s2 = load_yearly("DS_CO2_SCOPE_2_Y_2025.xlsx", our_isins)
co2 = co2_s1.add(co2_s2, fill_value=0).ffill(axis=1)
revenue = load_yearly("DS_REV_Y_2025.xlsx", our_isins).ffill(axis=1)

rf = pd.read_excel(DATA_DIR + "Risk_Free_Rate_2025.xlsx")
rf.columns = ["Date", "RF"]
rf["Date"] = pd.to_datetime(rf["Date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
rf = rf.set_index("Date")
rf["RF"] = rf["RF"] / 100 / 12  # annualised % -> monthly decimal

print(f"Loaded {len(our_isins)} AMER+EUR firms | {prices.shape[0]} monthly dates")

# ===========================================================================
# 2. DATA CLEANING AND RETURNS
# ===========================================================================
prices = prices.dropna(axis=1, how="all")                 # drop unmatched ISINs
mktcap = mktcap.reindex(columns=prices.columns)
prices = prices.where(prices >= 0.5)                       # low prices -> missing
returns = prices.pct_change().iloc[1:]                     # simple monthly returns

# Delisting: last valid price followed by NaN -> realised return of -100%
n_delist = 0
for col in returns.columns:
    last_valid = prices[col].last_valid_index()
    if last_valid is not None and last_valid < prices.index[-1]:
        nxt = prices.index[prices.index.get_loc(last_valid) + 1]
        if nxt in returns.index:
            returns.loc[nxt, col] = -1.0
            n_delist += 1

print(f"Returns matrix: {returns.shape[0]} months x {returns.shape[1]} firms "
      f"| {n_delist} delistings flagged")


# ===========================================================================
# 3. INVESTMENT SET
# ===========================================================================
def dec(year, index):
    """Return the December year-end date of `year` present in a DatetimeIndex."""
    d = index[(index.month == 12) & (index.year == year)]
    return d[0] if len(d) else None


_elig_cache = {}


def get_eligible(year):
    """Investment set at end of year Y (identical for all five portfolios).

    Criteria: AMER/EUR region, >=36 monthly returns over the 10y window,
    <=50% stale (zero) returns, valid year-end price and market cap, and
    carbon + revenue data available so the carbon footprint and intensity
    are well defined for every firm in the set.
    """
    if year in _elig_cache:
        return _elig_cache[year]
    win = returns.loc[dec(year - 10, returns.index):dec(year, returns.index)]
    valid = win.notna().sum()
    zero_pct = (win == 0).sum() / valid.replace(0, np.nan)
    d_year = dec(year, prices.index)
    has_price = prices.loc[d_year].notna()
    cap = mktcap.loc[d_year]
    has_cap = cap.notna() & (cap > 0)
    has_co2 = pd.Series({
        i: (co2.loc[i, :year].notna().any() if i in co2.index else False)
        for i in win.columns})
    has_rev = pd.Series({
        i: (i in revenue.index and year in revenue.columns
            and pd.notna(revenue.loc[i, year]))
        for i in win.columns})
    mask = (valid >= 36) & (zero_pct <= 0.5) & has_price & has_cap \
        & has_co2 & has_rev
    _elig_cache[year] = mask[mask].index.tolist()
    return _elig_cache[year]


def year_data(year):
    """Pre-compute everything needed for the allocation decided at end of Y."""
    elig = get_eligible(year)
    win = returns.loc[dec(year - 10, returns.index):dec(year, returns.index)]
    ret = win[elig].fillna(0.0)
    cov = ret.cov().values
    cov = cov + np.eye(len(elig)) * 1e-8          # regularise -> positive definite
    d_year = dec(year, prices.index)
    cap = mktcap.loc[d_year, elig].astype(float)  # year-end cap (million USD)
    emis = co2.loc[elig, year].astype(float)      # Scope 1+2 emissions (tonnes)
    rev_m = revenue.loc[elig, year].astype(float) / 1000.0  # thousands -> million USD
    return {
        "year": year,
        "elig": elig,
        "cov": cov,
        "cap": cap,
        "emis": emis,
        "ci": emis / rev_m,          # carbon intensity  (tCO2 / million USD revenue)
        "carb": emis / cap,          # carbon per cap    (tCO2 / million USD invested)
        "w_vw": cap / cap.sum(),     # value-weighted benchmark weights
    }


# ===========================================================================
# 4. OPTIMISERS
# ===========================================================================
def _solve(objective, n, extra_cons):
    """Solve a long-only QP; return cleaned weight vector and solver status."""
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
    """Long-only minimum-variance weights, optional carbon-footprint cap."""
    P = cp.psd_wrap(d["cov"])
    carb = d["carb"].values

    def extra(w):
        return [carb @ w <= cf_limit] if cf_limit is not None else []

    w, status = _solve(lambda w: cp.quad_form(w, P), len(d["elig"]), extra)
    return pd.Series(w, index=d["elig"]), status


def min_tracking_error(d, cf_limit=None):
    """Long-only tracking-error minimisation vs the VW benchmark, optional CF cap."""
    P = cp.psd_wrap(d["cov"])
    wvw = d["w_vw"].values
    carb = d["carb"].values

    def extra(w):
        return [carb @ w <= cf_limit] if cf_limit is not None else []

    w, status = _solve(lambda w: cp.quad_form(w - wvw, P),
                       len(d["elig"]), extra)
    return pd.Series(w, index=d["elig"]), status


# ===========================================================================
# 5. BUILD THE FIVE PORTFOLIOS (allocation weights + carbon metrics)
# ===========================================================================
yd = {Y: year_data(Y) for Y in YEARS}
print("Investment set size:", {Y: len(yd[Y]["elig"]) for Y in YEARS})

w_mv, w_mv05, w_vw05, w_nz = {}, {}, {}, {}
carbon = {p: {"cf": {}, "waci": {}} for p in ("vw", "mv", "mv05", "vw05", "nz")}
feasibility = {"mv05": {}, "vw05": {}, "nz": {}}
solver_status = {}


def _cf(d, w):
    return float(d["carb"].values @ np.asarray(w))


def _waci(d, w):
    return float(d["ci"].values @ np.asarray(w))


# Pass 1: unconstrained portfolios + VW metrics (net-zero baseline needs 2013)
for Y in YEARS:
    d = yd[Y]
    w_mv[Y], st = min_variance(d)
    solver_status[f"mv_{Y}"] = st
    for tag, w in (("vw", d["w_vw"]), ("mv", w_mv[Y])):
        carbon[tag]["cf"][Y] = _cf(d, w)
        carbon[tag]["waci"][Y] = _waci(d, w)

cf_vw_2013 = carbon["vw"]["cf"][2013]
print(f"VW carbon footprint 2013 (net-zero baseline): {cf_vw_2013:.2f} tCO2/$M")

# Pass 2: carbon-constrained portfolios
for Y in YEARS:
    d = yd[Y]
    min_cf = float(d["carb"].min())  # lowest CF a long-only portfolio can reach

    # ---- 3.2  P_mv(0.5): 50% below P_mv footprint -----------------------
    target = 0.5 * carbon["mv"]["cf"][Y]
    limit = max(target, min_cf * 1.0001)
    feasibility["mv05"][Y] = target >= min_cf
    w_mv05[Y], st = min_variance(d, cf_limit=limit)
    solver_status[f"mv05_{Y}"] = st
    carbon["mv05"]["cf"][Y] = _cf(d, w_mv05[Y])
    carbon["mv05"]["waci"][Y] = _waci(d, w_mv05[Y])

    # ---- 3.3  P_vw(0.5): 50% below P_vw footprint -----------------------
    target = 0.5 * carbon["vw"]["cf"][Y]
    limit = max(target, min_cf * 1.0001)
    feasibility["vw05"][Y] = target >= min_cf
    w_vw05[Y], st = min_tracking_error(d, cf_limit=limit)
    solver_status[f"vw05_{Y}"] = st
    carbon["vw05"]["cf"][Y] = _cf(d, w_vw05[Y])
    carbon["vw05"]["waci"][Y] = _waci(d, w_vw05[Y])

    # ---- 4.1  P_vw(NZ): cumulative 10%/year cut vs 2013 VW footprint ----
    target = (1 - THETA) ** (Y - 2013 + 1) * cf_vw_2013
    limit = max(target, min_cf * 1.0001)
    feasibility["nz"][Y] = target >= min_cf
    w_nz[Y], st = min_tracking_error(d, cf_limit=limit)
    solver_status[f"nz_{Y}"] = st
    carbon["nz"]["cf"][Y] = _cf(d, w_nz[Y])
    carbon["nz"]["waci"][Y] = _waci(d, w_nz[Y])

bad = [k for k, v in solver_status.items() if v not in ("optimal", "optimal_inaccurate")]
print("Solver issues:", bad if bad else "none")
for p in ("mv05", "vw05", "nz"):
    infeas = [Y for Y, ok in feasibility[p].items() if not ok]
    print(f"  {p}: target below min feasible CF in years {infeas}" if infeas
          else f"  {p}: all targets feasible")


# ===========================================================================
# 6. BACKTEST (monthly returns, Jan 2014 -> Dec 2024)
# ===========================================================================
def backtest_optimized(weights_by_year):
    """Monthly returns of a portfolio rebalanced annually with weight drift."""
    rows = []
    for Y, w_alloc in weights_by_year.items():
        elig = list(w_alloc.index)
        future = returns.loc[returns.index.year == Y + 1, elig].fillna(0.0)
        w = w_alloc.copy()
        for date, row in future.iterrows():
            rows.append({"Date": date, "Return": float((w * row).sum())})
            w = w * (1 + row)
            w = w / w.sum()
    return pd.DataFrame(rows).set_index("Date")["Return"]


def backtest_vw():
    """Value-weighted benchmark: monthly rebalanced on previous-month caps."""
    rows = []
    for Y in YEARS:
        elig = get_eligible(Y)
        future = returns.loc[returns.index.year == Y + 1, elig].fillna(0.0)
        for date in future.index:
            prev = mktcap.index[mktcap.index.get_loc(date) - 1]
            caps = mktcap.loc[prev, elig].fillna(0.0)
            w = caps / caps.sum()
            rows.append({"Date": date, "Return": float((w * future.loc[date]).sum())})
    return pd.DataFrame(rows).set_index("Date")["Return"]


ret_series = {
    "vw": backtest_vw(),
    "mv": backtest_optimized(w_mv),
    "mv05": backtest_optimized(w_mv05),
    "vw05": backtest_optimized(w_vw05),
    "nz": backtest_optimized(w_nz),
}
print(f"Backtest done: {len(ret_series['vw'])} monthly returns "
      f"({ret_series['vw'].index[0]:%Y-%m} to {ret_series['vw'].index[-1]:%Y-%m})")


# ===========================================================================
# 7. PERFORMANCE STATISTICS
# ===========================================================================
def summary(r):
    """Annualised performance statistics for a monthly return series."""
    rf_m = rf["RF"].reindex(r.index, method="nearest")
    excess = r - rf_m
    T = len(r)
    return {
        "Ann. avg return": r.mean() * 12,
        "Ann. volatility": r.std() * np.sqrt(12),
        "Ann. cum return": (1 + r).prod() ** (12 / T) - 1,
        "Sharpe ratio": excess.mean() / excess.std() * np.sqrt(12),
        "Min": r.min(),
        "Max": r.max(),
        "Cumulative return": (1 + r).prod() - 1,
    }


stats = {p: summary(r) for p, r in ret_series.items()}


def tracking_error(p_returns, bench_returns):
    """Annualised ex-post tracking error of a portfolio vs a benchmark."""
    diff = p_returns - bench_returns
    return diff.std() * np.sqrt(12)


te_ann = {
    "mv": tracking_error(ret_series["mv"], ret_series["vw"]),
    "mv05": tracking_error(ret_series["mv05"], ret_series["vw"]),
    "vw05": tracking_error(ret_series["vw05"], ret_series["vw"]),
    "nz": tracking_error(ret_series["nz"], ret_series["vw"]),
}

# Portfolio value path (V_2013 = 1 USD million, grown by realised annual returns)
value_path = {}
for p, r in ret_series.items():
    annual = (1 + r).groupby(r.index.year).prod()
    v = {2013: 1.0}
    for yr in sorted(annual.index):
        v[yr] = v[yr - 1] * annual[yr]
    value_path[p] = v


# ===========================================================================
# 8. CARBON CONTRIBUTORS (top-10 emitters of the VW benchmark)
# ===========================================================================
contrib = {}  # ISIN -> averaged WACI contribution to the VW portfolio
for Y in YEARS:
    d = yd[Y]
    c = d["w_vw"] * d["ci"]
    for isin, val in c.items():
        contrib[isin] = contrib.get(isin, 0.0) + val / len(YEARS)

top10 = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:10]
top10_rows = []
last = yd[2023]
for isin, avg_c in top10:
    top10_rows.append({
        "ISIN": isin,
        "Name": str(name_map.get(isin, "")),
        "Country": str(country_map.get(isin, "")),
        "Avg WACI contribution": avg_c,
        "CI 2023": float(last["ci"].get(isin, np.nan)),
        "VW weight 2023 %": float(last["w_vw"].get(isin, np.nan)) * 100,
    })


# ===========================================================================
# 9. COMPOSITION DIAGNOSTICS
# ===========================================================================
def effective_n(w):
    w = np.asarray(w)
    return 1.0 / np.sum(w ** 2)


def composition(weights_by_year, label):
    eff = [effective_n(weights_by_year[Y].values) for Y in YEARS]
    n_held = [int((weights_by_year[Y] > 1e-4).sum()) for Y in YEARS]
    return {"label": label,
            "avg_effective_n": float(np.mean(eff)),
            "avg_names_held": float(np.mean(n_held))}


comp = {
    "mv": composition(w_mv, "P_mv"),
    "mv05": composition(w_mv05, "P_mv(0.5)"),
    "vw05": composition(w_vw05, "P_vw(0.5)"),
    "nz": composition(w_nz, "P_vw(NZ)"),
}
# VW benchmark effective N (allocation-date weights)
comp["vw"] = {"label": "P_vw",
              "avg_effective_n": float(np.mean(
                  [effective_n(yd[Y]["w_vw"].values) for Y in YEARS])),
              "avg_names_held": float(np.mean(
                  [len(yd[Y]["elig"]) for Y in YEARS]))}

# Country tilt of P_vw(0.5) vs P_vw at the last allocation (2023)
last_d = yd[2023]
ctry_vw = (last_d["w_vw"].groupby(last_d["w_vw"].index.map(country_map)).sum())
ctry_vw05 = (w_vw05[2023].groupby(w_vw05[2023].index.map(country_map)).sum())
country_tilt = pd.DataFrame({"P_vw": ctry_vw, "P_vw(0.5)": ctry_vw05}).fillna(0.0)
country_tilt["Tilt"] = country_tilt["P_vw(0.5)"] - country_tilt["P_vw"]
country_tilt = country_tilt.sort_values("Tilt")


# ===========================================================================
# 10. FIGURES
# ===========================================================================
plt.style.use("seaborn-v0_8-whitegrid")
SUB = "Group BR  |  North America + Europe  |  Scope 1+2"


def _titles(ax, title):
    """Bold title with a small italic subtitle below it, no overlap."""
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=30)
    ax.text(0.0, 1.045, SUB, transform=ax.transAxes, fontsize=8.5,
            alpha=0.65, style="italic")


def _stagger(values, gap):
    """Spread label y-positions so none are closer than `gap`."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if pos[i] - pos[j] < gap:
            pos[i] = pos[j] + gap
    return pos


def growth_index(r):
    """Growth of 1 USD, anchored to 1.0 at the month before the first return."""
    cum = (1 + r).cumprod()
    start = r.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[start]), cum])


def fig_cumulative(series_keys, labels, title, fname):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ends = []
    for key, lab in zip(series_keys, labels):
        g = growth_index(ret_series[key])
        ax.plot(g.index, g.values, color=COL[key], linewidth=2.1, label=lab)
        ends.append(g.iloc[-1])
    label_y = _stagger(ends, gap=0.05 * (max(ends) - min(ends) + 0.6))
    last_x = growth_index(ret_series[series_keys[0]]).index[-1]
    for key, ev, ly in zip(series_keys, ends, label_y):
        ax.annotate(f"{ev:.2f}", xy=(last_x, ev), xytext=(8, 0),
                    textcoords="offset points", color=COL[key],
                    fontweight="bold", va="center", fontsize=9,
                    annotation_clip=False)
        if abs(ly - ev) > 1e-9:
            ax.plot([last_x], [ev], "o", color=COL[key], markersize=3)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Cumulative growth of 1 USD")
    ax.set_xlabel("Date")
    _titles(ax, title)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS_DIR + fname)


def fig_carbon(metric, series_keys, labels, title, ylabel, fname,
               target_line=None):
    fig, ax = plt.subplots(figsize=(11, 5.3))
    for key, lab in zip(series_keys, labels):
        vals = [carbon[key][metric][Y] for Y in YEARS]
        ax.plot(YEARS, vals, "o-", color=COL[key], linewidth=2.1,
                markersize=5, label=lab)
    if target_line is not None:
        ax.plot(YEARS, target_line, "--", color="black", linewidth=1.5,
                alpha=0.7, label="Net-zero target ceiling")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Allocation year Y")
    ax.set_xticks(YEARS)
    _titles(ax, title)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS_DIR + fname)


def fig_top10():
    fig, ax = plt.subplots(figsize=(11, 5.3))
    rows = top10_rows[::-1]
    labels = [r["Name"][:34] for r in rows]
    vals = [r["Avg WACI contribution"] for r in rows]
    ax.barh(labels, vals, color=COL["vw"], alpha=0.85)
    ax.set_xlabel("Average contribution to VW portfolio WACI (tCO2 / $M revenue)")
    _titles(ax, "Top-10 carbon-intensity contributors - value-weighted portfolio")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR + "top10_emitters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS_DIR + "top10_emitters.png")


# Part I
fig_cumulative(["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
               "Part I - Cumulative performance (2014-2024)",
               "cum_part1.png")
# 3.1 carbon of the two base portfolios
fig_carbon("waci", ["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
           "WACI - value-weighted vs minimum-variance portfolio",
           "WACI (tCO2 / million USD revenue)", "waci_base.png")
fig_carbon("cf", ["vw", "mv"], ["Value-Weighted", "Minimum-Variance"],
           "Carbon footprint - value-weighted vs minimum-variance portfolio",
           "Carbon footprint (tCO2 / million USD invested)", "cf_base.png")
fig_top10()
# 3.2
fig_cumulative(["mv", "mv05"], ["P_mv", "P_mv(0.5)"],
               "3.2 - Minimum-variance: standard vs 50% decarbonised",
               "cum_mv05.png")
fig_carbon("cf", ["mv", "mv05"], ["P_mv", "P_mv(0.5)"],
           "3.2 - Carbon footprint: P_mv vs P_mv(0.5)",
           "Carbon footprint (tCO2 / million USD invested)", "cf_mv05.png")
# 3.3
fig_cumulative(["vw", "vw05"], ["P_vw", "P_vw(0.5)"],
               "3.3 - Value-weighted vs 50% decarbonised tracking portfolio",
               "cum_vw05.png")
fig_carbon("cf", ["vw", "vw05"], ["P_vw", "P_vw(0.5)"],
           "3.3 - Carbon footprint: P_vw vs P_vw(0.5)",
           "Carbon footprint (tCO2 / million USD invested)", "cf_vw05.png")
# 4 net zero
nz_target = [(1 - THETA) ** (Y - 2013 + 1) * cf_vw_2013 for Y in YEARS]
fig_cumulative(["vw", "vw05", "nz"], ["P_vw", "P_vw(0.5)", "P_vw(NZ)"],
               "4.2 - Value-weighted family: benchmark, 50% cut, net-zero",
               "cum_netzero.png")
fig_carbon("cf", ["vw", "vw05", "nz"], ["P_vw", "P_vw(0.5)", "P_vw(NZ)"],
           "4.1 - Carbon footprint trajectories vs the net-zero glide path",
           "Carbon footprint (tCO2 / million USD invested)", "cf_netzero.png",
           target_line=nz_target)


# ===========================================================================
# 11. EXPORT RESULTS
# ===========================================================================
PORT_LABEL = {"vw": "P_vw", "mv": "P_mv", "mv05": "P_mv(0.5)",
              "vw05": "P_vw(0.5)", "nz": "P_vw(NZ)"}

results = {
    "sample": {"years": YEARS, "months": len(ret_series["vw"]),
               "n_firms_per_year": {Y: len(yd[Y]["elig"]) for Y in YEARS}},
    "stats": {PORT_LABEL[p]: stats[p] for p in stats},
    "tracking_error_ann": {PORT_LABEL[p]: te_ann[p] for p in te_ann},
    "carbon": {PORT_LABEL[p]: {
        "cf": carbon[p]["cf"], "waci": carbon[p]["waci"]} for p in carbon},
    "carbon_avg": {PORT_LABEL[p]: {
        "cf": float(np.mean(list(carbon[p]["cf"].values()))),
        "waci": float(np.mean(list(carbon[p]["waci"].values())))}
        for p in carbon},
    "cf_vw_2013_baseline": cf_vw_2013,
    "feasibility": feasibility,
    "composition": comp,
    "value_path": {PORT_LABEL[p]: value_path[p] for p in value_path},
    "top10_emitters": top10_rows,
    "country_tilt_2023": country_tilt.round(4).to_dict(),
    "n_delistings": n_delist,
}

with open("tools/results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print("saved tools/results.json")

with pd.ExcelWriter("Part_II_Results.xlsx") as xl:
    pd.DataFrame({PORT_LABEL[p]: stats[p] for p in stats}).to_excel(
        xl, sheet_name="Performance")
    carb_df = pd.DataFrame({
        f"{PORT_LABEL[p]} CF": carbon[p]["cf"] for p in carbon})
    waci_df = pd.DataFrame({
        f"{PORT_LABEL[p]} WACI": carbon[p]["waci"] for p in carbon})
    pd.concat([carb_df, waci_df], axis=1).to_excel(xl, sheet_name="Carbon")
    pd.DataFrame({PORT_LABEL[p]: ret_series[p] for p in ret_series}).to_excel(
        xl, sheet_name="Monthly returns")
    pd.DataFrame(top10_rows).to_excel(xl, sheet_name="Top10 emitters", index=False)
print("saved Part_II_Results.xlsx")

# ---------------------------------------------------------------------------
# CONSOLE SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PERFORMANCE SUMMARY (2014-2024, 132 months)")
print("=" * 72)
hdr = f"{'Metric':<20}" + "".join(f"{PORT_LABEL[p]:>13}" for p in stats)
print(hdr)
for k in ("Ann. avg return", "Ann. volatility", "Ann. cum return",
          "Sharpe ratio", "Min", "Max"):
    print(f"{k:<20}" + "".join(f"{stats[p][k]:>13.4f}" for p in stats))
print(f"{'Ann. tracking err':<20}" + f"{'-':>13}"
      + "".join(f"{te_ann[p]:>13.4f}" for p in ("mv", "mv05", "vw05", "nz")))
print("\nCARBON (sample average)")
for p in carbon:
    print(f"  {PORT_LABEL[p]:<12} CF={results['carbon_avg'][PORT_LABEL[p]]['cf']:8.2f}"
          f"   WACI={results['carbon_avg'][PORT_LABEL[p]]['waci']:8.2f}")
print("\nDone.")
