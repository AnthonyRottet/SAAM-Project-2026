# SAAM Project 2026 — Group BR — Deliverables Guide

Region: **North America + Europe** · Scope: **1 + 2** · Final deadline: **29 May 2026**

This file explains what has been produced and how to assemble the final
submission.

---

## 1. What is in the repository

| File / folder | Role |
|---|---|
| `SAAM_Project_2026.ipynb` | **Final notebook** — runs top to bottom, reproduces every report table and figure (Part I + Part II). Already executed; 0 errors. |
| `report/report.tex` | **Report** source (LaTeX, ~20 pages). |
| `report/sales_pitch.tex` | **Sales pitch** source (LaTeX, 1 page). |
| `report/video_script.md` | Timed 10-minute **video** presentation script. |
| `report/figures/` | The 10 figures used by the report; (re)generated here when the notebook runs. |
| `tools/saam_analysis.py` | Stand-alone analysis engine (verification copy). |
| `tools/build_notebook.py` | Rebuilds the notebook from source cells. |
| `tools/results.json` | All numeric results (used to fill the report). |
| `Part_II_Results.xlsx` | Performance, carbon and monthly-return tables. |
| `Part_1.xlsx`, `Part_I_Results.xlsx` | Part I result spreadsheets (kept for reference). |
| `data/` | Input Datastream files (needed to run the notebook). |

The Part I result spreadsheets (`Part_1.xlsx`, `Part_I_Results.xlsx`) are kept
for reference but are **superseded** by `SAAM_Project_2026.ipynb`, which covers
both parts on the corrected 2014–2024 sample.

---

## 2. Compiling the two PDFs on Overleaf

The `report/` folder is self-contained. Steps:

1. Zip the `report/` folder (it already contains `figures/`).
2. On [overleaf.com](https://www.overleaf.com): **New Project → Upload Project**
   and select the zip.
3. The report and the pitch are two separate documents in one project. Use the
   **Menu → Main document** selector to switch between them:
   - set main document to `report.tex` → **Recompile** → download `report.pdf`;
   - set main document to `sales_pitch.tex` → **Recompile** → download
     `sales_pitch.pdf`.
4. Overleaf's default compiler (pdfLaTeX, TeX Live) handles every package used
   (`makecell`, `booktabs`, `fancyhdr`, `hyperref`, …). No extra setup needed.

> Tip: add your three group members' names under `\author{...}` in both files
> before compiling — currently only "Group BR" is shown.

---

## 3. Running the notebook

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
jupyter notebook SAAM_Project_2026.ipynb   # then Run All
```

Runtime is ~2–3 minutes. It needs only the `data/` folder and writes
`report/figures/` and `Part_II_Results.xlsx`. All paths are relative — no manual edits.

---

## 4. Final submission folder (single folder, per the brief)

```
SAAM_Project_2026_GroupBR/
├── report.pdf                 # compiled from report/report.tex
├── sales_pitch.pdf            # compiled from report/sales_pitch.tex
├── SAAM_Project_2026.ipynb    # the final notebook
├── data/                      # so the grader can run the notebook
└── video.mp4                  # record using report/video_script.md
```

Still to do by the group: compile the two PDFs (Section 2), and record the
10-minute video (script in `report/video_script.md`).

---

## 5. Key results (back-test Jan 2014 – Dec 2024, 132 months)

| Portfolio | Ann. return | Volatility | Sharpe | Avg CF | CF vs P_vw |
|---|---|---|---|---|---|
| P_vw — value-weighted   | 10.62% | 14.62% | 0.72 | 119.1 | — |
| P_mv — minimum-variance |  5.96% | 14.04% | 0.42 | 599.6 | +404% |
| P_mv(0.5) — 50% cut     |  6.07% | 13.03% | 0.46 | 299.8 | +152% |
| P_vw(0.5) — 50% cut     | 10.16% | 14.63% | 0.69 |  59.2 | −50% |
| P_vw(NZ) — net-zero     | 10.23% | 14.64% | 0.69 |  92.7 | −22% |

**Headline.** The minimum-variance portfolio carries ~5× the benchmark's carbon
footprint (low-volatility = a hidden bet on carbon-heavy utilities). A 50%
footprint cut via a tracking-error overlay costs only 0.03 of Sharpe ratio and
1.1% tracking error; decarbonising the minimum-variance portfolio even improves
it. Over this sample, decarbonisation is close to free.
