# Video Presentation Script — SAAM Project 2026, Group BR
**North America + Europe · Scope 1 + 2 · Target length: 10:00 (hard cap)**

A speaker script with slide cues and timings. Aimed at a professional audience.
Read at a calm pace (~140 words/min). Bracketed `[...]` notes are stage
directions, not to be read aloud.

---

## Slide 1 — Title (0:00 – 0:30)

> Good morning. We are Group BR. Our project applies the climate-aware
> asset-management toolkit to a concrete portfolio problem: how much does it
> actually cost to decarbonise an equity portfolio?
>
> Our universe is North American and European equities; our emission perimeter
> is Scope 1 plus Scope 2. We back-test five portfolios over eleven years, from
> 2014 to 2024. The short answer we will defend today: done properly,
> decarbonisation is almost free — and sometimes it even pays.

---

## Slide 2 — Objective & roadmap (0:30 – 1:15)

> The project has two parts. In Part I we build two reference portfolios: the
> value-weighted benchmark, which is simply the market, and the global
> minimum-variance portfolio, the textbook low-risk strategy.
>
> In Part II we add a carbon objective and build three decarbonised portfolios:
> a minimum-variance portfolio with its footprint halved; a portfolio that
> tracks the benchmark while halving the footprint; and a net-zero portfolio
> that cuts its footprint by ten percent every year.
>
> We will compare them on two dimensions at once — financial performance and
> carbon — and that comparison is the heart of the talk.

---

## Slide 3 — Data & method, high level (1:15 – 2:45)

> A few words on the engine, kept deliberately high level.
>
> We start from raw Datastream data on about thirteen hundred firms. We clean
> it: securities with no match are dropped, prices below 0.5 are treated as
> missing, and delistings are booked as a minus-one-hundred-percent return so
> defaults are properly penalised.
>
> Each December we form an investment set — firms with enough return history,
> not stale, and with carbon data — and we keep the *same* set for every
> portfolio, so comparisons are clean.
>
> We estimate the covariance matrix from ten years of monthly returns. Every
> portfolio is a long-only quadratic program. The key technical point: the
> carbon footprint is *linear* in the weights, so a carbon ceiling is just one
> extra linear constraint — the problem stays easy to solve.
>
> We rebalance once a year and measure performance monthly. One note on the
> sample: following the instructor's consistency guidance, the last allocation
> is December 2023 and we stop the performance window in December 2024.

---

## Slide 4 — Part I results (2:45 – 4:00)

> [Show cumulative-performance chart.]
>
> Here is Part I. One dollar in the benchmark becomes 2.85 — an annualised
> ten percent, Sharpe ratio 0.72. The minimum-variance portfolio reaches only
> 1.72, a Sharpe of 0.42.
>
> Two things stand out. First, minimum variance barely reduced *realised*
> volatility — fourteen-point-zero versus fourteen-point-six percent. With a
> thousand firms and only a hundred-twenty months of data, the covariance
> matrix is noisy, and the optimiser concentrates in about thirty names whose
> past calm did not last. Its worst month was actually deeper than the market's.
>
> Second — and this is the bridge to Part II — minimum variance underperformed
> because, in a decade-long bull market, defensive stocks lagged.

---

## Slide 5 — The carbon surprise (4:00 – 5:15)

> [Show carbon-footprint chart: MV vs VW.]
>
> Now the carbon picture, and this is our most striking finding.
>
> The benchmark's footprint falls steadily over the decade — the market
> decarbonised on its own. But the minimum-variance portfolio is the red line:
> on average *five times* the benchmark's footprint, and wildly unstable —
> swinging from under two hundred to nearly seventeen hundred tonnes per
> million dollars.
>
> Why? The lowest-volatility stocks in this universe are regulated electric
> utilities — and utilities are extremely carbon-intensive. So an investor who
> simply minimises variance is, without knowing it, taking a huge carbon bet.
>
> [Show top-10 chart.] The intensity is concentrated: a handful of US
> utilities, plus Holcim in cement and the oil majors, drive most of it. That
> concentration is exactly what makes decarbonisation cheap — as we will see.

---

## Slide 6 — Decarbonised portfolios (5:15 – 7:15)

> [Show the three decarbonised results, one at a time.]
>
> Portfolio one: minimum variance with the footprint halved. The carbon cut
> costs nothing — in fact volatility *falls* and the Sharpe ratio *rises*, from
> 0.42 to 0.46. The carbon constraint pushes the optimiser off its concentrated
> utility bets, which also happened to be its risky bets. Decarbonising and
> de-risking pointed the same way.
>
> Portfolio two: the passive investor — stay close to the benchmark, halve the
> footprint. The result is almost magical: the two return paths are visually
> identical. The cost is three hundredths of a point of Sharpe ratio — 0.69
> versus 0.72 — and an annual
> tracking error of just one-point-one percent — for a permanent fifty-percent
> cut in emissions.
>
> Portfolio three: net zero — cut the footprint ten percent a year. Same story:
> Sharpe 0.69, tracking error one-point-one percent. The portfolio hugs its
> glide path year after year.
>
> The reason all of this is cheap: the carbon footprint sits in a small set of
> names. Re-weight those, leave the diversified core of a thousand stocks
> untouched, and performance barely moves.

---

## Slide 7 — Trade-offs, limitations, honesty (7:15 – 8:45)

> We want to be honest about what these numbers do *not* say.
>
> First, the result is regime-dependent. 2014 to 2024 rewarded low-carbon
> technology leaders. In a fossil-led market — think 2022 — a carbon screen
> would cost real return. The one-point-one-percent tracking-error budget is
> what bounds that risk.
>
> Second, estimation. The minimum-variance family rests on a noisy covariance
> matrix; shrinkage estimation would be our first improvement.
>
> Third, the net-zero glide path is anchored to 2013. Because the benchmark
> decarbonised so much on its own, that anchor becomes easy over time — a
> genuinely demanding net-zero rule should track a moving benchmark.
>
> Fourth, feasibility: a long-only portfolio cannot decarbonise forever — there
> is a mathematical floor.
>
> And fifth, the data: emissions are self-reported, lagged, and Scope 3 is
> excluded.

---

## Slide 8 — Takeaways (8:45 – 9:45)

> Three takeaways.
>
> One: low-volatility investing is a hidden carbon bet. Screening for carbon is
> not only an ethical choice — it is a useful risk control.
>
> Two: for a passive investor, halving the carbon footprint — and following a
> net-zero glide path — costs almost nothing, because emissions are
> concentrated in a few names.
>
> Three: *how* you decarbonise matters more than *whether* you do. A light
> overlay on a diversified benchmark is far more efficient than constraining an
> already concentrated portfolio.
>
> Over this sample, decarbonisation was close to a free lunch. We would not
> promise that in every market — but the evidence that it can be nearly costless
> is strong.
>
> Thank you. We are happy to take questions.

---

## Timing summary

| Segment | Slide | Window | Length |
|---|---|---|---|
| Title | 1 | 0:00–0:30 | 0:30 |
| Objective & roadmap | 2 | 0:30–1:15 | 0:45 |
| Data & method | 3 | 1:15–2:45 | 1:30 |
| Part I results | 4 | 2:45–4:00 | 1:15 |
| The carbon surprise | 5 | 4:00–5:15 | 1:15 |
| Decarbonised portfolios | 6 | 5:15–7:15 | 2:00 |
| Trade-offs & limitations | 7 | 7:15–8:45 | 1:30 |
| Takeaways | 8 | 8:45–9:45 | 1:00 |
| Buffer / questions | — | 9:45–10:00 | 0:15 |

**Suggested figures per slide** (all in `plots/`): Slide 4 → `cum_part1.png`;
Slide 5 → `cf_base.png` then `top10_emitters.png`; Slide 6 → `cum_mv05.png`,
`cum_vw05.png`, `cf_netzero.png`.
