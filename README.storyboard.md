# README Storyboard: financial-ml-regime-analysis

## 1. Purpose & reader
- Question the README answers: "What did this person build, what did they find, and can I trust it?" A recruiter must get that from the top screen without knowing what "v1", "arm", "stationary twin", "reading 2" or "primary pair" mean. [Filled, user: "it spams v1 and arm but they don't know what those are"]
- Reader: graduate quant / markets-research recruiter first, quant interviewer second. [Filled, `docs/spec.md:7` "I built this project to apply for graduate quant and markets research programmes"; user stories `docs/spec.md:24-48`]
- Stakes / why now: the recruiter skims for 30 seconds; a quant interviewer then checks rigour. Story 22 asks for the regime question "in plain language, so that I can follow the economic reasoning without reading code". [Filled, `docs/spec.md:47`]

## 2. Key message
I rebuilt my own trading-model project after finding its headline numbers were not real returns, ran a pre-registered backtest, and found the model has no edge over simply holding the stock part-time. [Filled, README.md headline bullets; `results/` via `docs/img/excess_ci.svg`]

## 3. Evidence
- Sources: 141 tests pass locally (`uv run pytest`, 63.9 s, run this session); CI green on `main` (`gh run list`, run 45906033349); official outputs in `results/` committed unopened at `dc2c165`, code tagged `v2-official-run` (`6391a43`); executed notebook `notebook/v2_analysis.ipynb` (`46074e6`); charts regenerable via `docs/img/make_charts.py`. [Filled]
- Caveats: result is a null (no edge); one illiquid Indonesian stock; data was not unseen, only the plan was fixed first; low statistical power (buy-and-hold Sharpe CI [−0.31, 1.71]). [Filled, README.md "What the pre-registration proves" and "Limitations"]

### Insight hierarchy
- Primary insight: neither model beats a fair benchmark (buy-and-hold scaled to the time spent in the stock); every confidence interval includes zero or sits below it. [Filled, `docs/img/excess_ci.svg`; README table: −4.97 bps/day [−12.37, +2.45] and −6.01 [−13.60, +1.48]]
- Supporting evidence: headline table (return, Sharpe, excess return, break-even); equity chart (−2.5% / +8.8% against benchmarks +50.3% / +80.6%); Test 1 both gates fail; cost stresses make it worse.
- Context: the original project's "Sharpe 9.59" was a hit score, not a return (`v1` tag, `calculate_weighted_returns`); how to reproduce with `uv`; repo layout.
- Interesting (non-essential): meta-labelling ablation, clustered pair, baselines, per-window table, KEEN price table, pre-registration commit table. Cut to `<details>`.

## 4. Narrative arc
### Hook (the problem)
My first version claimed a validation Sharpe of 9.59 and blamed its later collapse on a market "regime change". Those numbers were a +1/−1 hit score, not returns, and the pipeline leaked. [Filled, README "What v1 got wrong"; `v1` tag `acc6360`, 2025-10-30] | Visual: none

### Conflict (why it's hard)
A simpler explanation also fits: the model was fed raw prices, and tree models can't handle prices outside the range they trained on. To tell the two stories apart you need a fair comparison, real costs, and rules fixed before you look. [Filled, `docs/spec.md:13`, README "v1's numbers were not returns"] | Visual: none

### Insight (the key decision)
Run two models that differ in one thing only: one uses raw prices (as the original did), the other uses ratios to today's price (returns-based), through the same corrected, leak-free walk-forward backtest with Indonesian trading costs, and write the success criteria down before running. [Filled, `backtest/features.py:1,64-70`, `docs/spec.md` Solution; commits `6391a43` → `dc2c165` → `46074e6`] | Visual: `docs/img/pipeline.svg` (labels need the plain names)

### Resolution (the result)
Neither model beats the fair benchmark; raw vs scale-free makes no detectable difference; so there was no edge for a regime change to break. [Filled, README results table, `docs/img/excess_ci.svg`, `docs/img/headline.svg`] | Visual: `excess_ci.svg` first, results table, `headline.svg`

## 5. Design plan
- Section order and length budget: title + one-line message + "In plain terms" 3-bullet block + forest plot above the first `##`. Story sections each under ~120 words. All jargon defined once in a "Terms used below" line, then used consistently. Detailed tables in `<details>`.
- Naming (the core fix): [Filled, user: "Price-level / Returns-based"; charts regenerated; Amendment 6 drafted; details collapsed in README]
  - "v1" → "the original project (v1, Oct 2025)" on first use, then "the original".
  - "v1-style arm" → **Price-level model**; "stationary twin" → **Returns-based model**; "arm" → "model"; "primary pair" → "the two models".
  - "Reading 1/2/3" → **Question 1: does feature form matter? / 2: is there any edge? / 3: did performance change after 2024? (descriptive)**.
  - "exposure-matched benchmark" kept (glossary term, used in the charts), defined in plain words in a "Terms used below" line.
  - Spec story 23 still holds: the raw-price model is described as "the original's features inside the corrected pipeline".
- Visuals: `excess_ci.svg`, `headline.svg` regenerated from `results/` with the plain labels (`docs/img/make_charts.py`); `pipeline.svg` relabelled in `docs/img/pipeline.html`. No GIF, no demo.
- Palette and font: existing chart palette (`#0072B2` accent, grays), Inter, already per the Visuals defaults (`docs/img/make_charts.py:21-27`).

## 6. Call to action
Read the pre-registered spec, or reproduce in three commands with `uv`. [Filled, README "Reproducing", `docs/spec.md`]
