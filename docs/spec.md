> Copy of the v2 spec from [#1](https://github.com/Abhinawap/financial-ml-regime-analysis/issues/1). **#1 is the original**; if they disagree, #1 wins. This copy includes the amendments recorded in #1's comments.

> **Pre-registration.** This spec fixes the analysis plan, every free setting and the readings of the result before any v2 out-of-sample run. Terms follow `GLOSSARY.md`. The facts behind the cost and short-selling decisions are in `docs/research/idx-trading-frictions.md`.

## Problem Statement

I built this project to apply for graduate quant and markets research programmes. Its headline numbers (validation Sharpe 9.59, test −17.55%, walk-forward −124%, p = 0.011) would not survive a quant interviewer's first question, because none of them is a return:

- The "returns" are a **hit score**: +1 when the predicted label matched the actual label and −1 when not, times bet size. The "Sharpe" annualises those ±1 values with √252 across overlapping 10-day labels, and the walk-forward "total return" adds hit scores across windows.
- The 30 lag features are raw closing prices. Tree models cannot extrapolate to price levels they never trained on, so the 2024–25 failure may come from the features rather than from the regime change the README claims.
- Walk-forward feature selection reuses features chosen on the full 2019–2023 training set, which overlaps later test windows.
- The walk-forward meta-model trains on the primary model's in-sample predictions, and its target (`label != 0`) differs from the main pipeline's ("prediction equals label").
- The last 10 rows of every window carry padded "timeout, return 0" labels, and the barrier loop looks 9 days ahead instead of 10.
- In the main train/validation/test split, labels are computed on the full series before splitting, so labels near each boundary read prices from the next period.
- Consecutive walk-forward test windows overlap by half, so the 12 "independent" windows and their t-test overstate significance.
- There are no transaction costs, and the strategy takes short positions in a stock that retail investors could not short from March 2020.
- MDA uses an unpurged splitter and an unseeded shuffle. Cluster representatives are picked by raw variance, which depends on units. `class_weight='balanced'` is passed to XGBoost, which ignores it.
- The README has factual and housekeeping errors: "+105% peak-to-trough", a validation Sharpe of 9.59 presented as a success, a misnamed "ETF trick", wrong folder names, a missing data README, a placeholder clone URL, and a notebook that cannot find its data from the repo layout.

So the project's central claim, that a regime change broke a mean-reversion strategy, has never been tested. A reviewer who finds any one of these errors stops reading.

## Solution

Rebuild the evaluation as a corrected, pre-registered walk-forward backtest that asks one question: **did v1's out-of-sample failure come from raw price-level features or from a regime change?**

- A backtest module computes real strategy returns from prices: long-or-flat, entries and exits at the next close, costs from IDX market rules, and a continuous out-of-sample series built from non-overlapping test windows.
- Four feature arms run through the same corrected pipeline. The gated comparison is the **primary pair**: the v1-style arm (v1's lag handling) against its stationary twin, which differs only in the form of its features.
- Every setting, statistic, threshold and reading is fixed in this spec and committed before the official run. Development uses only synthetic prices and rows the out-of-sample series does not contain.
- The README is rewritten around whatever the results show. It leads with one figure that puts the v1-style arm, the stationary twin and the exposure-matched benchmark side by side, and it keeps a short section on what v1 got wrong.

## User Stories

### Graduate recruiter or quant interviewer reading the project

1. As a quant interviewer, I want the headline figure to show real compounded strategy returns, so that I can judge the strategy rather than a hit score.
2. As a quant interviewer, I want Sharpe ratios computed from daily strategy returns with a stated risk-free rate, so that I can compare them with figures I know.
3. As a quant interviewer, I want to see the strategy against an exposure-matched benchmark, so that I can tell skill from the stock's rise.
4. As a quant interviewer, I want excess return over that benchmark with a confidence interval, so that I can see whether any skill clears noise.
5. As a quant interviewer, I want results net of realistic IDX trading costs, so that I know whether an edge would survive trading.
6. As a quant interviewer, I want the break-even cost reported, so that I can see how much cost the strategy can absorb.
7. As a quant interviewer, I want cost stress tests at 150 and 200 bps round trip, so that the conclusion does not hang on one cost guess.
8. As a quant interviewer, I want the strategy to be long-only, so that it never takes positions that could not have been traded.
9. As a quant interviewer, I want entries and exits to fill at the close after the decision, so that the backtest uses no information it would not have had.
10. As a quant interviewer, I want the out-of-sample series to come from non-overlapping test windows joined in date order, so that its statistics are not inflated by reused days.
11. As a quant interviewer, I want block-bootstrap confidence intervals, so that the autocorrelation from multi-day holding periods is accounted for.
12. As a quant interviewer, I want simple baselines run through the same trade rules, so that I can see whether the ML adds anything over a rule of thumb.
13. As a quant interviewer, I want trade counts printed beside every result, so that I can discount results built on a handful of trades.
14. As a quant interviewer, I want a per-window table of strategy, benchmark and excess returns, so that I can see whether one window drives the total.
15. As a quant interviewer, I want a price-only table of the early and late periods, so that I can check v1's regime description against the price data itself.
16. As a quant interviewer, I want the readings of the result fixed before the run, so that I can trust they were not written to fit it.
17. As a quant interviewer, I want the README to state which data had already been seen before the plan was written, so that I know exactly what the pre-registration proves.
18. As a quant interviewer, I want a short section on v1's errors, so that I can see the author find and fix their own mistakes.
19. As a quant interviewer, I want the v1 errors linked to the exact code at a fixed tag, so that I can verify them myself.
20. As a quant interviewer, I want the known limitations listed, so that I can see the author knows where the model of trading falls short.
21. As a quant interviewer, I want the capacity of the strategy stated, so that I do not read the returns as scalable.
22. As a markets research recruiter, I want the README to explain what the regime question is and what the data says about it in plain language, so that I can follow the economic reasoning without reading code.
23. As a recruiter, I want the headline arm labelled as v1's features inside the corrected pipeline, so that I am not misled about how close it is to v1.
24. As a recruiter, I want the meta-labelling result reported even if it shows nothing, so that I can see the author reports null results.

### Reviewer reproducing the results

25. As a reviewer, I want the data file frozen and its hash published, so that I reproduce the exact numbers.
26. As a reviewer, I want every library version pinned in a lock file with the Python version recorded, so that a new release does not change the results.
27. As a reviewer, I want the commit that produced the official results tagged, so that I can check out and rerun it.
28. As a reviewer, I want all seeds fixed and listed, so that reruns are deterministic.
29. As a reviewer, I want the v2 notebook to read data from the repo's data folder, so that it runs from a fresh clone.
30. As a reviewer, I want any post-run fix logged as a dated deviation with both results shown, so that I can see how the result moved.
31. As a reviewer, I want exploratory changes kept apart from the headline, so that choices made after the run never replace the pre-registered result.
32. As a reviewer, I want the v1 notebook preserved at a tag and marked with a banner on main, so that I can tell which analysis I am reading.

### Project author

33. As the author, I want one gated comparison that tests whether raw price levels caused the failure, so that the project answers its own question.
34. As the author, I want the comparison to be two-sided, so that a result favouring the level features is reported as a rejection of the level-feature theory.
35. As the author, I want an inconclusive outcome defined in advance, so that I cannot argue a noisy gap into a finding.
36. As the author, I want selection noise measured with five seeds, so that I can tell real differences between arms from differences in which features MDA happened to pick.
37. As the author, I want a descriptive clustered pair, so that I can show what clustered MDA does when it is used as the method intends.
38. As the author, I want v1's hyperparameters frozen, so that only bug fixes and the agreed design changes separate v1 from v2.
39. As the author, I want the meta-labelling ablation trained on out-of-fold predictions, so that its result is not an artefact of in-sample memorisation.
40. As the author, I want the README repairs done in the rewrite, so that no housekeeping error undercuts the analysis.
41. As the author, I want my stated expectations written down before the run, so that the result is read against them rather than explained afterwards.
42. As the author, I want the volatility-scaled barrier run and the liquid-asset rerun specified separately before they run, so that the second pass is pre-registered too.

### Implementing agent

43. As the implementing agent, I want the trading rules written as glossary terms, so that the trade simulator has one unambiguous definition to implement.
44. As the implementing agent, I want the settings passed into the walk-forward run as an argument, so that tests can use small windows and few trees while the official run uses the frozen block.
45. As the implementing agent, I want a hand-checkable trade simulator seam, so that PnL and cost tests have known answers.
46. As the implementing agent, I want the walk-forward run to return a per-window training fingerprint and selected features, so that a purge leak is detectable by a test.
47. As the implementing agent, I want a statistics seam that takes return series and returns numbers, so that the bootstrap and excess-return functions are tested directly.
48. As the implementing agent, I want development restricted to synthetic prices and rows before the out-of-sample series, so that no official numbers are produced before the official run.
49. As the implementing agent, I want edge cases at the end of the data pinned, so that every strategy, buy-and-hold included, is costed the same way.
50. As the implementing agent, I want a determinism test to run before the corruption tests, so that nondeterminism does not make a leak test fail for the wrong reason.

## Implementation Decisions

### Research question and pre-registered readings

All readings use the **primary pair** (the v1-style arm against the stationary twin), each on its **seed-averaged series**.

- **Reading 1 (two-sided): levels against stationary form.** It passes only if both gates pass:
  1. The paired block-bootstrap CI on the daily return difference between the two arms excludes zero.
  2. The gap between the arms' mean per-seed Sharpe ratios is larger than the **selection spread**, taken as the larger of the two arms' max-minus-min per-seed Sharpe.
  - If the stationary twin is better, raw price levels hurt the strategy.
  - If the v1-style arm is better, the level-feature theory is rejected.
  - If either gate fails, the comparison is **inconclusive**.
- **Reading 2: the strategy never worked.** Neither arm's whole-series excess-return CI lies above zero, so there was no edge for a regime change to break. Reading 2 can hold together with reading 1.
- **Reading 3: descriptive only.** An arm beats the exposure-matched benchmark in the early windows and trails it in the late ones. That is consistent with a regime change but does not prove it. The boundary is 2024-01-01, the split v1's README claimed. A window that straddles it belongs to the side that holds most of its test days. Under the series rule below, windows 1–3 are early and 4–7 are late. With three windows against four, no test is run.
- No run tests the regime theory directly. The spec says so, and so does the README.

### Data and integrity

- The data is the existing KEEN daily CSV (dividend-adjusted, October 2019 to October 2025, 1,465 rows). It is frozen at sha256 `4a90304fa95b3e8adaf736e4876dfd03713585f7599e5d34bcd04a570d27f71d`. Yahoo rewrites adjusted prices after each dividend, so it is never re-downloaded.
- **Already seen.** v1's results, the price path, and every figure quoted while planning (per-window buy-and-hold returns, the early/late table, limit-down candidates) came from out-of-sample rows. The pre-registration proves that the analysis plan came before the v2 results. It does not prove the data was unseen. The README says this.
- **Development protocol.** The trade code and pipeline are built and tested only on synthetic prices and on rows 0–648, which lie before the out-of-sample series. The official run happens once, on a tagged commit.
- **Deviation rule.** A **bug** is code that does not do what this spec says. Only a bug triggers a fix, a full rerun, and a dated deviation entry in the README that says what changed and why and shows both results. A change to a choice this spec made, such as a threshold or a window length, appears as a labelled exploratory result and never replaces the headline.
- **Reproducibility.** A full lock file from the environment that produces the official run, with Python's version recorded (v1 ran on 3.12.11). pandas is pinned explicitly, because an unpinned install now gets 3.x. The indicator library is the package the code actually imports (`pandas-ta-classic`), not the one currently listed.
- **Tags.** `v1` marks the original analysis at `acc6360`. It exists locally and is pushed explicitly, since a plain push does not send tags. `v2-official-run` marks the commit that produces the headline results.

### Labels

- **Triple-barrier labels** are computed once on the full series. Each is measured from the **entry close**, the close after the signal day. The barriers are ±2% on closing prices and the **horizon** is 10 trading days after the entry close. Labels are +1, −1, or 0 on timeout.
- Rows without a complete horizon get no label. Nothing is padded.
- **Purge.** Every training row whose horizon extends past the end of its training window is dropped. Because labels start at the entry close, that is the last 11 rows of each window. Test windows need no labels, because strategy returns come from prices.

### Trading rules (trade simulator)

- **Position** is 0 or 1 in any single run. It is never negative, because retail short selling of KEEN was unavailable from March 2020. A prediction of −1 or 0 means flat.
- **One trade at a time.** A +1 prediction on a signal day while flat triggers an **entry** at the next close. Predictions while long matter only on review days.
- **Review day:** a day whose close crosses an open position's upper or lower barrier, or reaches its timeout. Barriers and timeout are measured from the trade's entry close, or from its last reset.
- **Reset:** if the prediction on a review day is +1, the position is kept, and a new 10-day horizon and new ±2% barriers start from the next close, at no cost. As a result, while the model stays at +1, a lower-barrier crossing does not close the position. The strategy reviews its position only on review days.
- **Exit:** if the prediction on a review day is not +1, the position goes flat at the next close. Exits are delayed one close just as entries are, so stops overshoot, and that is the honest figure.
- **Trade:** entry to exit, however many resets fall inside it. Costs are charged once at each end.
- **A +1 on the day an exit fills is an ordinary signal.** The position is flat at that close, and a new entry fills at the next close with the full buy cost. Only a review day can trigger a reset.
- **End of data.** A position still open on the last day closes at the final close and pays the sell cost. This applies to every strategy and baseline, buy-and-hold included. A signal on the final day never fills.
- **Strategy return** on each day is the previous close's position times that day's close-to-close return, minus the cost on any day a trade enters or exits. Returns are compounded into the equity curve and reported as percentage returns on a notional account.
- **Cost model.** The default is 45 bps on buys and 55 bps on sells, the small-order, one-tick case: 15/25 bps of fees and sales tax plus 30 bps of slippage per side. Stresses are 150 bps and 200 bps round trip, keeping the 10 bps sell-side tax gap (70/80 and 95/105 bps). Stresses run on the primary pair only. Costs never change a trading decision, so stress figures and break-even cost are recomputed from gross returns and trade dates without rerunning the models.
- **Capacity.** Valid for orders up to about 1% of KEEN's daily traded value, which the README states.
- **Deliberately not modelled:** tick rounding (it would need unadjusted prices), lot sizes, the order of barrier touches within a day, and intraday gap fills. Execution is close-to-close. The limit-down rule was dropped. The one out-of-sample day at the lower auto-rejection limit, 2022-09-07, is named in the README instead.

### Walk-forward

- Training windows are 600 rows, test windows 120 rows, and the step is 120, so test windows never overlap.
- **Series start.** The first training window begins at the first row where every feature exists, after about 49 rows of warm-up. The **out-of-sample series** starts at the row after that window ends, 2022-06-06 on the frozen data, and runs to the last row of the data. That gives six full test windows plus a short seventh of 96 rows (12% of the series), which is included.
- Window returns are measured from the close before each test window's first day.
- **Window handover.** A position open at the end of a test window carries into the next one, and the next window's retrained model makes its decisions from then on. There is no forced exit.
- Each window refits feature selection and every model using only its own purged training rows.

### Features and arms

- **The feature universe is everything v1 used**: v1's 36 technical indicators (31 computed indicators plus raw OHLCV, as v1 counted them) and 30 close lags: 66 columns. This includes `Open`, `High`, `Low`, `Close` and `Volume`, which v1 fed to the model because it selected every non-label column.
- **Level form** is the features as v1 used them.
- **Stationary form** replaces each level with a scale-free version. The relative information stays and the absolute price and volume level goes, and that removal is what the comparison tests.

| Type | Features | Stationary version |
|---|---|---|
| Price level | 30 close lags, `Open`, `High`, `Low`, `bb_mid/hi/lo`, `kc_upper/Lower`, `ema_8/21`, `sma_50`, `vwap` | log(x / close_t) |
| Close | `Close` | dropped |
| Price-unit spread | `macd`, `macd_signal`, `macd_hist`, `atr` | x / close_t |
| Cumulative volume | `obv`, `ad` | 20-day change ÷ 20-day volume sum |
| Volume level | `Volume`, `vol20` | volume_t / vol20 |
| Already stationary | `rsi`, `stoch_k/d`, `cci`, `roc`, `natr`, `bb_width`, `cmf`, `mfi`, `ema_ratio`, `adx`, `dm_plus/minus`, `aroon_up/down` | unchanged in both forms |

- **Duplicates in the stationary form**, all left for clustering to merge and noted in the write-up: `atr` ÷ close equals `natr`; the `ad` transform equals `cmf`; and `Volume` and `vol20` both become volume ÷ `vol20`.
- **Arms:**
  - **v1-style arm:** level form, with the 30 lags kept outside clustering as v1 did. It is described as *v1's lag handling inside the corrected pipeline*. It differs from v1 in per-window selection, purged and seeded MDA, the medoid rule, new labels and new trade rules, and it has no size or meta model in its base run.
  - **Stationary twin:** stationary form with its 30 lags outside clustering. The lag treatment matches the v1-style arm. Even so, the non-lag features cluster differently in each form and the twin has no `Close`. Both follow from the change of form, so they are part of what the comparison tests. The pair is not identical apart from the lags.
  - **Clustered level arm** and **clustered stationary arm:** each form with every feature, lags included, going through clustering. This **secondary pair** is reported descriptively, with no gates. In the first training window the 40 price-level features form a single cluster at the 0.6 threshold, so the clustered level arm keeps one price feature and probably has fewer than 15 candidates.
- **Feature selection,** refitted inside each training window:
  - Clustering uses Ward linkage on 1 − |correlation| at a distance threshold of 0.6, as in v1. Ward assumes Euclidean distances and this distance is not one. It is kept for comparability and listed as a limit.
  - Each cluster's representative is its **medoid**, the feature with the highest mean |correlation| to the rest of its cluster. Ties go to the earlier column, and single-feature clusters keep their feature. This replaces v1's highest-raw-variance rule, which depends on units: it picked `obv` and `ad` in v1 and would confound the arms.
  - MDA uses log-loss, a purged 5-split time-series splitter (11-row purge), 3 permutation repeats, and a seeded shuffle.
  - The top 15 features by MDA are kept, matching v1. If there are fewer than 15 candidates, all are kept. v1's importances fall below 0.001 after the eleventh feature, so the last few picks are noise, which the write-up notes.

### Models

- **Base run** in each arm: the primary XGBoost classifier with 3 classes and `multi:softprob`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, the same as v1. `class_weight` is removed, because XGBoost ignores it, so v1 never balanced classes either. The strategy goes long on +1 and stays flat otherwise. The size regressor is dropped, and the "ETF trick" goes with it.
- **Seeds.** Selection seeds 42–46 vary only the MDA permutation shuffle. The XGBoost seed is fixed at 42. v1 sets no subsampling, so that seed has no effect.
- **Meta-labelling ablation,** on the primary pair only:
  - Within each training window, the primary model's out-of-fold predictions come from the same purged 5-split splitter. That covers five-sixths of the window, roughly 190 +1 calls.
  - The meta target is whether a +1 call's triple-barrier label is +1.
  - The meta features are the three class probabilities only, since `pred_class` is constant and `confidence` equals `prob_profit` for a +1 call. The probabilities sum to one, so the meta-model can do little more than move the probability threshold.
  - Settings: `n_estimators=100`, `max_depth=4`, a fixed threshold of 0.5, and the strategy trades a +1 call only when meta probability ≥ 0.5.
  - Limit: the fold models train on as few as about 100 rows, while the model that trades trains on about 590. The meta-model therefore learns the mistakes of smaller models.
- **No tuning.** Every value above sits in one frozen settings block committed before the official run. There is no validation set and no threshold search.

### Evaluation and statistics

- The **seed-averaged series** is each arm's daily strategy returns averaged across the five seeds. It implies fractional positions. All headline statistics and plots use it.
- The **headline Sharpe** is the Sharpe of the seed-averaged series, with its bootstrap CI. The mean of the per-seed Sharpes, with its range, is reported as the selection spread. The risk-free rate is zero and stated. Indonesian cash yields were near 6% in this period, and excess return is unaffected by that choice.
- **Bootstrap.** Stationary block bootstrap with 10,000 resamples, a mean block length of 10 days and seed 0, using percentile intervals. Every paired test resamples the same days for both series. It is applied to the Sharpe CI, the daily return difference between arms, and excess return.
- **Exposure-matched benchmark.** Buy-and-hold scaled by the strategy's average position, with no costs. The headline uses the whole-series average position, and the per-window table uses each window's average. Scaling cannot change a Sharpe ratio, so skill is measured as **excess return**: mean daily strategy return minus benchmark return, with a paired bootstrap CI.
- **Break-even cost.** The round-trip cost at which mean excess return is zero, for each arm's seed-averaged series. "None" if the arm trails the benchmark before costs.
- **Baselines,** run through the trade simulator with default costs. Their parameters are fixed here:
  - Buy-and-hold: entry at the series start, exit at the final close.
  - Momentum: +1 when the 60-day return is above 0.
  - Mean reversion: +1 when RSI(14) is below 30. It fires in only about 11 episodes, so its trade count is printed next to its result.
- **Reported tables:**
  - The headline: strategy returns, Sharpe with CI, excess return with CI, break-even, and trade count for each primary-pair arm.
  - Per window: strategy, benchmark and excess return for each arm, not tested.
  - Price only, early and late. Early (2022-06-06 to 2023-12-31): buy-and-hold +72%, annualised volatility 53%, Sharpe 0.92. Late (2024 to October 2025): +30%, 37%, 0.60. KEEN fell 23% in calendar 2024. This table appears in the README whatever the model shows.
  - Cost stresses for the primary pair.
  - The meta-labelling ablation for the primary pair.
  - The secondary pair, descriptive.
  - Baselines, with trade counts.

### Modules and interfaces

- **A backtest module** holds all logic, behind three seams:
  - **Walk-forward run.** Takes the price frame, an arm, a selection seed and a settings object. Returns the out-of-sample daily strategy return series, the trade log, and for each window a fingerprint (a hash of the post-purge training features and labels) and the selected features. It produces predictions per window and passes one continuous prediction series to the trade simulator.
  - **Trade simulator.** Takes prices, a daily prediction series and a cost model. Returns daily strategy returns and a trade list (entry date, exit date, resets, gross and net return). It knows nothing about windows.
  - **Statistics.** Takes return series and returns numbers: Sharpe, block-bootstrap CIs (single and paired), excess return against a scaled benchmark, seed averaging, and break-even cost.
- **The settings object** holds the frozen block as its default. Tests pass small windows and few trees.
- **A new v2 analysis notebook** imports the module, reads data from the repo's data folder, runs every arm and seed, and produces all figures and tables.
- **The v1 notebook** stays as it is, apart from one banner cell at the top pointing to v2 and to the `v1` tag. Its conclusion still claims the implementation was correct.
- **The headline figure** is one plot with the v1-style arm (the **headline arm**), the stationary twin and the exposure-matched benchmark, all as equity curves of the seed-averaged series.

### README rewrite

- It opens with the headline figure and the research question. The headline arm is labelled *v1's features inside the corrected pipeline*, never *v1 measured properly*.
- It then gives the readings as pre-registered, and which one the results support, using the gate outcomes.
- **What v1 got wrong** (3–5 lines). It links to the hit-score function and the walk-forward meta training at the `v1` tag. The v1 dashboard image, which shows v1 numbers, moves here or is removed.
- A statement of what had already been seen, the deviation log, the frozen data hash, and the tags.
- **Limitations:** close-to-close execution, with no tick rounding, lots, intraday ordering or gap fills; the 2022-09-07 limit-down day; adjusted prices; a single illiquid stock; capacity of about 1% of daily traded value; Ward on a non-Euclidean distance; meta fold models weaker than the deployed model; noisy last few selected features; risk-free rate of zero; low statistical power (buy-and-hold's Sharpe over the series has a standard error of about 0.55–0.60, so only a large gap between arms is detectable).
- **Repairs:** the folder is `notebook/` (not `notebooks/`); the missing data README reference goes; the `yourusername` clone URL is fixed; "ETF trick" and the "+105%" regime claim go; the requirements list matches the lock file.

## Testing Decisions

- **What a good test is.** It checks behaviour at a seam (inputs to outputs), never internal steps. It uses synthetic prices, or rows 0–648 at most, never out-of-sample rows. It is fast, because it passes small settings, and deterministic, because seeds are fixed.
- **Prior art.** The repo has no tests yet. The look-ahead and purge tests follow the standard corruption pattern from backtesting practice: change the future and assert that the past does not move.

### Seam A: walk-forward run

1. **Determinism (runs first).** The same arm, seed and settings run twice give identical series, trade logs, fingerprints and selections.
2. **Look-ahead.** Corrupt every price after a date T on synthetic data and rerun. The out-of-sample series up to T must be unchanged.
3. **Purge.** Corrupt every price after a training window's last row. That window's training fingerprint and selected features must be unchanged. Without the purge, the last 11 labels would read the corrupted prices and change the fingerprint, which test 2 alone cannot see because every affected prediction falls after T.
4. **Window handover.** A position open at a window boundary carries into the next window with no exit, and the next window's predictions govern it.
5. **Series bounds.** The series starts at the row after the first training window, which itself starts after warm-up, and ends at the last row, including a short final window.

### Seam B: trade simulator (hand-checked)

1. A single trade hits the upper barrier. The entry and exit closes, the daily strategy returns and the costs match a hand calculation.
2. **Timing.** A signal on day t changes the position only from the close of t+1. The same holds for an exit decided on a review day.
3. **Reset.** A review day with a +1 prediction keeps the position, charges nothing, and restarts the horizon and barriers from the next close. The trade list shows one trade.
4. **Lower-barrier crossing with +1 held.** The position stays open.
5. **Signal on the exit-fill day.** The position goes flat at that close, then re-enters at the next close and pays the full buy cost.
6. **End of data.** An open position closes at the final close and pays the sell cost. A final-day signal never fills.
7. **Costs.** Charged exactly once at entry and once at exit, at 45/55 bps by default. Raising the cost changes returns but no trade dates.
8. **Buy-and-hold identity.** Strategy return equals the stock's return from start to end minus the buy and sell costs, which is an end-to-end check.
9. **Long-only.** A −1 or 0 prediction never produces a negative position.
10. **Break-even.** Recomputing excess return at the reported break-even cost gives zero.

### Seam C: statistics

1. A series paired with itself gives a difference of zero in every resample, so the CI is exactly [0, 0]. This also proves both series use the same resampled days.
2. The same bootstrap seed gives the same interval twice.
3. A strategy equal to the scaled benchmark has zero excess return.
4. Scaling a return series leaves its Sharpe unchanged.
5. Seed averaging of known series gives their daily mean.

## Out of Scope

- Regime detection or switching (HMM or clustering), LSTM or Transformer models, alternative data, and multi-asset portfolios.
- Hyperparameter tuning, threshold search, or changing Ward linkage.
- The size regressor and the "ETF trick".
- Short positions.
- Tick rounding, lot sizes, intraday barrier ordering, gap fills, limit-down modelling, and a liquidity or market-impact model.
- **Second pass, each with its own spec written and committed before it runs:**
  1. Volatility-scaled barriers as a second run alongside ±2%.
  2. A rerun on one liquid asset, with its spec written before its data is downloaded. It is the only data nobody has looked at.
- Live trading or any trading advice.

## Further Notes

- **Stated expectations, written before the run:**
  - Reading 2 (the strategy never worked) is the most likely outcome.
  - The selection-spread gate may fail in the v1-style arm. In the first training window the raw lags correlate at 0.76 or higher with each other, so MDA's choice among them is close to arbitrary. v1's picks were lags 29, 24, 28 and 23. If reading 1 comes back inconclusive for this reason, the finding is that v1's selection among its raw lags was noise.
  - The meta-labelling ablation is expected to show nothing. A null result is reported.
  - With about 1% round-trip costs on ±2% barriers, a winning trade gives back half its gross return to costs.
- **Statistical power is low.** Buy-and-hold's Sharpe over the series is about 0.6–0.8, with a standard error of about 0.55–0.60. The paired test will detect only a large gap between arms.
- **Short-selling and cost facts** come from `docs/research/idx-trading-frictions.md`. Several exchange figures there are cited through news reports of IDX decrees, because idx.co.id blocked automated access, and they are marked as secondary.
- **Workflow.** This spec, `GLOSSARY.md` and the research file are committed before development begins. The work then goes through `/to-tickets` and `/implement`, with `/tdd` driving each seam.

## Amendment (from #1's comments)

**Amendment, 2026-11-26, before development.** The spec has the v2 notebook run every arm and seed. Instead, one results command runs all arms, seeds, the meta-labelling ablation and the baselines, and saves the per-seed daily return series, trade logs, per-window fingerprints and selected features to files. The notebook only reads those files. A plotting fix after the official run then needs no model rerun and no deviation entry. Nothing else in the spec changes. The in-repo copy of the spec includes this amendment.

## Amendment 2 (from #1's comments)

**Amendment, 2026-11-26, during development, before the official run.** Two additions to the meta-labelling ablation. Nothing else in the spec changes.

- **Fallback for a window with no meta-model.** If a training window's out-of-fold predictions contain no +1 call, the meta-model has nothing to train on. That window skips the meta step and trades its +1 calls as the base run does. Vetoing them instead would make the meta run differ from the base run with no meta-model involved, and the ablation measures the meta-model. About 190 out-of-fold +1 calls per window are expected, so the fallback should not fire on KEEN. Each window records `oof_long_calls`, the number of out-of-fold +1 calls, and the README reports how many windows used the fallback.
- **Added limitation, next to "meta fold models weaker than the deployed model":** feature selection runs on the whole training window, labels included, before the out-of-fold predictions. The fold models never see the rows they predict, but the selected features were chosen with those rows' labels, so the out-of-fold probabilities the meta-model trains on look slightly better than the deployed model's. Test-window rows are never involved, so no reported return is affected. Selecting inside each fold, on about 100 rows, would be noisier than the leak.

## Amendment 3 (from #1's comments)

**Amendment, 2026-11-26, during development, before the official run.** Three guards on the official run. Nothing else in the spec changes.

- **Feature check.** Before the first model fit, the results command checks that every feature, in both level and stationary form, is finite on every row after warm-up. KEEN is illiquid, and the out-of-sample rows have not been checked for a flat bar (High equals Low) or a trading suspension, either of which can make a feature divide by zero. Development rows 0–648 pass. If the check stops the official run, the run produces no results. The fix for those values is then a dated amendment, not a deviation: the deviation rule shows both results, and there is no first result to show beside the fix. That amendment names the rows and features the check reported and states the fix before the run is repeated.
- **Clean tree.** `--official-run` refuses to start with uncommitted changes, so the commit recorded in `run.json` is the code that ran. `v2-official-run` is an annotated tag, so `git describe` records it in `run.json`.
- **No overwrite.** `--official-run` refuses to start if its output folder already holds a `run.json`. A rerun after a bug fix needs the first outputs moved or committed first, so both results can be shown as the deviation rule requires.

## Amendment 4 (from #1's comments)

**Amendment, 2026-11-26, after the official run, before its outputs were opened.** Two definitions the v2 notebook needs in order to compute the readings from rules. Nothing else in the spec changes.

- **Reading 1 direction.** When both gates pass, the better arm is the one gate 1 favours, given by the sign of the paired CI on the daily return difference. Reading 1 passes only if the gap in mean per-seed Sharpe points the same way. If the two gates favour different arms, the comparison is inconclusive.
- **Reading 3 measure.** An arm beats or trails its exposure-matched benchmark according to its mean daily excess return over the early windows and over the late windows. Each window's benchmark is scaled by that window's average position, as in the per-window table. The spec gives no side to a window with exactly half its test days on each side of the boundary, so the notebook stops instead of picking one. Under the series rule, no window splits evenly.

## Amendment 5 (from #1's comments)

**Amendment, 2026-11-26, after the official run and after its outputs were read.** A presentation change. No reading, rule, setting or number changes.

- **README lead figure.** The spec says the README leads with one figure showing the v1-style arm, the stationary twin and the exposure-matched benchmark. The README instead leads with a plot of each run's mean daily excess return and its 95% block-bootstrap interval, because reading 2 is the main finding and that plot shows its statistic directly. The spec's figure follows it, unchanged in content.
- **The lead plot was designed after the results were read.** It shows the pre-registered excess-return statistic and adds no new analysis.

## Amendment 6 (from #1's comments)

**Amendment, 2026-11-27, after the official run and after its outputs were read.** A presentation change. No reading, rule, setting or number changes.

- **Plain names on the README and charts.** The README, `docs/img/make_charts.py` and the pipeline diagram call the v1-style arm the **price-level model** and the stationary twin the **returns-based model**, "arm" becomes "model", and readings 1 to 3 are presented as three questions. Readers outside the project did not know what "v1", "arm" or "twin" meant. The README names the spec's terms once, so either name leads to the other. The spec, `GLOSSARY.md`, code and notebook keep their terms.
- **The headline model is still described as v1's features inside the corrected pipeline** (user story 23), never as v1 measured properly.

## Amendment 7 (from #1's comments)

**Amendment, 2026-11-27, after the official run and after its outputs were read.** Three descriptive additions, with each method fixed here before any of their numbers is computed. No reading, rule, setting or existing number changes. The notebook computes them from the saved results files, with no model rerun.

- **Drawdown in the headline table.** The headline table gains two columns, max drawdown and longest time under water, for both models. Each cell gives the model's value and then its exposure-matched benchmark's value, because a model that holds KEEN on 41% or 64% of days will always draw down less than buy-and-hold. Both are computed on the seed-averaged series net of default costs over all 816 out-of-sample days. The benchmark is the headline one, scaled by the whole-series average position, with no costs; the table note says so, since it means the comparison slightly favours the benchmark. Equity starts at 1 before the first out-of-sample day and compounds daily returns. Max drawdown is the largest fall from a running peak of equity to a later trough, as a percentage of that peak. Time under water is a run of consecutive trading days on which equity is below its previous running peak; the longest run is reported in trading days. A run still open on the final day counts to the final day and is marked "not recovered". Both are point estimates, with no interval. The per-window table gets no drawdown columns: its windows are 96 to 120 days long, so a period under water is cut off at a window's edges.
- **Dividend adjustment in the Limitations section.** The README's "Adjusted prices" bullet is rewritten to say the following. Yahoo's dividend adjustment multiplies every close before an ex-date by one minus the dividend divided by the unadjusted close on the day before the ex-date. A price level on day t therefore depends on dividends with ex-dates after t, which is look-ahead. A ratio of two closes on or before day t is scaled only by dividends between them, all of them before t. The stationary form divides every price level by the same day's close, and labels and strategy returns are ratios, so the leak reaches only the level-form arms: the v1-style arm and the clustered level arm. The bullet gives the size of the leak as the largest gap between an adjusted and an unadjusted close, one minus the product of the factors for every ex-date after a given day. That gap is largest on the first day of a span, so the bullet reports it on the file's first row (2019-10-01) and on the first out-of-sample day (2022-06-06). Look-ahead would normally flatter the v1-style arm, but a distortion of its features is not guaranteed to help it, so the bullet says both and gives the size bound for how much the leak could matter. The arm shows no edge with the leak in place. The bullet also says why KEEN was chosen: the question is whether v1's failure on KEEN came from a regime change, so the test has to be on KEEN. No arm is rerun on unadjusted closes. Dividend amounts and the unadjusted close before each ex-date come from Yahoo's history for KEEN.JK, fetched once by a script that does not touch or replace `data/KEEN_2019_2025.csv`, and checked against IDX announcements where those can be reached. The commit message records the script and the fetched values.
- **Block-length sensitivity.** The spec set the bootstrap's mean block length to 10 days without recording a reason. A table in a new collapsed README section, after "Higher costs", gives the 95% interval on mean daily excess return for both models of the primary pair at mean block lengths of 5, 10 and 20 days, with the same 10,000 resamples, seed 0 and percentile intervals. The frozen settings block does not change: the sensitivity passes a copy of the settings with only the block length changed. The 10-day row must reproduce the headline intervals exactly; if it does not, the notebook stops and nothing is published until the difference is explained. Block length cannot move the mean, only the interval's width. Every reading and every interval elsewhere stays on 10 days, and the table is published whatever it shows.
- **Two glossary entries,** under Performance: **Max drawdown** and **Time under water**, with the definitions above.

## Amendment 8 (from #1's comments)

**Amendment, 2026-11-27, after the official run and after its outputs were read.** Amendment 7's figures are computed by `scripts/amendment7.py` from the saved results files, not by the notebook, so the notebook stays as it ran on the official outputs. Where Amendment 7 says the notebook stops if the 10-day row does not reproduce the headline intervals, that script stops instead. Nothing else changes.

## Amendment 9 (from #1's comments)

**Amendment, 2026-11-28, after the official run and after its outputs were read.** Records a wording change made to the spec copy on 2026-11-26, before the official run, in commit 38e1143. No feature, rule, setting or number changes.

- **The feature universe is 66 columns.** The spec's feature-universe line reads: "v1's 36 technical indicators (31 computed indicators plus raw OHLCV, as v1 counted them) and 30 close lags: 66 columns." The earlier wording, "raw OHLCV, the 36 technical indicators, and 30 close lags", reads as 71 columns, because v1 counted OHLCV among its 36. The code, `GLOSSARY.md` and the official run have always used 66.
