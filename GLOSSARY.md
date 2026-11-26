# Financial ML Regime Analysis

A backtest of a machine-learning trading strategy on one Indonesian stock (KEEN), built to test whether its out-of-sample failure came from a market regime change or from price-level features.

## Performance

**Strategy return**:
The daily percentage change in account value: the previous day's position times the day's close-to-close return, minus costs on days a trade enters or exits. Compounded into an equity curve.
_Avoid_: Return (unqualified), PnL score, total return when summed

**Sharpe ratio**:
The mean of daily strategy returns divided by their standard deviation, times √252, with a risk-free rate of zero. The headline Sharpe is that of the seed-averaged series, not the mean of per-seed Sharpes.
_Avoid_: Sharpe of trade outcomes, per-window Sharpe

**Seed-averaged series**:
An arm's out-of-sample series averaged day by day across the five selection seeds. The series all headline statistics and plots use.

**Hit score**:
The v1 metric: +1 when the predicted label matched the actual label and −1 when not, times bet size. It is not a return and is only cited when explaining what v1 got wrong.
_Avoid_: Return, total return, Sharpe

**Out-of-sample series**:
The daily strategy returns from every test window placed end to end in date order. All headline statistics are computed on this series.
_Avoid_: Per-window results, walk-forward average

**Baseline**:
A simple rule-based strategy run through the same trade rules and cost model as the ML strategy, so the comparison isolates the model.

**Exposure-matched benchmark**:
Buy-and-hold scaled by the strategy's average position over the same period: the return a long-or-flat strategy earns from market exposure alone, with no skill.
_Avoid_: Benchmark (unqualified)

**Excess return**:
The daily strategy return minus the exposure-matched benchmark's daily return. The measure of skill, because scaling the benchmark cannot change its Sharpe ratio.
_Avoid_: Alpha, outperformance

## Trading

**Position**:
The fraction of the account held in KEEN at a close: 0 (flat) or 1 (long) in any single run. Never negative, because retail short selling of KEEN was unavailable from March 2020. The seed-averaged series implies fractional positions, since it averages five runs.
_Avoid_: Short, side, bet size

**Signal day**:
The day whose close produces the model's prediction. The trade it triggers fills at the next day's close.

**Entry**:
Moving from flat to long at the close of the day after the signal day.

**Review day**:
A day on which a barrier crossing or timeout is observed for an open position. The only days the strategy reconsiders a position it holds.

**Reset**:
On a review day where the prediction is still +1: the position is kept, and a new horizon and new barriers start from the next close, at no cost. While the model stays at +1, a lower-barrier crossing does not close the position.

**Exit**:
Moving from long to flat at the close of the day after a review day whose prediction is not +1.
_Avoid_: Stop fill, barrier fill

**Trade**:
One holding period from entry to exit, however many resets fall inside it. Costs are charged once at each end.

**Window handover**:
A position open at the end of a test window carries into the next one, and the next window's retrained model makes its decisions from then on. There is no forced exit at window boundaries.

**Cost model**:
The per-side trading cost charged at entry and exit. The default (45 bps buy, 55 bps sell) is the small-order, one-tick case; 150 and 200 bps round trip are the stress cases.
_Avoid_: Commission (it is only one part)

**Break-even cost**:
The round-trip cost at which mean excess return is zero. "None" when the strategy trails the exposure-matched benchmark before costs.

**Capacity**:
The largest order the cost model is valid for: about 1% of KEEN's daily traded value.

## Labels

**Triple-barrier label**:
The outcome of the long trade a signal day would trigger, measured from its entry close: +1 if a close reaches the upper barrier first, −1 if a close reaches the lower barrier first, 0 if neither happens within the horizon.
_Avoid_: Target, signal

**Horizon**:
The 10 trading days after the entry close that a triple-barrier label looks across.

**Purge**:
Dropping every training row whose horizon extends past the end of its training window, so no label uses prices the test window owns.
_Avoid_: Gap, embargo (a different technique)

## Evaluation

**Walk-forward window**:
One training window followed by the non-overlapping test window it is evaluated on.

**Ablation**:
A run that changes exactly one design choice against the main run, so a difference in results can be attributed to that choice.

**Arm**:
One feature configuration run through the whole walk-forward pipeline. Arms differ in the form of their features and in how close lags enter feature selection.

**Feature universe**:
Every column v1 fed the model: 31 computed indicators, raw OHLCV and 30 close lags, 66 columns in level form and 65 in stationary form (no `Close`). v1 and the spec call the 36 non-lag columns "the 36 technical indicators", counting OHLCV among them.

**Level form**:
Features with price and volume levels as v1 used them: raw OHLCV, raw close lags, bands, moving averages and cumulative volume.

**Stationary form**:
The same features with every level replaced by a scale-free version of itself (ratio to the current close, or to average volume). It keeps relative information but removes the absolute price and volume level.

**v1-style arm**:
Level form, with the 30 close lags kept outside clustering as v1 did: v1's lag handling inside the corrected pipeline.
_Avoid_: v1 (v1 is the original, uncorrected analysis)

**Stationary twin**:
Stationary form, with its 30 lags kept outside clustering, matching the v1-style arm's pipeline.

**Primary pair**:
The v1-style arm against the stationary twin. The only gated comparison: it tests whether v1's raw price levels caused its failure.

**Headline arm**:
The v1-style arm's base run: the equity curve and excess-return figure at the top of the README, always drawn beside the stationary twin and the exposure-matched benchmark. Described as v1's features inside the corrected pipeline, never as v1 measured properly.

**Clustered level arm** / **Clustered stationary arm**:
Level and stationary form with every feature, lags included, going through clustering.

**Secondary pair**:
The clustered level arm against the clustered stationary arm. Reported descriptively, with no gates.
_Avoid_: No-lags arm, return features

**Selection spread**:
The range of an arm's per-seed Sharpe ratios when only the feature-selection seed changes. A gap between arms smaller than this is noise, not evidence.

**Base run**:
Within one arm, the primary classifier alone, long when it predicts +1 and flat otherwise.
_Avoid_: Main run

**Meta-labelling ablation**:
An arm's base run plus a second model that predicts whether each +1 call reaches the upper barrier, used as a fixed-threshold filter. Run on the primary pair only.
