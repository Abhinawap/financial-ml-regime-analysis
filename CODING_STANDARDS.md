# Coding standards

Judgement calls for the reviewer. Mechanical rules are enforced by `tests/test_architecture.py`, ruff and vulture (`pyproject.toml`), not here.

## Settings

- **Tunables live in `Settings`.** Any value a ticket, ablation or stress could change (barriers, horizon, costs, seeds, windows, thresholds, bootstrap parameters, the CI level) is read from `backtest/settings.py`.
- **Frozen v1 feature definitions stay inline.** Indicator lengths and the column names pandas-ta derives from them (`ta.rsi(c, length=14)`, `"BBM_20_2.0"`) define v1's feature universe, which the spec freezes. They are not tunables.

## Tests

- **Every injectable default runs in at least one test.** A parameter like `model=None` that tests override for speed gets one test with the real default on tiny data. The default XGBoost path crashed on a fold missing a class because every test swapped in a smaller model.

## Bloat

- **No abstraction with one caller.** A helper, class or parameter needs a second caller or a spec requirement. Until it has one, inline it.
- **Leakage safety beats brevity.** Purge, labels and window handover stay explicit even when a shorter form exists. A simplification there ships with a test that fails on lookahead.
