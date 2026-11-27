# KEEN dividends and the size of the dividend-adjustment look-ahead

Research date: 2026-11-27. Purpose: the inputs to the dividend bound that [Amendment 7](../spec.md#amendment-7-from-1s-comments) asks for. This note needs the network, so it is not part of the tested code. `data/KEEN_2019_2025.csv` was not touched or re-downloaded.

## Inputs

Yahoo's history for KEEN.JK, fetched 2026-11-27 with yfinance (`auto_adjust=False`, `actions=True`), 2019-09-02 to 2025-10-24. Dᵢ is the cash dividend per share and Pᵢ the unadjusted close on the trading day before the ex-date. The adjustment factor is 1 − Dᵢ / Pᵢ. The Stock Splits column is zero on every row, so the amounts are not split-adjusted.

| Ex-date | Dᵢ (IDR) | Pᵢ date | Pᵢ (IDR) | Factor | Cross-check of amount and ex-date |
|---|---|---|---|---|---|
| 2020-08-10 | 2.93 | 2020-08-07 | 338 | 0.991331 | Ajaib dividend table [secondary] |
| 2021-09-09 | 3.88 | 2021-09-08 | 434 | 0.991060 | Bisnis, 2021-09-13 [secondary] |
| 2022-07-08 | 5.46 | 2022-07-07 | 446 | 0.987758 | Bisnis, 2022-06-29 [secondary] |
| 2023-06-27 | 6.50 | 2023-06-26 | 745 | 0.991275 | EmitenNews; Ajaib [secondary] |
| 2024-06-20 | 7.55 | 2024-06-19 | 695 | 0.989137 | Bisnis, 2024-06-07; Ajaib [secondary] |
| 2025-06-18 | 8.22 | 2025-06-17 | 805 | 0.989789 | Bisnis, 2025-06-05; Ajaib [secondary] |

Yahoo also lists 8.22 IDR with ex-date 2026-06-15 (reported by Kontan and BCA Sekuritas [secondary]). That is after the file's last row, 2025-10-24, so the file cannot contain it and the bound leaves it out.

The cross-checks come from search-result summaries of these pages, not from opening them:
[Ajaib](https://ajaib.co.id/saham/aset/KEEN/dividen),
[Bisnis 2021](https://market.bisnis.com/read/20210913/192/1441543/catat-ini-jadwal-pembagian-dividen-kencana-energi-keen),
[Bisnis 2022](https://market.bisnis.com/read/20220629/192/1549498/kencana-energi-keen-setor-dividen-us799-juta-atau-rp546-per-saham),
[EmitenNews 2023](https://www.emitennews.com/news/kencana-energi-keen-bagi-dividen-1105-dari-laba-per-saham-dapat-rp650-nih/10),
[Bisnis 2024](https://market.bisnis.com/read/20240607/192/1772175/rups-kencana-energi-keen-putuskan-dividen-rp755-per-saham),
[Bisnis 2025](https://market.bisnis.com/read/20250605/192/1882778/emiten-ebt-kencana-energi-keen-siapkan-dividen-rp822saham),
[Kontan 2026](https://investasi.kontan.co.id/news/kencana-energi-keen-bakal-bagi-dividen-tunai-final-rp-822-per-saham-cek-jadwalnya).

**No primary source was reached.** idx.co.id refuses automated access. The issuer's site (kencanaenergy.com) was not reachable from this session's network. KSEI's site was reached, but the four KSEI dividend notices that search results list for KEEN (2020-08 and 2021-09) all returned HTTP 404.

## Check of the frozen file against these inputs

If the frozen file was adjusted for exactly these six dividends, its close divided by Yahoo's unadjusted close on day t equals the product of the factors with ex-dates after t. Over all 1,465 rows the largest relative difference is 1.9 × 10⁻⁷, which is float rounding, and the ratio is exactly 1.0 from 2025-06-18 to 2025-10-24. The file carries these six dividends and no others, and the factor formula matches Yahoo's.

## The bound

The gap between an adjusted and an unadjusted close on day t is one minus the product of the factors with ex-dates after t. It shrinks over time, so it is largest on the first day of a span.

| Day | Ex-dates after it | Largest gap |
|---|---|---|
| 2019-10-01, first row of the file | all six | 5.82% |
| 2022-06-06, first out-of-sample day | last four | 4.14% |

So no adjusted price level in the file is more than 5.82% below the real traded price, and none in the out-of-sample period is more than 4.14% below it.
