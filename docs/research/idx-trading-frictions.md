# IDX trading frictions for the KEEN backtest (Oct 2019 – Oct 2025)

Research date: 2026-11-26. Scope: retail trading of PT Kencana Energi Lestari Tbk (KEEN / KEEN.JK) on the Indonesia Stock Exchange (IDX/BEI). Sources are labelled **[primary]** (regulator, exchange, SRO or tax-authority document) or **[secondary]** (news, broker blog, education site). Anything I could not confirm is marked **UNVERIFIED**. idx.co.id returned HTTP 403 to automated fetches, so several exchange facts are cited through news reports of IDX decrees.

---

## Bottom line for the backtest

1. **Explicit costs per side (all-in: commission + levy + VAT, plus the 0.1% sales tax on sells):**

   | Scenario | Buy | Sell | Round trip |
   |---|---|---|---|
   | Low (cheapest online / day-trade plans) | 10 bps | 20 bps | 30 bps |
   | **Typical (discount online broker, 2019–2025)** | **15 bps** | **25 bps** | **40 bps** |
   | High (full-service broker) | 30 bps | 40 bps | 70 bps |

   The 10 bps gap between buy and sell is the 0.1% final income tax, which only sellers pay. These rates held for the whole 2019–2025 period. The only change was VAT on commission, which went from 10% to 11% on 1 Apr 2022. That moves the all-in cost by about 0.15 bps, so ignore it.

2. **Add slippage separately. For KEEN it is larger than the fees.** KEEN's tick size is 40–99 bps of price (median 65 bps). Crossing a one-tick spread therefore costs about half a tick, roughly 20–50 bps, on every fill. **Recommended default: 15 bps + 30 bps slippage on buys (45 bps) and 25 bps + 30 bps on sells (55 bps), about 100 bps round trip.** Stress-test at 150 bps round trip for IDR 100 m positions. With ±2% profit and stop targets, a 1% round-trip cost takes half of every winning trade's gross return. This is the most important correction to the current backtest.

3. **Run long-only.** Retail short selling was not available for KEEN at any point in the backtest window. IDX withdrew every short-sell list on 2 Mar 2020, and short selling stayed banned until new rules arrived in Oct 2024. KEEN did appear on the Aug 2024 short-sell list. Even so, no broker was executing short sales before OJK postponed the launch in Mar 2025, and it stayed postponed past Oct 2025. Short positions in the backtest should become "flat" (cash), not negative positions.

4. **Other modelling rules:**
   - **Order of stop and target inside a bar.** On 548 of 1,464 days (37%), KEEN's intraday range was at least 4% of the prior close, so both the +2% target and the −2% stop could be touched on the same day. Daily bars cannot show which came first. Assume the stop was hit first; that is the conservative choice.
   - **Gaps through the stop.** On 39 days (2.7%) KEEN opened at least 2% below the prior close. If the open is below the stop, fill at the open, not at the stop price.
   - **Tick rounding.** A 2% move at Rp 600 is Rp 12, which is not a valid price under the Rp 5 tick, so the order would sit at Rp 10 (1.67%) or Rp 15 (2.5%). Round targets and stops to valid ticks using *unadjusted* prices. The `Close` column in `data/KEEN_2019_2025.csv` holds non-integer, dividend-adjusted values that are not real traded prices.
   - **Lots.** Round share counts down to multiples of 100. The effect is negligible above about IDR 5 m.
   - **Position size.** IDR 100 m was more than 1% of KEEN's daily traded value on 90% of days, and about 10% of the median day. Either cap orders at a few percent of trailing average daily value or apply the high-slippage scenario.

---

## 1. Retail round-trip trading costs

### Components

| Component | Rate (of trade value) | Side | Source |
|---|---|---|---|
| Exchange transaction fee (BEI) | 0.018% | both | KSEI/KPEI/BEI joint decree [primary], below |
| Clearing fee (KPEI) | 0.009% | both | same |
| Settlement fee (KSEI) | 0.003% | both | same |
| KPEI guarantee fund (dana jaminan) | 0.010% | both (regular market) | [secondary] |
| VAT on the SRO fees ("levy") | 10% of the 0.03%, i.e. ≈0.003% (11% from Apr 2022) | both | [secondary] |
| **SRO subtotal ("levy")** | **≈0.043%** | both | [secondary] |
| Final income tax on share sales (PPh final) | **0.1% of gross sale value** | **sell only** | [primary] |
| Broker commission | broker-specific, see below | both | [primary]/[secondary] |
| VAT on broker commission | 10% to 31 Mar 2022; 11% from 1 Apr 2022; nominally 12% from 1 Jan 2025 but on an 11/12 base, so effectively still 11% | both | [primary] |
| Stamp duty (e-Meterai) | Rp 10,000 per trade confirmation over Rp 10 m | per confirmation | [secondary] |

**Exchange, clearing and settlement fees.** The joint BEI/KPEI/KSEI decree Kep-00120/BEI/08-2026 (14 Aug 2026) sets the negotiated-market fee at 0.03%: 0.018% BEI, 0.009% KPEI, 0.003% KSEI. Its recitals cite the standing rules. KPEI Regulation II-5 sets clearing at 0.009%, and KSEI Regulation VI-A (KEP-0017/DIR/KSEI/0422, 26 Apr 2022) sets settlement at 0.003% for equities. The decree also refers to its 2020 predecessor, Kep-00052/BEI/08-2020, which had the same 0.03% structure. [primary] [KSEI PDF](https://www.ksei.co.id/storage/208531/SKB-Direksi-SRO-Kebijakan-Keringanan-Biaya-Transaksi-dan-Penyelesaian-Transaksi-di-Pasar-Negosiasi.pdf). The regular-market split (0.018/0.009/0.003 + 0.010 guarantee fund + VAT, totalling 0.043% on buys and 0.143% on sells *before* broker commission) comes from a Feb 2019 university page that reproduces IDX's trading-mechanism table [secondary] ([BINUS Accounting, 2019](https://accounting.binus.ac.id/?p=13586)). Broker pages state the same total, for example Ajaib's 0.0433% [secondary] ([Ajaib](https://ajaib.co.id/biaya), as summarized in search; the page returned 403 to direct fetch). The 0.010% guarantee-fund rate is **UNVERIFIED against a KPEI primary document**.

**Sales tax.** PP 41/1994 as amended by PP 14/1997 sets the rate at "0,1% (satu per seribu) dari jumlah bruto nilai transaksi penjualan" for every share sale on the exchange, ordinary or founder shares alike [primary] ([PP 14/1997, BPK](https://peraturan.bpk.go.id/Download/45728/PP%20No.%2014%20Th%201997.pdf)). KMK 282/KMK.04/1997 implements it at "0,1% ... dari jumlah bruto nilai transaksi penjualan saham", collected by the exchange through the broker [primary text, hosted by DDTC] ([KMK 282/1997](https://perpajakan.ddtc.co.id/sumber-hukum/peraturan-pusat/keputusan-menteri-keuangan-282kmk-041997)). The tax falls only on the sell side. The extra 0.5% founder-share tax does not apply to a retail secondary-market buyer. No rate change took effect in 2019–2025.

**VAT on commission.** VAT rose from 10% to 11% on 1 Apr 2022 under Article 7 of UU 7/2021 (HPP) [primary] ([DJP press release SP-22/2022](https://www.pajak.go.id/sites/default/files/2022-04/SP-22%20%202022%20Reformasi%20PPN%20Kedepankan%20Rasa%20Keadilan.pdf); [Setkab](https://setkab.go.id/tarif-ppn-jadi-11-persen-per-1-april-2022-menkeu-indonesia-tidak-berlebih-lebihan/)). From 1 Jan 2025, PMK 131/2024 applies a 12% rate on a base of 11/12 of the invoice for non-luxury goods and services, which works out to 11% [primary] ([PMK 131/2024, JDIH Kemenkeu](https://jdih.kemenkeu.go.id/download/b6655071-d62e-4733-a9e2-72a8b4d233cd/131%20Tahun%202024.pdf); [DJP explainer](https://www.pajak.go.id/id/artikel/pmk-1312024-tarif-ppn-sebelas-dua-belas)). IDX's trading director confirmed that the effective rate on securities transaction fees is still 11% ("finalnya sama dengan PPN 11%") [secondary] ([Katadata, 8 Jan 2025](https://katadata.co.id/amp/finansial/bursa/677e3ea9de97f/ojk-buka-suara-soal-imbas-ppn-12-ke-perdagangan-saham/)).

### Broker commission

- **Pluang:** 0.15% buy and 0.25% sell, "inclusive of taxes and third party fees". The breakdown lists levy 0.03% + KPEI 0.01% + transaction fee 0.0961% + 11% VAT on levy and fee, plus 0.1% withholding tax on sales only [primary: broker's own page] ([Pluang fee page](https://pluang.com/en/biaya/id-stocks)). The components add up to about 0.151% buy and 0.251% sell, which confirms that the headline numbers are all-in.
- **Stockbit and Mirae Asset:** 0.15% buy / 0.25% sell. **IPOT (Indo Premier):** 0.19% buy / 0.29% sell. Mirae's day-trade plan: 0.08% / 0.18% [secondary] ([Rankia: IPOT vs Stockbit](https://rankia.id/ipot-vs-stockbit); [Rankia: cheapest brokers](https://rankia.id/sekuritas-fee-termurah-indonesia/)). I could not read the brokers' own pages (403/404 or JavaScript-rendered), so these rates and whether they include levy and VAT are **UNVERIFIED** from primary sources. The consistent 0.10-point gap between buy and sell matches the 0.1% sales tax, which points to all-in quotes.
- **Full-service brokers, roughly 0.25–0.35% commission:** **UNVERIFIED**. No primary fee schedule was found. This range underpins the "High" scenario.
- **Minimum fees:** none stated on the pages I read. **UNVERIFIED** for other brokers.

### Recommended explicit cost

Typical all-in: **buy 15 bps, sell 25 bps (40 bps round trip)**. Low: 10/20. High: 30/40. Floor check: the SRO levy (≈4.3 bps) plus the sales tax (10 bps) give 4.3 bps buy and 14.3 bps sell before any commission, so nothing below about 5/15 bps is achievable. Changes over the period: VAT on commission went from 10% to 11% on 1 Apr 2022 (effect under 0.2 bps). Nothing else that matters changed.

---

## 2. Short selling

**Regulation.**
- Before Oct 2024, margin and short selling ran under OJK and IDX rules, with IDX publishing a monthly list of eligible securities.
- POJK 6/2024 (Apr 2024) replaced that framework. IDX then issued Rule II-H, Kep-00157/BEI/10-2024, on margin and short-sell requirements, and Kep-00170/BEI/10-2024 on short-sell limits, both effective 3 Oct 2024 [secondary] ([Hukumonline](https://www.hukumonline.com/klinik/a/aturan-dan-batasan-transaksi-short-selling-di-indonesia-cl4663/); [Okezone, 3 Oct 2024](https://economy.okezone.com/read/2024/10/03/278/3070434/pengumuman-buat-investor-saham-ada-2-aturan-baru-soal-transaksi-margin-dan-short-selling?page=all)).
- Retail customers may short only through an OJK-licensed, IDX-approved broker with a short-sell financing account and initial collateral. Naked shorting is prohibited.
- Eligibility under Rule II-H III.1 includes free float of at least 20% over the trailing 6 months.
- Daily short volume per broker is capped at 0.01–0.03% of listed shares, and aggregate outstanding short positions at 0.04% of listed shares per stock [secondary] ([Hukumonline](https://www.hukumonline.com/klinik/a/aturan-dan-batasan-transaksi-short-selling-di-indonesia-cl4663/)).

**Timeline in the backtest window.**

| Period | Status |
|---|---|
| Oct 2019 – 1 Mar 2020 | Short selling legally allowed for eligible stocks via approved brokers. KEEN was on IDX's short-sell list effective 2 Mar 2020, published 28 Feb 2020 [secondary] ([Kontan, 2 Mar 2020](https://investasi.kontan.co.id/news/simak-saham-saham-yang-bisa-ditransaksikan-secara-marjin-dan-shortsell-bulan-maret?page=all)). KEEN listed on 2 Sep 2019; whether it was on the Jan/Feb 2020 lists is **UNVERIFIED**. Whether any retail KEEN short sale ever happened is **UNVERIFIED**; IDX short-sell volume was historically negligible (also **UNVERIFIED**). |
| 2 Mar 2020 – Oct 2024 | **Banned.** IDX revoked every short-sellable security by letter S-01419/BEI.POP/03-2020 on 2 Mar 2020, the same day KEEN's listing took effect. OJK: "Pelarangan Transaksi Short Selling bagi semua Anggota Bursa mulai tanggal 2 Maret 2020 s.d. batas waktu yang ditetapkan OJK" [primary] ([OJK joint press release, 23 Mar 2020](https://www.ojk.go.id/id/berita-dan-kegiatan/siaran-pers/Pages/Siaran-Pers-Bersama-OJK-dan-SRO-Jaga-Keberlangsungan-Aktivitas-Perdagangan-Bursa-Efek-yang-Teratur.aspx)). IDX reaffirmed the ban in Jan 2021 [secondary] ([Bisnis, 30 Jan 2021](https://market.bisnis.com/read/20210130/7/1349958/bei-tegaskan-larangan-short-selling-transaksi-margin-boleh)). |
| Jun – Aug 2024 | IDX published new lists ahead of launch: 116 stocks (May 2024), 118 (Jun 2024), 113 (Aug 2024). **KEEN is #65 on the Aug 2024 list** [secondary] ([CNBC Indonesia, 15 Aug 2024](https://www.cnbcindonesia.com/market/20240815112122-17-563264/siap-siap-shortsell-launching-oktober-ini-daftar-113-sahamnya)). |
| 3 Oct 2024 – Mar 2025 | New rules in force, but brokers were still being approved; the first ones were named in Feb 2025 [secondary] ([Bisnis, 12 Feb 2025](https://market.bisnis.com/read/20250212/7/1838873/mansek-hingga-ajaib-sekuritas-masuk-daftar-jadi-ab-short-selling)). The launch was planned for Q2 2025 with only 10 LQ45 stocks, not including KEEN [secondary] ([Katadata, 13 Feb 2025](https://katadata.co.id/finansial/bursa/67ad2bc6e235a/daftar-saham-yang-masuk-lis-short-selling-bei-berlaku-mulai-kuartal-ii-2025)). |
| 3 Mar 2025 – end of window | **Postponed.** OJK and IDX deferred implementation on 3 Mar 2025, first to 26 Sep 2025, then to 17 Mar 2026 (and later again) [secondary] ([Detik](https://finance.detik.com/bursa-dan-valas/d-7885380/short-selling-ditunda-hingga-september-2025); [BCA Sekuritas](https://bcasekuritas.co.id/latest-news/news/bei-tunda-implementasi-short-selling-hingga-maret-2026)). Retail short selling finally began on 1 Oct 2026 with five large caps (ASII, BBCA, BRPT, BUMI, INDF), outside the backtest window [secondary] ([Industry.co.id, 1 Oct 2026](https://www.industry.co.id/read/156816/investor-saham-wajib-tahu-skema-transaksi-short-selling-di-bei-resmi-berlaku-mulai-hari-ini)). |

**Plain answer.** Short selling KEEN was effectively unavailable to a retail trader for the whole Oct 2019 – Oct 2025 window. KEEN appeared on two eligibility lists, Mar 2020 and Aug 2024. The first was revoked on the day it took effect, and the second came before any short-sale execution actually went live. The backtest should be long-only (long/flat).

---

## 3. Lot size and tick size

- **Lot:** 1 lot = 100 shares since 6 Jan 2014, under IDX Kep-00071/BEI/11-2013 (previously 500) [secondary] ([Kontan](https://investasi.kontan.co.id/news/hari-ini-perubahan-aturan-lot-saham-berlaku); [Detik](https://finance.detik.com/bursa-dan-valas/d-2457743/satu-lot-jadi-100-lembar-pekan-depan-bei-lakukan-uji-coba)).
- **Tick (fraksi harga):** under IDX Kep-00023/BEI/04-2016 (Rule II-A), in force throughout 2019–2025 [secondary] ([CNN Indonesia, 23 May 2016](https://www.cnnindonesia.com/ekonomi/20160523135607-78-132774/bei-sebut-penambahan-fraksi-harga-dongkrak-transaksi-saham/)):

  | Price | Tick |
  |---|---|
  | < Rp 200 | Rp 1 |
  | Rp 200 – < 500 | Rp 2 |
  | Rp 500 – < 2,000 | Rp 5 |
  | Rp 2,000 – < 5,000 | Rp 10 |
  | ≥ Rp 5,000 | Rp 25 |

  I found no tick change between 2019 and 2025. IDX revised Rule II-A in Sep–Oct 2026, outside the window. That the April 2025 Rule II-A reissue (Kep-00003/BEI/04-2025) left the tick bands unchanged is **UNVERIFIED** from the primary text.

- **Effect on KEEN.** KEEN's (adjusted) close ranged from Rp 245 to Rp 1,135 and sat below Rp 500 on 43% of days. One tick was 40–99 bps of price, median 65 bps (computed from `data/KEEN_2019_2025.csv`). **That is material for ±2% targets.** A 2% move covers only 2–5 ticks. Targets snap to valid prices, for example 1.67% or 2.5% at Rp 600 instead of 2.0%. Crossing the spread costs about half a tick, 20–50 bps, per fill. Lot size is not material: at Rp 700 one lot is Rp 70,000, so rounding an IDR 10 m order loses under 0.7% of the position.

---

## 4. Auto-rejection limits (ARA/ARB), 2019–2025

Limits are percentages from the reference (previous close) price. KEEN traded in the Rp 200–5,000 band throughout.

| Effective | ARA (upper), Rp 200–5,000 band | ARB (lower) | Instrument | Source |
|---|---|---|---|---|
| before 10 Mar 2020 | 25% (35% for Rp 50–200, 20% above Rp 5,000) | 25%, symmetric | Rule II-A | [secondary] [Bisnis, 14 Aug 2023](https://market.bisnis.com/read/20230814/7/1684509/auto-rejection-simetris-mulai-september-2023-ara-arb-saham-bisa-35-persen) |
| 10 Mar 2020 | 25% (unchanged) | **10%**, all bands | Kep-00023/BEI/03-2020, ordered by OJK S-273/PM.21/2020 | [secondary] [Kontan, 9 Mar 2020](https://investasi.kontan.co.id/news/ihsg-jatuh-658-bei-ubah-batasan-auto-rejection-dan-ini-rinciannya) |
| 13 Mar 2020 | 25% (unchanged) | **7%**, all bands (also: pre-opening session suspended for all stocks) | Kep-00025/BEI/03-2020, OJK S-281/PM.21/2020 | [secondary] [Detik, 13 Mar 2020](https://finance.detik.com/bursa-dan-valas/d-4936869/ingat-saham-anjlok-7-langsung-disetop-bei-mulai-hari-ini) |
| 5 Jun 2023 | 25% | **15%**, all bands (phase I) | Kep-00055/BEI/03-2023 (number as reported) | [secondary] [Kontan](https://investasi.kontan.co.id/news/bei-pastikan-auto-rejection-simetris-berlaku-pada-4-september-2023) |
| 4 Sep 2023 | 25% | **25%**, symmetric again (phase II) | same | [secondary] [Kontan](https://investasi.kontan.co.id/news/bei-pastikan-auto-rejection-simetris-berlaku-pada-4-september-2023); [Bareksa, 31 Aug 2023](https://www.bareksa.com/berita/saham/2023-08-31/bei-aturan-baru-batasan-persentase-auto-rejection-simetris-berlaku-mulai-4-september-2023) |
| 8 Apr 2025 | 25% (no change reported) | **15%**, all bands; IHSG halt thresholds of 8% (30 min), 15% (30 min), 20% (suspension) | Emergency guideline and Rule II-A amendment, Apr 2025 | [secondary, exchange association] [ASEAN Exchanges, 8 Apr 2025](https://www.aseanexchanges.org/content/idx-adjusted-rules-on-lower-limit-of-auto-rejection-and-rules-on-temporary-halt-on-securities-trading/) |

Notes:
- **Decree numbers disagree for April 2025.** The ASEAN Exchanges page cites Kep-00196/BEI/12-2024 and Kep-00024/BEI/03-2020 as the regulations amended. News reports name the new decrees Kep-00002/BEI/04-2025 and Kep-00003/BEI/04-2025. The latter is confirmed as the Rule II-A of 8 Apr 2025 by the recitals of the KSEI decree above [primary].
- **ARA in April 2025:** an unchanged upper limit is **UNVERIFIED** from primary text.
- **Out of window:** IDX cut ARB to an asymmetric 7% from 19 Oct 2026 (Kep-00157/BEI/10-2026) [secondary] ([BCA Sekuritas](https://bcasekuritas.co.id/latest-news/news/bei-sesuaikan-batas-bawah-auto-rejection-efek-bersifat-ekuitas)).
- **Pre-opening suspension in 2020:** while the pre-opening session was suspended for all stocks, the daily "Open" was the first continuous-session trade, not an auction price. When the pre-opening session was restored is **UNVERIFIED**.

**Can a 2% stop be blocked or gapped through?**
- ARB itself never blocks a 2% stop, because the tightest limit in the window was 7%.
- Gaps are the real risk. KEEN opened at least 2% below the prior close on 39 of 1,464 days (2.7%). A resting stop fills at or below the open on those days.
- On a day locked at the lower limit (ARB) with no bids, sell orders cannot fill at all. KEEN closed down at least 6.9% on 9 days, which are candidates for limit-down. Whether those days were actually locked with zero bids is **UNVERIFIED**, since there is no order-book data.
- Rule: fill stops at min(stop, open). When the close is at the day's lower limit, treat the exit as delayed to the next session's open.

---

## 5. Liquidity

From the repo's Yahoo data (`data/KEEN_2019_2025.csv`, traded value = Close × Volume; adjusted close slightly understates raw value). The source is secondary data; I could not reach IDX's own statistics pages (HTTP 403).

| Year | Median daily value (IDR bn) | 10th pct | 90th pct |
|---|---|---|---|
| 2019 (Oct–Dec) | 3.18 | 1.42 | 10.29 |
| 2020 | 0.67 | 0.06 | 0.98 |
| 2021 | 0.56 | 0.24 | 1.07 |
| 2022 | 0.88 | 0.36 | 20.45 |
| 2023 | 4.45 | 1.53 | 16.21 |
| 2024 | 0.94 | 0.26 | 6.64 |
| 2025 (to 24 Oct) | 2.54 | 0.44 | 7.63 |
| **All** | **0.96** | 0.33 | — |

Order of magnitude: **about IDR 1 bn per day on a median day, often IDR 0.3–0.7 bn in 2020–2021**. Market cap was roughly IDR 2–3 tn [secondary] ([Bareksa](https://www.bareksa.com/id/saham/detail/KEEN/kencana_energi_lestari_tbk); [Kabar Bursa](https://www.kabarbursa.com/market-hari-ini/profil-keen-emiten-karbon-dengan-kapitalisasi-pasar-rp2328-miliar)). IDX issued unusual market activity (UMA) notices on KEEN, so volume bursts are episodic [secondary] ([BCA Sekuritas](https://bcasekuritas.co.id/latest-news/news/bei-cermati-pergerakan-saham-hajj-dan-keen); [EmitenNews](https://emitennews.com/news/bei-telisik-kencana-energi-keen-laba-drop-saham-melonjak)).

**Slippage by position size:**
- **IDR 10 m** is about 1% of median daily value. Expect to pay roughly the spread (half a tick, 20–50 bps) with little extra impact.
- **IDR 100 m** is about 10% of median daily value and above 1% of daily value on 1,320 of 1,464 days. It will often walk the book beyond the best price, especially in 2020–2021. Model it with the high scenario (≥50 bps per side slippage) or split execution.
- No bid-ask or order-book data was available, so the spread estimate (one tick) is an assumption: **UNVERIFIED**.
