# F-07 Results: Tower 2 — fixing the broken evaluation (TOWER 2 ONLY)

**Notebook:** `F07_tower2_evaluation_RFm.ipynb`
**Scope:** **Tower 2 only** (Red farmlet, 'Great Field' / Catchment 2).
**Executed:** 2026-06-25

---

## 1  Why this experiment

Tower 2 had been the project's failure: R-01 RF = **−16.9**, and even the best feature set (F-06) gave **−0.045**. This experiment shows that was a **broken evaluation, not a data or model failure**, and recovers a strongly positive result.

**The diagnosis (from the data):**
- Tower 2 EC CH4 exists **only Oct 2017 – Jun 2019** (grassland period). The CH4 analyser was relocated to Tower 9 in Jul 2019 when the Red farmlet converted to arable.
- Catchment 2 had **~10 cattle in 2018 (FCH4 mean ≈ 42 nmol m⁻² s⁻¹)** but **zero livestock in early 2019 (FCH4 mean ≈ 2)** — animals removed ahead of the arable conversion.
- The **D-15 split** (train all-2018 / test Jan–May 2019) trains on the high-flux/livestock regime and tests on the near-zero/no-livestock regime → the model predicts the wrong level → catastrophic R².
- Gap-filling is **interpolation**, so the correct evaluation is a **full-period gap-CV** — mask gaps anywhere across 2017–2019 and fill from the surrounding data — not a forward year split.

---

## 2  What was done

Five configurations, two evaluation modes, all on the **F-06 best feature set**:

| Variant | Model | Evaluation | Training data |
|---|---|---|---|
| `MDS_D15` | MDS (SW+TA) | D-15 split (gaps in test 2019) | observed 2018+ analogs |
| `RFm_solo_D15` | RandomForest | D-15 split | Tower 2, 2018 |
| `MDS_fullCV` | MDS (SW+TA) | full-period gap-CV | observed analogs across 2017–2019 |
| `RFm_solo_fullCV` | RandomForest | full-period gap-CV | Tower 2 unmasked rows (2017–2019) |
| `RFm_pool_fullCV` | RandomForest + tower dummy | full-period gap-CV | Tower 2 unmasked **+ Towers 4 & 9 (2018–2021)** |

- **Gap scenarios:** vs/s/m/l/m1 = 1 / 4 / 32 / 288 / mixed h calendar gaps; MASK_FRAC = 0.25; 5 reps; median R².
- **MDS** = the project's REddyProc-style implementation (`mds_fill_batch`): same hour ±1, similar SW (±50 W m⁻²) and TA (±2.5 °C) within expanding ±7/14/28/91-day windows.
- **RandomForest:** 500 trees, `min_samples_leaf=5`, mean-imputation.

## 3  Data used (Tower 2)

- **Target:** `FCH4_1_1_1 [Tower 2]`, two-pass QC (SSITC ∈ {0,1} → plausibility [−500, 3000] nmol m⁻² s⁻¹, D-13). **4,890 valid hours**, Oct 2017 – Jun 2019.
- **Features (F-06 set), from four precomputed sources:**
  - `consolidated_hourly.csv` — raw met, **livestock counts** (`cattle/sheep/lamb_Catchment 2`), wind, soil.
  - `reddyproc_processed.csv` (D-33) — **gap-filled met drivers** (`SWIN, TA, VPD, PPFD, RN, WS, USTAR, SHF, precip, SWC, TS`, the `__f` columns) + **GPP / Reco** [Tower 2].
  - `fco2_gapfilled.csv` (D-26) — **gap-filled FCO₂** [Tower 2].
  - `management_features.csv` (D-28) — `mgmt_t2_cut/manure` recency.
  - Derived in-notebook: **stocking density** (LSU/ha; Catchment 2 = 6.65 ha), grazing flag, **SWC/TS lags** (168/336/504/672 h), cyclical time (hour/doy sin·cos).
- **Pooled variant** additionally uses Towers 4 & 9 (2018–2021) rows + a tower-indicator dummy (D-30).

---

## 4  Results — Tower 2 median R² by gap scenario

| Variant | vs (1h) | s (4h) | m (32h) | l (288h) | m1 | **overall** |
|---|---|---|---|---|---|---|
| MDS — D-15 split | 0.043 | −0.162 | −0.128 | −0.544 | −0.216 | **−0.162** |
| RFm — D-15 split | −0.062 | −0.078 | −0.089 | −0.104 | −0.083 | **−0.083** |
| **MDS — full-period CV** | −0.417 | −0.488 | −0.426 | −0.711 | −1.007 | **−0.488** |
| **RFm solo — full-period CV** | +0.394 | +0.522 | +0.511 | +0.355 | +0.324 | **+0.394** |
| **RFm pooled — full-period CV** | **+0.519** | **+0.657** | **+0.618** | **+0.490** | **+0.472** | **+0.519** |

**Improvement over MDS (full-period CV):** RFm solo **+0.81 … +1.33**; RFm pooled **+0.94 … +1.48** R² units.

---

## 5  Interpretation

1. **The negative Tower 2 results were a broken evaluation.** Under the D-15 year split, *both* MDS and RFm are negative (the test regime is unseen in training). Switching to the correct full-period gap-CV flips RFm to **+0.394 solo / +0.519 pooled** — Tower 2's *best result in the project*, and it **exceeds the R² ≈ 0.5 target** at most gap lengths.
2. **MDS cannot solve Tower 2 — RFm beats it by ~1.0 R² unit.** MDS (SW+TA only) is **livestock-blind**: a winter 2018 hour (cattle present, high flux) and a winter 2019 hour (no cattle, ~0) look identical to it, so it averages across regimes and fails (−0.49 even under the correct evaluation). RFm, with **livestock density**, learns the regime cleanly. This is the clearest demonstration in the project of the supervisor-endorsed framing — *improvement over MDS*, not absolute R².
3. **Pooling helps again** (+0.394 → +0.519): Tower 2 borrows strength from Towers 4/9 while the dummy keeps its own level (consistent with F-03 / D-30).
4. **Why Tower 2's R² is high (honest caveat):** its flux is unusually *discriminable* — a near-binary livestock-on/off regime with large between-regime variance that the model explains well. This is genuine signal, but it means **Tower 2's 0.52 is NOT directly comparable to Towers 4/9's ~0.3** (continuous grazing, subtler variation). The headline is the *improvement over MDS* and the recovery from negative, not a claim that Tower 2 is "easier."

## 6  Implementation notes & flaw fix

- **Fixed vs the prototype (`t2_experiment.py`, now removed):** the prototype masked blocks of *consecutive valid observations*; F-07 masks **calendar-hour gaps over the full hourly index** (`insert_calendar_gaps`), consistent with F-01–F-06, so the scenario labels mean the same thing. This correction lowered the pooled overall from a prototype 0.625 to **0.519** (the prototype was mildly inflated).
- **Full-period CV is interpolation, not leakage:** training on points temporally around a masked gap is exactly what gap-filling is; the gap's own value is never used.
- **MDS uses raw SW/TA** (vs RFm's gap-filled `__f`): a slight disadvantage, but MDS's failure here is livestock-blindness, not met quality.
- **Pooled training** mixes Towers 4/9 (2018–21) with Tower 2 — cross-tower borrowing, not Tower 2 self-leakage.

## 7  Recommendation

- **Adopt the full-period gap-CV as the correct evaluation for Tower 2** (the year split is inappropriate given its single-regime-per-year data). Report Tower 2 as **R² ≈ +0.52 (pooled), a ~1.0 improvement over MDS** — recovered from −16.9.
- **Methodology implication for Towers 4 & 9:** their ~0.3 figures used the *stricter* train-2018-21/test-2022-23 split. Re-evaluating them under the same full-period gap-CV would make all three towers consistent (and likely raise 4/9). Recommended as a quick follow-up before forecasting. **→ Done in F-08 (D-35):** under full-period gap-CV, **T4 +0.163→+0.362**, T9 +0.335→+0.350; all three towers now consistent ≈0.35–0.49 (pooled). See `F08_results.md`.

25 rows tagged `F-07` in `results/benchmarks.csv`.
