# F-06 Results: REddyProc-style met gap-filling + GPP — does it beat mean-imputation? (RFm)

**Notebook:** `F06_reddyproc_pipeline_RFm.ipynb`
**Pipeline:** `src/data/reddyproc_pipeline.py` → `data/Hourly/reddyproc_processed.csv`
**Executed:** 2026-06-16
**Motivation:** prompted by the NWFP/REddyProc EC processing report (RPubs `970790`). Every model so far **mean-imputed** the meteorological drivers (SWIN ~52–75% present). We re-implemented the REddyProc-style pipeline in Python (D-33): gap-fill the met drivers properly (linear interp + mean-diurnal-course), estimate u*-thresholds, and partition CO₂ flux → GPP/Reco (nighttime Lloyd-Taylor). This A/B/C tests the **F-05 best config** (partial pool + density + lags + pruned management) varying **only** the driver preprocessing — identical training rows + identical gap scenarios.

- **imputed** — raw met + mean-imputation (baseline)
- **metfill** — REddyProc-gap-filled met drivers (preserve diurnal cycle)
- **metfill+gpp** — + GPP & Reco features

---

## 1  Pipeline outputs (sanity)
Met drivers gap-filled to **100%** within each tower (from 52–78% raw); gap-filled SWIN preserves the diurnal cycle (Tower 4 night ≈ 0.7 → midday ≈ 387 W m⁻²). u*-thresholds: T2 = 0.043, T4 = 0.106, T9 = 0.121 m s⁻¹. Partitioning E0 = 194 / 330 / 285 K; GPP ≥ 0, coverage 72–79%.

## 2  Headline — overall median R² and Δ vs imputed

| Tower | imputed | metfill | **metfill+gpp** |
|---|---|---|---|
| Tower 2 | −0.126 | −0.050 (**+0.076**) | **−0.045** (+0.081) |
| Tower 4 | +0.116 | +0.138 (+0.022) | **+0.163** (+0.047) |
| Tower 9 | +0.270 | +0.287 (+0.017) | **+0.335** (+0.065) |

## 3  Median R² by scenario (metfill+gpp = best)

**Tower 9** | vs | s | m | l | m1 |
|---|---|---|---|---|---|
| imputed | 0.302 | 0.267 | 0.325 | 0.192 | 0.265 |
| metfill | 0.323 | 0.298 | 0.338 | 0.203 | 0.282 |
| **metfill+gpp** | **0.356** | **0.316** | **0.374** | **0.269** | **0.332** |

**Tower 4** | vs | s | m | l |
|---|---|---|---|---|
| imputed | 0.232 | 0.150 | 0.227 | 0.060 |
| **metfill+gpp** | **0.250** | **0.187** | **0.301** | **0.081** |

---

## 4  Interpretation — the first addition since pooling that genuinely helps

1. **Proper met gap-filling beats mean-imputation** (Δ +0.017 to +0.076 overall), **largest at the coverage-poorest tower** (Tower 2, Δ+0.076 — lowest met availability). Crucially this is **not** the "redundant on the rich base" pattern of F-04 (lags) / F-05 (management): those *added* information the base already had; **met-fill *fixes the inputs*** (better SWIN/TA/… values → better predictions), which the base could not compensate for.
2. **GPP/Reco adds more on top** (a further Δ up to +0.048; Tower 9 metfill +0.287 → +gpp **+0.335**). GPP (gross productivity, from CO₂ partitioning) is a genuinely new biophysical driver — substrate supply for methanogenesis — and a far better productivity signal than the crude `SWIN×TA` proxy tried in F-01's P5.
3. **metfill+gpp is the new best config across all towers:** **Tower 9 ≈ +0.335 (new project best)**, Tower 4 +0.163, Tower 2 −0.045 (best Tower 2 yet, still negative — D-15 split).
4. **Why this worked where lags/management didn't:** met-fill and GPP attack different things than the redundant features — input quality and a missing productivity driver — so they are *not* redundant with FCO₂ + density + pooling.

**Note on the baseline:** F-06's `imputed` uses mean-imputation on target-valid rows (not F-05's met-dropna), so it sits slightly above F-05's reported numbers; the clean comparison is imputed→metfill→+gpp on **identical** rows.

## 5  u*-threshold filtering (reported separately)
Per-tower binned-plateau u*-thresholds flagged nighttime low-turbulence flux: T2 ≈ 1,257 h, T4 ≈ 9,319 h, T9 ≈ 9,556 h. **Not applied to the CH4 R² ablation** — u*-filtering for CH4 is debated (ebullition / non-turbulent transport); its primary role is the CO₂ partitioning. Available as a flag for future sensitivity tests.

## 6  Recommendation
- **Adopt the REddyProc met-fill + GPP** as standard. New best config: **partial pool + density + lags + pruned management + gap-filled met drivers + GPP/Reco**.
- Carry the gap-filled drivers + GPP into the forecasting phase.
- *Caveats:* pragmatic simplifications of REddyProc's bootstrap u* (Papale 2006) and partitioning (Reichstein 2005 / Lasslop 2010) — documented in `reddyproc_pipeline.py`; partitioning uses nighttime method only.

225 rows tagged `F-06` in `results/benchmarks.csv` (total 2740): `model = RFm_imputed / RFm_metfill / RFm_metfill_gpp`.
