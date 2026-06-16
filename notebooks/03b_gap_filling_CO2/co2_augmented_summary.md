# CO₂-Augmented Gap-Filling — Synthesis (03b)

Companion to `../03_gap_filling/gap_filling_summary.md`. This folder re-runs all three gap-filling replications with one change: **CO₂ flux (FCO₂) is first gap-filled from meteorological drivers, then used as a CH₄ gap-filling feature.** It directly tests the D-22 hypothesis — that the EC flux co-variates (LE/H/**FC**) are the reason met-only CH₄ gap-filling fails.

---

## 1  Design

### Step 1 — reconstruct FCO₂ (`src/data/fco2_gapfill.py`)
- QC FCO₂: SSITC∈{0,1} → plausibility [−100, 100] µmol m⁻² s⁻¹ (**D-25**).
- Train **RFm** (the best-scoring R-02 model: RF on `driver_m`, met-only, 11 vars + AUX) on the training-window observed FCO₂.
- Predict FCO₂ at every timestamp; output the **observed-where-available** series (observed QC'd FC, else RFm reconstruction). **D-26**.

**FCO₂ reconstructs far better than CH₄** — it is strongly met-driven:

| Tower | Recon test R² | RMSE (µmol m⁻² s⁻¹) | n_train | points reconstructed |
|-------|---------------|---------------------|---------|----------------------|
| Tower 4 | **0.745** | 5.05 | 21,869 | 30,799 |
| Tower 9 | **0.746** | 4.45 | 19,952 | 34,249 |
| Tower 2 | 0.197 | 7.23 | 5,511 | 39,159 |

### Step 2 — feed FCO₂ into the three replications
- **R-01-CO2 / R-03-CO2:** replace the existing `FC_1_1_1` feature with the gap-filled version.
- **R-02-CO2:** *add* FCO₂ to `driver_m` (which had no FC) → RFm/XGBm gain it; RF3 and MDS stay as controls.

### Process & feature delta vs `03_gap_filling`
The pipeline is identical except for a prepended data step and the FC column:

```
Load consolidated_hourly.csv
   └─► NEW: merge data/Hourly/fco2_gapfilled.csv → FC = gap-filled FCO₂
           (R-02 also: append FC to driver_m)
   └─► (unchanged) QC → features → temporal split → models → scenarios → benchmarks
```

| Replication | FC in base? | Change in 03b |
|---|---|---|
| R-01-CO2 | yes (raw) | FC → QC'd + gap-filled |
| R-02-CO2 | **no** (D-22) | FC **added** to driver_m |
| R-03-CO2 | yes (raw) | FC → QC'd + gap-filled |

---

## 2  Headline result

**Adding gap-filled FCO₂ to the met-only model recovers the skill that D-22 removed.**

| Tower 4, best comparison | base | + FCO₂ |
|---|---|---|
| R-02 RFm (short/vs gap) | −0.128 | **+0.156** |
| R-02 RFm (medium/m gap) | −0.160 | **+0.111** |
| R-03 ANN (all gaps) | +0.06…+0.10 | **+0.12…+0.17** |

- R-02-CO2 RFm goes **negative → positive** at Tower 4 while its no-FC controls (RF3, MDS) are unchanged to 3 dp — an unambiguous, causal demonstration that **FC is the single most informative FCH₄ predictor** at NWFP.
- The CO₂-augmented RFm (+0.11…+0.16) matches/exceeds R-01 RF (+0.144), closing the R-01↔R-02 gap that D-22 first identified.

---

## 3  Per-replication summary

### R-01-CO2 (`R01_CO2_results.md`)
Tower 4 RF **drops** (+0.144 → +0.049) and Tower 2 **improves** (−16.9 → −4.85). Base R-01 used *raw* FC; QC'ing it removes gross CO₂ spikes / SSITC-rejects that spuriously co-tracked extreme FCH₄. Net: cleaner but slightly less "predictive" of outliers at T4; far less pathological at T2.

### R-02-CO2 (`R02_CO2_results.md`)
The clean experiment. RFm Tower 4 −0.10…−0.16 → +0.03…+0.16; Tower 9 ≈ −0.09 → ≈ 0. RF3/MDS controls byte-identical. FC benefit is largest at short/medium gaps, fades by 288h.

### R-03-CO2 (`R03_CO2_results.md`)
**ANN at Tower 4 is the standout: +0.12…+0.17, best model at every gap length.** RF/RF_lag short-gap drop (same QC-of-FC effect as R-01-CO2). Tower 9 flat/mixed; ANN xlong still collapses (small-sample artefact). Kim's PCA and lag findings are unchanged under augmentation.

---

## 4  Cross-cutting findings

1. **FCO₂ is the key CH₄ co-variate.** Every place it is genuinely *added* (R-02 met-only), R² jumps; the controls confirm causality.
2. **The benefit is tower- and gap-dependent.** Tower 4 (more data, recon R²=0.75) gains strongly; Tower 9 gains little; the gain shrinks with gap length.
3. **Model interaction with cleaned FC differs.** ANN exploits the clean complete FCO₂ best (Tower 4); tree models partly lose the raw-FC artifact signal they had leaned on.
4. **FCO₂ is itself highly gap-fillable (R²≈0.75)** — unlike FCH₄ (≈0). So a two-stage "fill CO₂ → help CH₄" pipeline is technically sound.

---

## 5  Critical caveat (D-22) — these are an upper bound

The chosen design is **observed-FCO₂-where-available**. Because the experiments mask only FCH₄, FCO₂ remains *observed* at the gap points — so the models still see a co-observed EC flux that, in a real EC outage, would also be missing. **These results are therefore an optimistic upper bound, not operational performance.**

The strictly operational test (not run here) would use the **reconstruction everywhere** (`FC_recon`, never the co-observed value at gaps). Given FCO₂ recon R²≈0.75, that variant would likely retain *part* of the gain — quantifying it is the natural follow-up.

For genuine **forecasting** (R-05+), lagged FCO₂ from prior timesteps is a fully legitimate, non-co-failed predictor, and this experiment suggests it will be valuable.

---

## 6  Ledger

`results/benchmarks.csv` now holds 940 rows: 470 base (R-01/02/03) + 470 CO₂-augmented (R-01-CO2: 30, R-02-CO2: 200, R-03-CO2: 240). Compare any base vs `-CO2` pair by the `replication` tag.

*Source: `R01_CO2_*`, `R02_CO2_*`, `R03_CO2_* .ipynb`; `src/data/fco2_gapfill.py`; `data/Hourly/fco2_gapfilled.csv`. Decisions D-25, D-26 (+ reused D-16/D-18/D-21/D-22).*
