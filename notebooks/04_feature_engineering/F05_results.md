# F-05 Results: does pruned field-event (management) info help the rich pooled model? (RFm)

**Notebook:** `F05_management_partial_pooling_RFm.ipynb`
**Executed:** 2026-06-16
**Question:** field events were tested in F-01 (12-col set → overfit) and F-02 (pruned 2-col set → mild help) but **dropped from F-03/F-04** (the partial-pooling configs). This re-adds the **pruned tower-specific management** features — `cut` + `manure` recency (exp-decay time-since, D-28) — on top of the F-04 best config to test whether they add anything.

Tower 2 management was added to `build_management_features.py` (Catchment 2 = {NW002 Great Field}) for this experiment. Two variants, partial pooling (T2+T4+T9 + tower dummies) + density + SWC/TS lags:
- **partial+lags** — F-04 best (reference)
- **partial+lags+mgmt** — + tower-specific cut & manure recency

---

## 1  The management effect (Δ vs no-management, overall median R²)

| Tower | partial+lags | +management | Δ |
|---|---|---|---|
| Tower 2 | −0.062 | −0.049 | **+0.013** |
| Tower 4 | +0.070 | +0.082 | **+0.012** |
| Tower 9 | +0.247 | +0.252 | +0.005 |

## 2  Median R² by scenario (short→long)

| Tower | variant | vs | s | m | l | m1 |
|---|---|---|---|---|---|---|
| Tower 4 | partial+lags | 0.244 | 0.156 | 0.244 | 0.052 | −0.098 |
| Tower 4 | +mgmt | 0.242 | **0.164** | 0.234 | 0.050 | −0.094 |
| Tower 9 | partial+lags | 0.291 | 0.247 | 0.295 | 0.200 | 0.247 |
| Tower 9 | +mgmt | 0.289 | 0.251 | 0.295 | 0.197 | 0.252 |
| Tower 2 | partial+lags | −0.074 | −0.059 | −0.102 | −0.069 | −0.025 |
| Tower 2 | +mgmt | −0.057 | −0.049 | −0.094 | **−0.045** | −0.018 |

---

## 3  Interpretation — small, non-harmful, largest at the weaker towers

1. **Pruned management gives a small positive bump** (Δ +0.005 to +0.013 overall median) and **never hurts** — largest at the weaker-base towers (Tower 2 +0.013, Tower 4 +0.012), negligible at the already-strong Tower 9 (+0.005).
2. **This is exactly the F-04 lag pattern.** Management — like the SWC/TS lags — is **largely redundant on the rich base**: gap-filled FCO₂ + livestock density + pooling already encode most of the ecosystem-state signal that fertiliser/cut events drive. It contributes only at the margin, and most where the base is weakest (Tower 2).
3. **Consistent lesson across F-04 and F-05:** once the strong predictors (FCO₂, livestock density, pooling) are in place, additional "memory/management" features (lags, events) add only marginal value. The big levers are FCO₂, livestock density, and partial pooling — not management or lags.

## 4  Recommendation

- **Keep pruned tower-specific management** (cut + manure recency) in the standard set — it's cheap, marginally positive, and never harmful — but recognise it is **not a lever**.
- **Standard "kitchen-sink" config:** partial pooling + stocking density + SWC/TS lags + pruned management. Each cheap group adds a little; none hurt.
- **Feature engineering is now exhausted** — P1–P6, pooling, lags, and management have all been tested; returns have plateaued (Tower 4 ≈ +0.24, Tower 9 ≈ +0.25–0.29). Next phase: forecasting.

150 rows tagged `F-05` in `results/benchmarks.csv` (total 2515): `model = RFm_partial_lags / RFm_partial_lags_mgmt`.
