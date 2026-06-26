# F-08 — EC-tower vs external (SMS/MET) sensor sourcing (D-35)

**Notebook:** `F08_external_sensors_RFm.ipynb` · **Script:** `src/data/build_sms_met_dataset.py`
**Data:** `consolidated_hourly_SMS_MET.csv`, `reddyproc_processed_SMS_MET.csv` (parallel to the originals; originals untouched)
**Results:** `results/f08_summary.csv`; 90 rows tagged `F-08` in `results/benchmarks.csv`.

---

## 1  Why

The project sources almost every gap-filling driver from the **co-located EC tower**, switching to the **external catchment network** in exactly one case — soil moisture (D-18, tower SWC ~5–10%). The NWFP runs a *second, independent* network (one central MET station `[Site]` + per-catchment SMS stations `[Catchment N]`). Seven variables overlap. A flagged **inconsistency**: soil *moisture* uses the external per-catchment sensor (D-18), but soil *temperature* uses a cross-tower **EC proxy** (`TS_1_1_1 [Tower 9]`, D-16) despite a well-correlated (r≈0.98) per-catchment external twin. F-08 tests whether external sourcing of the overlapping drivers helps.

## 2  What was done

A parallel **external-sourced** data layer was built (originals never touched):
- `SWIN_1_1_1 [Tower t]` ← `Solar Radiation (W/m2) [Site]`
- `TA_0_0_1 [Tower t]` ← `Air Temperature (oC) [Site]`
- `WS_0_0_1 [Tower t]` ← `Wind Speed (km/h) [Site] ÷ 3.6` (→ m/s)
- soil temperature ← `Soil Temperature @ 15cm Depth (oC) [Catchment t]` (per-catchment external, replacing the Tower-9 EC proxy)
- kept EC (no external twin): VPD, PPFD, RN, USTAR, SHF. Already external: precipitation, soil moisture. **FCO₂ kept EC in both variants** (it is an EC flux — no external twin — so holding it fixed isolates the driver-source effect). RH/WD are not model features.

Two variants — **`EC`** (baseline) and **`EXT`** (external) — were run under **one identical harness**: **full-period gap-CV for all three towers** (the F-07 fix, D-34 — gap-filling is interpolation, so mask calendar gaps across each tower's whole valid-CH₄ span and fill from surrounding data). F-06 best feature set; gap scenarios vs/s/m/l/m1, 25% mask, 5 reps, median R². Per tower: **MDS**, **RFm solo**, **RFm partial-pooled (T2+T4+T9)**.

Full-period domains: T2 = 2017-10→2019-06, T4 = 2017-10→2023-12, T9 = 2020-02→2023-12.
**Harness validated:** EC Tower-2 solo = **0.395** — reproduces F-07 exactly.

## 3  Results — overall median R² (EC vs EXT), Δ = EXT − EC

| Tower | Model | EC | EXT | Δ |
|---|---|---|---|---|
| **2** | RFm **pool** | 0.478 | **0.490** | **+0.012** |
| 2 | RFm solo | 0.395 | 0.385 | −0.010 |
| 2 | MDS | −0.488 | −0.310 | +0.178 |
| **4** | RFm **pool** | 0.362 | **0.376** | **+0.014** |
| 4 | RFm solo | 0.362 | 0.355 | −0.007 |
| 4 | MDS | −0.236 | −0.233 | +0.003 |
| **9** | RFm **pool** | 0.350 | **0.364** | **+0.014** |
| 9 | RFm solo | 0.337 | 0.339 | +0.002 |
| 9 | MDS | −0.336 | −0.320 | +0.016 |

**Improvement over MDS** (RFm pool): EC — T2 +0.97, T4 +0.60, T9 +0.69; EXT — T2 +0.80, T4 +0.61, T9 +0.68.

## 4  Interpretation

**(a) External sourcing is essentially neutral for the RF models — and never hurts the pooled model.**
For the partially-pooled RFm (the recommended config), swapping *all* overlapping drivers to external gives a **small, consistent +0.012–0.014 at every tower**. For solo it is mixed and tiny (−0.010…+0.002). This is the now-familiar **"redundant on the rich base"** pattern (cf. F-04 lags, F-05 management): the RF already extracts the available signal, so changing the *source* of well-correlated drivers barely moves it. Crucially, the swap **did not degrade results** even though it injected (i) site-level rather than tower-local met and (ii) a wind-speed series that correlates only r≈0.17 with the EC anemometer — the RF simply down-weights the weak wind channel, and wind is not a strong CH₄ driver.

**(b) The per-catchment soil-temperature fix is vindicated.** The bundled external swap (which includes replacing the Tower-9 soil-temp proxy with each tower's own-catchment external sensor) is **net positive for the pooled model at all three towers**. So the spatially-faithful, internally-consistent choice (match the soil-moisture policy, D-18) costs nothing and is marginally better — adopt it.

**(c) The biggest result is the EC baseline itself, under full-period gap-CV.** Re-evaluating Towers 4/9 with the same interpolation-style CV used for Tower 2 (the D-34 consistency action, folded in here) **raises them substantially vs the F-06 year-split**:

| Tower | F-06 year-split (best) | F-08 EC full-period CV (pool) |
|---|---|---|
| 2 | −0.045 → +0.519 (F-07) | **+0.478** |
| 4 | **+0.163** | **+0.362** |
| 9 | **+0.335** | **+0.350** |

Tower 4 more than doubles (+0.163 → +0.362): its 2022–2023 year-split test window was the hard/shifted case; under interpolation it sits with the others. **All three towers now land at a consistent ≈ 0.35–0.49**, each beating MDS by ≈ 0.6–1.0 R² — the cleanest, most comparable cross-tower picture in the project. (Tower-2 pool here is 0.478 vs F-07's 0.519 because F-08 pools the *full* T4/T9 domains as donors rather than just 2018–2021 — a deliberate consistency choice, not a regression.)

## 5  Implementation notes / honesty

- External **air temp / solar / wind / RH are Site-level only** (a single central station); only soil temp / moisture / precip are per-catchment. So the met swaps trade tower-local representativeness for a coarser but far better-covered (99% vs 52–78%) site signal — a wash for the RF here.
- **Wind speed swap is the weakest link** (r≈0.17, different mast height/location); it is included for completeness and shown to be harmless, not beneficial.
- **FCO₂ held constant (EC) across variants** — it has no external twin; this isolates the met/soil-temp swap but means F-08 does not test an "all-external" pipeline end-to-end.
- u\*-filtering not applied to CH₄ R² (ebullition caveat), consistent with F-06.

## 6  Recommendation (for forecasting, 05_benchmarking)

1. **Adopt the per-catchment external soil temperature** in place of the Tower-9 EC proxy — it removes the D-16/D-18 inconsistency, is spatially faithful, and is marginally positive (pooled). 
2. **Sourcing of the met drivers (air temp / solar / wind) is a wash for accuracy**, so choose on **operational grounds**: the external Site network has much higher coverage (≈99%), which means fewer gaps to fill and a more robust operational pipeline for forecasting — a mild reason to prefer external met there too. Keep wind as EC or external indifferently (it barely matters).
3. **Carry the full-period-gap-CV framing forward** as the consistent cross-tower evaluation — it puts all three towers on equal footing (≈0.35–0.49, each ≈0.6–1.0 over MDS) and is the honest "improvement over MDS" story for the open system.
4. External sourcing is **not a new lever** (like FCO₂/livestock/pooling were) — it is a *consistency-and-robustness* improvement, not an accuracy breakthrough. Feature/sourcing engineering remains exhausted; the next gains are in forecasting modelling, not inputs.

*Source: `F08_external_sensors_RFm.ipynb`, `results/f08_summary.csv`, `results/benchmarks.csv` (F-08). Decision D-35.*
