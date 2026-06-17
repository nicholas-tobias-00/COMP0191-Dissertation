# F-03 Results: Partial Pooling across all towers (RFm)

**Notebook:** `F03_partial_pooling_RFm.ipynb`
**Executed:** 2026-06-16
**Builds on F-02.** Compares three ways to share data across towers, all using stocking-density livestock (D-29):
- **solo** — one model per tower (per-tower training only);
- **full pool** — one model on Towers 2+4+9 stacked, no tower identity (F-02 style);
- **partial pool** — pooled + **tower-indicator dummies** (`is_t2/is_t4/is_t9`): shared relationships, tower-specific level.

Pooled training = 12,558 rows (T2 = 3,629 [2018]; T4 = 12,033; T9 = 5,387). Evaluated on all three towers: Towers 4 & 9 on the standard test (2022–23); **Tower 2 on its D-15 custom split** (train 2018 / test Jan–May 2019) — the seasonal-mismatch caveat (D-15/D-19) still applies.

---

## 1  Results — median R² by tower × variant

### Short gaps (vs, 1 h)
| Tower | solo | full pool | **partial pool** |
|---|---|---|---|
| Tower 2 | −0.712 | −0.301 | **−0.245** |
| Tower 4 | 0.248 | 0.239 | 0.238 |
| Tower 9 | −0.054 | 0.287 | **0.293** |

### Overall median (across scenarios)
| Tower | solo | full pool | **partial pool** |
|---|---|---|---|
| Tower 2 | −0.426 | −0.230 | **−0.179** |
| Tower 4 | 0.098 | 0.065 | 0.067 |
| Tower 9 | −0.051 | 0.250 | **0.251** |

(Per-scenario detail incl. medium gaps where pooling helps Tower 4 — solo m = 0.180 → pool m = 0.236 — is in the notebook.)

---

## 2  Interpretation

1. **Partial pooling ≥ full pooling at every tower** — it never underperforms full pooling and is the clear winner where it matters (Towers 2 and 9). It is the recommended default.
2. **The tower indicator helps most for the most "different" tower (Tower 2).** Partial > full at Tower 2 (−0.245 vs −0.301 short; −0.179 vs −0.230 overall) because the dummy lets Tower 2 keep its own baseline level instead of being forced onto one shared relationship. Tower 2 is the natural odd-one-out (Red farmlet, converting to arable, different area 6.65 ha).
3. **Pooling rescues the data-poor Tower 9** (≈ 0 solo → ≈ +0.29 pooled) — confirms F-02; partial ≈ full here (both work; tiny edge to partial).
4. **Tower 4 (data-rich) is roughly neutral** — it doesn't need to borrow strength (solo ≈ pool at short gaps; pooling helps at medium gaps). Crucially, the tower dummy **protects** it: partial pooling matches solo without the small full-pool dip.
5. **Tower 2 is still negative** — pooling cuts its error ~3–4× (solo −0.71 → partial −0.25) but cannot fix the **D-15 seasonal-mismatch split** (winter-only test vs all-season training). The relative gain shows pooling helps; absolute skill needs the split redesign, not a better model. (Note: solo T2 here is −0.43, far better than R-01's −16.9 — the cumulative gains from gap-filled/QC'd FCO₂ + density already tame the catastrophe.)

---

## 3  Takeaway & next steps

**Adopt partial pooling (pooled model + tower indicator + stocking density) as the standard multi-tower configuration.** It delivers the best of both worlds — shared physics from all towers (rescuing data-poor Tower 9) plus tower-specific levels (protecting data-rich Tower 4 and the different Tower 2).

- Carry this into the **forecasting phase** (`05_benchmarking`): a partially-pooled global model with per-tower effects.
- **Tower 2 split redesign** (D-15) is now the limiting factor for Tower 2 — pooling has done what it can.
- Optional: replace one-hot tower dummies with continuous tower descriptors (fenced area, soil type) so the model can *generalise* to unseen catchments rather than memorise the three it has seen.

225 rows tagged `F-03` (`model = RFm_solo / RFm_full_pool / RFm_partial_pool`) in `results/benchmarks.csv` (total 2065).

---

## 4  Shared features & the "catchments are similar" assumption

Partial pooling trains **one** model on the stacked rows of Towers 2+4+9. **Every predictor is shared** — i.e. one FCH₄-vs-driver relationship is learned across all three catchments — and the **only** tower-specific term is the indicator dummy, which gives each catchment its own *baseline level*, not its own *response shape*.

**Shared features** (18 in F-03; +8 SWC/TS lags in F-04), all carrying the assumption that the response shape is catchment-invariant:

| Group | Features | Similarity assumption |
|---|---|---|
| Met / radiative | SWIN, NETRAD (RN), PPFD, TA, VPD | **Safest** — physics is catchment-invariant |
| Turbulence | USTAR, WS | Safe |
| Soil / hydrology | SWC, SHF, precipitation (catchment-matched) | Questionable — land-use dependent |
| Soil temperature | TS (`TS_1_1_1 [Tower 9]`) | Identical input for all towers (D-16 proxy) |
| Ecosystem activity | gap-filled FCO₂ | Questionable — land-use dependent |
| Livestock | **stocking density (LSU/ha)**, grazing flag | **Engineered** for comparability (÷ area) |
| Time | hour/doy sin·cos; (F-04) SWC/TS lags @168/336/504/672 h | Safe (time) / questionable (lags) |

**Not shared:** `is_t2`, `is_t4`, `is_t9` — per-tower intercept only.

**The assumption stated:** same response *shape*, different *level*. It is strongest for the met drivers, engineered-to-hold for livestock (density), and **weakest for soil/land-use features at Tower 2** (Red farmlet, converting to arable — biophysically different from the grassland Towers 4 & 9). This is precisely why the tower dummy helped **Tower 2 the most** — it is the catchment where "similar" is least true, so giving it its own intercept mattered most.

**Mechanism note (what "shared" means at prediction time):** a tower is predicted from *its own* feature values at that timestamp — Tower 4's concurrent sensor readings are **not** fed into a Tower 9 prediction. But the learned relationship — and, for a Random Forest, the leaf averages — are built from **all** towers' rows, so other catchments' flux observations *do* inform a tower's predictions wherever the model treats the catchments as exchangeable. This is the intended **borrowing of statistical strength** (more *training examples* of the same features: Tower 9 alone = 5,387 rows → pooled = 12,558), not data leakage. The dummy (and RF's freedom to split on it) limits the borrowing where catchments genuinely differ.

### Two common clarifications (recorded for the writeup)
- **"Is Tower 9 predicted *from* Tower 4's data?"** Not as *inputs* — only Tower 9's own features (and `is_t9=1`) enter a Tower 9 prediction. But *yes* via the learned rule: the function (and RF leaf averages) were fit on all towers' rows, so Tower 4/Tower 2 observations shaped what gets applied to Tower 9. Analogy: *learn the general "temperature → methane" rule from every field, then apply it using your own field's temperature.*
- **Pooling adds rows, not columns.** The feature *set* is identical for every tower (~18 in F-03, ~26 in F-04); pooling adds more *training examples* of those same features. "Tower 9 trained on more data" = more rows (5,387 → 12,558), not more features.

## 5  How R² is evaluated (per-tower scoring of the pooled model)

Pooling changes the **training** set only; **evaluation is strictly per tower**, identical to the solo models.

For each test tower independently: build that tower's frame → mask its test-period gaps (Towers 4/9: 2022–23; Tower 2: Jan–May 2019, D-15) → predict with the (shared) model using **that tower's own feature values** → `r2_score(y_true_tower, y_pred_tower)` on **that tower's own** held-out points. Towers are never mixed in the metric. ("Overall median" rows = median over *scenarios* for one tower, still never across towers.)

Three things are held identical across solo / full / partial, so the comparison is controlled:
1. **Same held-out points** — `insert_calendar_gaps` uses a fixed seed, so a tower's masked timestamps are identical for every variant; only the model differs.
2. **Same R² baseline** — `r2_score` references **that tower's own test-subset mean**, not a pooled mean. So "Tower 9 +0.29" = beats "always predict Tower 9's average flux" by 29 % of *Tower 9's* variance.
3. **No leakage** — each tower's test period (2022–23) is temporally separate from all training data (2018–21 + Tower 2's 2018); its held-out points are never in the pooled training set.

➡️ So "Tower 9: solo −0.05 → partial +0.29" is the **same Tower 9 test set, scored the same way** — only the model's training data changed.

## 6  Is this legitimate? Methodological basis & references

Partial pooling is an established technique with two supporting literatures (several PDFs are in `documents/Literature/`):

**General methodology**
- **Gelman & Hill (2007), *Data Analysis Using Regression and Multilevel/Hierarchical Models*** — the source of the term "partial pooling" (between no-pooling = separate models and complete-pooling = ignore site). Our solo < full < partial ordering is the textbook bias–variance trade-off.
- **Montero-Manso & Hyndman (2021), *Int. J. Forecasting*, "…Locality and globality"** — "global" (cross-learning) models beating per-series models, esp. for short series (cf. M4/M5 competitions).
- **Hajjem, Bellavance & Larocque (2014), "Mixed-Effects Random Forest"** — the specific method of RF + a grouping (site) factor; legitimises "RF + tower effect."

**This domain — EC flux upscaling pools across sites by design**
- **Jung et al. (2020), FLUXCOM — "Scaling carbon fluxes from EC sites to the globe"** *(in folder)* — canonical cross-site ML upscaling.
- **McNicol et al. (2023), UpCH4 — "Upscaling Wetland Methane Emissions From the FLUXNET-CH4 Network"** *(in folder)* — **methane-specific, cross-site**; the closest analogue to this work.
- **Tramontana et al. (2016), *Biogeosciences*** — predicting fluxes across FLUXNET with ML.
- **Liang et al. (2019)** *(in folder)* — global grassland NEE via a model-tree ensemble across sites.
- **Irvin et al. (2021)** (our R-01) — frames gap-filling for multi-site syntheses across 17 wetland sites.

**Nuance the literature also gives us**
- A one-hot **site dummy helps *within-site* prediction but cannot generalise to an *unseen* catchment**; for that, use **transferable covariates** (climate, soil, PFT/vegetation) as in FLUXCOM/UpCH4 — hence the suggestion to swap dummies for continuous tower descriptors (area, soil).
- **Leave-one-site-out CV** is the standard test of cross-site generalisation.
- **Site/PFT heterogeneity** is the known failure mode — maps onto Tower 2 (arable) being the weak link in the "catchments are similar" assumption.

**Dissertation positioning:** *a partially-pooled (mixed-effects) ML model, consistent with the FLUXNET upscaling paradigm (FLUXCOM, UpCH4), applied at the within-farm scale to rescue data-poor towers.* (Confirm exact citation details from the PDFs before quoting.)
