# F-09b — outlier-correction technique comparison: winsorization vs Hampel filter vs hard truncation (D-51)

**Script:** ad-hoc, scratchpad only (`f09b_outlier_technique_check.py`, not committed — same precedent as F-09a).
**Results:** `results/f09b_series_diagnostics.csv` (32 rows), `results/f09b_gapfill_summary.csv` (80 rows).

Follow-up to D-50, which found two new confirmed met-driver contamination sources (`WS_0_0_1`, `TA_0_0_1`
at Tower 2) and recommended the same hard-truncation fix already used for USTAR/VPD (D-48) — staged but not
applied. Before applying that fix, this experiment asks the user's question directly: are winsorization or
a dedicated outlier-correction algorithm (Hampel filter) better alternatives to hard truncation? Uses the
two already-confirmed contamination cases as real ground truth — no synthetic outliers needed.

## 0. Scoping finding (verified before running the comparison — corrects D-50)

D-50's audit ran against `data/Hourly/consolidated_hourly.csv` (the raw EC-tower file), where the WS/TA
contamination is real and severe. But `src/data/build_sms_met_dataset.py` (D-35/F-08) **swaps**
`WS_0_0_1`/`TA_0_0_1` to external **Site**-station readings for all three towers before any gap-filling
happens (`ext_driver_map()`: `SITE_TA`/`SITE_WS` mapped in, comments confirm "holds Site air temp/wind after
swap"). Directly verified by reading `consolidated_hourly_SMS_MET.csv` (the actual EXT dataset production
consumes): WS/TA are **already clean** there — max 32.2 m/s, range −7.24…32.1°C, identical across all 3
towers (one shared Site station) — well inside the D-50-proposed bounds. **The D-50-confirmed contamination
does not currently reach the production forecasting/gap-filling pipeline** (unlike USTAR/VPD/PPFD/RN/SHF,
which stay EC-sourced with no external twin — exactly why D-48's fix mattered there). This experiment
therefore evaluates each correction technique on the **EC-tower-sourced** WS/TA (where the contamination
genuinely lives), with production (Site-sourced) as a reference/ceiling row, not one of the techniques
under test. **This corrects D-50**, which did not check whether its flagged contamination actually reaches
the pipeline that's used — it's a real bug in the raw EC data, but currently a non-issue for production.

## 1. Configs tested

| Config | Technique |
|---|---|
| A_baseline | Uncorrected EC-sourced WS/TA (reproduces the D-50-documented contamination) |
| B_truncate | Hard truncation — D-50's staged recommendation (`WS_0_0_1:(0,40)`, `TA_0_0_1:(-20,40)`, same mechanism as USTAR/VPD, D-48) |
| C_winsorize | Percentile capping — WS upper-only at 99th pct; TA (Tower 2) lower-only at 10th pct (widened from a naive 1st pct given 6.8% contamination) |
| D_hampel | Rolling MAD despiking, `k=3.0`, ±12h window (25h), replaces flagged points with the local rolling median |
| D2_hampel_wide_T2 | Same as D but ±84h window (169h), Tower 2 TA only — a stress test for long contiguous fault blocks |
| E_production | Reference/ceiling — current production (Site-sourced WS/TA), not a technique under test |

Parameters were chosen from the actual contamination statistics (`results/d50_met_outlier_audit.csv` plus a
fresh contiguous-run-length analysis): WS contamination is short/scattered (median run 1h, max 11h — a 25h
Hampel window should comfortably span it); TA-at-Tower-2 contamination is a sensor stuck for extended
stretches (median run 11.5h, mean 159.6h, **max ~1805h / ~75 days**, spanning both Jan and Aug — a fixed
fault value, not weather) — 35% of runs ≥25h, 15% ≥72h, predicting Hampel would only partially correct it
even with a much wider window.

## 2. Result 1 — does each technique actually clean the series?

| Variable | Tower | A (uncorrected) max/min | B (truncate) | C (winsorize) | D (Hampel 25h) | D2 (Hampel 169h) |
|---|---|---|---|---|---|---|
| WS_0_0_1 | 2 | max 1370.6 | max 39.8 (bound-compliant) | max 10.6 (99th pct) | max **135.8** (partial) | — |
| WS_0_0_1 | 4 | max 1277.2 | max 39.7 | max 8.3 | max **114.7** (partial) | — |
| WS_0_0_1 | 9 | max 703.5 | max 39.6 | max 9.6 | max **140.0** (partial) | — |
| TA_0_0_1 | 2 | min −39.61 | min −19.7 (bound-compliant) | min 2.1 (10th pct, safely >−30) | min **−39.61 (unchanged)** | min **−39.61 (unchanged)** |

**Hard truncation (B) and winsorization (C) both fully resolve the contamination**, by construction — every
value is guaranteed within the target range. **Hampel (D/D2) is a genuine partial failure, exactly as
predicted**: WS improves dramatically (1370→~135 m/s, an ~90% reduction) but does not fully clear the
physical bound — some residual spike magnitude survives the local-median replacement (max observed
contiguous WS-spike run was only 11h, comfortably inside the 25h window, so the residual is somewhat
surprising and was not investigated further — candidate cause: closely-spaced separate short spike clusters
within the same window may inflate the local MAD estimate, raising the effective detection threshold; not
confirmed). **TA at Tower 2 is essentially untouched by Hampel at either window size** (−39.607°C before and
after, both 25h and 169h) — the fault block is long and internally consistent enough that the rolling local
median/MAD become themselves dominated by the fault value, leaving the filter with no clean local reference
to compare against. This confirms the pre-registered prediction: **Hampel-style local despiking is not
suited to a long, internally-consistent stuck-sensor fault** (as opposed to short transient spikes, where it
works better, if still not perfectly).

## 3. Result 2 — does it matter downstream? (gap-filling R², median across 5 gap-scenarios)

| Config | Tower 2 | Tower 4 | Tower 9 |
|---|---|---|---|
| A_baseline (uncorrected) | 0.567 | 0.408 | 0.418 |
| B_truncate | 0.568 | 0.408 | 0.417 |
| C_winsorize | 0.567 | 0.408 | 0.418 |
| D_hampel | 0.569 | 0.408 | 0.416 |
| D2_hampel_wide_T2 | 0.569 | — | — |
| E_production (reference) | 0.574 | 0.402 | 0.418 |

**Null result — none of the correction techniques move downstream gap-filling R² outside noise, and
neither does leaving WS/TA completely uncorrected.** Every config, at every tower and every one of the 5
gap-length scenarios (full breakdown in `f09b_gapfill_summary.csv`), sits within ~0.01 R² of every other
config. Even `A_baseline` — WS up to 1370 m/s, TA stuck at −39.6°C for weeks at a time — scores
indistinguishable from every corrected variant and from clean production data.

**Caveat on the strength of this "null result" claim**: with only `N_REPS=2`, this is an impression from
close numbers, not a statistically rigorous equivalence test — there is no real variance estimate behind
"within noise." A tighter version of this experiment would rerun at `N_REPS=5` (matching F-08's original
design) to put an actual confidence bound on the claim.

**This is a genuinely different outcome from D-48's USTAR/VPD fix**, which visibly moved AR-feature means
and fixed clearly spurious multi-hundred-nmol spikes. Two candidate explanations (not distinguished further
in this bounded experiment):

1. **Contamination severity/leverage is smaller.** D-48's own SHAP attribution found USTAR contributed
   +244 nmol of a +380 nmol spurious spike vs WS's +107 nmol — USTAR was the dominant driver even in the
   case where WS also mattered. With USTAR/VPD already fixed (active in every config here, since the base
   `MET_PLAUS` dict is never removed, only added to), WS/TA's incremental contamination may simply carry
   much less predictive leverage on FCH4 than USTAR did.
2. **This evaluation harness may not exercise the failure mode that made USTAR/VPD damaging.** D-48's worst
   damage came from `mdc_gapfill`'s *last-resort median/mean fallback* — which only fires during extended
   real-data blackouts with nothing nearby to interpolate from (e.g. Tower 2's entire Jun–Dec-2019 stretch).
   This experiment's calendar-gap CV masks chunks up to 288h (12 days) *within* a tower's otherwise-available
   domain — shorter and less pathological than a genuine multi-month blackout, so the fallback tier where
   contamination does the most damage may rarely get exercised here. **This is a real caveat on this
   experiment's null result, not just on WS/TA specifically** — a clean/positive result under this harness
   does not fully rule out a real-world impact during an actual extended blackout. A direct follow-up test
   (mask one gap matching Tower 2's actual multi-month blackout shape, rather than the standard calendar
   scenarios) would distinguish this from explanation 1 — not done here.

## Recommendation

- **No change to production** — WS/TA in the actual pipeline are already clean (Site-sourced, §0), so there
  is no pending fix to apply there regardless of this result.
- **If the EC-tower-sourced WS/TA are ever used directly** (e.g. a future EC-only variant, or if the
  Site-station swap is ever reverted) — **prefer hard truncation (B) or winsorization (C) over a Hampel
  filter (D)** for this project's contamination profile: both fully resolve the known issue by construction,
  while Hampel only partially corrects WS and does not correct TA's long-duration fault blocks at all. Of
  the two full-resolution options, truncation is simpler and already has an established, working precedent
  in this codebase (D-48); winsorization's one real advantage is not clipping to a hard physical bound but to
  the data's own distribution, which matters more for variables without an obvious physical ceiling than for
  WS/TA (which do have clear physical bounds) — no strong reason to prefer it here.
- **D-50's flagged fix is downgraded in urgency** — it remains worth doing eventually for the raw EC-tower
  data's own correctness/honesty (and for any future EC-only variant), but is **not** a pending production
  bug and should not be prioritized over other work.
- **Caveat for any future contamination audit**: the calendar-gap CV methodology used throughout the
  gap-filling phase (F-07 onward) may be a poor tool for detecting the impact of contamination whose damage
  concentrates in genuine long real-data blackouts — a clean/positive result under this harness does not
  fully rule out a real-world impact during an actual extended blackout (as D-48's USTAR/VPD case was).
- **If more rigor is wanted later**: rerun at `N_REPS=5` for a real confidence bound, and/or add a
  blackout-shaped masked gap to directly test explanation 2 above.

## Files / scope
Fully additive, read-only w.r.t. production files — no changes to `src/data/reddyproc_pipeline.py`,
`src/models/gapfill_rfm.py`, `src/data/build_sms_met_dataset.py`, or any `data/Hourly/*.csv` file. All reuse
via unmodified imports (`plausibility_filter`, `mdc_gapfill`, `MET_PLAUS`, `load_ext`, `frame`, `fit`), plus
a runtime-only, self-restoring monkeypatch of `MET_PLAUS` for config B. Config E's Tower-2 row was verified
to match `results/f09a_summary.csv` exactly (regression check on the copied CV harness) before trusting
configs A–D's numbers.

*Source: `f09b_outlier_technique_check.py` (scratchpad), `results/f09b_series_diagnostics.csv`,
`results/f09b_gapfill_summary.csv`. Decision D-51. Corrects a scoping gap in D-50.*
