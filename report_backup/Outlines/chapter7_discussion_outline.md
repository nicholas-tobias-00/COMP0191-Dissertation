# Chapter 7: Discussion — Finalized Outline

<!-- _Working reference, not chapter prose. All figures/numbers illustrative only —
will be independently validated before anything goes into the actual report._ -->

## 7.1 Main Findings

### Model ranking: Tabular > Ensemble > Deep Learning
- **Two separate, already-established mechanisms stacked, not one blanket
  explanation:**
  1. **Tabular > Ensemble — missingness-compensation.** 
     <!-- 60%+ of the underlying
     data is missing. Parameter-fitting methods (RF/XGB/LSTM/SARIMAX) are
     starved of clean signal to learn from. TabPFN/TabICL are pretrained on
     synthetic priors and don't need to fit parameters to this specific
     dataset at inference time — structurally insulated from a problem that
     hurts every other model family. This is why heavy missingness is a
     central thesis-wide theme, not just a gap-filling-chapter detail. -->
  2. **Ensemble > DL — Zeng et al.'s finding** 
     <!-- (already cited Ch2, grounds the
     "architectural superiority cannot be assumed" theme, substantiated by
     Ch5's own point-forecasting results): attention-based/sequential
     architectures lose to trees; permutation-invariant self-attention
     destroys temporal ordering. -->
<!-- - **Scope/wording, checked against Ch4's actual table:** TabICL leads at
  **two of three towers** in Gap-Filling (T9: RFm edges it narrowly) —
  state as "leads at 2/3 towers," not an unqualified "wins," to avoid a
  checkable contradiction. Forecasting's TabPFN win is unambiguous, no
  qualifier needed there.
- "Productionized/adopted" status is irrelevant to this ranking — this is a
  thesis, not a deployed system. Raw benchmark numbers are what count. -->

### FCH4 gap-filling and forecasting remains a genuinely difficult task
- Performance comparison to follow (numbers TBD/validate before writing).

## 7.2 Limitations

1. **Data quality and reliance on artificial (gap-filled) data** — root-cause
   link to forecasting's weaker performance. Tower 2 flagged as worst data of
   the three (full detail — land-use conversion, R²-incomparability,
   UQ-uncalibratability — already owned by Ch3/Ch4; this is a summary line,
   not a re-telling).
2. **Scenario projection as a pure inference task:**
   no future ground truth exists for validation, *and* no existing diagnostic
   substitutes for that absence — PICP/MPIW measure interval calibration, not
   trustworthiness under distribution shift; nothing in the current pipeline
   catches a smooth, confident, but structurally wrong extrapolation.
3. **CMIP6-vs-NWFP historical mismatch** — separate, explicit item, pulled up
   from Ch6. Independent of model choice.
4. **Compute-constrained UQ/scenario scope, reworded from the original
   draft:** UQ and scenario projection were built for the models suited to
   that architecture's specific requirements (one-shot inference, no fitting
   step, vectorized full-trajectory prediction) and within compute limits —
   **not** "only the best-performing models got UQ." Important distinction:
   TabICLv2 is not the dissertation's forecasting champion (that's still
   TabPFN), so the original phrasing would have directly contradicted Ch6's
   own stated note.
<!-- 5. **[DEFERRED — not yet added, explicit placeholder]** TabICLv2's untested
   extrapolation behavior as a limitation of the dissertation's *headline*
   scenario result (not a secondary footnote) — flagged extensively in Ch6,
   confirmed by two independent codebase audits, deliberately held back from
   this chapter for further discussion before being written in. -->

## 7.3 Future Work

1. **Livestock geolocation tracking** (e.g. via Google Earth Engine) relative
   to tower location — could explain spike occurrence directly, improving
   both forecasting and gap-filling performance.
2. **Rerun once more NWFP data is available** — the platform's data collection
   is ongoing/continuous, so this is a genuinely actionable, low-effort
   extension rather than a generic "more data would help" placeholder.

---
<!-- 
## Open items

- **Item 5 above (TabICLv2 extrapolation limitation) — deferred by explicit
  request, to be resumed later.** This is arguably the single largest open
  risk left in the dissertation at this point; flagging so it doesn't get
  lost given how much other ground this chapter covers. -->
