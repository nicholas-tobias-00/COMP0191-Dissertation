# prompts/templates.md
_Copy-paste these at the start of a session. Replace [BRACKETED] placeholders._

---

## Template 1 — Standard session start

```
--- CONTEXT ---
[paste full contents of CONTEXT.md here]
--- END CONTEXT ---

Task for this session (one sentence):
[e.g. "Build the EDA notebook: distributions, missingness, and temporal coverage for measurements.csv and greenhouse.csv"]
```

---

## Template 2 — Replication session

```
--- CONTEXT ---
[paste full contents of CONTEXT.md here]
--- END CONTEXT ---

--- PAPER ---
[paste abstract or key methods section of the paper being replicated]
--- END PAPER ---

Replication ID: [R-XX]
Task: Implement [specific method/result] from the paper above using the NWFP dataset.
Key differences from paper's dataset to flag: [e.g. different sensor resolution, missing years, etc.]
```

---

## Template 3 — Debugging / code review

```
--- CONTEXT ---
[paste full contents of CONTEXT.md here]
--- END CONTEXT ---

Problem: [one sentence description]
File: [path:line]
Error / unexpected behaviour:
[paste traceback or describe what's wrong]
```

---

## Template 4 — End-of-session checklist

Before committing, run through:
- [ ] Updated `CONTEXT.md` → Current status, Replications table, and Next task fields
- [ ] Added any new decisions to `DECISIONS.md` (one D-XX entry per decision)
- [ ] Appended new rows to `results/benchmarks.csv` if a model was evaluated
- [ ] Created or updated `notebooks/03_gap_filling/R0X_results.md` if a replication ran
- [ ] Updated any prior `R0X_results.md` files if new findings recontextualize earlier results
- [ ] Committed with a meaningful message

## Template 5 — Replication result cross-check

After running a new replication, check:
- [ ] Are R² / RMSE values in the expected range for the paper and site type?
- [ ] Are training set sizes reasonable (not dramatically reduced by stricter dropna)?
- [ ] Do the gap scenario results show the expected pattern (e.g., ML > MDS for long gaps)?
- [ ] Are driver variables realistic for operational gap-filling? (flag LE/H/FC as co-failed — D-22)
- [ ] Is MDS fill rate reported? (should be ~100% with 4+ years of background data)
- [ ] Are MBE values near zero for MDS and plausibly signed for ML models?
