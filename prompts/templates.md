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
- [ ] Added any new decisions to `DECISIONS.md`
- [ ] Appended new rows to `results/benchmarks.csv` if a model was evaluated
- [ ] Committed with a meaningful message
