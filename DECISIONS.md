# DECISIONS.md
_Log every methodological choice here — one entry per decision, in order. Never delete entries; add a "superseded by D-XX" note if a decision is reversed._

---

## Format

| Field | Guidance |
|---|---|
| ID | Sequential: D-01, D-02, … |
| Date | YYYY-MM-DD |
| Area | Data / Features / Modelling / Evaluation / Infrastructure |
| Decision | What was decided (one sentence) |
| Rationale | Why — cite paper or constraint if applicable |
| Alternatives considered | What else was on the table |

---

## Log

### D-01 — 2026-06-12 — Infrastructure
**Decision:** Compile all annual CSV slices into single multi-year files (`data/Compiled/`) before analysis, rather than loading per-year files inline in each notebook.  
**Rationale:** Avoids repeating glob+concat boilerplate across notebooks; compiled files are deduplicated and have consistent datetime parsing applied once.  
**Alternatives considered:** Load Consolidated files directly in each notebook with a shared utility function.

### D-02 — 2026-06-12 — Data
**Decision:** Transform livestock weight and condition score data from wide format (one column per weighing date) to long format (`livestock_weight_long.csv`, `livestock_condition_score_long.csv`).  
**Rationale:** Long format is required for time-series analysis and joining against environmental covariates by date; wide format has hundreds of sparse date columns.  
**Alternatives considered:** Keep wide format and melt on-the-fly in downstream notebooks.

### D-03 — 2026-06-12 — Data
**Decision:** Keep livestock location data in wide format (rows = animals, columns = dates) rather than melting to long.  
**Rationale:** Location data is used primarily to derive field-occupancy counts (already in `Animal_location_counts_*.csv`); keeping it wide preserves the original structure for any individual-animal lookups.  
**Alternatives considered:** Melt to long — deferred until a clear downstream use case requires it.

---

_[Add new entries below this line]_
