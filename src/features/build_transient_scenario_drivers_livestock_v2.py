"""S-05 livestock ladder redesign (supervisor feedback, 2026-08-13): replaces the original flat
1x/2x/3x multiplier (`build_transient_scenario_drivers_species.py`'s COMBOS in
`s05_trajectory_2050.py`) with absolute, externally-anchored stocking-density levels, after the
supervisor flagged that 3x (applied to the smoothed day-of-year climatology curve, not raw data)
is not obviously plausible given real catchment capacity.

**Quantified plausibility problem the redesign fixes** (checked directly this session): the OLD
3x scenario's SMOOTHED, full-year-sustained stocking density EXCEEDED each catchment's own
real historical INSTANTANEOUS peak (the single highest day ever recorded, any year) --
T4: 7.44 vs. real max 4.99 LSU/ha (+49%); T9: 7.74 vs. real max 5.65 (+37%). Conversely at T2,
3x UNDERSHOOT both benchmarks (2.13 LSU/ha vs. a 3.0 literature ceiling and T2's own 4.51 real
max) -- the same flat multiplier was simultaneously too extreme for T4/T9 and too mild for T2.

**New levels, absolute stocking-density targets (LSU/ha) instead of relative multipliers:**
- `half`      -- 0.5x the real climatology curve (extensification / reduced stocking).
- `baseline`  -- 1.0x, unmodified (status quo, matches the original design).
- `lit_ceil`  -- literature-grounded "best UK grassland conditions" ceiling, uniform 3.0 LSU/ha
  across towers. Sources: UK grassland stocking-rate literature describes >3 GLU/ha as reachable
  only under "very best growing conditions + near-optimal N fertiliser" (typical conventional
  range 1.5-2.5 GLU/ha); NVZ manure-N regulation caps at 170 kg N/ha/yr (up to 250 under grassland
  derogation), commonly corresponding to ~2 GLU/ha; an AHDB rotational-grazing case study achieved
  2.4 LSU/ha as a real high-end example. 3.0 LSU/ha sits just above all of these as a defensible
  "genuinely intensive, near the edge of what's been documented as achievable" ceiling.
- `own_max`   -- per-tower's own real historical INSTANTANEOUS peak stocking density, i.e. "as
  dense as this specific catchment has itself actually been, at least once": T2=4.511,
  T4=4.987, T9=5.652 LSU/ha (from `forecast_daily_v3.csv`'s real `fx_lsu_dens`, max over all
  years). The strongest, most farm-specific plausibility anchor -- "this catchment has done it
  before," no external assumption needed.

**Two families, same 4 levels, mirroring the original design's "cattle alone" vs "all species
together" split** (D-84's headline: cattle dominates far beyond its own LSU-weight share, so this
distinction remains the interesting axis -- not re-litigated here, just re-anchored):
- `all_species` -- cattle/sheep/lamb all scaled by the SAME solved multiplier `m`.
- `cattle_alone` -- only cattle scaled by `m`; sheep/lamb held at their own real baseline (1x).

For `lit_ceil`/`own_max`, `m` is solved (bisection on the climatology curves, not assumed) so the
resulting fx_lsu_dens curve's OWN peak hits the named target exactly -- this is why `m` differs
between families and across towers (e.g. T4 all_species lit_ceil needs only m=1.21, since baseline
is already close to 3.0 LSU/ha; T2 cattle_alone own_max needs m=6.95, since T2's baseline is far
below its own historical spike). `half`/`baseline` need no solving (m=0.5/1.0 uniformly -- scaling
DOWN or leaving unmodified never raises a plausibility question).

Reuses `build_climatology_base_species()`/`overlay_transient_species()`
(`build_transient_scenario_drivers_species.py`) unchanged -- this module only supplies the new
multiplier table, no new driver-construction logic.
"""
import sys

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src\features")

from build_transient_scenario_drivers_species import (  # noqa: E402
    build_climatology_base_species, overlay_transient_species, SPECIES,
)

TOWERS = [2, 4, 9]

# Real historical instantaneous max fx_lsu_dens (forecast_daily_v3.csv, all years) -- checked
# directly this session.
OWN_MAX = {2: 4.511, 4: 4.987, 9: 5.652}
LIT_CEILING = 3.0  # LSU/ha, uniform -- see module docstring for the literature grounding.

# Multipliers solved by bisection (see docstring) so the resulting fx_lsu_dens climatology curve's
# own peak hits the named absolute target exactly -- computed once this session, hardcoded here
# (cheap, deterministic, no reason to re-solve at every scenario run).
MULTIPLIERS = {
    # level: {tower: {"all_species": m, "cattle_alone": m}}
    "half":     {t: {"all_species": 0.5, "cattle_alone": 0.5} for t in TOWERS},
    "baseline": {t: {"all_species": 1.0, "cattle_alone": 1.0} for t in TOWERS},
    "lit_ceil": {
        2: {"all_species": 4.2074, "cattle_alone": 4.5855},
        4: {"all_species": 1.2116, "cattle_alone": 1.2225},
        9: {"all_species": 1.1628, "cattle_alone": 1.2113},
    },
    "own_max": {
        2: {"all_species": 6.3269, "cattle_alone": 6.9548},
        4: {"all_species": 2.0141, "cattle_alone": 2.0663},
        9: {"all_species": 2.1905, "cattle_alone": 2.5265},
    },
}

LEVELS_NON_BASELINE = ["half", "lit_ceil", "own_max"]
FAMILIES = ["all_species", "cattle_alone"]

# Combo name -> (level, family). "baseline" collapses both families into one row (m=1.0 either
# way), matching the original design's single "baseline_1x1x1x" combo.
COMBOS = {"baseline": ("baseline", "all_species")}
for lvl in LEVELS_NON_BASELINE:
    for fam in FAMILIES:
        COMBOS[f"{lvl}__{fam}"] = (lvl, fam)


def multiplier_for(tower, combo):
    """Returns (mult_cattle, mult_sheep, mult_lamb) for a given (tower, combo) -- combo is a key
    of COMBOS above."""
    level, family = COMBOS[combo]
    m = MULTIPLIERS[level][tower][family]
    if family == "all_species":
        return m, m, m
    return m, 1.0, 1.0  # cattle_alone: sheep/lamb held at real baseline


def overlay_combo(tower, T, year, combo):
    """One-call convenience: build the climatology base for (tower, year) and apply `combo`'s
    solved multipliers -- mirrors overlay_transient_species()'s signature minus the transient
    weather year (caller still needs to merge that in via overlay_transient_species itself)."""
    mult_cattle, mult_sheep, mult_lamb = multiplier_for(tower, combo)
    clim = build_climatology_base_species(tower, T, year)
    return clim, mult_cattle, mult_sheep, mult_lamb
