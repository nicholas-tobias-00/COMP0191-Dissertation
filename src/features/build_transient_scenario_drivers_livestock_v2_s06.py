"""S-06-only correction: `lit_ceil`'s target was 3.0 LSU/ha (D-97's own synthesized round number,
set just above several converging but indirect estimates -- a ScienceDirect Topics tertiary
aggregator, an AHDB case study, NVZ regulation). User found a better, directly-citable primary
source: UK Countryside Stewardship regulation (Annex 8, non-SDA land) states a hard ceiling of
**2.5 livestock units/ha** -- more precise and more authoritative than the prior derivation.

Scoped to S-06 only, per direct instruction -- S-05's original `lit_ceil` (target=3.0, already
computed and documented, D-97) is left untouched, not retroactively invalidated. This module
overrides ONLY the `lit_ceil` multipliers (both families, all 3 towers); `half`/`baseline`/
`own_max` are unaffected by this correction and reused unchanged from
`build_transient_scenario_drivers_livestock_v2.py`.

Solved by the same bisection method as the original (see that module's docstring). Notable finding
kept, not smoothed over: **Tower 9's own baseline climatology peak (2.58 LSU/ha) already exceeds
2.5** -- its solved multiplier is <1.0 (0.969 all-species / 0.959 cattle-alone), meaning `lit_ceil`
sits BELOW `baseline` for T9 specifically, not above it. T9 is, on real historical evidence, already
operating above the UK non-SDA regulatory stocking ceiling -- a genuine result, not a construction
artifact.
"""
LIT_CEILING_S06 = 2.5  # UK Countryside Stewardship, Annex 8, non-SDA land

LIT_CEIL_MULTIPLIERS_S06 = {
    2: {"all_species": 3.5062, "cattle_alone": 3.8016},
    4: {"all_species": 1.0097, "cattle_alone": 1.0102},
    9: {"all_species": 0.9690, "cattle_alone": 0.9593},
}


def multiplier_for_s06(tower, combo):
    """Same interface as build_transient_scenario_drivers_livestock_v2.multiplier_for(), but
    returns the corrected lit_ceil multipliers for S-06; delegates every other combo unchanged."""
    from build_transient_scenario_drivers_livestock_v2 import COMBOS, multiplier_for
    level, family = COMBOS[combo]
    if level == "lit_ceil":
        m = LIT_CEIL_MULTIPLIERS_S06[tower][family]
        if family == "all_species":
            return m, m, m
        return m, 1.0, 1.0
    return multiplier_for(tower, combo)
