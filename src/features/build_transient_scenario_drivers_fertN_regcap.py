"""New fertiliser scenario level: `reg_cap` -- rate scaled so the TRUE (area-weighted, per-field)
typical-year N loading hits exactly the UK NVZ N-max for grassland, 300 kg N/ha/yr (gov.uk,
"Using nitrogen fertilisers in nitrate vulnerable zones" -- standard limit; the raised 340 kg/ha
limit for grassland cut >=3 times/yr is not used since NWFP fields are not routinely cut that
often). Additive alongside `historical`/`plus50pct_rate`/`plus50pct_freq` -- those are untouched.

**Why this needs its own module, not just a new `FERT_LEVELS` entry**: every existing FERT_LEVELS
entry is a single (rate_mult, freq_mult) tuple applied UNIFORMLY across all 3 towers (e.g. "+50%"
means +50% everywhere). `reg_cap` can't work that way -- 300 kg N/ha/yr is an ABSOLUTE target, and
each tower's real historical baseline sits at a very different distance from it, so each needs its
own multiplier (same reasoning as D-97/D-104's livestock `lit_ceil`/`own_max` levels, which are also
absolute-anchored and also needed per-tower multipliers).

**Why "true area-weighted" and not the base FERTN_TEMPLATE's own pooled n_events x mean_rate**:
verified directly against `Field_Event_Data_Format_1.csv` that T4 and T9 are each TWO independently
fertilised sub-fields (T4: NW005 Bottom Burrows 1.26ha + NW006 Burrows 6.49ha; T9: NW013 Dairy South
6.45ha + NW039 Dairy Corner 1.30ha), and `FERTN_TEMPLATE` pools both fields' events into one
catchment-level frequency/rate with NO area-weighting -- this overstates the true catchment-average
annual loading by roughly 2x at T4/T9 (naive pooled ~294-323 kg N/ha/yr vs. true area-weighted
~146-156 kg N/ha/yr; no real field, in any real year 2017-2024, ever exceeds ~245 kg N/ha/yr). T2 is
a single field (NW002), so pooled == true there, no correction needed. This module computes the
CORRECT area-weighted true typical-year total per tower, then sets reg_cap's rate_mult so that
target = pooled_rate x rate_mult, when correctly re-interpreted through the same area-weighting,
equals 300. Concretely: rate_mult = 300 / true_typical_total[tower].

Field areas (ha), primary source: NWFP_UG_Design_Develop.pdf, Appendix D (Hawkins/Griffith/Sint/
Harris 2023) -- same source already used to verify T4/T9 catchment totals (7.75 ha each) elsewhere
in this project.
"""
import numpy as np
import pandas as pd

import build_transient_scenario_drivers_practices as _p

# Captured at import time, BEFORE the caller monkey-patches p.synthetic_fertN_events -- delegating
# to p.synthetic_fertN_events directly (post-patch) would recurse into this same wrapper.
_ORIGINAL_SYNTHETIC_FERTN_EVENTS = _p.synthetic_fertN_events

FIELD_AREA_HA = {"NW002": 6.65, "NW005": 1.26, "NW006": 6.49, "NW013": 6.45, "NW039": 1.30}

REG_CAP_TARGET_KGN_HA = 300.0  # gov.uk NVZ N-max, standard grassland

# True area-weighted typical-year N loading (kg N/ha/yr), computed directly from
# Field_Event_Data_Format_1.csv via build_management_features.classify()/parsing (true-N events,
# N%>0 only), summed per field-year, area-weighted by FIELD_AREA_HA, averaged over each tower's
# real event-history year span. rate_mult = 300 / this value; freq unchanged (rate-only lever,
# matching the existing plus50pct_rate axis-isolation convention).
TRUE_TYPICAL_TOTAL_KGN_HA = {2: 155.76, 4: 145.50, 9: 27.53}
REG_CAP_RATE_MULT = {tower: REG_CAP_TARGET_KGN_HA / v for tower, v in TRUE_TYPICAL_TOTAL_KGN_HA.items()}


def synthetic_fertN_events_regcap(tower, years, level):
    """Same interface as build_transient_scenario_drivers_practices.synthetic_fertN_events(), but
    level=="reg_cap" uses REG_CAP_RATE_MULT's tower-specific multiplier instead of a shared
    FERT_LEVELS tuple. Every other level delegates unchanged."""
    if level != "reg_cap":
        return _ORIGINAL_SYNTHETIC_FERTN_EVENTS(tower, years, level)

    tpl = _p.FERTN_TEMPLATE[tower]
    n_events = max(1, round(tpl["n_events"]))
    doys = np.linspace(tpl["doy_min"], tpl["doy_max"], n_events)
    rate = tpl["mean_rate_kgN_ha"] * REG_CAP_RATE_MULT[tower]

    dates, rates = [], []
    for yr in years:
        for d in doys:
            dates.append(pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=int(round(d)) - 1))
            rates.append(rate)
    return dates, rates
