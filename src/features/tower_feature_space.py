"""Per-tower full feature-space column selection for the imputation-revisited phase (08).

Reuses the D-18 spatial-alignment rule (Tower N = Catchment N) to assemble every raw
measured column relevant to a tower from consolidated_hourly.csv: EC/met (``[Tower N]``),
catchment water-quality/soil (``[Catchment N]``), and livestock (``{species}_Catchment N``).
Unlike src/data/fco2_gapfill.py's curated 11-column driver_m set, this is the *complete*
column inventory for that tower -- deliberately unfiltered, since the goal (IMP-01) is a
full missingness survey, not a curated predictor set.
"""
from __future__ import annotations

import re

CATCHMENT_SUFFIX = {
    2: "Catchment 2",
    4: "Catchment 4 After  2013/08/13",
    9: "Catchment 9",
}

TARGET_COL = {n: f"FCH4_1_1_1 [Tower {n}]" for n in (2, 4, 9)}

# Ordered (first match wins) label -> regex, used to tag columns for plotting/analysis.
GROUP_PATTERNS = [
    ("EC_flux",         r"^(FCH4_|FC_|CH4_|CO2_|H2O_|H_1_1_1|LE_1_1_1|TAU_1_1_1)"),
    ("QC_flag",         r"SSITC_TEST"),
    ("EC_met",          r"^(WS_|WD_|TA_|RH_|PA_|VPD_|PPFD_|SWIN_|SWOUT_|LWIN_|LWOUT_|RN_|"
                        r"USTAR_|MO_LENGTH_|ZL_|T_SONIC|U_SIGMA_|V_SIGMA_|W_SIGMA_)"),
    ("EC_soil",         r"^(TS_|SHF_|SWC_)"),
    ("EC_fetch",        r"^FETCH_"),
    ("Catchment_water", r"^(Ammonia|Ammonium|Conductivity|Dissolved Oxygen|Flow |"
                        r"Fluorescent|Nitrite|Ortho Phosphorus|Total Phosphorus|"
                        r"Turbidity|Water Temperature|pH \(|Pump)"),
    ("Catchment_soil",  r"^(Soil Moisture|Soil Temperature)"),
    ("Precipitation",   r"^Precipitation"),
    ("Livestock",       r"^(cattle|sheep|lamb)_"),
]


def classify(col: str) -> str:
    """Tag a column with its variable-family group (used for plot coloring/legends)."""
    for label, pat in GROUP_PATTERNS:
        if re.search(pat, col):
            return label
    return "Other"


def get_tower_columns(all_columns, tower: int) -> dict:
    """Return the full feature-space column set for one tower.

    ``{'target': FCH4 column, 'columns': sorted column list, 'groups': {col: group_label}}``
    """
    catch = CATCHMENT_SUFFIX[tower]
    tower_tag = f"[Tower {tower}]"
    catch_tag = f"[{catch}]"

    cols = [c for c in all_columns if c.endswith(tower_tag)]
    cols += [c for c in all_columns if catch_tag in c]
    for species in ("cattle", "sheep", "lamb"):
        col = f"{species}_{catch}"
        if col in all_columns:
            cols.append(col)

    cols = sorted(set(cols))
    groups = {c: classify(c) for c in cols}
    return {"target": TARGET_COL[tower], "columns": cols, "groups": groups}
