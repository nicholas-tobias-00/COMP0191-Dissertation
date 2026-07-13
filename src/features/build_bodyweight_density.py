"""Family (e), F-10 bonus: per-catchment estimated total liveweight density (kg/ha), from
last-observed-carried-forward per-animal location x weight joins. Additive, new file only --
does not modify any existing livestock/gap-filling script.

Feasibility confirmed empirically (2026-07, F-10 planning): the per-animal location files'
values genuinely include real NWFP field codes (e.g. "NW002"), not just shed/sale labels, so a
per-animal location -> CATCHMENT_FIELDS -> catchment join is possible (this was the open
feasibility question flagged when this stretch item was designed).

Method, explicitly bounded/exploratory:
  1. Melt each species' wide location file (Official tag x sparse date columns) to long form,
     forward-fill each animal's RAW location label (not yet mapped to a catchment) across its
     own observed date span only -- ffilling the raw label first (not the catchment) is the
     correct order: a shed/sale record must interrupt catchment membership, not be silently
     skipped, or an animal that later returns to a different catchment would be misattributed
     to its old one through the shed period.
  2. Map the ffilled daily location to a tower via FIELD_TO_TOWER; keep only rows resolving to
     one of this project's 3 EC towers (2, 4, 9).
  3. Bring forward each animal's last-known Weight_kg (merge_asof, backward) onto its own
     tower-resident days only.
  4. Sum resolved animal-day weights per (tower, date), divide by AREA[tower] -> kg/ha.

No growth-curve modeling, no cross-animal imputation. Reports the resolved animal-day fraction
explicitly (F-10 verification checklist item for family (e)) so the feature's own reliability is
visible, not asserted.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COMPILED = ROOT / "data" / "Compiled"
HOURLY = ROOT / "data" / "Hourly"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_management_features import CATCHMENT_FIELDS, TOWER_CATCHMENT  # noqa: E402

AREA = {2: 6.65, 4: 7.75, 9: 7.75}   # same AREA constants as gapfill_rfm.py (D-29)

FIELD_TO_TOWER = {}
for tw, cat in TOWER_CATCHMENT.items():
    for f in CATCHMENT_FIELDS[cat]:
        FIELD_TO_TOWER[f] = tw

SPECIES_FILES = {
    "Cattle": "Cattle_Location_Data_Format_1.csv",
    "Breeding Sheep": "Breeding_Sheep_Location_Data_Format_1.csv",
    "Lamb": "Lamb_Location_Data_Format_1.csv",
}


def melt_location(path):
    df = pd.read_csv(path, low_memory=False)
    date_cols = [c for c in df.columns if c != "Official tag"]
    long = df.melt(id_vars="Official tag", value_vars=date_cols,
                    var_name="date_str", value_name="location")
    long = long.dropna(subset=["location"])
    long["date"] = pd.to_datetime(long["date_str"], format="%d/%m/%Y", errors="coerce")
    long = long.dropna(subset=["date"])
    return long[["Official tag", "date", "location"]].rename(columns={"Official tag": "tag"})


def daily_location_per_animal(loc_long):
    """Forward-fill each animal's RAW location label across its own observed date span only."""
    frames = []
    for tag, sub in loc_long.sort_values("date").groupby("tag"):
        sub = sub.drop_duplicates("date", keep="last")
        idx = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
        loc = sub.set_index("date")["location"].reindex(idx).ffill()
        frames.append(pd.DataFrame({"tag": tag, "date": idx, "location": loc.values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["tag", "date", "location"])


def attach_weight(tower_days, weight_long, species):
    """merge_asof (backward, by=tag) to bring the last-known weight onto each tower-day."""
    w = weight_long[weight_long["Species"] == species].copy()
    w["Record_Date"] = pd.to_datetime(w["Record_Date"], errors="coerce")
    w = w.dropna(subset=["Record_Date", "Weight_kg"]).sort_values("Record_Date")
    w = w.rename(columns={"Official tag": "tag"})
    td = tower_days.sort_values("date")
    merged = pd.merge_asof(td, w[["tag", "Record_Date", "Weight_kg"]],
                            left_on="date", right_on="Record_Date", by="tag", direction="backward")
    return merged


def main():
    weight_long = pd.read_csv(COMPILED / "livestock_weight_long.csv", low_memory=False)

    all_animal_days = []
    for species, fname in SPECIES_FILES.items():
        loc_long = melt_location(COMPILED / fname)
        n_tags = loc_long["tag"].nunique()
        daily = daily_location_per_animal(loc_long)
        daily["tower"] = daily["location"].map(FIELD_TO_TOWER)
        tower_days = daily.dropna(subset=["tower"]).copy()
        tower_days["tower"] = tower_days["tower"].astype(int)
        merged = attach_weight(tower_days, weight_long, species)
        n_resolved = merged["Weight_kg"].notna().sum()
        n_total = len(merged)
        print(f"{species}: {n_tags} animals, {len(tower_days):,} tower-resident animal-days, "
              f"{n_resolved:,}/{n_total:,} ({100*n_resolved/max(n_total,1):.1f}%) have a resolvable weight")
        merged["species"] = species
        all_animal_days.append(merged[["tag", "date", "tower", "Weight_kg", "species"]])

    animal_days = pd.concat(all_animal_days, ignore_index=True)
    resolved = animal_days.dropna(subset=["Weight_kg"])

    agg = resolved.groupby(["date", "tower"])["Weight_kg"].sum().reset_index()
    agg["fx_total_liveweight_dens"] = agg["tower"].map(AREA)
    agg["fx_total_liveweight_dens"] = agg["Weight_kg"] / agg["fx_total_liveweight_dens"]
    agg = agg.rename(columns={"tower": "tower_check"})

    out = agg[["date", "tower_check", "fx_total_liveweight_dens"]].rename(
        columns={"date": "Datetime", "tower_check": "tower"})
    out = out.sort_values(["tower", "Datetime"]).reset_index(drop=True)

    dest = HOURLY / "bodyweight_density.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({out.shape[0]:,} rows x {out.shape[1]} cols)")
    for t in [2, 4, 9]:
        sub = out[out.tower == t]
        cov_days = sub["Datetime"].nunique()
        print(f"  Tower {t}: {cov_days:,} distinct days with a resolved liveweight-density value "
              f"({sub['fx_total_liveweight_dens'].mean():.1f} kg/ha mean when present)")


if __name__ == "__main__":
    main()
