"""Build hourly management-event features for the 04 feature-engineering experiment (D-28).

Parses NWFP Field Events (data/Compiled/Field_Event_Data_Format_1.csv), classifies each
operation into a CH4-relevant channel, and emits hourly **time-since-event recency**
features (exponential decay) at two spatial scopes:

  - site   : all NWFP fields (robust, always available)
  - tower-area : events on the tower's own management area
        Tower 9  = Catchment 9  = {NW013 Dairy South, NW039 Dairy Corner}  (confident, Appendix A)
        Tower 4  = Green farmlet = {NW005,6,9,16,17,45,46,47}              (farmlet-level inference)

Channels (e-folding tau, days):
  fertN  (14)  inorganic N fertiliser  -> recency + rate-weighted magnitude
  manure (30)  organic fertiliser / FYM
  cut    (21)  mow / silage / forage harvest
  lime   (90)  liming
  cultiv (30)  plough / drill / power-harrow / reseed

Recency(t) = exp(-(days since most recent event <= t) / tau); 0 before the first event.
Output: data/Hourly/management_features.csv (indexed to consolidated_hourly's timeline).

Spatial mapping (complete) from NWFP_UG_Design_Develop.pdf, Appendix D — see CATCHMENT_FIELDS.
Tower management area = its own catchment (D-18):
  Tower 4 = Catchment 4 = {NW005 Bottom Burrows, NW006 Burrows}
  Tower 9 = Catchment 9 = {NW013 Dairy South, NW039 Dairy Corner}
(Earlier draft over-broadly used the whole Green farmlet for Tower 4; corrected per Appendix D, D-28.)
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
HOURLY  = ROOT / "data" / "Hourly"
EVENTS  = ROOT / "data" / "Compiled" / "Field_Event_Data_Format_1.csv"

TAU = {"fertN": 14.0, "manure": 30.0, "cut": 21.0, "lime": 90.0, "cultiv": 30.0}

# Complete field -> catchment map (NWFP_UG_Design_Develop.pdf, Appendix D, current/post-2013).
# Catchment N = Tower N (D-18). Field codes match those in Field_Event_Data.
CATCHMENT_FIELDS = {
    1:  {"NW001", "NW038", "NW047"},          # Pecketsford / Little Pecketsford / Pecketsford whole
    2:  {"NW002"},                              # Great Field
    3:  {"NW003", "NW004"},                     # Poor Field / Ware Park
    4:  {"NW005", "NW006"},                     # Bottom Burrows / Burrows  (Top Burrows NW007 removed 2013)
    5:  {"NW008", "NW045", "NW046"},            # Orchard Dean North / South
    6:  {"NW009"},                              # Golden Rove
    7:  {"NW012"},                              # Lower Wyke Moor
    8:  {"NW010", "NW011"},                     # Higher / Middle Wyke Moor
    9:  {"NW013", "NW039"},                     # Dairy South / Dairy Corner
    10: {"NW015"},                              # Lower Wheaty
    11: {"NW014"},                              # Dairy East
    12: {"NW016"},                              # Dairy North
    13: {"NW017"},                              # Longlands South
    14: {"NW018"},                              # Longlands North
    15: {"NW019"},                              # Longlands East
}
# Tower management area = its own catchment (D-18, D-28 revised).
# Tower 2 = Catchment 2 = {NW002 Great Field} added for the F-05 pooled experiment.
TOWER_CATCHMENT = {2: 2, 4: 4, 9: 9}

# Total fenced area (ha) per catchment (NWFP_UG_Design_Develop.pdf, Appendix D, post-Aug 2013).
# Reference data for stocking-density features (D-29). Note Catchments 4 and 9 are both 7.75 ha.
CATCHMENT_AREA_HA = {
    1: 4.81, 2: 6.65, 3: 6.62, 4: 7.75, 5: 6.47, 6: 3.86, 7: 2.60, 8: 7.02,
    9: 7.75, 10: 1.82, 11: 1.76, 12: 1.78, 13: 1.75, 14: 1.72, 15: 1.54,
}

CUT_OPS = {"Mow", "Rowing up", "Forage harvest", "Trailers (silage)", "Top",
           "Baling (silage)", "Hay turning"}
CULTIV_OPS = {"Plough", "Power harrow", "Drill Seed", "Broadcast Seed",
              "Grass seeding (overseeding)", "Cambridge/Ring roll", "Sub soiling / ripping",
              "Chain harrow", "Cultivate / Level"}


def classify(row):
    """Return channel name or None for a field-event row."""
    app = str(row.get("Application", "")).lower()
    op  = str(row.get("Field_Operation", "")).strip()
    if "inorganic fertiliser" in app:
        return "fertN"
    if "organic fertiliser" in app:
        return "manure"
    if "liming" in app:
        return "lime"
    if op in CUT_OPS:
        return "cut"
    if op in CULTIV_OPS or op.startswith("Drilling"):
        return "cultiv"
    # "Apply Fertiliser" op with no Application label -> treat as inorganic N
    if op == "Apply Fertiliser":
        return "fertN"
    return None


def recency_series(index, event_times, event_mag, tau):
    """exp(-days_since_last_event/tau) and mag-weighted version, over `index`."""
    idx_ns = index.values.astype("datetime64[ns]")
    if len(event_times) == 0:
        z = np.zeros(len(index))
        return z, z.copy()
    ev = np.sort(np.array(event_times, dtype="datetime64[ns]"))
    order = np.argsort(np.array(event_times, dtype="datetime64[ns]"))
    mag = np.asarray(event_mag, dtype=float)[order]
    pos = np.searchsorted(ev, idx_ns, side="right") - 1   # index of most recent event <= t
    rec = np.zeros(len(index))
    recmag = np.zeros(len(index))
    valid = pos >= 0
    days = (idx_ns[valid] - ev[pos[valid]]) / np.timedelta64(1, "D")
    r = np.exp(-days / tau)
    rec[valid] = r
    recmag[valid] = r * np.nan_to_num(mag[pos[valid]], nan=0.0)
    return rec, recmag


def main():
    # --- timeline from consolidated_hourly ---
    base = pd.read_csv(HOURLY / "consolidated_hourly.csv", usecols=["Datetime"], low_memory=False)
    base["Datetime"] = pd.to_datetime(base["Datetime"], format="mixed")
    index = pd.DatetimeIndex(base["Datetime"])
    out = pd.DataFrame(index=index)
    out.index.name = "Datetime"

    # --- events ---
    fe = pd.read_csv(EVENTS, low_memory=False)
    fe["dt"] = pd.to_datetime(fe["Event_Date"], errors="coerce")
    fe["channel"] = fe.apply(classify, axis=1)
    fe["field"] = fe["Field"].astype(str).str.strip()
    fe["rate"] = pd.to_numeric(fe["Application_rate_per_ha"], errors="coerce")
    fe = fe.dropna(subset=["dt", "channel"])

    scopes = {"site": lambda f: True}
    for tw, cat in TOWER_CATCHMENT.items():
        fields = CATCHMENT_FIELDS[cat]
        scopes[f"t{tw}"] = (lambda fs: (lambda f: f in fs))(fields)   # own catchment only

    coverage = {}
    for scope, sel in scopes.items():
        sub = fe[fe["field"].map(sel)]
        coverage[scope] = len(sub)
        for ch, tau in TAU.items():
            evs = sub[sub["channel"] == ch]
            rec, recmag = recency_series(index, evs["dt"].tolist(),
                                         evs["rate"].tolist(), tau)
            out[f"mgmt_{scope}_{ch}_recency"] = np.round(rec, 5)
            if ch == "fertN":   # magnitude only meaningful for fertiliser N rate
                out[f"mgmt_{scope}_fertN_rate"] = np.round(recmag, 3)

    dest = HOURLY / "management_features.csv"
    out.to_csv(dest)
    print(f"Events used by scope (rows): {coverage}")
    print(f"Channels per scope: {list(TAU)} (+ fertN_rate)")
    print(f"Wrote {dest}  ({out.shape[0]:,} rows x {out.shape[1]} cols)")
    # quick non-zero coverage sanity
    nz = (out != 0).mean().sort_values(ascending=False)
    print("\nTop non-zero feature coverage:")
    print((100 * nz.head(8)).round(1).to_string())


if __name__ == "__main__":
    main()
