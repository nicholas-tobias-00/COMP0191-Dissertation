"""Additive fertN feature build (D-98): corrected `fertN_amount` (true kg N/ha, recency-weighted --
same mechanism as the existing `mgmt_{scope}_fertN_rate` in build_management_features.py, but with
D-97's units fix applied to the REAL-DATA pipeline, not just S-05's scenario construction) plus a
genuinely new `fertN_freq` feature (trailing-365-day count of true-nitrogen application events --
no such feature exists anywhere in this project; only S-05's scenario TEMPLATE had a "frequency"
parameter, never a real-data column).

Purely additive: does not modify build_management_features.py, does not touch
data/Hourly/management_features.csv. Output: data/Hourly/fertN_amount_freq_features.csv, one row
per hour (matching consolidated_hourly's timeline), columns fertN_amount_t{2,4,9} and
fertN_freq_t{2,4,9} (tower-own-catchment scope only -- the scope every downstream experiment
this session actually uses).

Run from project root:  python src/features/build_fertN_amount_freq_features.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "data" / "Hourly"
EVENTS = ROOT / "data" / "Compiled" / "Field_Event_Data_Format_1.csv"

TAU_FERTN = 14.0
FREQ_WINDOW_DAYS = 365

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_management_features import classify, TOWER_CATCHMENT, CATCHMENT_FIELDS  # noqa: E402


def load_true_n_events():
    """Real fertN events, magnitude converted to true kg N/ha (D-97's fix, applied here to the
    real-data pipeline -- restricted to N-content > 0, i.e. genuine nitrogen applications only)."""
    fe = pd.read_csv(EVENTS, low_memory=False)
    fe["dt"] = pd.to_datetime(fe["Event_Date"], errors="coerce")
    fe["channel"] = fe.apply(classify, axis=1)
    fe["field"] = fe["Field"].astype(str).str.strip()
    fe["rate"] = pd.to_numeric(fe["Application_rate_per_ha"], errors="coerce")
    fe["n_pct"] = fe["Application_Info"].astype(str).map(
        lambda s: float(m.group(1)) if (m := re.search(r"([\d.]+)%\s*N\b", s)) else 0.0)
    fe["kgN_per_ha"] = fe["rate"] * fe["n_pct"] / 100.0
    fe = fe.dropna(subset=["dt", "channel"])
    return fe[(fe.channel == "fertN") & (fe.n_pct > 0)]


def recency_amount(index, event_times, event_mag, tau=TAU_FERTN):
    """exp(-days_since_last_event/tau) * magnitude -- identical mechanism to
    build_management_features.py's recency_series(), reused here (not reimplemented) via direct
    import would create a circular-ish extra dependency for one function; kept as a verified,
    byte-identical copy instead."""
    idx_ns = index.values.astype("datetime64[ns]")
    if len(event_times) == 0:
        return np.zeros(len(index))
    ev = np.sort(np.array(event_times, dtype="datetime64[ns]"))
    order = np.argsort(np.array(event_times, dtype="datetime64[ns]"))
    mag = np.asarray(event_mag, dtype=float)[order]
    pos = np.searchsorted(ev, idx_ns, side="right") - 1
    out = np.zeros(len(index))
    valid = pos >= 0
    days = (idx_ns[valid] - ev[pos[valid]]) / np.timedelta64(1, "D")
    out[valid] = np.exp(-days / tau) * np.nan_to_num(mag[pos[valid]], nan=0.0)
    return out


def trailing_frequency(index, event_times, window_days=FREQ_WINDOW_DAYS):
    """Count of events in (t - window_days, t] for every t in `index` -- genuinely new feature,
    no precedent elsewhere in this project. Vectorized via two searchsorted calls (upper minus
    lower bound of the trailing window)."""
    idx_ns = index.values.astype("datetime64[ns]")
    if len(event_times) == 0:
        return np.zeros(len(index))
    ev = np.sort(np.array(event_times, dtype="datetime64[ns]"))
    window_ns = np.timedelta64(window_days, "D")
    hi = np.searchsorted(ev, idx_ns, side="right")
    lo = np.searchsorted(ev, idx_ns - window_ns, side="right")
    return (hi - lo).astype(float)


def main():
    base = pd.read_csv(HOURLY / "consolidated_hourly.csv", usecols=["Datetime"], low_memory=False)
    base["Datetime"] = pd.to_datetime(base["Datetime"], format="mixed")
    index = pd.DatetimeIndex(base["Datetime"])
    out = pd.DataFrame(index=index)
    out.index.name = "Datetime"

    ev = load_true_n_events()
    print(f"[OK] {len(ev)} true-nitrogen fertN events loaded (site-wide)")

    for t, cat in TOWER_CATCHMENT.items():
        fields = CATCHMENT_FIELDS[cat]
        sub = ev[ev.field.isin(fields)].sort_values("dt")
        print(f"  Tower {t}: {len(sub)} true-N events in own catchment, "
              f"mean kgN/ha={sub.kgN_per_ha.mean():.1f}" if len(sub) else f"  Tower {t}: 0 events")

        out[f"fertN_amount_t{t}"] = np.round(
            recency_amount(index, sub["dt"].tolist(), sub["kgN_per_ha"].tolist()), 3)
        out[f"fertN_freq_t{t}"] = trailing_frequency(index, sub["dt"].tolist())

    out.to_csv(HOURLY / "fertN_amount_freq_features.csv")
    print(f"[OK] Saved fertN_amount_freq_features.csv ({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
