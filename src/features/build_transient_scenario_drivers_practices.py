"""S-05 follow-up: farming-practice scenario levers -- grazing timing (season-extension) and
fertilizer application schedule (rate/frequency) -- as management scenarios layered on top of
S-05's FX_A_SPECIES feature set, at BASELINE livestock (1x/1x/1x real climatology), matching the
project's "isolate one axis at a time" convention (same reasoning as S-05 holding two species at
1x while varying the third).

Unlike livestock density (a continuous quantity, scaled by a multiplier), both grazing_active/
days_since_grazing and mgmt_fertN_recency/rate are DERIVED features -- built from a presence
pattern (grazing) or discrete event list (fertilizer), not directly scalable. Both reuse the
EXACT functions the real historical columns are built from (days_since_grazing() from
build_forecasting_matrix_v2.py; recency_series() from build_management_features.py), applied to a
synthetic scenario input instead of real data -- not reimplemented.

GRAZING: real day-of-year species-density climatology (already built for FX_A_SPECIES) is
phase-shifted at the season edges to simulate earlier turnout / later housing, without inventing
tower-specific boundary-detection logic: for the first half of the year (DOY<=182), sample the
real climatology at doy+shift_days (pulling a later, more in-season value earlier); for the second
half, sample at doy-shift_days (pulling an earlier value later). For a real, single-humped annual
profile (rises then falls, ~0 at both ends), this extends the shoulder seasons while leaving deep
summer/winter days -- where nearby days already look similar -- largely unaffected, without
needing per-tower/species turnout-date detection.

FERTILIZER: real event history (Field_Event_Data_Format_1.csv, fertN channel, per tower's own
catchment) is summarized into a "typical year" template (event count, DOY range, mean rate) rather
than replaying one arbitrary specific year (avoids overfitting to that year's idiosyncrasies) --
see FERTN_TEMPLATE below, derived directly from the real per-tower event data (documented, not
guessed). Scenario levels scale the template's rate or event count; recency_series() (tau=14,
exp(-days_since_event/14)) is then run over pre-anchor REAL events + the synthetic future schedule.
"""
import sys

import numpy as np
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT + r"\src\features")
import models.recursive_rollout as rr  # noqa: E402
from build_forecasting_matrix_v2 import days_since_grazing  # noqa: E402
from build_management_features import recency_series, TAU  # noqa: E402
from build_transient_scenario_drivers_species import SPECIES  # noqa: E402

EVENTS_PATH = ROOT + r"\data\Compiled\Field_Event_Data_Format_1.csv"

# ---- Grazing timing scenario levels: shift (days) applied at each season boundary ----
GRAZING_SHIFT_LEVELS = {"historical": 0, "plus2wk": 14, "plus4wk": 28}

# ---- Fertilizer "typical year" template, derived directly from real per-tower event history
# (Field_Event_Data_Format_1.csv, fertN channel).
#
# UNITS FIX (supervisor feedback, 2026-08-13, scenario-scope only -- see D-97):
# `Application_rate_per_ha` is kg of PRODUCT per ha, not kg of nitrogen per ha -- and the
# `classify()` "fertN" channel (defined in build_management_features.py, used project-wide) lumps
# in ANY inorganic fertiliser event, including P/K/S/Mg-only products with 0% nitrogen content
# (e.g. Triple Superphosphate, Muriate of Potash -- 31% of all "fertN"-tagged events site-wide).
# The template below is recomputed using true kg N/ha = Application_rate_per_ha x (N-content-% /
# 100), parsed from `Application_Info` (e.g. "34.5% N"), and restricted to events with N% > 0 --
# so both `mean_rate_kgN_ha` (rate) and `n_events` (frequency) now mean what their names say.
# This materially changes the numbers, especially at T9 (most of its "fertN" events turn out to be
# non-nitrogen products): T2 3.3 true-N events/yr (was 5, all-product), T4 7.4 (was 8), T9 1.55
# (was 4). Scenario-scope fix only -- build_management_features.py's own "fertN" channel/rate
# feature (used by every prior experiment) is UNCHANGED; this template applies to S-05's fertilizer
# scenario axis specifically. See D-97 for the full derivation.
FERTN_TEMPLATE = {
    2: {"n_events": 3.3, "doy_min": 55, "doy_max": 184, "mean_rate_kgN_ha": 51.9},
    4: {"n_events": 7.4, "doy_min": 82, "doy_max": 234, "mean_rate_kgN_ha": 43.6},
    9: {"n_events": 1.55, "doy_min": 87, "doy_max": 204, "mean_rate_kgN_ha": 26.1},
}
# rate = kg true N/ha per application; frequency = applications/year -- independently scalable,
# isolating "how much N per event" from "how often N is applied" (the supervisor's exact question).
FERT_LEVELS = {"historical": (1.0, 1.0), "plus50pct_rate": (1.5, 1.0), "plus50pct_freq": (1.0, 1.5)}


def shifted_species_climatology_base(tower, T, year, shift_days):
    """Same 3-species day-of-year climatology as build_transient_scenario_drivers_species.py's
    build_climatology_base_species(), but phase-shifted at season edges by `shift_days` (0 = real,
    unmodified). Also derives fx_grazing_active/fx_days_since_grazing from the SAME shifted
    presence pattern (any species density > 0), reusing days_since_grazing() unchanged -- matches
    the real construction (build_forecasting_matrix_v2.py's daily_table(): ga = any-species-present,
    days_since_grazing(ga))."""
    target_dates = pd.date_range(f"{year}-01-01", periods=365, freq="D")
    doy = target_dates.dayofyear.values

    out = pd.DataFrame(index=target_dates)
    out["fx_DOY_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["fx_DOY_cos"] = np.cos(2 * np.pi * doy / 365.0)
    month = target_dates.month
    out["fx_is_growing"] = np.isin(month, [4, 5, 6, 7, 8, 9]).astype(float)
    out["fx_is_winter"] = np.isin(month, [12, 1, 2]).astype(float)

    dft = T[tower]
    species_present = pd.Series(False, index=target_dates)
    for s in SPECIES:
        hist = dft[f"fx_{s}_dens"].dropna()
        clim_365 = rr.doy_climatology(hist, pd.date_range(f"{year}-01-01", periods=365, freq="D"), window=7)
        if shift_days == 0:
            shifted = clim_365
        else:
            shift_doy = np.where(doy <= 182, np.minimum(doy + shift_days, 365), np.maximum(doy - shift_days, 1))
            lookup = pd.Series(clim_365, index=doy)
            shifted = lookup.reindex(shift_doy).values
        out[f"fx_{s}_dens_base"] = shifted
        species_present |= (shifted > 0.01)

    out["fx_grazing_active"] = species_present.astype(float).values
    out["fx_days_since_grazing"] = days_since_grazing(species_present).values

    return out


def overlay_transient_practices(clim_base, transient_year_df):
    """Baseline (1x/1x/1x) livestock overlay -- real CMIP6 weather + unmultiplied species density
    (no scaling axis here, unlike species_marginal_response; the scenario lever is grazing timing
    or fertilizer schedule, layered on top of otherwise-baseline livestock)."""
    out = clim_base.copy()
    out["fx_TA_min"] = transient_year_df["MIN"].values
    out["fx_TA_max"] = transient_year_df["MAX"].values
    out["fx_TA_mean"] = (out["fx_TA_min"] + out["fx_TA_max"]) / 2.0
    out["fx_PRECIP_sum"] = transient_year_df["RAIN"].values
    out["fx_SWIN_mean"] = transient_year_df["RAD"].values * 11.574  # RAD_MJ_TO_WM2, D-52

    for s in SPECIES:
        out[f"fx_{s}_dens"] = out[f"fx_{s}_dens_base"]
    out["fx_lsu_dens"] = 1.0 * out["fx_cattle_dens"] + 0.1 * out["fx_sheep_dens"] + 0.05 * out["fx_lamb_dens"]

    return out.drop(columns=[f"fx_{s}_dens_base" for s in SPECIES])


def load_real_fertN_events(tower):
    """Real pre-anchor fertN events for `tower`'s own catchment -- for recency_series()'s
    pre-anchor context, so decay carries correctly across the anchor boundary (same convention as
    every other AR/history boundary in this project). UNITS FIX (D-97, scenario-scope only):
    restricted to true-nitrogen events (N-content-% > 0, parsed from Application_Info) and
    magnitude converted to true kg N/ha -- previously included 0%-N (P/K/S/Mg-only) products at
    their raw product-mass rate, mislabeling them as nitrogen applications."""
    import re
    sys.path.insert(0, ROOT + r"\src\features")
    from build_management_features import classify, TOWER_CATCHMENT, CATCHMENT_FIELDS

    fe = pd.read_csv(EVENTS_PATH, low_memory=False)
    fe["dt"] = pd.to_datetime(fe["Event_Date"], errors="coerce")
    fe["channel"] = fe.apply(classify, axis=1)
    fe["field"] = fe["Field"].astype(str).str.strip()
    fe["rate"] = pd.to_numeric(fe["Application_rate_per_ha"], errors="coerce")
    fe["n_pct"] = fe["Application_Info"].astype(str).map(
        lambda s: float(m.group(1)) if (m := re.search(r"([\d.]+)%\s*N\b", s)) else 0.0)
    fe["kgN_per_ha"] = fe["rate"] * fe["n_pct"] / 100.0
    fe = fe.dropna(subset=["dt", "channel"])

    fields = CATCHMENT_FIELDS[TOWER_CATCHMENT[tower]]
    sub = fe[(fe.field.isin(fields)) & (fe.channel == "fertN") & (fe.n_pct > 0)].sort_values("dt")
    return sub["dt"].tolist(), sub["kgN_per_ha"].tolist()


def synthetic_fertN_events(tower, years, level):
    """Synthetic future fertN event schedule for `years`, per FERTN_TEMPLATE's typical-year
    pattern, scaled by `level`'s (rate_mult, freq_mult). Events evenly spaced within
    [doy_min, doy_max] each year -- same schedule repeats every scenario year (day-of-year
    climatology's own convention, applied to a discrete event list instead of a continuous
    quantity)."""
    tpl = FERTN_TEMPLATE[tower]
    rate_mult, freq_mult = FERT_LEVELS[level]
    n_events = max(1, round(tpl["n_events"] * freq_mult))
    doys = np.linspace(tpl["doy_min"], tpl["doy_max"], n_events)
    rate = tpl["mean_rate_kgN_ha"] * rate_mult

    dates, rates = [], []
    for yr in years:
        for d in doys:
            dates.append(pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=int(round(d)) - 1))
            rates.append(rate)
    return dates, rates


def fertN_recency_frame(tower, target_dates, level, anchor):
    """fx_mgmt_fertN_recency/_rate over target_dates, for the given scenario level -- real
    pre-anchor events (unmodified) + synthetic future events for target_dates' years, run through
    the EXACT recency_series() the real columns use (tau=14, exp decay, rate-weighted)."""
    real_dates, real_rates = load_real_fertN_events(tower)
    real_dates = [d for d in real_dates if d <= anchor]
    real_rates = real_rates[:len(real_dates)]

    years = sorted(set(d.year for d in target_dates))
    syn_dates, syn_rates = synthetic_fertN_events(tower, years, level)

    all_dates = real_dates + syn_dates
    all_rates = real_rates + syn_rates
    rec, recmag = recency_series(pd.DatetimeIndex(target_dates), all_dates, all_rates, TAU["fertN"])
    return pd.DataFrame({"fx_mgmt_fertN_recency": rec, "fx_mgmt_fertN_rate": recmag}, index=target_dates)
