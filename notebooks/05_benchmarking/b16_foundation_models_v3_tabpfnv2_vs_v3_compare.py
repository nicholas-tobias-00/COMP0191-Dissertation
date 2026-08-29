"""D-8x: compares TabPFN v2 (forced generic tabular checkpoint) against the standing v3 results
(TS-finetuned checkpoint, already on file), climatology-scored (D-80 convention) -- pure
arithmetic on already-saved MAE/n columns, no reruns.

Run from project root:  python notebooks/05_benchmarking/b16_foundation_models_v3_tabpfnv2_vs_v3_compare.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

clim = pd.read_csv(RESULTS / "_today_climatology_baseline.csv")[
    ["tower", "anchor_year", "bin", "MAE_climatology"]
]

sources = [
    (RESULTS / "b16_foundation_models_v3_tabpfnv2_summary.csv", None),
    (RESULTS / "b16_foundation_models_v3_tabpfnv2_gf_summary.csv", None),
    (RESULTS / "b16_foundation_models_v3_summary.csv", ["TabPFN"]),
    (RESULTS / "b16_foundation_models_v3_gf_summary.csv", ["TabPFN_gf"]),
]

all_rows = []
for path, model_filter in sources:
    df = pd.read_csv(path)
    df = df[df.target == "observed"]
    if model_filter is not None:
        df = df[df.model.isin(model_filter)]
    df = df.merge(clim, on=["tower", "anchor_year", "bin"], how="left")
    df = df.dropna(subset=["MAE", "MAE_climatology", "n"])
    df = df[(df.n > 0) & (df.MAE_climatology > 0)]
    df["MASE_climatology"] = df["MAE"] / df["MAE_climatology"]
    for (model, cfg), g in df.groupby(["model", "config"]):
        all_rows.append({
            "model": model, "config": cfg,
            "MASE_climatology": np.average(g["MASE_climatology"], weights=g["n"]),
            "R2": np.average(g["R2"], weights=g["n"]),
            "n_total": int(g["n"].sum()),
        })

out = pd.DataFrame(all_rows)
out.to_csv(RESULTS / "b16_foundation_models_v3_tabpfnv2_vs_v3_compare.csv", index=False)

pd.set_option("display.width", 120)
print("=== Full table, sorted by MASE (climatology) ===")
print(out.sort_values("MASE_climatology").round(3).to_string(index=False))

print("\n=== v2 vs v3 delta, matched by config (observed-context) ===")
v2 = out[out.model == "TabPFN_v2"].set_index("config")["MASE_climatology"]
v3 = out[out.model == "TabPFN"].set_index("config")["MASE_climatology"]
delta = pd.DataFrame({"MASE_v2": v2, "MASE_v3": v3, "delta_v2_minus_v3": v2 - v3}).dropna()
print(delta.round(3).sort_index().to_string())

print("\n=== v2_gf vs v3_gf delta, matched by config (gap-filled-context) ===")
v2gf = out[out.model == "TabPFN_v2_gf"].set_index("config")["MASE_climatology"]
v3gf = out[out.model == "TabPFN_gf"].set_index("config")["MASE_climatology"]
delta_gf = pd.DataFrame({"MASE_v2_gf": v2gf, "MASE_v3_gf": v3gf,
                          "delta_v2_minus_v3": v2gf - v3gf}).dropna()
print(delta_gf.round(3).sort_index().to_string())

print("\n=== Headline: BASE+species (current champion config) ===")
for model in ["TabPFN", "TabPFN_v2", "TabPFN_gf", "TabPFN_v2_gf"]:
    row = out[(out.model == model) & (out.config == "BASE+species")]
    if len(row):
        print(f"  {model:15s} MASE={row.MASE_climatology.values[0]:.3f}  R2={row.R2.values[0]:.3f}")

print(f"\nSaved {RESULTS / 'b16_foundation_models_v3_tabpfnv2_vs_v3_compare.csv'}")
