"""I-03 figures: overall ranking bar chart + per-tower comparison, from the already-saved
i03_tabpfn_species_importance*.csv outputs. No new model calls."""
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = rf"{ROOT}\results"
FIGDIR = rf"{ROOT}\results\figures"
os.makedirs(FIGDIR, exist_ok=True)

overall = pd.read_csv(f"{RESULTS}/i03_tabpfn_species_importance_ranked.csv", index_col=0)
by_tower = pd.read_csv(f"{RESULTS}/i03_tabpfn_species_importance_by_tower.csv")

SPECIES = {"fx_lsu_dens", "fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens",
           "fx_grazing_active", "fx_total_liveweight_dens", "fx_days_since_grazing"}

# ---------------------------------------------------------------- Figure 1: overall top-15
top15 = overall.head(15).iloc[::-1]  # reverse for horizontal bar (largest at top)
colors = ["#c0392b" if f in SPECIES else "#7f8c8d" for f in top15.index]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top15.index, top15["mean_importance"], color=colors)
ax.set_xlabel("Mean permutation importance (|Δ mean forecast|, nmol m⁻² s⁻¹)")
ax.set_title("I-03: TabPFN+species champion — feature importance\n(pooled across 3 towers × 5 anchors)")
red_patch = plt.matplotlib.patches.Patch(color="#c0392b", label="Livestock-related")
gray_patch = plt.matplotlib.patches.Patch(color="#7f8c8d", label="Other")
ax.legend(handles=[red_patch, gray_patch], loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/i03_overall_ranking.png", dpi=150)
print("Saved i03_overall_ranking.png")

# ---------------------------------------------------------------- Figure 2: per-tower top-8
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=False)
for ax, tower in zip(axes, [2, 4, 9]):
    sub = (by_tower[by_tower.eval_tower == tower]
           .sort_values("importance", ascending=False).head(8).iloc[::-1])
    colors_t = ["#c0392b" if f in SPECIES else "#7f8c8d" for f in sub["feature"]]
    ax.barh(sub["feature"], sub["importance"], color=colors_t)
    ax.set_title(f"Tower {tower}")
    ax.set_xlabel("Mean importance")
fig.suptitle("Per-tower top-8 features", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/i03_per_tower_top8.png", dpi=150, bbox_inches="tight")
print("Saved i03_per_tower_top8.png")

# ---------------------------------------------------------------- Figure 3: species split zoom
species_only = overall.loc[["fx_lsu_dens", "fx_cattle_dens", "fx_sheep_dens", "fx_lamb_dens",
                             "fx_total_liveweight_dens"]].iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = ["#2c3e50", "#c0392b", "#f1c40f", "#e67e22", "#2c3e50"]
ax.barh(species_only.index, species_only["mean_importance"], color=bar_colors[::-1])
ax.set_xlabel("Mean permutation importance")
ax.set_title("I-03: livestock-density features — cattle drives the species-split gain,\nnot sheep/lamb")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/i03_species_split.png", dpi=150)
print("Saved i03_species_split.png")
