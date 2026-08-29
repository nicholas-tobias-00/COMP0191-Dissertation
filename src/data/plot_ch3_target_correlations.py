"""Generate Chapter 3 target-correlation results from the gap-filling frame.

Correlations use the exact 30-predictor feature construction shared by the RFm
and TabICL gap-filling experiments. The target is observed, QC-valid FCH4; it is
never replaced with gap-filled values for this diagnostic.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))

from gapfill_rfm import TOWERS, feat_list, frame, load_ext  # noqa: E402


START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2024-01-01")
FIGURE = ROOT / "report" / "Figures" / "ch3_fch4_target_correlations.png"
FULL_FIGURE = (
    ROOT / "report" / "Figures" / "ch3_fch4_target_correlations_full.png"
)
RESULTS = ROOT / "results" / "ch3_fch4_target_correlations.csv"
TOP_K = 5

LABELS = {
    "SWIN_1_1_1": "Incoming shortwave radiation",
    "TA_0_0_1": "Air temperature",
    "VPD_0_0_1": "Vapour pressure deficit",
    "PPFD_1_1_1": "Photosynthetic photon flux",
    "RN_1_1_1": "Net radiation",
    "WS_0_0_1": "Wind speed",
    "USTAR_0_0_1": "Friction velocity",
    "SHF_1_1_1": "Soil heat flux",
    "Precipitation (mm)": "Precipitation",
    "Soil Temperature @ 15cm Depth (oC)": "Soil temperature (15 cm)",
    "Soil Moisture @ 10cm Depth (%)": "Soil moisture (10 cm)",
    "fc": r"FCO$_2$",
    "_hs": "Hour sine",
    "_hc": "Hour cosine",
    "_ds": "Day-of-year sine",
    "_dc": "Day-of-year cosine",
    "lsu_dens": "Livestock density",
    "graze": "Grazing active",
    "swc_l168": "Soil moisture lag (7 d)",
    "swc_l336": "Soil moisture lag (14 d)",
    "swc_l504": "Soil moisture lag (21 d)",
    "swc_l672": "Soil moisture lag (28 d)",
    "ts_l168": "Soil temperature lag (7 d)",
    "ts_l336": "Soil temperature lag (14 d)",
    "ts_l504": "Soil temperature lag (21 d)",
    "ts_l672": "Soil temperature lag (28 d)",
    "mgmt_cut": "Cutting recency",
    "mgmt_manure": "Manure recency",
    "gpp": "Gross primary productivity",
    "reco": "Ecosystem respiration",
}


def draw_heatmap(matrix: pd.DataFrame, output: Path, title: str, compact: bool) -> None:
    """Draw one consistently scaled tower-correlation heatmap."""
    figsize = (7.6, 4.8) if compact else (8.0, 8.5)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-0.4,
        vmax=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8 if compact else 7.5},
        linewidths=0.45,
        linecolor="white",
        cbar_kws={
            "label": r"Spearman correlation ($\rho$)",
            "shrink": 0.82 if compact else 0.78,
        },
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("EC tower")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    if compact:
        fig.subplots_adjust(left=0.37, right=0.93, top=0.91, bottom=0.10)
    else:
        fig.subplots_adjust(left=0.38, right=0.93, top=0.95, bottom=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_ext()
    features = feat_list()
    rows = []

    for tower in TOWERS:
        analysis = frame(tower, pooled=False, d=data)
        analysis = analysis.loc[
            analysis.index.to_series().between(START, END, inclusive="left")
        ]
        subset = analysis[["target", *features]]
        pearson = subset.corr(method="pearson", min_periods=100)["target"]
        spearman = subset.corr(method="spearman", min_periods=100)["target"]
        pair_counts = subset[features].notna().multiply(
            subset["target"].notna(), axis=0
        ).sum()
        for feature in features:
            rows.append(
                {
                    "tower": tower,
                    "feature": feature,
                    "label": LABELS[feature],
                    "n_pairs": int(pair_counts[feature]),
                    "pearson_r": pearson[feature],
                    "spearman_rho": spearman[feature],
                }
            )

    results = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS, index=False)

    full_matrix = (
        results.pivot(index="feature", columns="tower", values="spearman_rho")
        .reindex(features)
        .rename(index=LABELS, columns={tower: f"T{tower}" for tower in TOWERS})
    )

    # Select each tower's five largest absolute correlations, then display the
    # union. Every selected predictor retains all three tower coefficients so
    # that the compact view still supports direct cross-tower comparison.
    selected_features = set()
    for tower in TOWERS:
        tower_results = results.loc[results["tower"] == tower]
        selected_features.update(
            tower_results.loc[
                tower_results["spearman_rho"].abs().nlargest(TOP_K).index,
                "feature",
            ]
        )
    selected_order = (
        results.loc[results["feature"].isin(selected_features)]
        .assign(abs_rho=lambda x: x["spearman_rho"].abs())
        .groupby("feature")["abs_rho"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    compact_matrix = full_matrix.reindex(
        [LABELS[feature] for feature in selected_order]
    )

    sns.set_theme(style="white", font_scale=0.82)
    draw_heatmap(
        compact_matrix,
        FIGURE,
        rf"Union of {TOP_K} strongest absolute FCH$_4$ correlations per tower",
        compact=True,
    )
    draw_heatmap(
        full_matrix,
        FULL_FIGURE,
        r"All gap-filling predictors correlated with observed QC-valid FCH$_4$",
        compact=False,
    )


if __name__ == "__main__":
    main()
