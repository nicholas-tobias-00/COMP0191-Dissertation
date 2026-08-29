"""Create the Chapter 3 TICA/UMAP figure from the gap-filling pipeline.

TICA is refitted using the D5 pipeline's exact environmental feature set,
tower-domain segmentation, pooled scaling and 24-hour lag. UMAP coordinates
and local-agreement results are read from the saved D5 outputs so the report
visualises the same fitted embedding evaluated by the experiment.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deeptime.decomposition import TICA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from gap_cv import dom_mask  # noqa: E402
from gapfill_rfm import TOWERS, feat_list, frame, load_ext  # noqa: E402


D5_DATA = ROOT / "notebooks" / "03c_gap_filling_revisited" / "_data"
UMAP_RESULTS = D5_DATA / "d5_nonlinear_embedding_sample.csv"
AGREEMENT_RESULTS = D5_DATA / "d5_embedding_local_agreement.csv"
FIGURE = ROOT / "report" / "Figures" / "ch3_tica_umap_embeddings.png"
TICA_DIAGNOSTICS = ROOT / "results" / "ch3_tica_diagnostics.csv"
TICA_LOADINGS = ROOT / "results" / "ch3_tica_loadings.csv"

TICA_LAG = 24
N_TICS = 3
ENVIRONMENTAL_FEATURES = feat_list()[:11]

SHORT_LABELS = {
    "SWIN_1_1_1": "Shortwave radiation",
    "TA_0_0_1": "Air temperature",
    "VPD_0_0_1": "VPD",
    "PPFD_1_1_1": "Photon flux",
    "RN_1_1_1": "Net radiation",
    "WS_0_0_1": "Wind speed",
    "USTAR_0_0_1": "Friction velocity",
    "SHF_1_1_1": "Soil heat flux",
    "Precipitation (mm)": "Precipitation",
    "Soil Temperature @ 15cm Depth (oC)": "Soil temperature",
    "Soil Moisture @ 10cm Depth (%)": "Soil moisture",
}

TOWER_COLOURS = {2: "#d62728", 4: "#2ca02c", 9: "#1f77b4"}
SEASON_COLOURS = {
    "Winter": "#4C72B0",
    "Spring": "#55A868",
    "Summer": "#DD8452",
    "Autumn": "#8172B2",
}
TARGET_COLOURS = {"FCH4 observed": "#4C72B0", "FCH4 missing": "#C44E52"}


def contiguous_segments(data: pd.DataFrame, features: list[str]) -> list[dict]:
    """Keep temporally contiguous, complete segments for valid lag pairs."""
    segments = []
    for tower in TOWERS:
        tower_frame = frame(tower, pooled=True, d=data)
        tower_frame = tower_frame.loc[dom_mask(tower_frame.index, tower)]
        values = tower_frame[features].to_numpy(dtype=float)
        finite = pd.Series(np.isfinite(values).all(axis=1), index=tower_frame.index)
        segment_id = (~finite).cumsum()
        for _, block in tower_frame.loc[finite, features].groupby(
            segment_id.loc[finite]
        ):
            if len(block) >= TICA_LAG + 1:
                segments.append(
                    {"tower": tower, "index": block.index, "X": block.to_numpy()}
                )
    return segments


def fit_tica(data: pd.DataFrame):
    segments = contiguous_segments(data, ENVIRONMENTAL_FEATURES)
    scaler = StandardScaler().fit(np.vstack([segment["X"] for segment in segments]))
    estimator = TICA(lagtime=TICA_LAG, dim=N_TICS, scaling=None)
    n_pairs = 0
    for segment in segments:
        scaled = scaler.transform(segment["X"])
        segment["X_scaled"] = scaled
        estimator.partial_fit((scaled[:-TICA_LAG], scaled[TICA_LAG:]))
        n_pairs += len(scaled) - TICA_LAG
    model = estimator.fetch_model()

    n_components = min(N_TICS, model.feature_component_correlation.shape[1])
    component_names = [f"TIC{i + 1}" for i in range(n_components)]
    loadings = pd.DataFrame(
        model.feature_component_correlation[:, :n_components],
        index=ENVIRONMENTAL_FEATURES,
        columns=component_names,
    )
    loadings.index.name = "feature"

    timescales = model.timescales(k=n_components, lagtime=TICA_LAG)
    diagnostics = pd.DataFrame(
        {
            "component": component_names,
            "lag_hours": TICA_LAG,
            "autocorrelation_strength": model.singular_values[:n_components],
            "implied_timescale_hours": timescales,
            "n_lagged_pairs": n_pairs,
        }
    )

    projected_parts = []
    for segment in segments:
        projection = model.transform(segment["X_scaled"])[:, :n_components]
        projected = pd.DataFrame(
            projection, index=segment["index"], columns=component_names
        )
        projected["tower"] = segment["tower"]
        projected_parts.append(projected)
    return loadings, diagnostics, pd.concat(projected_parts)


def scatter_groups(ax, data, x, y, group_col, colours, title, size=5, alpha=0.4):
    for label, group in data.groupby(group_col, observed=True):
        legend_label = f"T{label}" if group_col == "tower" else str(label)
        ax.scatter(
            group[x],
            group[y],
            s=size,
            alpha=alpha,
            color=colours[label],
            label=legend_label.replace("FCH4", r"FCH$_4$"),
            rasterized=True,
        )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2.5, fontsize=7.5, frameon=True)


def main() -> None:
    data = load_ext()
    loadings, diagnostics, projection = fit_tica(data)

    TICA_DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(TICA_DIAGNOSTICS, index=False)
    loadings.reset_index().to_csv(TICA_LOADINGS, index=False)

    embedding = pd.read_csv(UMAP_RESULTS, parse_dates=["Datetime"])
    agreement = pd.read_csv(AGREEMENT_RESULTS)
    umap_agreement = agreement.loc[agreement["embedding"] == "UMAP"]
    if set(umap_agreement["label"]) != {"tower", "season", "target_state"}:
        raise ValueError("Saved UMAP agreement results do not match the D5 output schema")

    sns.set_theme(style="white", font_scale=0.88)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))

    display_loadings = loadings.rename(index=SHORT_LABELS)
    sns.heatmap(
        display_loadings,
        ax=axes[0, 0],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Feature-component correlation", "shrink": 0.72},
    )
    axes[0, 0].set_title(
        "(a) TICA feature-component correlations", fontsize=10, fontweight="bold"
    )
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].tick_params(axis="x", rotation=0, labelsize=8)
    axes[0, 0].tick_params(axis="y", labelsize=7.5)

    projection_sample = pd.concat(
        [projection.loc[projection["tower"] == tower].iloc[::6] for tower in TOWERS]
    )
    scatter_groups(
        axes[0, 1],
        projection_sample,
        "TIC1",
        "TIC2",
        "tower",
        TOWER_COLOURS,
        "(b) TICA environmental state by tower",
        size=3,
        alpha=0.25,
    )
    axes[0, 1].set_xlabel("TIC1", fontsize=8)
    axes[0, 1].set_ylabel("TIC2", fontsize=8)

    scatter_groups(
        axes[1, 0],
        embedding,
        "UMAP1",
        "UMAP2",
        "season",
        SEASON_COLOURS,
        "(c) UMAP environmental state by season",
    )
    scatter_groups(
        axes[1, 1],
        embedding,
        "UMAP1",
        "UMAP2",
        "target_state",
        TARGET_COLOURS,
        r"(d) UMAP observed versus missing FCH$_4$ support",
    )

    fig.suptitle(
        r"Low-dimensional structure of the gap-filling environmental predictors",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Wrote {FIGURE}; TICA used {len(projection):,} rows and "
        f"UMAP used {len(embedding):,} saved D5 rows"
    )


if __name__ == "__main__":
    main()
