"""D-7x: merges the 5 gap-filled-target/-context ablation columns (DLinear_gf, LSTM_gf, TFT_gf,
TabPFN_gf, TabICLv2_gf -- from `b10_b13_dl_gf_extension.py`, `b10_b13_tft_gf_extension.py`,
`b10_b13_foundation_gf_extension.py`) into `results/b10_b13_full_chains.csv` in place, following the
exact backup/verify-row-count-unchanged discipline `b10_b13_climatology_gf_baseline.py` established
for its own `Climatology_gf` column addition.
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
RESULTS = Path(ROOT) / "results"

NEW_COLS = {
    "b10_b13_dl_gf_extension_chains.csv": ["DLinear_gf", "LSTM_gf"],
    "b10_b13_tft_gf_extension_chains.csv": ["TFT_gf"],
    "b10_b13_foundation_gf_extension_chains.csv": ["TabPFN_gf", "TabICLv2_gf"],
}


def main():
    full_path = RESULTS / "b10_b13_full_chains.csv"
    chains = pd.read_csv(full_path)
    before_rows, before_cols = len(chains), len(chains.columns)

    backup_path = RESULTS / "b10_b13_full_chains_backup_pre_gf.csv"
    shutil.copy(full_path, backup_path)
    print(f"[OK] Backed up {full_path.name} -> {backup_path.name}")

    for fname, cols in NEW_COLS.items():
        for c in cols:
            if c in chains.columns:
                raise RuntimeError(f"b10_b13_full_chains.csv already has a {c} column")
        src = pd.read_csv(RESULTS / fname)[["date", "tower", "anchor_year"] + cols]
        chains = chains.merge(src, on=["date", "tower", "anchor_year"], how="left")
        print(f"[OK] Merged {cols} from {fname}")

    assert len(chains) == before_rows, f"merge changed row count: {before_rows} -> {len(chains)}"
    chains.to_csv(full_path, index=False)
    print(f"[OK] Wrote {full_path.name}: {len(chains)} rows (unchanged), "
          f"{before_cols} -> {len(chains.columns)} columns")


if __name__ == "__main__":
    main()
