"""Master script: coordinate B-14 complete execution (waits for notebook, extracts, runs multi-anchor)."""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = r"c:\Users\Nicholas\Documents\COMP0191 MSc Artificial Intelligence for Sustainable Development Project"
NOTEBOOK_PATH = Path(ROOT) / "notebooks/05_benchmarking/B14_hyperparameter_tuning.ipynb"
EXTRACT_SCRIPT = Path(ROOT) / "notebooks/05_benchmarking/extract_b14_winners.py"
MULTI_ANCHOR_SCRIPT = Path(ROOT) / "notebooks/05_benchmarking/b14_multi_anchor.py"
RESULTS_DIR = Path(ROOT) / "results"

def wait_for_notebook(timeout_min=45):
    """Wait for notebook to complete (check for completion flag or timeout)."""
    print(f"Waiting for notebook to complete (max {timeout_min} min)...")
    start_time = time.time()
    max_wait = timeout_min * 60

    csv_files = ["b14_tree_grid_search.csv", "b14_dl_grid.csv", "b14_sarimax_grid.csv"]

    while time.time() - start_time < max_wait:
        all_done = all((RESULTS_DIR / f).exists() for f in csv_files)
        if all_done:
            print("✓ All grid search CSV files generated")
            return True

        elapsed = int((time.time() - start_time) / 60)
        print(f"  [{elapsed} min elapsed] Waiting for results CSVs...")
        time.sleep(30)

    print(f"ERROR: Notebook did not complete within {timeout_min} minutes")
    return False

def extract_hyperparameters():
    """Extract winning hyperparameters from grid search results."""
    print("\n✓ Extracting winning hyperparameters...")
    try:
        # Simple extraction from CSVs
        import pandas as pd

        winners = {}

        # Tree models
        tree_grid = pd.read_csv(RESULTS_DIR/"b14_tree_grid_search.csv")
        for model_name in ["RF", "XGB", "LGB"]:
            best = tree_grid[tree_grid["model"] == model_name].nlargest(1, "val_r2").iloc[0]
            winners[model_name] = dict(best.drop(["model", "val_r2"]))
            print(f"  {model_name}: {dict(best.drop(['model', 'val_r2']))}")

        # DL models
        dl_grid = pd.read_csv(RESULTS_DIR/"b14_dl_grid.csv")
        for model in ["DLinear", "LSTM"]:
            best = dl_grid[dl_grid["model"] == model].nsmallest(1, "val_loss").iloc[0]
            winners[model] = dict(best.drop(["model", "val_loss"]))
            print(f"  {model}: {dict(best.drop(['model', 'val_loss']))}")

        # SARIMAX
        sarimax_grid = pd.read_csv(RESULTS_DIR/"b14_sarimax_grid.csv")
        best = sarimax_grid.nsmallest(1, "AIC").iloc[0]
        winners["SARIMAX"] = {"order": (int(best["p"]), 1, int(best["q"]))}
        print(f\"  SARIMAX: order={winners['SARIMAX']['order']}\")

        return winners
    except Exception as e:
        print(f"ERROR extracting hyperparameters: {e}")
        return None

def update_multi_anchor_script(winners):
    \"\"\"Update b14_multi_anchor.py with tuned hyperparameters.\"\"\"
    if not winners:
        print("Skipping multi-anchor update (no winning hyperparams)")
        return False

    print("\n✓ Updating multi-anchor script with tuned hyperparameters...")

    # Read the script
    with open(MULTI_ANCHOR_SCRIPT) as f:
        content = f.read()

    # Update TUNED_PARAMS dict
    new_tuned_params = f'''TUNED_PARAMS = {{
    "RF": {winners["RF"]},
    "XGB": {winners["XGB"]},
    "LightGBM": {winners["LGB"]},
    "SARIMAX": {winners["SARIMAX"]},
}}'''

    # Find and replace the old TUNED_PARAMS
    import re
    content = re.sub(
        r'TUNED_PARAMS = \{[^}]+?\}',
        new_tuned_params,
        content,
        flags=re.DOTALL
    )

    with open(MULTI_ANCHOR_SCRIPT, 'w') as f:
        f.write(content)

    print("✓ Multi-anchor script updated")
    return True

def run_multi_anchor():
    \"\"\"Execute the multi-anchor validation script.\"\"\"
    print(f"\n✓ Running multi-anchor validation (5 anchors, all models)...")
    try:
        os.chdir(str(Path(ROOT) / "notebooks/05_benchmarking"))
        result = subprocess.run(
            [sys.executable, "b14_multi_anchor.py"],
            capture_output=False,
            timeout=2400  # 40 minutes
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR running multi-anchor: {e}")
        return False

def main():
    print("="*70)
    print("B-14 COMPLETE EXECUTION: Grid Search → Multi-Anchor → Results")
    print("="*70)

    # Wait for notebook
    if not wait_for_notebook():
        print("Aborting: notebook did not finish")
        sys.exit(1)

    # Extract hyperparameters
    winners = extract_hyperparameters()
    if not winners:
        print("Aborting: could not extract hyperparameters")
        sys.exit(1)

    # Update multi-anchor script
    if not update_multi_anchor_script(winners):
        print("Warning: could not update multi-anchor script, proceeding anyway")

    # Run multi-anchor validation
    if not run_multi_anchor():
        print("Warning: multi-anchor script may have failed")

    print("\n" + "="*70)
    print("B-14 execution complete")
    print("Results: results/b14_tuned_rollout_summary.csv")
    print("="*70)

if __name__ == "__main__":
    main()
