"""Extract winning hyperparameters from B-14 grid search results and prepare for multi-anchor run."""

import pandas as pd
from pathlib import Path

RESULTS = Path("../../results")

def extract_winners():
    """Extract best hyperparameters from grid search CSVs."""

    winners = {}

    # RF
    try:
        rf_grid = pd.read_csv(RESULTS/"b14_tree_grid_search.csv")
        rf_best = rf_grid[rf_grid["model"] == "RF"].nlargest(1, "val_r2").iloc[0]
        winners["RF"] = {
            "max_features": float(rf_best["max_features"]),
            "min_samples_leaf": int(rf_best["min_samples_leaf"])
        }
        print(f"RF: {winners['RF']}")
    except Exception as e:
        print(f"Error reading RF grid: {e}")
        winners["RF"] = {"max_features": 0.5, "min_samples_leaf": 10}

    # XGB
    try:
        xgb_grid = pd.read_csv(RESULTS/"b14_tree_grid_search.csv")
        xgb_best = xgb_grid[xgb_grid["model"] == "XGB"].nlargest(1, "val_r2").iloc[0]
        winners["XGB"] = {
            "max_depth": int(xgb_best["max_depth"]),
            "learning_rate": float(xgb_best["learning_rate"]),
            "min_child_weight": int(xgb_best["min_child_weight"])
        }
        print(f"XGB: {winners['XGB']}")
    except Exception as e:
        print(f"Error reading XGB grid: {e}")
        winners["XGB"] = {"max_depth": 2, "learning_rate": 0.02, "min_child_weight": 10}

    # LGB
    try:
        lgb_grid = pd.read_csv(RESULTS/"b14_tree_grid_search.csv")
        lgb_best = lgb_grid[lgb_grid["model"] == "LGB"].nlargest(1, "val_r2").iloc[0]
        winners["LGB"] = {
            "num_leaves": int(lgb_best["num_leaves"]),
            "min_child_samples": int(lgb_best["min_child_samples"]),
            "learning_rate": float(lgb_best["learning_rate"])
        }
        print(f"LGB: {winners['LGB']}")
    except Exception as e:
        print(f"Error reading LGB grid: {e}")
        winners["LGB"] = {"num_leaves": 7, "min_child_samples": 10, "learning_rate": 0.02}

    # DL models
    try:
        dl_grid = pd.read_csv(RESULTS/"b14_dl_grid.csv")
        for model in ["DLinear", "LSTM"]:
            best_dl = dl_grid[dl_grid["model"] == model].nsmallest(1, "val_loss").iloc[0]
            winners[model] = {
                "hidden": int(best_dl["hidden"]),
                "lr": float(best_dl["lr"]),
                "wd": float(best_dl["wd"])
            }
            print(f"{model}: {winners[model]}")
    except Exception as e:
        print(f"Error reading DL grid: {e}")
        winners["DLinear"] = {"hidden": 28, "lr": 1e-3, "wd": 0}
        winners["LSTM"] = {"hidden": 28, "lr": 1e-3, "wd": 0}

    # SARIMAX order
    try:
        sarimax_grid = pd.read_csv(RESULTS/"b14_sarimax_grid.csv")
        sarimax_best = sarimax_grid.nsmallest(1, "AIC").iloc[0]
        winners["SARIMAX"] = {
            "order": (int(sarimax_best["p"]), 1, int(sarimax_best["q"]))
        }
        print(f"SARIMAX: order={winners['SARIMAX']['order']}")
    except Exception as e:
        print(f"Error reading SARIMAX grid: {e}")
        winners["SARIMAX"] = {"order": (2, 1, 1)}

    return winners

if __name__ == "__main__":
    print("Extracting B-14 grid search winners...")
    winners = extract_winners()
    print("\nAll winners extracted!")
