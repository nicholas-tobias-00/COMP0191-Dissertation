import time
import gc
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

CKPT_DIR = DATA_DIR / "d100_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CHAMPION_R2 = {2: 0.576, 4: 0.404, 9: 0.426}   # standing production champion (D-77), hardcoded reference only

# sklearn-R2 numbers already on disk (_data/model_comparison.csv) -- used as the verification
# checkpoint (this run's own sklearn R2 must reproduce these bit-for-bit before R2_OLS is trusted).
ORIGINAL_R2 = {
    ("LightGBM", 2): 0.522, ("LightGBM", 4): 0.410, ("LightGBM", 9): 0.422,
    ("XGBoost", 2): 0.551, ("XGBoost", 4): 0.349, ("XGBoost", 9): 0.369,
    ("TabICL", 2): 0.558, ("TabICL", 4): 0.423, ("TabICL", 9): 0.364,
    ("SAITS", 2): 0.358, ("SAITS", 4): 0.293, ("SAITS", 9): 0.285,
    ("BI-LSTM", 2): 0.237, ("BI-LSTM", 4): 0.155, ("BI-LSTM", 9): 0.146,
}


def mets_ols(y, p):
    """mets() + R2_OLS (squared Pearson r, Zhu et al. 2023a convention, bounded [0,1]) -- same
    formula as temp_gap_filling_pipeline.ipynb's own mets(), ported here since this notebook's
    mets() predates that addition."""
    r2, rmse, mae, mbe = mets(y, p)
    y = np.asarray(y, float); p = np.asarray(p, float)
    if np.var(y) > 0 and np.var(p) > 0:
        slope, _, r, _, _ = linregress(y, p)
        r2_ols = r ** 2
    else:
        slope, r2_ols = np.nan, np.nan
    return r2, rmse, mae, mbe, r2_ols, slope


def med_metrics_ols(rows):
    if not rows:
        return {k: np.nan for k in ["R2", "RMSE", "MAE", "MBE", "R2_OLS", "OLS_slope"]}
    a = np.array(rows, float)
    return {"R2": np.nanmedian(a[:, 0]), "RMSE": np.median(a[:, 1]), "MAE": np.median(a[:, 2]),
            "MBE": np.median(a[:, 3]), "R2_OLS": np.nanmedian(a[:, 4]), "OLS_slope": np.nanmedian(a[:, 5])}


def run_model_capture(t, fit_fn, model_name, pooled=True):
    """Verbatim copy of cell 103's run_model(), extended to also stash raw (Datetime, actual,
    pred) rows and score with mets_ols instead of mets. Same insert_calendar_gaps(seed=0) fold
    structure -- must reproduce ORIGINAL_R2[(model_name, t)] exactly."""
    feat = FEATURES + (DUM if pooled else [])
    towers = TOWERS if pooled else [t]
    frames = {tt: frame(tt, pooled, d_all) for tt in towers}
    g = frames[t]; dm = dom_mask(g.index, t)
    raw_parts = []; scn_metrics = {}
    for sc, gh in SCENARIOS.items():
        rr = []
        for fi, gt in enumerate(insert_calendar_gaps(g, "target", dm, gh)):
            if len(gt) < 5:
                continue
            base = g[dm & g["target"].notna().values]; trd = base.drop(index=gt, errors="ignore")
            if pooled:
                others = []
                for tt in towers:
                    if tt == t:
                        continue
                    gg = frames[tt]; dmt = dom_mask(gg.index, tt)
                    others.append(gg[dmt & gg["target"].notna().values])
                trd = pd.concat([trd] + others, ignore_index=True)
            m, imp = fit_fn(feat, trd)
            yp = m.predict(imp.transform(g.loc[gt, feat].values))
            y_true = g.loc[gt, "target"].values
            rr.append(mets_ols(y_true, yp))
            raw_parts.append(pd.DataFrame({"actual": y_true, "pred": yp, "rep": fi, "scenario": sc}, index=gt))
        scn_metrics[sc] = med_metrics_ols(rr)
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    raw_df = raw_df.reset_index().rename(columns={"index": "Datetime"})
    raw_df["tower"] = t; raw_df["model"] = model_name
    return scn_metrics, raw_df


def fit_score_saits_capture(t, sc, gh, seed=42):
    """Verbatim copy of cell 111's fit_score_saits_scenario(), extended to also stash raw pairs
    and score with mets_ols."""
    channels = SAITS_FEAT_COLS + ["target"]
    target_idx = channels.index("target")
    torch.manual_seed(seed)

    g_t = frame(t, pooled=False, d=d_all)
    dm_t = dom_mask(g_t.index, t)
    reps = insert_calendar_gaps(g_t, "target", dm_t, gh, n_reps=N_REPS, seed=0)
    union_ts = pd.DatetimeIndex(sorted(set().union(*[set(r) for r in reps])))

    g_gapped = g_t.copy()
    y_true_all = g_gapped["target"].copy()
    g_gapped.loc[union_ts, "target"] = np.nan

    scaler = StandardScaler().fit(g_gapped.loc[DOMAIN[t][0]:DOMAIN[t][1], channels])
    g_scaled = g_gapped.copy()
    g_scaled[channels] = scaler.transform(g_gapped[channels])
    X_t, idx_t, starts_t = make_windows(g_scaled, channels, DOMAIN[t])

    train_set, val_set = build_train_val(X_t)
    model = SAITS(
        n_steps=WINDOW, n_features=len(channels), n_layers=3, d_model=256, n_heads=4,
        d_k=64, d_v=64, d_ffn=512, dropout=0.1, epochs=100, patience=10, batch_size=32,
        device=DEVICE, saving_path=None, model_saving_strategy=None, verbose=False,
        training_loss=SpikeWeightedMAE,
    )
    model.fit(train_set, val_set)
    result = model.predict({"X": X_t})
    imputation = np.asarray(result["imputation"])   # detach from any GPU-backed object before model is freed
    del model, result, train_set, val_set, X_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_mean, t_std = scaler.mean_[target_idx], scaler.scale_[target_idx]
    rep_metrics = []; raw_parts = []
    for rep_i, rep_ts in enumerate(reps):
        preds_scaled = extract_at_timestamps(imputation, idx_t, starts_t, WINDOW, target_idx, rep_ts)
        ts_scored = [ts for ts in rep_ts if ts in preds_scaled]
        if len(ts_scored) < 5:
            continue
        y_pred = np.array([preds_scaled[ts] for ts in ts_scored]) * t_std + t_mean
        y_obs = y_true_all.loc[ts_scored].values
        rep_metrics.append(mets_ols(y_obs, y_pred))
        raw_parts.append(pd.DataFrame({"actual": y_obs, "pred": y_pred, "rep": rep_i, "scenario": sc},
                                       index=pd.DatetimeIndex(ts_scored)))
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    return med_metrics_ols(rep_metrics), raw_df


def fit_score_bilstm_capture(t, sc, gh, seed=42):
    """Verbatim copy of cell 114's fit_score_bilstm_scenario(), extended to also stash raw pairs
    and score with mets_ols."""
    channels = SAITS_FEAT_COLS + ["target"]
    target_idx = channels.index("target")

    g_t = frame(t, pooled=False, d=d_all)
    dm_t = dom_mask(g_t.index, t)
    reps = insert_calendar_gaps(g_t, "target", dm_t, gh, n_reps=N_REPS, seed=0)
    union_ts = pd.DatetimeIndex(sorted(set().union(*[set(r) for r in reps])))

    g_gapped = g_t.copy()
    y_true_all = g_gapped["target"].copy()
    g_gapped.loc[union_ts, "target"] = np.nan

    scaler = StandardScaler().fit(g_gapped.loc[DOMAIN[t][0]:DOMAIN[t][1], channels])
    g_scaled = g_gapped.copy()
    g_scaled[channels] = scaler.transform(g_gapped[channels])
    X_t, idx_t, starts_t = make_windows(g_scaled, channels, DOMAIN[t])

    n = X_t.shape[0]
    n_val = max(1, int(n * 0.1))
    X_train, X_val = X_t[:n - n_val], X_t[n - n_val:]

    model = train_bilstm(X_train, X_val, seed=seed)
    imputation = predict_bilstm(model, X_t)   # already .cpu().numpy() inside predict_bilstm
    del model, X_train, X_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t_mean, t_std = scaler.mean_[target_idx], scaler.scale_[target_idx]
    rep_metrics = []; raw_parts = []
    for rep_i, rep_ts in enumerate(reps):
        preds_scaled = extract_at_timestamps(imputation, idx_t, starts_t, WINDOW, 0, rep_ts)
        ts_scored = [ts for ts in rep_ts if ts in preds_scaled]
        if len(ts_scored) < 5:
            continue
        y_pred = np.array([preds_scaled[ts] for ts in ts_scored]) * t_std + t_mean
        y_obs = y_true_all.loc[ts_scored].values
        rep_metrics.append(mets_ols(y_obs, y_pred))
        raw_parts.append(pd.DataFrame({"actual": y_obs, "pred": y_pred, "rep": rep_i, "scenario": sc},
                                       index=pd.DatetimeIndex(ts_scored)))
    raw_df = pd.concat(raw_parts) if raw_parts else pd.DataFrame(columns=["actual", "pred", "rep", "scenario"])
    return med_metrics_ols(rep_metrics), raw_df


# ============================== driver ==============================
RESULTS_ROWS = []
ALL_RAW = []


def _report(model_name, t, scn):
    ov = {k: np.nanmedian([scn[s][k] for s in SCENARIOS]) for k in ["R2", "RMSE", "MAE", "MBE", "R2_OLS", "OLS_slope"]}
    orig = ORIGINAL_R2[(model_name, t)]
    match = "OK" if abs(ov["R2"] - orig) < 0.002 else "MISMATCH"
    print(f"{model_name:10s} T{t}: R2_sklearn={ov['R2']:+.3f} (orig {orig:+.3f}, {match})  "
          f"R2_OLS={ov['R2_OLS']:.3f}  champion_R2={CHAMPION_R2[t]:.3f}", flush=True)
    RESULTS_ROWS.append({"model": model_name, "tower": t, **ov, "orig_R2_sklearn": orig,
                          "verification": match, "champion_R2": CHAMPION_R2[t]})


print("=== D-100: recalculating R2_OLS (scipy/Zhu-style) for D-78's 5 non-TabPFN models ===", flush=True)


def _ckpt_load_or_run(key, compute_fn):
    """Checkpoint-gated: if (raw_df, scn_dict) was already saved for this key, load it instead of
    recomputing -- so a killed/restarted run doesn't repay already-finished work."""
    raw_path = CKPT_DIR / f"{key}_raw.csv"
    scn_path = CKPT_DIR / f"{key}_scn.csv"
    if raw_path.exists() and scn_path.exists():
        print(f"  [checkpoint hit] {key}", flush=True)
        raw_df = pd.read_csv(raw_path, parse_dates=["Datetime"])
        scn = pd.read_csv(scn_path).set_index("scenario").to_dict("index")
        return scn, raw_df
    scn, raw_df = compute_fn()
    raw_df.to_csv(raw_path, index=False)
    pd.DataFrame(scn).T.reset_index().rename(columns={"index": "scenario"}).to_csv(scn_path, index=False)
    return scn, raw_df


t0 = time.time()
for model_name, fit_fn in [("LightGBM", fit_lgbm), ("XGBoost", fit_xgb), ("TabICL", fit_tabicl)]:
    for t in TOWERS:
        scn, raw_df = _ckpt_load_or_run(f"{model_name}_{t}", lambda: run_model_capture(t, fit_fn, model_name, pooled=True))
        _report(model_name, t, scn)
        raw_df.to_csv(DATA_DIR / f"d100_raw_{model_name}_{t}.csv", index=False)
        ALL_RAW.append(raw_df)
print(f"Tree/foundation models done [{time.time()-t0:.0f}s elapsed]", flush=True)

for t in TOWERS:
    def _run_saits_tower(t=t):
        scn_t = {}; raw_parts_t = []
        for sc, gh in SCENARIOS.items():
            m, raw = fit_score_saits_capture(t, sc, gh)
            scn_t[sc] = m; raw["scenario"] = sc; raw_parts_t.append(raw)
            print(f"    SAITS T{t} scenario={sc}: R2={m['R2']:+.3f} R2_OLS={m['R2_OLS']:.3f} [{time.time()-t0:.0f}s elapsed]", flush=True)
        raw_df_t = pd.concat(raw_parts_t).reset_index().rename(columns={"index": "Datetime"})
        raw_df_t["tower"] = t; raw_df_t["model"] = "SAITS"
        return scn_t, raw_df_t
    scn, raw_df = _ckpt_load_or_run(f"SAITS_{t}", _run_saits_tower)
    _report("SAITS", t, scn)
    raw_df.to_csv(DATA_DIR / f"d100_raw_SAITS_{t}.csv", index=False)
    ALL_RAW.append(raw_df)
print(f"SAITS done [{time.time()-t0:.0f}s elapsed]", flush=True)

for t in TOWERS:
    def _run_bilstm_tower(t=t):
        scn_t = {}; raw_parts_t = []
        for sc, gh in SCENARIOS.items():
            m, raw = fit_score_bilstm_capture(t, sc, gh)
            scn_t[sc] = m; raw["scenario"] = sc; raw_parts_t.append(raw)
            print(f"    BI-LSTM T{t} scenario={sc}: R2={m['R2']:+.3f} R2_OLS={m['R2_OLS']:.3f} [{time.time()-t0:.0f}s elapsed]", flush=True)
        raw_df_t = pd.concat(raw_parts_t).reset_index().rename(columns={"index": "Datetime"})
        raw_df_t["tower"] = t; raw_df_t["model"] = "BI-LSTM"
        return scn_t, raw_df_t
    scn, raw_df = _ckpt_load_or_run(f"BILSTM_{t}", _run_bilstm_tower)
    _report("BI-LSTM", t, scn)
    raw_df.to_csv(DATA_DIR / f"d100_raw_BILSTM_{t}.csv", index=False)
    ALL_RAW.append(raw_df)
print(f"BI-LSTM done [{time.time()-t0:.0f}s elapsed]", flush=True)

RESULTS_DF = pd.DataFrame(RESULTS_ROWS)
RESULTS_DF.to_csv(DATA_DIR / "d100_ols_recalc_summary.csv", index=False)
ALL_RAW_DF = pd.concat(ALL_RAW, ignore_index=True)
ALL_RAW_DF.to_csv(DATA_DIR / "d100_ols_recalc_raw_predictions.csv", index=False)
print(f"\nSaved d100_ols_recalc_summary.csv ({len(RESULTS_DF)} rows) and "
      f"d100_ols_recalc_raw_predictions.csv ({len(ALL_RAW_DF):,} rows) [{time.time()-t0:.0f}s total]", flush=True)

print("\n=== Final table: sklearn R2 vs R2_OLS (scipy/Zhu-style), per model/tower ===")
print(RESULTS_DF[["model", "tower", "R2", "orig_R2_sklearn", "verification", "R2_OLS", "champion_R2"]]
      .to_string(index=False))
