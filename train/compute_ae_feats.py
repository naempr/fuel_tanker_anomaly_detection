
import argparse, numpy as np, pandas as pd
from tensorflow import keras
import tensorflow as tf
from src.utils import load_config, ensure_dir, load_scaler, make_run_dirs
from src.windowing import make_windows
from src.models_ae import build_ae

def window_stats(X, feat_names):
    stats = {}
    per = [10, 90]
    for i, n in enumerate(feat_names):
        x = X[:, :, i]
        stats[f"{n}_mean"] = x.mean(axis=1)
        stats[f"{n}_std"]  = x.std(axis=1)
        stats[f"{n}_min"]  = x.min(axis=1)
        stats[f"{n}_max"]  = x.max(axis=1)
        for p in per:
            stats[f"{n}_p{p}"] = np.percentile(x, p, axis=1)
    return stats

def proc(csv_path, split_name, cfg, wcfg, ae, mean, std, feat_cols):
    d = pd.read_csv(csv_path)
    rows = []
    for tid, g in d.groupby("trip_id"):
        X, y, starts, _ = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X)==0: continue
        Xs = (X - mean)/std
        Xhat = ae.predict(Xs, verbose=0)

        mae_f = np.mean(np.abs(X - Xhat), axis=1)
        total = np.mean(np.abs(X - Xhat), axis=(1,2))

        st = window_stats(X, feat_cols)
        out = pd.DataFrame({"split": split_name,
                            "trip_id": [tid]*len(X),
                            "win_start_idx": starts,
                            "ae_total_mae": total})
        for i, n in enumerate(feat_cols):
            out[f"ae_mae_{n}"] = mae_f[:, i]
        for k,v in st.items():
            out[k] = v
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--run_tag", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    paths, wcfg = cfg["paths"], cfg["windowing"]
    runp = make_run_dirs(paths, a.scenario, a.run_tag)

    df0 = pd.read_csv(paths["train_csv"])
    first_trip = next(iter(df0.groupby("trip_id")))[1]
    X0, _, _, feat_cols = make_windows(
        first_trip, wcfg["window_sec"], wcfg["stride_sec"],
        cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
    )
    if len(X0)==0: raise SystemExit("No windows to infer AE shape.")

    ae = build_ae(X0.shape[1], X0.shape[2], cfg.get("ae", {}))
    ae.compile(optimizer="rmsprop", loss="mae")
    # build once
    ae(tf.zeros((1, X0.shape[1], X0.shape[2])))
    ae.load_weights(runp["ae_weights"])

    mean, std = load_scaler(runp["scaler_path"])

    train_out = proc(paths["train_csv"], "train", cfg, wcfg, ae, mean, std, feat_cols)
    test_out  = proc(paths["test_csv"], "test",  cfg, wcfg, ae, mean, std, feat_cols)

    df_all = pd.concat([train_out, test_out], ignore_index=True)
    df_all.to_csv(runp["ae_feats_csv"], index=False)

    print("Wrote AE features to", runp["ae_feats_csv"], "Rows:", len(df_all))
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
