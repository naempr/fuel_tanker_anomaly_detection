
import argparse, numpy as np, pandas as pd, joblib
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from src.utils import load_config, ensure_dir, load_scaler, make_run_dirs
from src.windowing import make_windows
from train.train_lstm_encoder import build_lstm_encoder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--run_tag", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    paths, wcfg = cfg["paths"], cfg["windowing"]
    runp = make_run_dirs(paths, a.scenario, a.run_tag)

    df = pd.read_csv(paths["train_csv"])
    Xs, ys, metas = [], [], []
    for tid, g in df.groupby("trip_id"):
        X, y, starts, _ = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X)==0: continue
        Xs.append(X); ys.append(y); metas.append(pd.DataFrame({"split":"train","trip_id":[tid]*len(X),"win_start_idx":starts}))
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    meta_all = pd.concat(metas, ignore_index=True)

    classes = sorted(np.unique(y).tolist())
    lab2id = {v:i for i,v in enumerate(classes)}
    id2lab = {i:v for v,i in lab2id.items()}
    y_id = np.array([lab2id[v] for v in y], dtype=int)

    mean, std = load_scaler(runp["scaler_path"])
    Xs = (X - mean) / std

    enc = build_lstm_encoder(timesteps=X.shape[1], seq_feats=X.shape[2], num_classes=len(classes))
    enc.load_weights(runp["lstm_encoder_weights"])
    feat_model = keras.Model(enc.input, enc.get_layer("embed").output)
    Z = feat_model.predict(Xs, verbose=0)

    ae_df = pd.read_csv(runp["ae_feats_csv"])
    ae_train = ae_df[ae_df["split"]=="train"].copy()
    aux_cols = [c for c in ae_train.columns if c not in ["split","trip_id","win_start_idx"]]
    merged = meta_all.merge(ae_train, on=["split","trip_id","win_start_idx"], how="left").reindex(columns=["split","trip_id","win_start_idx"]+aux_cols)
    merged[aux_cols] = merged[aux_cols].fillna(0.0)
    Aux = merged[aux_cols].to_numpy(float)

    F = np.concatenate([Z, Aux], axis=1)

    rf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=cfg["seed"], class_weight="balanced_subsample")
    rf.fit(F, y_id)

    bundle = {"rf": rf, "classes": classes, "lab2id": lab2id}
    joblib.dump(bundle, runp["rf_model_path"])
    print("Saved RF model to", runp["rf_model_path"], " / Feat dim:", F.shape[1])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
