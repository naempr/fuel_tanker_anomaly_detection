
import argparse, os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from src.utils import load_config, ensure_dir, load_scaler, make_run_dirs
from src.windowing import make_windows
from src.models_seq import build_seq_attn_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--run_tag", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    paths, wcfg = cfg["paths"], cfg["windowing"]
    runp = make_run_dirs(paths, a.scenario, a.run_tag)
    ensure_dir(runp["eval_dir"])

    df = pd.read_csv(paths["train_csv"])
    ae_df = pd.read_csv(runp["ae_feats_csv"])
    ae_train = ae_df[ae_df["split"]=="train"].copy()
    aux_cols = [c for c in ae_train.columns if c not in ["split","trip_id","win_start_idx"]]

    Xs, AUX, ys = [], [], []
    for tid, g in df.groupby("trip_id"):
        X, y, starts, feat_cols = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X)==0: continue
        meta = pd.DataFrame({"split":"train","trip_id":[tid]*len(X),"win_start_idx": starts})
        merged = meta.merge(ae_train, on=["split","trip_id","win_start_idx"], how="left")\
                     .reindex(columns=["split","trip_id","win_start_idx"] + aux_cols)
        merged[aux_cols] = merged[aux_cols].fillna(0.0)

        Xs.append(X)
        AUX.append(merged[aux_cols].to_numpy(dtype=float))
        ys.append(y)

    X = np.concatenate(Xs, axis=0)
    Aux = np.concatenate(AUX, axis=0)
    y = np.concatenate(ys, axis=0)

    classes = sorted(np.unique(y).tolist())
    id2lab = {i: c for i, c in enumerate(classes)}
    lab2id = {v: k for k, v in id2lab.items()}
    y_id = np.array([lab2id[v] for v in y], dtype=int)

    mean, std = load_scaler(runp["scaler_path"])
    X = (X - mean) / std

    Xtr, Xva, Atr, Ava, ytr, yva = train_test_split(
        X, Aux, y_id, test_size=0.2, stratify=y_id, random_state=cfg["seed"]
    )

    model = build_seq_attn_model(
        timesteps=X.shape[1],
        seq_feats=X.shape[2],
        aux_feats=Aux.shape[1],
        num_classes=len(classes)
    )

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(runp["attn_weights"], save_best_only=True, monitor="val_accuracy", save_weights_only=True)
    ]
    model.fit({"seq": Xtr, "aux": Atr}, ytr,
              validation_data=({"seq": Xva, "aux": Ava}, yva),
              epochs=80, batch_size=256, verbose=2, callbacks=cb)

    json.dump({str(k): v for k, v in id2lab.items()}, open(runp["label_mapping_json"], "w", encoding="utf-8"), ensure_ascii=False)
    print("Saved attention weights to", runp["attn_weights"], "Aux features:", Aux.shape[1])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
