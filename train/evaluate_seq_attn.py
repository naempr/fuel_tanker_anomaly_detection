
import os, argparse, json
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report
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

    id2lab = {int(k): v for k, v in json.load(open(runp["label_mapping_json"], "r", encoding="utf-8")).items()}
    lab2id = {v: k for k, v in id2lab.items()}

    ae_df = pd.read_csv(runp["ae_feats_csv"])
    ae_test = ae_df[ae_df["split"]=="test"].copy()
    aux_cols = [c for c in ae_test.columns if c not in ["split","trip_id","win_start_idx"]]

    df = pd.read_csv(paths["test_csv"])
    mean, std = load_scaler(runp["scaler_path"])

    Xs, AUX, ys, metas = [], [], [], []
    for tid, g in df.groupby("trip_id"):
        X, y, starts, _ = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X)==0: continue
        meta = pd.DataFrame({"split":"test","trip_id":[tid]*len(X),"win_start_idx": starts})
        m = meta.merge(ae_test, on=["split","trip_id","win_start_idx"], how="left")\
                .reindex(columns=["split","trip_id","win_start_idx"] + aux_cols)
        m[aux_cols] = m[aux_cols].fillna(0.0)

        Xs.append((X - mean) / std)
        AUX.append(m[aux_cols].to_numpy(float))
        ys.append(y)
        metas.append(meta)

    X = np.concatenate(Xs, axis=0)
    Aux = np.concatenate(AUX, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = pd.concat(metas, ignore_index=True)

    y_id = np.array([lab2id[v] for v in y], dtype=int)
    classes = sorted(id2lab.keys())
    names = [id2lab[i] for i in classes]

    model = build_seq_attn_model(timesteps=X.shape[1], seq_feats=X.shape[2], aux_feats=Aux.shape[1], num_classes=len(classes))
    model.load_weights(runp["attn_weights"])

    y_prob = model.predict({"seq": X, "aux": Aux}, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    rep = classification_report(y_id, y_pred, target_names=names, digits=4, zero_division=0)
    print(rep)
    open(os.path.join(runp["eval_dir"], "classification_report.txt"), "w", encoding="utf-8").write(rep)

    out = meta.copy()
    out["true_label"] = [id2lab[i] for i in y_id]
    out["pred_label"] = [id2lab[i] for i in y_pred]
    for i, n in zip(classes, names):
        out[f"prob_{n}"] = y_prob[:, i]
    out.to_csv(os.path.join(runp["eval_dir"], "predictions_window_level.csv"), index=False)
    print("Saved eval artifacts to", runp["eval_dir"])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
