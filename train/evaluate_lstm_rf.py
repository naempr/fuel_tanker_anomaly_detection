
import os, argparse, numpy as np, pandas as pd, joblib
from sklearn.metrics import classification_report
from tensorflow import keras
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
    ensure_dir(runp["rf_eval_dir"])

    d0 = pd.read_csv(paths["train_csv"])
    first_trip = next(iter(d0.groupby("trip_id")))[1]
    X0, _, _, _ = make_windows(first_trip, wcfg["window_sec"], wcfg["stride_sec"], cfg["simulation"]["sample_period_s"], wcfg["feat_cols"])
    enc = build_lstm_encoder(timesteps=X0.shape[1], seq_feats=X0.shape[2], num_classes=3)
    enc.load_weights(runp["lstm_encoder_weights"])
    feat_model = keras.Model(enc.input, enc.get_layer("embed").output)

    bundle = joblib.load(runp["rf_model_path"])
    rf = bundle["rf"]; classes = bundle["classes"]; lab2id = bundle["lab2id"]; id2lab = {v:k for k,v in lab2id.items()}

    ae_df = pd.read_csv(runp["ae_feats_csv"])
    ae_test = ae_df[ae_df["split"]=="test"].copy()
    aux_cols = [c for c in ae_test.columns if c not in ["split","trip_id","win_start_idx"]]

    df = pd.read_csv(paths["test_csv"])
    mean, std = load_scaler(runp["scaler_path"])

    Xs, AUX, ys, metas = [], [], [], []
    for tid, g in df.groupby("trip_id"):
        X, y, starts, _ = make_windows(g, wcfg["window_sec"], wcfg["stride_sec"], cfg["simulation"]["sample_period_s"], wcfg["feat_cols"])
        if len(X)==0: continue
        meta = pd.DataFrame({"split":"test","trip_id":[tid]*len(X), "win_start_idx": starts})
        m = meta.merge(ae_test, on=["split","trip_id","win_start_idx"], how="left").reindex(columns=["split","trip_id","win_start_idx"]+aux_cols)
        m[aux_cols] = m[aux_cols].fillna(0.0)
        Xs.append((X - mean)/std); AUX.append(m[aux_cols].to_numpy(float)); ys.append(y); metas.append(meta)

    X = np.concatenate(Xs, axis=0); Aux = np.concatenate(AUX, axis=0); y = np.concatenate(ys, axis=0)
    y_id_true = np.array([lab2id[v] for v in y], dtype=int)

    Z = feat_model.predict(X, verbose=0)
    F = np.concatenate([Z, Aux], axis=1)

    y_pred_ids = rf.predict(F)
    rep = classification_report(y_id_true, y_pred_ids, labels=rf.classes_, target_names=[id2lab[i] for i in rf.classes_], digits=4, zero_division=0)
    print(rep)
    open(os.path.join(runp["rf_eval_dir"], "classification_report.txt"),"w",encoding="utf-8").write(rep)

    y_prob = rf.predict_proba(F)
    id_to_pos = {cid:i for i,cid in enumerate(rf.classes_)}
    out = pd.DataFrame({"true_label":[id2lab[i] for i in y_id_true], "pred_label":[id2lab[i] for i in y_pred_ids]})
    for cid in rf.classes_:
        out[f"prob_{id2lab[cid]}"] = y_prob[:, id_to_pos[cid]]
    out.to_csv(os.path.join(runp["rf_eval_dir"], "predictions_window_level.csv"), index=False)
    print("Saved RF eval artifacts to", runp["rf_eval_dir"])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
