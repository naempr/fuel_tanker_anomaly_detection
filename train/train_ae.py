
import argparse, numpy as np, pandas as pd
from tensorflow import keras
from src.utils import load_config, ensure_dir, save_scaler, make_run_dirs
from src.windowing import make_windows
from src.models_ae import build_ae

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
    Xs = []
    for _, g in df.groupby("trip_id"):
        X, y, _, _ = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X) == 0: continue
        m = (y == "normal")
        if m.any():
            Xs.append(X[m])

    if not Xs:
        raise SystemExit("No normal windows for AE.")

    X = np.concatenate(Xs, axis=0)
    mean = X.mean(axis=(0,1))
    std  = X.std(axis=(0,1)); std = np.where(std < 1e-8, 1.0, std)
    save_scaler(runp["scaler_path"], mean, std)
    X = (X - mean) / std

    ae = build_ae(X.shape[1], X.shape[2], cfg.get("ae", {}))
    ae.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss="mae")

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(runp["ae_weights"], save_weights_only=True, save_best_only=True, monitor="val_loss")
    ]
    ae.fit(X, X, epochs=80, batch_size=64, validation_split=0.2, verbose=2, callbacks=cb)
    open(runp["ae_model_summary"],"w",encoding="utf-8").write("")
    print("Saved AE weights to", runp["ae_weights"])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
