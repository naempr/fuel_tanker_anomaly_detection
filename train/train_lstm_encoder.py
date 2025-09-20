
import argparse, numpy as np, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from src.utils import load_config, ensure_dir, load_scaler, make_run_dirs
from src.windowing import make_windows

def build_lstm_encoder(timesteps, seq_feats, num_classes=3, rnn_units=96, embed_dim=64, dropout=0.2):
    inp = keras.Input(shape=(timesteps, seq_feats), name="seq")
    x = layers.Masking()(inp)
    x = layers.LSTM(rnn_units)(x)
    z = layers.Dense(embed_dim, activation="relu", name="embed")(x)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(num_classes, activation="softmax")(z)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

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
    Xs, ys = [], []
    for _, g in df.groupby("trip_id"):
        X, y, _, _ = make_windows(
            g, wcfg["window_sec"], wcfg["stride_sec"],
            cfg["simulation"]["sample_period_s"], wcfg["feat_cols"]
        )
        if len(X)==0: continue
        Xs.append(X); ys.append(y)
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)

    classes = sorted(np.unique(y).tolist())
    lab2id = {v:i for i,v in enumerate(classes)}
    y_id = np.array([lab2id[v] for v in y], dtype=int)

    mean, std = load_scaler(runp["scaler_path"])
    X = (X - mean) / std

    Xtr, Xva, ytr, yva = train_test_split(X, y_id, test_size=0.2, stratify=y_id, random_state=cfg["seed"])

    model = build_lstm_encoder(timesteps=X.shape[1], seq_feats=X.shape[2], num_classes=len(classes))
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(runp["lstm_encoder_weights"], save_best_only=True, monitor="val_accuracy", save_weights_only=True)
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=80, batch_size=256, verbose=2, callbacks=cb)
    print("Saved LSTM encoder weights to", runp["lstm_encoder_weights"])
    print("RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
