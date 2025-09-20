
import argparse, numpy as np, pandas as pd
from src.utils import load_config, ensure_dir

def make_split(cfg, n_trips, seed):
    from src.simulate import simulate_trip
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(n_trips):
        df = simulate_trip(cfg, rng).copy()
        df["trip_id"] = tid
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--n_trips", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override_theft_prob", type=float, default=None)
    p.add_argument("--override_test_speed_kmh", type=float, default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    sim = cfg["simulation"]
    n_train, n_test = sim["n_train_trips"], sim["n_test_trips"]
    if a.n_trips is not None:
        n_train = int(a.n_trips * 0.8)
        n_test  = int(a.n_trips * 0.2)

    cfg_train = {**cfg}
    if a.override_theft_prob is not None:
        sim2 = {**cfg["simulation"], "theft_prob": float(a.override_theft_prob)}
        cfg_train = {**cfg, "simulation": sim2}

    train_df = make_split(cfg_train, n_train, a.seed)

    if a.override_test_speed_kmh is not None:
        cfg_test = {**cfg, "simulation": {**cfg["simulation"], "base_speed_kmh": float(a.override_test_speed_kmh)}}
        test_df = make_split(cfg_test, n_test, a.seed+1)
    else:
        test_df = make_split(cfg, n_test, a.seed+1)

    paths = cfg["paths"]
    ensure_dir(paths["train_csv"]); ensure_dir(paths["test_csv"])
    train_df.to_csv(paths["train_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)
    print(f"Wrote: {paths['train_csv']} {len(train_df)} rows")
    print(f"Wrote: {paths['test_csv']} {len(test_df)} rows")

if __name__ == "__main__":
    main()
