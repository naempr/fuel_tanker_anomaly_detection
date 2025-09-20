
import os, json, yaml, time

def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path_or_file):
    base, ext = os.path.splitext(path_or_file)
    if ext:
        d = os.path.dirname(path_or_file)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    else:
        if not os.path.exists(path_or_file):
            os.makedirs(path_or_file, exist_ok=True)

def save_scaler(path, mean, std):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mean": [float(x) for x in mean], "std": [float(x) for x in std]}, f)

def load_scaler(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    import numpy as np
    mean = np.array(d["mean"], dtype=float)
    std  = np.array(d["std"], dtype=float)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std

def make_run_dirs(paths, scenario="baseline", run_tag=None):
    base = paths["artifacts_dir"]
    if run_tag is None:
        run_tag = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base, scenario, run_tag)
    d = {
        "run_dir": run_dir,
        "scaler_path": os.path.join(run_dir, "standard_scaler.json"),
        "ae_weights": os.path.join(run_dir, "ae.weights.h5"),
        "ae_model_summary": os.path.join(run_dir, "ae_model.txt"),
        "ae_feats_csv": os.path.join(run_dir, "window_ae_features.csv"),
        "attn_weights": os.path.join(run_dir, "seq_attn.weights.h5"),
        "label_mapping_json": os.path.join(run_dir, "label_mapping.json"),
        "eval_dir": os.path.join(run_dir, "eval"),
        "lstm_encoder_weights": os.path.join(run_dir, "lstm_encoder.weights.h5"),
        "rf_model_path": os.path.join(run_dir, "lstm_rf.joblib"),
        "rf_eval_dir": os.path.join(run_dir, "rf_eval"),
        "manifest_json": os.path.join(run_dir, "manifest.json")
    }
    for v in d.values():
        if isinstance(v, str) and v != run_dir:
            ensure_dir(v)
    ensure_dir(run_dir)
    m = {"scenario": scenario, "run_tag": run_tag, **d}
    with open(d["manifest_json"], "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    latest = os.path.join(base, scenario, "latest.txt")
    # ensure dir exists
    os.makedirs(os.path.dirname(latest), exist_ok=True)
    with open(latest, "w", encoding="utf-8") as f:
        f.write(run_tag)
    return d
