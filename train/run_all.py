
import argparse, subprocess
from src.utils import load_config, make_run_dirs

def sh(cmd):
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--scenario", default="baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_trips", type=int, default=120)
    p.add_argument("--override_theft_prob", type=float, default=None)
    p.add_argument("--override_test_speed_kmh", type=float, default=None)
    p.add_argument("--run_tag", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    runp = make_run_dirs(cfg["paths"], a.scenario, a.run_tag)
    tag = runp['run_dir'].split('/')[-1]
    print("RUN_DIR:", runp["run_dir"])

    # data
    cmd = f"PYTHONPATH=. python train/make_data.py --config {a.config} --n_trips {a.n_trips} --seed {a.seed}"
    if a.override_theft_prob is not None:
        cmd += f" --override_theft_prob {a.override_theft_prob}"
    if a.override_test_speed_kmh is not None:
        cmd += f" --override_test_speed_kmh {a.override_test_speed_kmh}"
    sh(cmd)

    # ae
    sh(f"PYTHONPATH=. python train/train_ae.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")
    sh(f"PYTHONPATH=. python train/compute_ae_feats.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")

    # seq+attn
    sh(f"PYTHONPATH=. python train/train_seq_attn.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")
    sh(f"PYTHONPATH=. python train/evaluate_seq_attn.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")

    # lstm+rf
    sh(f"PYTHONPATH=. python train/train_lstm_encoder.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")
    sh(f"PYTHONPATH=. python train/extract_embeddings_and_train_rf.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")
    sh(f"PYTHONPATH=. python train/evaluate_lstm_rf.py --config {a.config} --scenario {a.scenario} --run_tag {tag}")

    print("DONE. RUN_DIR:", runp["run_dir"])

if __name__ == "__main__":
    main()
