

```markdown
# Fuel Tanker E2E — Scenario-tagged & Run-tagged

This project implements an **end-to-end pipeline** for simulation, feature extraction, model training, and evaluation in the task of detecting suspicious fuel tanker trips (**theft**, **delivery**, **normal**).

---

## Project Structure

```

fuel\_tanker\_e2e/
│
├── configs/                # YAML config files (base project + scenarios)
├── data/                   # Simulated data (train/test CSVs)
├── src/                    # Core modules (simulate, utils, windowing, models, attention)
├── train/                  # Training & evaluation scripts
│   ├── make\_data.py        # Generate simulated trips
│   ├── train\_ae.py         # Train Autoencoder on time windows
│   ├── compute\_ae\_feats.py # Extract AE features for each window
│   ├── train\_seq\_attn.py   # Train Seq+Attention model
│   ├── evaluate\_seq\_attn.py# Evaluate Seq+Attention model
│   ├── train\_lstm\_encoder.py # Train LSTM encoder for RF
│   ├── extract\_embeddings\_and\_train\_rf.py # Train RandomForest on LSTM embeddings
│   ├── evaluate\_lstm\_rf.py # Evaluate LSTM+RF
│   ├── post\_eval\_metrics.py# Compute additional metrics (PR-AUC, ROC-AUC, …)
│   ├── run\_all.py          # Run full pipeline for a given scenario
│   └── collect\_runs.py     # Aggregate results across multiple runs
└── artifacts/              # Saved models, reports, and evaluation outputs

````

---

## Algorithms

### 1. Seq-Attn Model (end-to-end baseline)

* **Input**: time-series sensor features + auxiliary features.  
* **Architecture**: LSTM → Attention → Dense layers.  
* **Output**: trip class (normal / delivery / theft).  
* **Strength**: directly learns temporal dependencies, end-to-end.

### 2. LSTM + RandomForest (comparison baseline)

* LSTM encoder compresses each trip into an embedding.  
* RandomForest classifier is trained on embeddings.  
* **Strength**: simpler, interpretable, strong traditional baseline.

---

## Scenarios

### Baseline
* Default theft probability (e.g., 0.2).  
* Test distribution same as training.

### Scenario A: Class Imbalance
* Reduced theft probability (e.g., 0.05).  
* Tests robustness to **severe imbalance**.  
* Key metrics: **Precision, Recall, F1 for theft**, **PR-AUC**.

### Scenario B: Distribution Shift
* Test trips have shifted speeds (e.g., -20 km/h or +40 km/h).  
* Tests robustness to **distribution shift**.  
* Key metrics: drop in accuracy and F1 compared to baseline.

---

## Evaluation Metrics

Stored under `artifacts/<scenario>/<run_tag>/eval/`:

1. **Accuracy** (overall correctness).  
2. **Precision, Recall, F1-score** per class (via `classification_report.txt`).  
   → Critical for minority class **theft**.  
3. **Confusion Matrix** (raw + normalized) as PNG images.  
4. **Precision–Recall Curves (PR curves) + AUC** for each class.  
   → Especially informative for imbalanced data.  
5. **ROC Curves + AUC** for each class.  

Additionally, `metrics_summary.csv` aggregates the main results.

---

## How to Run

### Baseline
```bash
PYTHONPATH=. python train/run_all.py --scenario baseline --seed 42 --n_trips 120
````

### Scenario A (class imbalance)

```bash
PYTHONPATH=. python train/run_all.py --scenario scenA_imbalance \
  --seed 42 --n_trips 120 --override_theft_prob 0.05
```

### Scenario B (distribution shift)

```bash
PYTHONPATH=. python train/run_all.py --scenario scenB_speedshift \
  --seed 42 --n_trips 120 --override_test_speed_kmh 40
```

### Aggregate results across runs

```bash
PYTHONPATH=. python train/collect_runs.py \
  --artifacts_dir artifacts --out_csv runs/summary_all_runs.csv
```

### Post-evaluation metrics for a run

```bash
PYTHONPATH=. python train/post_eval_metrics.py --run_dir artifacts/baseline/<run_tag>
```

```

---

👉 Do you want me to package this `README.md` into a proper zip so you can just drop it into your project folder?
```
