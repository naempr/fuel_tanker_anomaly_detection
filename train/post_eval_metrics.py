# train/post_eval_metrics.py
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)

plt.rcParams["figure.dpi"] = 180

def load_preds(pred_csv):
    df = pd.read_csv(pred_csv)
    classes = [c.replace("prob_", "") for c in df.columns if c.startswith("prob_")]
    if not classes:
        classes = sorted(pd.unique(pd.concat([df["true_label"], df["pred_label"]]).astype(str)))
    return df, classes

def _annotated_green_cm(mat, classes, out_path, title, vmin=0.0, vmax=None, fmt="d"):
    H, W = mat.shape
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(mat, cmap="Greens", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    vmax = mat.max() if vmax is None else vmax
    thr = vmin + 0.55 * (vmax - vmin)

    for (i, j), v in np.ndenumerate(mat):
        txt = f"{int(v)}" if fmt == "d" else f"{v:.2f}"
        color = "white" if v >= thr else "black"
        weight = "bold" if i == j else "normal"
        ax.text(j, i, txt, ha="center", va="center", color=color, fontweight=weight, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def compute_curves_and_metrics(df, classes, out_dir):
    y_true_labels = df["true_label"].astype(str).values
    y_pred_labels = df["pred_label"].astype(str).values
    rows = [{
        "class": "_overall_",
        "accuracy": accuracy_score(y_true_labels, y_pred_labels),
        "f1_macro": f1_score(y_true_labels, y_pred_labels, average="macro"),
        "f1_weighted": f1_score(y_true_labels, y_pred_labels, average="weighted"),
        "precision_macro": precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0),
    }]

    for cls in classes:
        bin_true = (y_true_labels == cls).astype(int)
        col = f"prob_{cls}"
        if col not in df.columns: 
            continue
        score = df[col].values

        p, r, _ = precision_recall_curve(bin_true, score)
        pr_auc = auc(r, p) if (len(r) > 1 and len(p) > 1) else float("nan")
        fig = plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR — {cls} (AUC={pr_auc:.4f})")
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"pr_curve_{cls}.png")); plt.close(fig)

        fpr, tpr, _ = roc_curve(bin_true, score)
        roc_auc = auc(fpr, tpr) if (len(fpr) > 1 and len(tpr) > 1) else float("nan")
        fig = plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC — {cls} (AUC={roc_auc:.4f})")
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"roc_curve_{cls}.png")); plt.close(fig)

        rows.append({"class": cls, "pr_auc": pr_auc, "roc_auc": roc_auc})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

def process_pair(eval_dir_seq, eval_dir_rf):
    """برای یک سناریو، هر دو مدل را با یک مقیاس رنگی مشترک رسم می‌کند."""
    # بارگذاری
    df_seq, classes_seq = load_preds(os.path.join(eval_dir_seq, "predictions_window_level.csv"))
    df_rf,  classes_rf  = load_preds(os.path.join(eval_dir_rf,  "predictions_window_level.csv"))
    # اطمینان از ترتیب کلاس یکسان
    classes = [c for c in classes_seq if c in classes_rf]

    # ماتریس‌ها
    y_true_s = df_seq["true_label"].astype(str).values
    y_pred_s = df_seq["pred_label"].astype(str).values
    y_true_r = df_rf["true_label"].astype(str).values
    y_pred_r = df_rf["pred_label"].astype(str).values
    cm_seq = confusion_matrix(y_true_s, y_pred_s, labels=classes)
    cm_rf  = confusion_matrix(y_true_r, y_pred_r, labels=classes)

    # vmax مشترک بین دو مدل (برای RAW). برای normalized ثابت 0..1 است.
    vmax_raw = int(max(cm_seq.max(), cm_rf.max(), 1))

    # رسم هر دو با یک مقیاس
    _annotated_green_cm(cm_seq, classes, os.path.join(eval_dir_seq, "confusion_matrix.png"),
                        "Confusion Matrix — Seq-Attn", vmin=0, vmax=vmax_raw, fmt="d")
    _annotated_green_cm(cm_rf, classes, os.path.join(eval_dir_rf, "confusion_matrix.png"),
                        "Confusion Matrix — LSTM+RF", vmin=0, vmax=vmax_raw, fmt="d")

    # normalized
    cmn_seq = cm_seq.astype(float) / np.maximum(cm_seq.sum(axis=1, keepdims=True), 1.0)
    cmn_rf  = cm_rf.astype(float)  / np.maximum(cm_rf.sum(axis=1,  keepdims=True), 1.0)
    _annotated_green_cm(cmn_seq, classes, os.path.join(eval_dir_seq, "confusion_matrix_normalized.png"),
                        "Normalized Confusion Matrix — Seq-Attn", vmin=0.0, vmax=1.0, fmt=".2f")
    _annotated_green_cm(cmn_rf, classes, os.path.join(eval_dir_rf, "confusion_matrix_normalized.png"),
                        "Normalized Confusion Matrix — LSTM+RF", vmin=0.0, vmax=1.0, fmt=".2f")

    # منحنی‌ها و خلاصه متریک‌ها
    compute_curves_and_metrics(df_seq, classes, eval_dir_seq)
    compute_curves_and_metrics(df_rf,  classes, eval_dir_rf)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="artifacts/<scenario>/<run_tag>")
    a = p.parse_args()

    eval_seq = os.path.join(a.run_dir, "eval")
    eval_rf  = os.path.join(a.run_dir, "rf_eval")

    if os.path.isdir(eval_seq) and os.path.isdir(eval_rf):
        process_pair(eval_seq, eval_rf)
        print("[OK] wrote PR/ROC & (shared-scale) confusions for both models.")
    else:
        # اگر فقط یکی وجود داشت، همان قبلی را هم پشتیبانی می‌کنیم
        ok = False
        for sub in ["eval", "rf_eval"]:
            d = os.path.join(a.run_dir, sub)
            pred_csv = os.path.join(d, "predictions_window_level.csv")
            if os.path.isdir(d) and os.path.exists(pred_csv):
                df, classes = load_preds(pred_csv)
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(df["true_label"].astype(str), df["pred_label"].astype(str), labels=classes)
                vmax_raw = int(max(cm.max(), 1))
                _annotated_green_cm(cm, classes, os.path.join(d, "confusion_matrix.png"),
                                    f"Confusion Matrix — {sub}", vmin=0, vmax=vmax_raw, fmt="d")
                cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
                _annotated_green_cm(cmn, classes, os.path.join(d, "confusion_matrix_normalized.png"),
                                    f"Normalized Confusion Matrix — {sub}", vmin=0.0, vmax=1.0, fmt=".2f")
                compute_curves_and_metrics(df, classes, d)
                ok = True
        if not ok:
            raise SystemExit("No eval folders with predictions found.")

if __name__ == "__main__":
    main()
