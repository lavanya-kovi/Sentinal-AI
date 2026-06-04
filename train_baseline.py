"""
Checkpoint 1 - Baseline training (no differential privacy yet)

This trains the MLP and Logistic Regression on credit risk data.
DP-SGD will be added in Phase 2 using Opacus. For now this is just
to get a working baseline and confirm our architecture and pipeline
are correct before we layer in the privacy stuff.

Usage:
    # Quick demo on synthetic data (no downloads needed):
    python train_baseline.py --demo

    # On the real Home Credit dataset:
    python train_baseline.py --data_dir data/home_credit/

Outputs go to results/:
    mlp_baseline.pt          - saved model weights
    lr_baseline.pkl          - logistic regression
    baseline_metrics.json    - all the numbers in one place
    training_history.png     - loss and AUC curves
    mlp_confusion.png        - confusion matrix on test set
    privacy_utility_stub.png - placeholder showing planned DP experiments
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import generate_synthetic_data, load_home_credit, preprocess
from models.mlp import get_model

DEVICE       = torch.device("cpu")
BATCH_SIZE   = 256
EPOCHS       = 50
LR           = 1e-3
PATIENCE     = 5
RANDOM_STATE = 42
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model, X, y, threshold=0.5, label=""):
    """Get AUC, F1, accuracy from the PyTorch model."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = model.predict_proba(X_t).cpu().numpy()
    y_prob = probs[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y, y_prob)
    f1     = f1_score(y, y_pred, average="weighted")
    f1_def = f1_score(y, y_pred, zero_division=0)
    acc    = accuracy_score(y, y_pred)
    if label:
        print(f"  [{label}] AUC={auc:.4f}  F1={f1:.4f}  F1(default)={f1_def:.4f}  Acc={acc:.4f}")
    return {"auc": auc, "f1_weighted": f1, "f1_default": f1_def, "accuracy": acc,
            "y_prob": y_prob, "y_pred": y_pred}


def evaluate_sklearn(clf, X, y, label=""):
    """Same as above but for sklearn models."""
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    auc    = roc_auc_score(y, y_prob)
    f1     = f1_score(y, y_pred, average="weighted")
    f1_def = f1_score(y, y_pred, zero_division=0)
    acc    = accuracy_score(y, y_pred)
    if label:
        print(f"  [{label}] AUC={auc:.4f}  F1={f1:.4f}  F1(default)={f1_def:.4f}  Acc={acc:.4f}")
    return {"auc": auc, "f1_weighted": f1, "f1_default": f1_def, "accuracy": acc}


def train_mlp(splits: dict) -> tuple:
    """
    Standard MLP training loop with early stopping.
    No DP here - this is the baseline we'll compare against in Phase 2.
    """
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]

    n_features = X_train.shape[1]
    model = get_model(n_features=n_features).to(DEVICE)

    # weighted loss because only ~8% of loans are defaults (class imbalance)
    class_counts  = np.bincount(y_train)
    class_weights = torch.tensor(
        [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_loss": [], "val_auc": []}
    best_val_auc = 0.0
    patience_counter = 0

    print(f"\n[Training] MLP | features={n_features} | batch={BATCH_SIZE} | max_epochs={EPOCHS}")
    print("-" * 60)

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
        scheduler.step()

        avg_loss = epoch_loss / len(y_train)
        history["train_loss"].append(avg_loss)

        val_metrics = evaluate_model(model, X_val, y_val)
        val_auc = val_metrics["auc"]
        history["val_auc"].append(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | val_auc={val_auc:.4f} | t={elapsed:.0f}s")

        # save best and check patience
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "mlp_baseline.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    # reload the best weights before returning
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "mlp_baseline.pt"), map_location=DEVICE)
    )
    print(f"[Training] Done. Best val AUC: {best_val_auc:.4f}")
    return model, history


def train_logistic_regression(splits: dict):
    """
    Simple logistic regression as a linear comparison point.
    We'll also train a DP version (diffprivlib) in Phase 2 to see how
    much accuracy linear models lose compared to MLP under the same epsilon.
    """
    print("\n[Training] Logistic Regression ...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
        C=1.0,
    )
    clf.fit(splits["X_train"], splits["y_train"])
    joblib.dump(clf, os.path.join(RESULTS_DIR, "lr_baseline.pkl"))
    # also save a copy for the sklearn predict endpoint in the API
    joblib.dump(clf, os.path.join(RESULTS_DIR, "mlp_baseline_sklearn.pkl"))
    print(f"  Saved to {RESULTS_DIR}/lr_baseline.pkl")
    return clf


def plot_training_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("SentinelAI - MLP Baseline Training", fontsize=13, fontweight="bold")

    ax1.plot(history["train_loss"], color="#0055aa", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(history["val_auc"], color="#009944", linewidth=2)
    ax2.axhline(y=0.76, color="#cc3300", linestyle="--", label="Target AUC (0.76)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC-ROC")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_dp_stub():
    """
    Placeholder for the privacy-utility curve we'll build in Phase 2.
    The estimated values here come from published DP-SGD papers on
    similar tabular datasets (Naresh & Ayyappa 2026, Rahman et al. 2018).
    We'll replace these with our actual measured values.
    """
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
    estimated_auc = [0.62, 0.66, 0.69, 0.72, 0.73, 0.76]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(epsilons)), estimated_auc, "o--",
            color="#0055aa", linewidth=2, markersize=8, label="Estimated from literature")
    ax.axhline(y=0.76, color="#cc3300", linestyle=":", linewidth=1.5, label="Baseline (no DP): 0.76")
    ax.axhline(y=0.70, color="#ff8800", linestyle=":", linewidth=1.5, label="Min acceptable AUC: 0.70")

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(["e=0.5", "e=1.0", "e=2.0", "e=4.0", "e=8.0", "e=inf\n(baseline)"])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Privacy-Utility Trade-off (Estimated)\n[Will be updated with real values in Phase 2]", fontsize=11)
    ax.legend(loc="lower right")
    ax.set_ylim(0.55, 0.82)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "privacy_utility_stub.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to Home Credit CSV files")
    parser.add_argument("--demo", action="store_true",
                        help="Run on synthetic data (no download needed)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    print("=" * 60)
    print("  SentinelAI - Checkpoint 1: Baseline (no DP)")
    print("=" * 60)

    if args.demo or args.data_dir is None:
        print("\n[Mode] Demo - using synthetic data (50K samples)")
        df = generate_synthetic_data(n_samples=50000)
    else:
        print(f"\n[Mode] Real data from: {args.data_dir}")
        df = load_home_credit(args.data_dir)

    print("\n[Stage 1] Preprocessing ...")
    splits = preprocess(df, output_dir=RESULTS_DIR)

    print("\n[Stage 2a] Training MLP ...")
    mlp, history = train_mlp(splits)
    plot_training_history(history)

    print("\n[Eval] MLP on all splits:")
    mlp_train = evaluate_model(mlp, splits["X_train"], splits["y_train"], label="Train")
    mlp_val   = evaluate_model(mlp, splits["X_val"],   splits["y_val"],   label="Val  ")
    mlp_test  = evaluate_model(mlp, splits["X_test"],  splits["y_test"],  label="Test ")
    plot_confusion_matrix(splits["y_test"], mlp_test["y_pred"], "MLP - Test Set", "mlp_confusion.png")

    print("\n[Stage 2b] Training Logistic Regression ...")
    lr_clf = train_logistic_regression(splits)

    print("\n[Eval] Logistic Regression:")
    lr_train = evaluate_sklearn(lr_clf, splits["X_train"], splits["y_train"], label="Train")
    lr_val   = evaluate_sklearn(lr_clf, splits["X_val"],   splits["y_val"],   label="Val  ")
    lr_test  = evaluate_sklearn(lr_clf, splits["X_test"],  splits["y_test"],  label="Test ")

    print("\n[Plotting] Privacy-utility stub ...")
    plot_dp_stub()

    metrics = {
        "checkpoint": 1,
        "mode": "demo" if (args.demo or args.data_dir is None) else "real",
        "dataset": "synthetic_50k" if (args.demo or args.data_dir is None) else "home_credit_307k",
        "mlp_baseline": {
            "train_auc": round(mlp_train["auc"], 4),
            "val_auc":   round(mlp_val["auc"],   4),
            "test_auc":  round(mlp_test["auc"],  4),
            "test_f1_weighted": round(mlp_test["f1_weighted"], 4),
            "test_f1_default":  round(mlp_test["f1_default"],  4),
            "test_accuracy":    round(mlp_test["accuracy"],    4),
        },
        "lr_baseline": {
            "train_auc": round(lr_train["auc"], 4),
            "val_auc":   round(lr_val["auc"],   4),
            "test_auc":  round(lr_test["auc"],  4),
            "test_f1_weighted": round(lr_test["f1_weighted"], 4),
            "test_f1_default":  round(lr_test["f1_default"],  4),
            "test_accuracy":    round(lr_test["accuracy"],    4),
        },
        "targets_for_real_data": {
            "mlp_auc": ">=0.76",
            "lr_auc":  ">=0.72",
            "note": "Demo on synthetic data will give different numbers. Real targets need Home Credit download."
        },
    }

    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  CHECKPOINT 1 RESULTS")
    print("=" * 60)
    print(f"\n  MLP Baseline:")
    print(f"    Test AUC:      {mlp_test['auc']:.4f}  (target on real data: >=0.76)")
    print(f"    Test F1 (wtd): {mlp_test['f1_weighted']:.4f}")
    print(f"    Test Accuracy: {mlp_test['accuracy']:.4f}")
    print(f"\n  Logistic Regression:")
    print(f"    Test AUC:      {lr_test['auc']:.4f}  (target on real data: >=0.72)")
    print(f"    Test F1 (wtd): {lr_test['f1_weighted']:.4f}")
    print(f"    Test Accuracy: {lr_test['accuracy']:.4f}")
    print(f"\n  Saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
