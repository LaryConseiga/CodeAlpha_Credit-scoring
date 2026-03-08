"""Train, compare and save the credit scoring models.

Experiment design
-----------------
The original benchmark CSV (16,714 rows) is an exact, class-balanced subsample of Kaggle's
"Give Me Some Credit" (150,000 rows). To measure whether the extra Kaggle data helps, GMSC is
split once into train/test (80/20, stratified). Every model is then trained twice:

* **benchmark**: only the benchmark rows that fall in the GMSC training split (~13k rows);
* **full**: the whole GMSC training split (~120k rows);

and both variants are scored on the *same* held-out GMSC test set.

Usage: python -m src.train
"""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.data import RAW_FEATURES, ROOT, TARGET, benchmark_ids, load_benchmark, load_gmsc
from src.features import FEATURE_NAMES
from src.models import SEED, fit_nn, make_models, predict_nn

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
NN_NAME = "Neural Network"


def split_data():
    gmsc = load_gmsc()
    bench_ids = benchmark_ids(gmsc, load_benchmark())
    train, test = train_test_split(gmsc, test_size=0.2, stratify=gmsc[TARGET], random_state=SEED)
    splits = {
        "benchmark": train.loc[train.index.isin(bench_ids)],
        "full": train,
    }
    return splits, test, bench_ids


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def compare(splits, test):
    """Train every model on each training set and score it on the shared test set."""
    X_test, y_test = test[RAW_FEATURES], test[TARGET]
    rows, test_proba, fitted = [], {}, {}
    for data_name, train in splits.items():
        X, y = train[RAW_FEATURES], train[TARGET]
        for name, model in make_models().items():
            t0 = time.time()
            model.fit(X, y)
            proba = model.predict_proba(X_test)[:, 1]
            rows.append(_score(name, data_name, len(train), y_test, proba, time.time() - t0))
            if data_name == "full":
                test_proba[name], fitted[name] = proba, model
        t0 = time.time()
        pre, nn, _ = fit_nn(X, y)
        proba = predict_nn(pre, nn, X_test)
        rows.append(_score(NN_NAME, data_name, len(train), y_test, proba, time.time() - t0))
        if data_name == "full":
            test_proba[NN_NAME], fitted[NN_NAME] = proba, (pre, nn)
    return pd.DataFrame(rows), test_proba, fitted


def _score(model, data, n_train, y_true, proba, seconds):
    row = {
        "model": model, "training_data": data, "n_train": n_train,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "fit_seconds": round(seconds, 1),
    }
    print(f"  {model:<20} [{data:<9}] ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}")
    return row


def cross_validate_best(name, train):
    """5-fold out-of-fold predictions on the full training set, used to pick the threshold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(
        make_models()[name], train[RAW_FEATURES], train[TARGET], cv=cv, method="predict_proba"
    )[:, 1]
    return oof


def plot_all(results, test, test_proba, best_name, best_model, threshold):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_test = test[TARGET]
    sns.set_theme(style="whitegrid")

    # 1. Benchmark-only vs full Kaggle training data (dot plot: one line per model).
    wide = results.pivot(index="model", columns="training_data", values="roc_auc").sort_values("full")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(wide))
    ax.hlines(y, wide["benchmark"], wide["full"], color="#cbd2d9", lw=3, zorder=1)
    ax.scatter(wide["benchmark"], y, s=80, color="#9aa5b1", zorder=2, label="Benchmark only (~13k rows)")
    ax.scatter(wide["full"], y, s=80, color="#2b6cb0", zorder=3, label="Benchmark + Kaggle (~120k rows)")
    for i, (b, f) in enumerate(zip(wide["benchmark"], wide["full"])):
        ax.annotate(f"{f:.3f} (+{f - b:.3f})", (f, i), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#2b6cb0")
    ax.set_yticks(y, wide.index)
    ax.set_xlim(wide.values.min() - 0.005, wide.values.max() + 0.012)
    ax.set_xlabel("ROC-AUC on the shared held-out test set (30,000 rows)")
    ax.set_title("Effect of adding the Kaggle data")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_comparison.png", dpi=120)
    plt.close(fig)

    # 2. ROC and precision-recall curves (models trained on full data).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for name, proba in test_proba.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax1.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, proba):.3f})")
        p, r, _ = precision_recall_curve(y_test, proba)
        ax2.plot(r, p, label=f"{name} ({average_precision_score(y_test, proba):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (AUC)")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y_test.mean():.3f})")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (average precision)")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    # 3. Confusion matrix of the selected model at the chosen threshold.
    y_pred = (test_proba[best_name] >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No default", "Default"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{best_name} - threshold {threshold:.2f}")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # 4. Permutation importance of the raw inputs (model-agnostic).
    sample = test.sample(n=min(20_000, len(test)), random_state=SEED)
    imp = permutation_importance(
        best_model, sample[RAW_FEATURES], sample[TARGET], scoring="roc_auc",
        n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp = pd.Series(imp.importances_mean, index=RAW_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#2b6cb0")
    ax.set(xlabel="Mean drop in ROC-AUC when the feature is shuffled", title=f"Feature importance - {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    return imp


def export_nn(pre, nn):
    import tensorflow as tf

    nn_dir = MODELS_DIR / "neural_network"
    nn_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, nn_dir / "preprocessor.joblib")
    nn.save(nn_dir / "credit_nn.keras")
    tflite = tf.lite.TFLiteConverter.from_keras_model(nn).convert()
    (nn_dir / "credit_nn.tflite").write_bytes(tflite)


def main():
    print("Loading data ...")
    splits, test, bench_ids = split_data()
    for k, v in splits.items():
        print(f"  train[{k}]: {len(v):,} rows, default rate {v[TARGET].mean():.1%}")
    print(f"  test: {len(test):,} rows, default rate {test[TARGET].mean():.1%}")

    print("\nComparing models (test set scores) ...")
    results, test_proba, fitted = compare(splits, test)

    full = results[results.training_data == "full"].set_index("model")
    best_name = full["roc_auc"].idxmax()
    if best_name == NN_NAME:
        # Keep a scikit-learn model as the main artefact; the NN is exported separately anyway.
        best_name = full.drop(index=NN_NAME)["roc_auc"].idxmax()
    print(f"\nSelected model: {best_name}")

    print("Choosing the decision threshold from 5-fold out-of-fold predictions ...")
    oof = cross_validate_best(best_name, splits["full"])
    threshold = best_f1_threshold(splits["full"][TARGET], oof)
    print(f"  threshold (max F1): {threshold:.3f}  | OOF ROC-AUC: {roc_auc_score(splits['full'][TARGET], oof):.4f}")

    best_model = fitted[best_name]
    y_pred = (test_proba[best_name] >= threshold).astype(int)
    report = classification_report(test[TARGET], y_pred, target_names=["No default", "Default"], output_dict=True)
    print(classification_report(test[TARGET], y_pred, target_names=["No default", "Default"]))

    print("Saving figures and models ...")
    importance = plot_all(results, test, test_proba, best_name, best_model, threshold)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "credit_model.joblib", compress=3)
    export_nn(*fitted[NN_NAME])

    # Threshold for the NN from its (in-sample) training predictions: never tune on the test set.
    nn_train_proba = predict_nn(*fitted[NN_NAME], splits["full"][RAW_FEATURES])
    nn_threshold = best_f1_threshold(splits["full"][TARGET], nn_train_proba)
    # Score on the benchmark rows of the test set only, to compare with the original balanced notebook.
    in_bench = test.index.isin(bench_ids)
    bench_auc = roc_auc_score(test.loc[in_bench, TARGET], test_proba[best_name][in_bench])
    print(f"ROC-AUC on the {in_bench.sum():,} benchmark rows of the test set: {bench_auc:.4f}")
    metadata = {
        "model": best_name,
        "threshold": threshold,
        "nn_threshold": nn_threshold,
        "nn_test_roc_auc": float(full.loc[NN_NAME, "roc_auc"]),
        "raw_features": RAW_FEATURES,
        "engineered_features": FEATURE_NAMES,
        "train_rows": len(splits["full"]),
        "test_rows": len(test),
        "test_default_rate": float(test[TARGET].mean()),
        "test_roc_auc": float(full.loc[best_name, "roc_auc"]),
        "test_pr_auc": float(full.loc[best_name, "pr_auc"]),
        "benchmark_test_rows": int(in_bench.sum()),
        "benchmark_test_roc_auc": float(bench_auc),
        "sklearn_version": sklearn.__version__,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    REPORTS_DIR.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({
        "selected_model": best_name,
        "threshold": threshold,
        "classification_report_test": report,
        "permutation_importance": importance.sort_values(ascending=False).round(4).to_dict(),
    }, indent=2))
    print("\n" + results.round(4).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
