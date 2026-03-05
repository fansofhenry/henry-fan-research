"""
run_benchmark.py
----------------
Main experiment runner.

Usage:
    python run_benchmark.py                        # full benchmark
    python run_benchmark.py --datasets iris wine   # subset of datasets
    python run_benchmark.py --models logistic_regression svm
    python run_benchmark.py --cv 10 --seed 42
    python run_benchmark.py --fast                 # skip MNIST for speed

Outputs:
    results/results.csv        — per-fold metrics
    results/summary.csv        — mean ± std per dataset × model
"""

import argparse
import time
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data.loaders    import load_dataset, ALL_DATASETS
from models.registry import get_model,   ALL_MODELS


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(model, X_train, X_test, y_train, y_test, n_classes):
    """
    Fit model, return dict of metrics + timing.
    """
    # Train
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # Inference
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    avg = "binary" if n_classes == 2 else "macro"
    f1  = f1_score(y_test, y_pred, average=avg, zero_division=0)

    # AUC-ROC (needs probabilities)
    auc = float("nan")
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=np.unique(y_test))
                auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        except Exception:
            pass

    return dict(
        accuracy=round(acc, 4),
        f1=round(f1, 4),
        auc_roc=round(auc, 4) if not np.isnan(auc) else None,
        train_time_s=round(train_time, 4),
        infer_time_ms=round(infer_time * 1000, 4),
    )


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(dataset_name: str, model_name: str, n_folds: int, seed: int):
    """
    Run k-fold cross-validation for one (dataset, model) pair.
    Returns a list of per-fold result dicts.
    """
    print(f"  [{dataset_name:15s} × {model_name:22s}]", end=" ", flush=True)

    X, y, meta = load_dataset(dataset_name)
    n_classes   = meta["n_classes"]
    skf         = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = get_model(model_name)
        metrics = compute_metrics(
            model,
            X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            n_classes,
        )
        fold_results.append({
            "dataset":    meta["name"],
            "n_samples":  meta["n_samples"],
            "n_features": meta["n_features"],
            "n_classes":  n_classes,
            "task":       meta["task"],
            "model":      model.model_name,
            "fold":       fold_idx + 1,
            **metrics,
        })

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    print(f"acc={mean_acc:.3f}")
    return fold_results


# ── Full benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    datasets: list,
    models:   list,
    n_folds:  int  = 5,
    seed:     int  = 42,
    out_dir:  Path = Path("results"),
):
    out_dir.mkdir(exist_ok=True)

    all_rows = []
    total    = len(datasets) * len(models)
    done     = 0

    for ds_name in datasets:
        for model_name in models:
            done += 1
            print(f"[{done:2d}/{total}]", end=" ")
            rows = run_experiment(ds_name, model_name, n_folds, seed)
            all_rows.extend(rows)

    # ── Per-fold CSV ──────────────────────────────────────────────────────────
    df_raw = pd.DataFrame(all_rows)
    raw_path = out_dir / "results.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"\n✓ Per-fold results  → {raw_path}")

    # ── Summary CSV (mean ± std) ──────────────────────────────────────────────
    numeric_cols = ["accuracy", "f1", "auc_roc", "train_time_s", "infer_time_ms"]
    group_cols   = ["dataset", "n_samples", "n_features", "n_classes", "task", "model"]

    agg = df_raw.groupby(group_cols)[numeric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Readable "mean±std" columns
    for col in numeric_cols:
        m, s = f"{col}_mean", f"{col}_std"
        if m in agg.columns:
            agg[f"{col}_summary"] = agg.apply(
                lambda r: f"{r[m]:.4f} ± {r[s]:.4f}" if pd.notna(r[m]) else "—",
                axis=1,
            )

    summ_path = out_dir / "summary.csv"
    agg.to_csv(summ_path, index=False)
    print(f"✓ Summary results   → {summ_path}")

    # ── Pretty console table ──────────────────────────────────────────────────
    print_table(agg)

    return df_raw, agg


def print_table(agg: pd.DataFrame):
    print("\n" + "═" * 80)
    print(f"{'Dataset':<18} {'Model':<24} {'Accuracy':>12} {'F1':>10} {'Train(s)':>10}")
    print("─" * 80)

    for _, row in agg.iterrows():
        acc  = row.get("accuracy_mean", float("nan"))
        f1   = row.get("f1_mean",       float("nan"))
        trt  = row.get("train_time_s_mean", float("nan"))
        print(
            f"{row['dataset']:<18} {row['model']:<24} "
            f"{acc:>12.4f} {f1:>10.4f} {trt:>10.4f}s"
        )
    print("═" * 80)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ML Small-Dataset Benchmark")
    p.add_argument(
        "--datasets", nargs="+",
        default=list(ALL_DATASETS.keys()),
        choices=list(ALL_DATASETS.keys()),
        help="Datasets to benchmark",
    )
    p.add_argument(
        "--models", nargs="+",
        default=list(ALL_MODELS.keys()),
        choices=list(ALL_MODELS.keys()),
        help="Models to benchmark",
    )
    p.add_argument("--cv",    type=int, default=5,  help="Number of CV folds")
    p.add_argument("--seed",  type=int, default=42, help="Random seed")
    p.add_argument("--fast",  action="store_true",  help="Skip MNIST")
    p.add_argument(
        "--out", type=Path, default=Path("results"),
        help="Output directory for CSVs",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    datasets = args.datasets
    if args.fast and "mnist" in datasets:
        datasets = [d for d in datasets if d != "mnist"]
        print("⚡ Fast mode: skipping MNIST")

    print(f"\n{'ML Small-Dataset Benchmark':^80}")
    print(f"{'Datasets: ' + ', '.join(datasets):^80}")
    print(f"{'Models:   ' + ', '.join(args.models):^80}")
    print(f"{'CV folds: ' + str(args.cv) + '  |  Seed: ' + str(args.seed):^80}\n")

    run_benchmark(
        datasets=datasets,
        models=args.models,
        n_folds=args.cv,
        seed=args.seed,
        out_dir=args.out,
    )
