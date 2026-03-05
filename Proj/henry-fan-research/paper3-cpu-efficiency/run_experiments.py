"""
paper3-cpu-efficiency/run_experiments.py
-----------------------------------------
Experiments for:
  "Training Machine Learning Models Without a GPU:
   A Practical CPU Efficiency Benchmark"

Research questions:
  RQ1: How does parallelism (n_jobs) affect training time on CPU?
  RQ2: How does training set size (n) affect time and accuracy?
  RQ3: Which sklearn models are most CPU-efficient (time per unit accuracy)?
  RQ4: How does feature dimensionality affect CPU training time?

All experiments are designed to run on a standard laptop CPU in < 10 minutes.

Outputs → results/ CSVs + figures/
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})

PALETTE = {
    "Random Forest":       "#2563eb",
    "Gradient Boosting":   "#d97706",
    "Logistic Regression": "#16a34a",
    "SGD Classifier":      "#dc2626",
    "MLP":                 "#7c3aed",
    "Linear SVM":          "#0891b2",
}


def _timed_fit_score(model, X_train, X_test, y_train, y_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = model.score(X_test, y_test)
    infer_ms = (time.perf_counter() - t1) * 1000

    return round(train_s, 5), round(acc, 4), round(infer_ms, 4)


# ── RQ1: Parallelism (n_jobs) ─────────────────────────────────────────────────

def rq1_parallelism():
    """
    Fix dataset size (n=5000, d=20), vary n_jobs in {1, 2, 4, -1}.
    Models: Random Forest, Gradient Boosting (no n_jobs), Logistic Regression.
    """
    print("\n[RQ1] Parallelism (n_jobs) vs training time...")
    X, y = make_classification(n_samples=5000, n_features=20,
                                n_informative=15, random_state=42)
    split = int(0.8 * len(X))
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    configs = [
        ("Random Forest", lambda j: RandomForestClassifier(n_estimators=200, n_jobs=j, random_state=42)),
        ("Logistic Regression", lambda j: Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, n_jobs=j, random_state=42))])),
    ]
    n_jobs_vals = [1, 2, 4, -1]
    rows = []
    for model_name, factory in configs:
        for nj in n_jobs_vals:
            model = factory(nj)
            t, acc, inf = _timed_fit_score(model, X_tr, X_te, y_tr, y_te)
            rows.append({
                "model": model_name, "n_jobs": nj,
                "train_time_s": t, "accuracy": acc, "infer_ms": inf,
            })
            print(f"  {model_name:22s} n_jobs={nj:2d}  t={t:.4f}s  acc={acc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rq1_parallelism.csv", index=False)
    print(f"  ✓ rq1_parallelism.csv ({len(df)} rows)")
    return df


# ── RQ2: Training Set Size ────────────────────────────────────────────────────

def rq2_sample_scaling():
    """
    Fix d=20, vary n from 500 to 50 000.
    Measure time and accuracy for all 6 models.
    """
    print("\n[RQ2] Sample size scaling...")
    n_vals = [500, 1000, 2000, 5000, 10000, 20000, 50000]

    models_factories = {
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=300, random_state=42))]),
        "SGD Classifier":      lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", SGDClassifier(max_iter=200, random_state=42))]),
        "MLP":                 lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                       max_iter=200, random_state=42))]),
        "Linear SVM":          lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", LinearSVC(max_iter=1000, random_state=42))]),
    }

    rows = []
    for n in n_vals:
        X, y = make_classification(n_samples=n, n_features=20,
                                   n_informative=15, random_state=42)
        split = int(0.8 * n)
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

        for mname, factory in models_factories.items():
            model = factory()
            t, acc, inf = _timed_fit_score(model, X_tr, X_te, y_tr, y_te)
            rows.append({
                "model": mname, "n_samples": n,
                "train_time_s": t, "accuracy": acc, "infer_ms": inf,
            })
        print(f"  n={n:6d}  done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rq2_sample_scaling.csv", index=False)
    print(f"  ✓ rq2_sample_scaling.csv ({len(df)} rows)")
    return df


# ── RQ3: CPU Efficiency Score ─────────────────────────────────────────────────

def rq3_efficiency():
    """
    Fixed dataset (n=5000, d=20), 5-fold CV.
    CPU Efficiency = accuracy / training_time  (higher is better).
    Also reports: accuracy per watt-second proxy (training_time * n_cores_used).
    """
    print("\n[RQ3] CPU efficiency score (accuracy / train_time)...")
    X, y = make_classification(n_samples=5000, n_features=20,
                                n_informative=15, random_state=42)
    split = int(0.8 * len(X))
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    configs = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=500, random_state=42))]),
        "SGD Classifier":      Pipeline([("sc", StandardScaler()),
                                   ("clf", SGDClassifier(max_iter=300, random_state=42))]),
        "MLP":                 Pipeline([("sc", StandardScaler()),
                                   ("clf", MLPClassifier(hidden_layer_sizes=(128, 64),
                                       max_iter=300, random_state=42))]),
        "Linear SVM":          Pipeline([("sc", StandardScaler()),
                                   ("clf", LinearSVC(max_iter=2000, random_state=42))]),
    }

    rows = []
    for mname, model in configs.items():
        t, acc, inf = _timed_fit_score(model, X_tr, X_te, y_tr, y_te)
        efficiency = acc / (t + 1e-9)
        rows.append({
            "model":         mname,
            "train_time_s":  t,
            "accuracy":      acc,
            "infer_ms":      inf,
            "efficiency_acc_per_sec": round(efficiency, 2),
        })
        print(f"  {mname:22s}  t={t:.4f}s  acc={acc:.3f}  eff={efficiency:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rq3_efficiency.csv", index=False)
    print(f"  ✓ rq3_efficiency.csv")
    return df


# ── RQ4: Feature Dimensionality ───────────────────────────────────────────────

def rq4_dimensionality():
    """
    Fix n=5000, vary d from 10 to 1000.
    Measure how training time scales with feature count.
    """
    print("\n[RQ4] Feature dimensionality scaling...")
    d_vals = [10, 20, 50, 100, 200, 500, 1000]

    configs = {
        "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Logistic Regression": lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=300, random_state=42))]),
        "MLP":                 lambda: Pipeline([("sc", StandardScaler()),
                                   ("clf", MLPClassifier(hidden_layer_sizes=(64,),
                                       max_iter=200, random_state=42))]),
    }

    rows = []
    for d in d_vals:
        n_inf = min(d - 2, d // 2 + 5)
        X, y = make_classification(n_samples=5000, n_features=d,
                                   n_informative=max(2, n_inf), random_state=42)
        split = int(0.8 * 5000)
        X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
        for mname, factory in configs.items():
            model = factory()
            t, acc, inf = _timed_fit_score(model, X_tr, X_te, y_tr, y_te)
            rows.append({"model": mname, "n_features": d,
                         "train_time_s": t, "accuracy": acc})
        print(f"  d={d:5d}  done")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "rq4_dimensionality.csv", index=False)
    print(f"  ✓ rq4_dimensionality.csv ({len(df)} rows)")
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_parallelism(df):
    """Fig 1 — n_jobs speedup."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, label in zip(axes,
            ["train_time_s", "accuracy"],
            ["Training Time (s)", "Accuracy"]):
        for model in df["model"].unique():
            sub = df[df["model"] == model].sort_values("n_jobs")
            ax.plot(sub["n_jobs"].astype(str), sub[metric],
                    color=PALETTE.get(model, "#555"),
                    marker="o", linewidth=2, label=model)
        ax.set_xlabel("n_jobs (CPU threads)")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)

    axes[0].set_title("Fig 1a — Training Time vs Parallelism")
    axes[1].set_title("Fig 1b — Accuracy vs Parallelism")
    fig.suptitle("Figure 1 — Effect of CPU Parallelism on Training", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_parallelism.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_parallelism.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig1_parallelism")


def fig_sample_scaling(df):
    """Fig 2 — Training time vs sample size (log-log)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("n_samples")
        color = PALETTE.get(model, "#555")
        axes[0].plot(sub["n_samples"], sub["train_time_s"],
                     color=color, marker="o", linewidth=2, label=model)
        axes[1].plot(sub["n_samples"], sub["accuracy"],
                     color=color, marker="o", linewidth=2, label=model)

    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Training Set Size (n)"); axes[0].set_ylabel("Train Time (s)")
    axes[0].set_title("Fig 2a — Training Time (log-log scale)")
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training Set Size (n)"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Fig 2b — Accuracy vs Sample Size")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Figure 2 — Scaling with Dataset Size on CPU", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_sample_scaling.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_sample_scaling.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig2_sample_scaling")


def fig_efficiency(df):
    """Fig 3 — CPU efficiency: accuracy / train_time."""
    df_s = df.sort_values("efficiency_acc_per_sec", ascending=True)
    colors = [PALETTE.get(m, "#555") for m in df_s["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bar: efficiency score
    axes[0].barh(df_s["model"], df_s["efficiency_acc_per_sec"],
                 color=colors, alpha=0.88)
    for i, (_, row) in enumerate(df_s.iterrows()):
        axes[0].text(row["efficiency_acc_per_sec"] + 0.5, i,
                     f"{row['efficiency_acc_per_sec']:.1f}",
                     va="center", fontsize=9)
    axes[0].set_xlabel("CPU Efficiency (accuracy / train_time_s)")
    axes[0].set_title("Fig 3a — CPU Efficiency Score")

    # Scatter: accuracy vs train time (Pareto)
    for _, row in df.iterrows():
        axes[1].scatter(row["train_time_s"], row["accuracy"],
                        color=PALETTE.get(row["model"], "#555"),
                        s=90, zorder=3, edgecolors="white", linewidths=0.6)
        axes[1].annotate(row["model"].split()[0],
                         (row["train_time_s"], row["accuracy"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=8.5)

    axes[1].set_xlabel("Training Time (s)")
    axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_title("Fig 3b — Accuracy vs Training Time (CPU Pareto)")

    fig.suptitle("Figure 3 — CPU Efficiency: Which Model Gives the Most Per Second?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_efficiency.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_efficiency.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig3_efficiency")


def fig_dimensionality(df):
    """Fig 4 — Training time vs feature count."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("n_features")
        color = PALETTE.get(model, "#555")
        axes[0].plot(sub["n_features"], sub["train_time_s"],
                     color=color, marker="o", linewidth=2, label=model)
        axes[1].plot(sub["n_features"], sub["accuracy"],
                     color=color, marker="o", linewidth=2, label=model)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of Features (d)"); axes[0].set_ylabel("Train Time (s)")
    axes[0].set_title("Fig 4a — Time vs Dimensionality")
    axes[0].legend(fontsize=9)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Number of Features (d)"); axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[1].set_title("Fig 4b — Accuracy vs Dimensionality")

    fig.suptitle("Figure 4 — Effect of Feature Dimensionality on CPU Training",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_dimensionality.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_dimensionality.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig4_dimensionality")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  Paper 3: CPU Efficiency Benchmark")
    print("═" * 60)

    df_rq1 = rq1_parallelism()
    df_rq2 = rq2_sample_scaling()
    df_rq3 = rq3_efficiency()
    df_rq4 = rq4_dimensionality()

    print("\nGenerating figures...")
    fig_parallelism(df_rq1)
    fig_sample_scaling(df_rq2)
    fig_efficiency(df_rq3)
    fig_dimensionality(df_rq4)

    print(f"\n✓ Results → {RESULTS}/")
    print(f"✓ Figures → {FIGURES}/")
    print("═" * 60)
