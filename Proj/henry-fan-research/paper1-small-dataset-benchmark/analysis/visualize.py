"""
analysis/visualize.py
---------------------
Generates all figures for the paper from results/summary.csv.

Usage:
    python analysis/visualize.py
    python analysis/visualize.py --results results/summary.csv --out figures/

Outputs (saved as both .pdf for LaTeX and .png for README):
    fig1_accuracy_heatmap.{pdf,png}
    fig2_accuracy_vs_dataset.{pdf,png}
    fig3_train_time_comparison.{pdf,png}
    fig4_accuracy_vs_traintime.{pdf,png}
    fig5_f1_comparison.{pdf,png}
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "Logistic Regression":   "#2563eb",   # blue
    "Random Forest":         "#16a34a",   # green
    "SVM (RBF)":             "#dc2626",   # red
    "Gradient Boosting":     "#d97706",   # amber
    "Neural Network (MLP)":  "#7c3aed",   # violet
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

MODEL_ORDER = list(PALETTE.keys())


def _save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  ✓ {stem}.pdf/png")
    plt.close(fig)


# ── Fig 1: Accuracy Heatmap ───────────────────────────────────────────────────

def fig_accuracy_heatmap(df: pd.DataFrame, out_dir: Path):
    pivot = df.pivot(index="model", columns="dataset", values="accuracy_mean")

    # Sort rows by MODEL_ORDER, columns by dataset size
    present_models = [m for m in MODEL_ORDER if m in pivot.index]
    pivot = pivot.reindex(present_models)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = LinearSegmentedColormap.from_list("blu", ["#dbeafe", "#1e40af"])
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Cell annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color="white" if val > 0.8 else "#1e3a5f",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Accuracy (5-fold CV)")
    ax.set_title("Figure 1 — Accuracy Heatmap: All Models × All Datasets", fontsize=13, pad=12)
    ax.grid(False)
    _save(fig, out_dir, "fig1_accuracy_heatmap")


# ── Fig 2: Grouped Bar — Accuracy per Dataset ─────────────────────────────────

def fig_accuracy_by_dataset(df: pd.DataFrame, out_dir: Path):
    datasets = df["dataset"].unique()
    models   = [m for m in MODEL_ORDER if m in df["model"].unique()]
    x        = np.arange(len(datasets))
    width    = 0.14

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(models):
        sub   = df[df["model"] == model].set_index("dataset")
        means = [sub.loc[d, "accuracy_mean"] if d in sub.index else 0.0 for d in datasets]
        stds  = [sub.loc[d, "accuracy_std"]  if d in sub.index else 0.0 for d in datasets]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model,
                      color=PALETTE[model], alpha=0.88, yerr=stds,
                      capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Mean Accuracy (5-fold CV)")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Figure 2 — Model Accuracy by Dataset", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    _save(fig, out_dir, "fig2_accuracy_by_dataset")


# ── Fig 3: Training Time ──────────────────────────────────────────────────────

def fig_train_time(df: pd.DataFrame, out_dir: Path):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    means  = [df[df["model"] == m]["train_time_s_mean"].mean() for m in models]
    stds   = [df[df["model"] == m]["train_time_s_std"].mean()  for m in models]
    colors = [PALETTE[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(models, means, xerr=stds, color=colors, alpha=0.88,
                   capsize=4, error_kw={"linewidth": 1})

    for bar, val in zip(bars, means):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}s", va="center", fontsize=10)

    ax.set_xlabel("Mean Training Time (seconds)")
    ax.set_title("Figure 3 — Average Training Time per Model (all datasets)", fontsize=13, pad=12)
    ax.invert_yaxis()
    _save(fig, out_dir, "fig3_train_time")


# ── Fig 4: Accuracy vs Training Time (scatter) ───────────────────────────────

def fig_acc_vs_time(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    for model in models:
        sub = df[df["model"] == model]
        ax.scatter(
            sub["train_time_s_mean"],
            sub["accuracy_mean"],
            s=80,
            color=PALETTE[model],
            label=model,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        # Label each point with dataset name
        for _, row in sub.iterrows():
            ax.annotate(
                row["dataset"],
                (row["train_time_s_mean"], row["accuracy_mean"]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=7.5, alpha=0.7,
            )

    ax.set_xlabel("Mean Training Time (s)")
    ax.set_ylabel("Mean Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Figure 4 — Accuracy vs Training Time: Does Deep Learning Win?", fontsize=13, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, out_dir, "fig4_acc_vs_traintime")


# ── Fig 5: F1 Score Comparison ────────────────────────────────────────────────

def fig_f1_comparison(df: pd.DataFrame, out_dir: Path):
    models   = [m for m in MODEL_ORDER if m in df["model"].unique()]
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 5),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub    = df[df["dataset"] == dataset].set_index("model")
        f1s    = [sub.loc[m, "f1_mean"]  if m in sub.index else 0.0 for m in models]
        stds   = [sub.loc[m, "f1_std"]   if m in sub.index else 0.0 for m in models]
        colors = [PALETTE[m] for m in models]

        bars = ax.bar(range(len(models)), f1s, color=colors, alpha=0.85,
                      yerr=stds, capsize=3)
        ax.set_title(dataset, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split()[0] for m in models], rotation=45,
                           ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    axes[0].set_ylabel("Mean F1 Score (macro)")
    fig.suptitle("Figure 5 — F1 Score per Dataset × Model", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_f1_comparison")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/summary.csv"))
    p.add_argument("--out",     type=Path, default=Path("figures"))
    args = p.parse_args()

    if not args.results.exists():
        print(f"✗ Results file not found: {args.results}")
        print("  Run `python run_benchmark.py` first.")
        return

    args.out.mkdir(exist_ok=True)
    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}\n")

    print("Generating figures...")
    fig_accuracy_heatmap(df, args.out)
    fig_accuracy_by_dataset(df, args.out)
    fig_train_time(df, args.out)
    fig_acc_vs_time(df, args.out)
    fig_f1_comparison(df, args.out)
    print(f"\n✓ All figures saved to {args.out}/")


if __name__ == "__main__":
    main()
