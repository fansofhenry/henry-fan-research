"""
analysis/stats.py
-----------------
Statistical significance tests for the paper's claims.

Runs:
  1. Friedman test  — is there a significant difference across models?
  2. Wilcoxon signed-rank — pairwise comparisons (Bonferroni-corrected)
  3. Effect sizes (Cohen's d)

Usage:
    python analysis/stats.py
    python analysis/stats.py --results results/results.csv
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size."""
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-10)


def interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def run_friedman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Friedman test across all models, per dataset.
    Tests H0: all models perform equally.
    """
    rows = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        groups = [
            sub[sub["model"] == m]["accuracy"].values
            for m in sub["model"].unique()
        ]
        # Need at least 3 models and same n_folds
        min_len = min(len(g) for g in groups)
        groups  = [g[:min_len] for g in groups]
        if len(groups) < 3:
            continue
        try:
            stat, p = stats.friedmanchisquare(*groups)
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({"dataset": dataset, "friedman_stat": stat, "p_value": p,
                     "significant_p05": p < 0.05})
    return pd.DataFrame(rows)


def run_pairwise(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon signed-rank pairwise tests, Bonferroni corrected.
    """
    rows = []
    models   = df["model"].unique()
    datasets = df["dataset"].unique()
    n_pairs  = len(list(combinations(models, 2)))

    for dataset in datasets:
        sub = df[df["dataset"] == dataset]
        for m1, m2 in combinations(models, 2):
            s1 = sub[sub["model"] == m1]["accuracy"].values
            s2 = sub[sub["model"] == m2]["accuracy"].values
            min_len = min(len(s1), len(s2))
            s1, s2  = s1[:min_len], s2[:min_len]

            try:
                stat, p = stats.wilcoxon(s1, s2)
            except Exception:
                stat, p = float("nan"), float("nan")

            p_bonf = min(p * n_pairs, 1.0)
            d = cohen_d(s1, s2)

            rows.append({
                "dataset":    dataset,
                "model_a":    m1,
                "model_b":    m2,
                "mean_a":     round(np.mean(s1), 4),
                "mean_b":     round(np.mean(s2), 4),
                "wilcoxon_stat": round(stat, 4),
                "p_raw":      round(p, 4),
                "p_bonferroni": round(p_bonf, 4),
                "significant":  p_bonf < alpha,
                "cohen_d":    round(d, 4),
                "effect_size": interpret_d(d),
            })

    return pd.DataFrame(rows)


def print_summary(friedman: pd.DataFrame, pairwise: pd.DataFrame):
    print("\n" + "═" * 70)
    print("FRIEDMAN TEST — Is there a significant difference across models?")
    print("─" * 70)
    print(f"{'Dataset':<20} {'Stat':>8} {'p-value':>10} {'Sig?':>8}")
    print("─" * 70)
    for _, r in friedman.iterrows():
        sig = "✓ YES" if r["significant_p05"] else "✗ no"
        print(f"{r['dataset']:<20} {r['friedman_stat']:>8.3f} {r['p_value']:>10.4f} {sig:>8}")

    print("\n" + "═" * 70)
    print("PAIRWISE WILCOXON (Bonferroni-corrected) — Significant pairs only")
    print("─" * 70)
    sig = pairwise[pairwise["significant"]]
    if sig.empty:
        print("No significant pairwise differences found after Bonferroni correction.")
    else:
        cols = ["dataset", "model_a", "model_b", "mean_a", "mean_b", "p_bonferroni", "effect_size"]
        print(sig[cols].to_string(index=False))
    print("═" * 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/results.csv"))
    p.add_argument("--out",     type=Path, default=Path("results"))
    args = p.parse_args()

    if not args.results.exists():
        print(f"✗ File not found: {args.results}. Run run_benchmark.py first.")
        return

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    friedman = run_friedman(df)
    pairwise = run_pairwise(df)

    friedman.to_csv(args.out / "stats_friedman.csv",  index=False)
    pairwise.to_csv(args.out / "stats_pairwise.csv",  index=False)
    print(f"✓ Saved stats_friedman.csv and stats_pairwise.csv to {args.out}")

    print_summary(friedman, pairwise)


if __name__ == "__main__":
    main()
