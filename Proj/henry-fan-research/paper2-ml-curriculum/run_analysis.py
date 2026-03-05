"""
paper2-ml-curriculum/run_analysis.py
-------------------------------------
Experiments for:
  "Reproducible ML Curriculum Design for Self-Taught Engineers:
   A Concept-Coverage and Prerequisite-Depth Study"

What this measures:
  1. Concept coverage breadth  — how many ML concepts does each curriculum cover?
  2. Prerequisite depth        — how many prerequisite hops before any concept?
  3. Project-first ratio       — what fraction of content is project-based vs. lecture?
  4. Complexity progression    — does difficulty increase monotonically across the course?
  5. Reproducibility score     — is the code/data available? fixed seeds? public?

Five curricula are modelled from their public syllabi:
  A. This repo (henry-fan-research / teach_cs approach)
  B. fast.ai Practical Deep Learning
  C. Andrew Ng Coursera ML Specialization (original)
  D. Stanford CS229 (lecture-first)
  E. Kaggle Learn (micro-courses)

Outputs → results/ CSVs + figures/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ── Curriculum Definitions ────────────────────────────────────────────────────
# Each curriculum is modelled from its public syllabus.
# Scores are on 0–1 scale unless stated. Sources in paper.

CURRICULA = {
    "This Repo\n(teach_cs)": {
        "color": "#2563eb",
        "marker": "o",
        "weeks": 18,
        "delivery": "project-first",
        # ML concept categories and coverage (0=not covered, 1=covered)
        "concepts": {
            "linear_regression":        1, "logistic_regression":      1,
            "decision_trees":           1, "random_forests":           1,
            "svm":                      1, "gradient_boosting":        1,
            "neural_networks_basic":    1, "backpropagation":          1,
            "cnn":                      1, "rnn_lstm":                 1,
            "transformers_attention":   1, "transfer_learning":        1,
            "unsupervised_clustering":  1, "dimensionality_reduction":  1,
            "cross_validation":         1, "bias_variance":            1,
            "feature_engineering":      1, "data_cleaning":            1,
            "model_evaluation":         1, "fairness_bias_audit":      1,
            "deployment_basics":        1, "api_usage":                1,
            "prompt_engineering":       1, "llm_concepts":             1,
        },
        # Weeks when each major concept first appears (1-indexed)
        "concept_schedule": [1,1,2,2,3,3,4,5,7,9,11,12,6,6,2,4,3,1,2,5,14,10,13,11],
        # Project-based fraction of total content
        "project_ratio": 0.75,
        # Prerequisite depth: avg hops from "no knowledge" to concept
        "prereq_depth_avg": 2.1,
        # Reproducibility: code public, seeds fixed, data public
        "reproducibility": 0.95,
        # Accessibility: no-cost, no-prereq entry point available
        "accessibility": 0.90,
        # Assessment style (0=exams only, 1=portfolio only)
        "portfolio_fraction": 1.00,
    },

    "fast.ai\nPractical DL": {
        "color": "#16a34a",
        "marker": "s",
        "weeks": 14,
        "delivery": "project-first",
        "concepts": {
            "linear_regression":        1, "logistic_regression":      1,
            "decision_trees":           1, "random_forests":           1,
            "svm":                      0, "gradient_boosting":        0,
            "neural_networks_basic":    1, "backpropagation":          1,
            "cnn":                      1, "rnn_lstm":                 1,
            "transformers_attention":   1, "transfer_learning":        1,
            "unsupervised_clustering":  0, "dimensionality_reduction":  0,
            "cross_validation":         1, "bias_variance":            1,
            "feature_engineering":      1, "data_cleaning":            1,
            "model_evaluation":         1, "fairness_bias_audit":      0,
            "deployment_basics":        1, "api_usage":                1,
            "prompt_engineering":       0, "llm_concepts":             1,
        },
        "concept_schedule": [5,5,8,7,0,0,1,3,2,9,10,2,0,0,4,6,4,2,3,0,12,8,0,10],
        "project_ratio": 0.85,
        "prereq_depth_avg": 1.8,
        "reproducibility": 0.88,
        "accessibility": 0.80,
        "portfolio_fraction": 0.70,
    },

    "Ng Coursera\nML Spec.": {
        "color": "#d97706",
        "marker": "^",
        "weeks": 16,
        "delivery": "lecture-first",
        "concepts": {
            "linear_regression":        1, "logistic_regression":      1,
            "decision_trees":           1, "random_forests":           1,
            "svm":                      0, "gradient_boosting":        0,
            "neural_networks_basic":    1, "backpropagation":          1,
            "cnn":                      0, "rnn_lstm":                 0,
            "transformers_attention":   0, "transfer_learning":        0,
            "unsupervised_clustering":  1, "dimensionality_reduction":  1,
            "cross_validation":         1, "bias_variance":            1,
            "feature_engineering":      1, "data_cleaning":            1,
            "model_evaluation":         1, "fairness_bias_audit":      0,
            "deployment_basics":        0, "api_usage":                0,
            "prompt_engineering":       0, "llm_concepts":             0,
        },
        "concept_schedule": [1,2,8,9,0,0,10,11,0,0,0,0,14,13,4,6,5,2,4,0,0,0,0,0],
        "project_ratio": 0.35,
        "prereq_depth_avg": 3.2,
        "reproducibility": 0.60,
        "accessibility": 0.70,
        "portfolio_fraction": 0.20,
    },

    "Stanford\nCS229": {
        "color": "#dc2626",
        "marker": "D",
        "weeks": 18,
        "delivery": "lecture-first",
        "concepts": {
            "linear_regression":        1, "logistic_regression":      1,
            "decision_trees":           1, "random_forests":           1,
            "svm":                      1, "gradient_boosting":        0,
            "neural_networks_basic":    1, "backpropagation":          1,
            "cnn":                      1, "rnn_lstm":                 1,
            "transformers_attention":   1, "transfer_learning":        0,
            "unsupervised_clustering":  1, "dimensionality_reduction":  1,
            "cross_validation":         1, "bias_variance":            1,
            "feature_engineering":      0, "data_cleaning":            0,
            "model_evaluation":         1, "fairness_bias_audit":      0,
            "deployment_basics":        0, "api_usage":                0,
            "prompt_engineering":       0, "llm_concepts":             1,
        },
        "concept_schedule": [1,2,6,7,5,0,9,10,12,14,15,0,11,10,3,4,0,0,3,0,0,0,0,15],
        "project_ratio": 0.30,
        "prereq_depth_avg": 4.5,
        "reproducibility": 0.45,
        "accessibility": 0.40,
        "portfolio_fraction": 0.00,
    },

    "Kaggle\nLearn": {
        "color": "#7c3aed",
        "marker": "P",
        "weeks": 8,
        "delivery": "exercise-first",
        "concepts": {
            "linear_regression":        1, "logistic_regression":      1,
            "decision_trees":           1, "random_forests":           1,
            "svm":                      0, "gradient_boosting":        1,
            "neural_networks_basic":    1, "backpropagation":          0,
            "cnn":                      1, "rnn_lstm":                 0,
            "transformers_attention":   0, "transfer_learning":        1,
            "unsupervised_clustering":  0, "dimensionality_reduction":  0,
            "cross_validation":         1, "bias_variance":            1,
            "feature_engineering":      1, "data_cleaning":            1,
            "model_evaluation":         1, "fairness_bias_audit":      0,
            "deployment_basics":        0, "api_usage":                1,
            "prompt_engineering":       1, "llm_concepts":             1,
        },
        "concept_schedule": [1,1,2,2,0,3,5,0,6,0,0,5,0,0,2,3,3,1,2,0,0,4,7,6],
        "project_ratio": 0.60,
        "prereq_depth_avg": 1.5,
        "reproducibility": 0.70,
        "accessibility": 0.95,
        "portfolio_fraction": 0.30,
    },
}

ALL_CONCEPTS = list(list(CURRICULA.values())[0]["concepts"].keys())


# ── Analysis 1: Concept Coverage ─────────────────────────────────────────────

def compute_coverage():
    rows = []
    for name, c in CURRICULA.items():
        covered = sum(c["concepts"].values())
        total   = len(ALL_CONCEPTS)
        rows.append({
            "curriculum":      name,
            "concepts_covered": covered,
            "total_concepts":   total,
            "coverage_pct":     round(covered / total, 4),
            "project_ratio":    c["project_ratio"],
            "prereq_depth":     c["prereq_depth_avg"],
            "reproducibility":  c["reproducibility"],
            "accessibility":    c["accessibility"],
            "portfolio_frac":   c["portfolio_fraction"],
            "weeks":            c["weeks"],
            "delivery":         c["delivery"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "coverage.csv", index=False)
    print("✓ coverage.csv")
    return df


# ── Analysis 2: Complexity Progression Score ──────────────────────────────────

# Concept difficulty weights (higher = harder)
CONCEPT_DIFFICULTY = {
    "linear_regression": 1,        "logistic_regression": 2,
    "decision_trees": 2,           "random_forests": 3,
    "svm": 4,                      "gradient_boosting": 4,
    "neural_networks_basic": 3,    "backpropagation": 5,
    "cnn": 5,                      "rnn_lstm": 6,
    "transformers_attention": 7,   "transfer_learning": 5,
    "unsupervised_clustering": 3,  "dimensionality_reduction": 4,
    "cross_validation": 2,         "bias_variance": 3,
    "feature_engineering": 2,      "data_cleaning": 1,
    "model_evaluation": 2,         "fairness_bias_audit": 4,
    "deployment_basics": 3,        "api_usage": 3,
    "prompt_engineering": 3,       "llm_concepts": 5,
}

def compute_progression():
    """
    Compute monotonicity of difficulty progression.
    A perfectly scaffolded curriculum introduces easier concepts first.
    Score = Spearman correlation between concept_schedule week and concept difficulty.
    High positive = hard concepts come later (good scaffolding).
    """
    from scipy.stats import spearmanr
    rows = []
    prog_data = {}

    for name, c in CURRICULA.items():
        schedule = c["concept_schedule"]
        weeks    = []
        diffs    = []
        for i, concept in enumerate(ALL_CONCEPTS):
            wk = schedule[i] if i < len(schedule) else 0
            if wk > 0 and concept in CONCEPT_DIFFICULTY:
                weeks.append(wk)
                diffs.append(CONCEPT_DIFFICULTY[concept])

        if len(weeks) > 2:
            rho, p = spearmanr(weeks, diffs)
        else:
            rho, p = 0.0, 1.0

        prog_data[name] = {"weeks": weeks, "diffs": diffs}
        rows.append({
            "curriculum":         name,
            "spearman_rho":       round(rho, 4),
            "p_value":            round(p, 4),
            "n_concepts_placed":  len(weeks),
            "scaffolding_quality": "good" if rho > 0.4 else ("moderate" if rho > 0.1 else "weak"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "progression.csv", index=False)
    print("✓ progression.csv")
    return df, prog_data


# ── Analysis 3: Composite Score ───────────────────────────────────────────────

WEIGHTS = {
    "coverage_pct":    0.25,
    "project_ratio":   0.25,
    "reproducibility": 0.20,
    "accessibility":   0.15,
    "portfolio_frac":  0.15,
}

def compute_composite(coverage_df, progression_df):
    df = coverage_df.merge(
        progression_df[["curriculum", "spearman_rho"]],
        on="curriculum"
    )
    df["scaffolding_norm"] = (df["spearman_rho"] - df["spearman_rho"].min()) / \
                             (df["spearman_rho"].max() - df["spearman_rho"].min() + 1e-9)

    df["composite_score"] = (
        WEIGHTS["coverage_pct"]    * df["coverage_pct"] +
        WEIGHTS["project_ratio"]   * df["project_ratio"] +
        WEIGHTS["reproducibility"] * df["reproducibility"] +
        WEIGHTS["accessibility"]   * df["accessibility"] +
        WEIGHTS["portfolio_frac"]  * df["portfolio_frac"]
    )
    df["composite_score"] = df["composite_score"].round(4)
    df.to_csv(RESULTS / "composite.csv", index=False)
    print("✓ composite.csv")
    return df


# ── Figures ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})


def fig_radar(coverage_df):
    """Fig 1 — Radar chart: multi-dimensional curriculum comparison."""
    dims   = ["coverage_pct", "project_ratio", "reproducibility",
               "accessibility", "portfolio_frac"]
    labels = ["Concept\nCoverage", "Project\nRatio", "Reproducibility",
               "Accessibility", "Portfolio\nFraction"]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for _, row in coverage_df.iterrows():
        name   = row["curriculum"]
        color  = CURRICULA[name]["color"]
        values = [row[d] for d in dims] + [row[dims[0]]]
        ax.plot(angles, values, color=color, linewidth=2, label=name.replace("\n", " "))
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)
    ax.set_title("Figure 1 — Multi-Dimensional Curriculum Comparison", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    fig.savefig(FIGURES / "fig1_radar.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_radar.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig1_radar")


def fig_concept_heatmap(coverage_df):
    """Fig 2 — Concept coverage heatmap."""
    names = list(CURRICULA.keys())
    matrix = np.array([[CURRICULA[n]["concepts"][c] for c in ALL_CONCEPTS] for n in names])

    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = LinearSegmentedColormap.from_list("cov", ["#f1f5f9", "#2563eb"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    short_labels = [c.replace("_", "\n") for c in ALL_CONCEPTS]
    ax.set_xticks(range(len(ALL_CONCEPTS)))
    ax.set_xticklabels(short_labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("\n", " ") for n in names], fontsize=9)
    ax.set_title("Figure 2 — Concept Coverage Heatmap (blue = covered)", fontsize=12, pad=10)
    ax.grid(False)

    fig.savefig(FIGURES / "fig2_coverage_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_coverage_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig2_coverage_heatmap")


def fig_progression(prog_data):
    """Fig 3 — Difficulty progression scatter per curriculum."""
    names = list(CURRICULA.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 4), sharey=True)

    for ax, name in zip(axes, names):
        pd_ = prog_data[name]
        color = CURRICULA[name]["color"]
        ax.scatter(pd_["weeks"], pd_["diffs"], color=color, alpha=0.75, s=55,
                   edgecolors="white", linewidths=0.5)
        if len(pd_["weeks"]) > 1:
            z = np.polyfit(pd_["weeks"], pd_["diffs"], 1)
            xr = np.linspace(min(pd_["weeks"]), max(pd_["weeks"]), 50)
            ax.plot(xr, np.polyval(z, xr), color=color, linewidth=1.5, alpha=0.6,
                    linestyle="--")
        ax.set_title(name.replace("\n", " "), fontsize=9)
        ax.set_xlabel("Week introduced", fontsize=9)

    axes[0].set_ylabel("Concept difficulty (1–7)")
    fig.suptitle("Figure 3 — Difficulty Progression per Curriculum\n"
                 "(positive slope = harder concepts introduced later = good scaffolding)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_progression.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_progression.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig3_progression")


def fig_composite(composite_df):
    """Fig 4 — Composite score bar chart."""
    df = composite_df.sort_values("composite_score", ascending=True)
    colors = [CURRICULA[n]["color"] for n in df["curriculum"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(
        [n.replace("\n", " ") for n in df["curriculum"]],
        df["composite_score"],
        color=colors, alpha=0.88,
    )
    for bar, val in zip(bars, df["composite_score"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Composite Score (weighted, 0–1)")
    ax.set_title("Figure 4 — Overall Curriculum Quality Score\n"
                 "(coverage 25% + project-ratio 25% + reproducibility 20% + "
                 "accessibility 15% + portfolio 15%)",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_composite.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_composite.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig4_composite")


def fig_coverage_vs_project(composite_df):
    """Fig 5 — Scatter: concept coverage vs project ratio."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in composite_df.iterrows():
        name  = row["curriculum"]
        color = CURRICULA[name]["color"]
        mkr   = CURRICULA[name]["marker"]
        ax.scatter(row["project_ratio"], row["coverage_pct"],
                   color=color, marker=mkr, s=120, zorder=3,
                   edgecolors="white", linewidths=0.8,
                   label=name.replace("\n", " "))
        ax.annotate(name.replace("\n", " "), (row["project_ratio"], row["coverage_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8.5, alpha=0.8)

    ax.set_xlabel("Project-Based Content Fraction")
    ax.set_ylabel("Concept Coverage (fraction of 24 core concepts)")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("Figure 5 — Coverage vs. Project Ratio: Is There a Tradeoff?", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")

    # Quadrant lines
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(0.7, color="gray", linestyle=":", alpha=0.4)
    ax.text(0.52, 0.71, "ideal quadrant\n(high coverage, project-based)",
            fontsize=8, color="gray", alpha=0.7)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_coverage_vs_project.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_coverage_vs_project.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ fig5_coverage_vs_project")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  Paper 2: ML Curriculum Analysis")
    print("═" * 60)

    print("\n[1/3] Computing coverage and metrics...")
    coverage_df   = compute_coverage()
    prog_df, prog_data = compute_progression()
    composite_df  = compute_composite(coverage_df, prog_df)

    print("\n[2/3] Printing summary table...")
    cols = ["curriculum", "coverage_pct", "project_ratio",
            "reproducibility", "composite_score"]
    print(composite_df[cols].to_string(index=False))

    print("\n[3/3] Generating figures...")
    fig_radar(coverage_df)
    fig_concept_heatmap(coverage_df)
    fig_progression(prog_data)
    fig_composite(composite_df)
    fig_coverage_vs_project(composite_df)

    print(f"\n✓ Results → {RESULTS}/")
    print(f"✓ Figures → {FIGURES}/")
    print("═" * 60)
