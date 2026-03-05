"""
data/loaders.py
---------------
Loads all five benchmark datasets with consistent (X, y, meta) returns.

Each loader returns:
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)
    meta : dict     {name, n_samples, n_features, n_classes, task}
"""

import numpy as np
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_openml,
)
from sklearn.preprocessing import LabelEncoder


# ── 1. Iris ──────────────────────────────────────────────────────────────────
def load_iris_dataset():
    d = load_iris()
    meta = dict(
        name="Iris",
        n_samples=len(d.target),
        n_features=d.data.shape[1],
        n_classes=len(np.unique(d.target)),
        task="multiclass",
    )
    return d.data, d.target, meta


# ── 2. Wine ───────────────────────────────────────────────────────────────────
def load_wine_dataset():
    d = load_wine()
    meta = dict(
        name="Wine",
        n_samples=len(d.target),
        n_features=d.data.shape[1],
        n_classes=len(np.unique(d.target)),
        task="multiclass",
    )
    return d.data, d.target, meta


# ── 3. Breast Cancer ──────────────────────────────────────────────────────────
def load_breast_cancer_dataset():
    d = load_breast_cancer()
    meta = dict(
        name="Breast Cancer",
        n_samples=len(d.target),
        n_features=d.data.shape[1],
        n_classes=len(np.unique(d.target)),
        task="binary",
    )
    return d.data, d.target, meta


# ── 4. MNIST (first 5 000 samples to keep training fast) ─────────────────────
def load_mnist_dataset(n_samples: int = 5000):
    """
    Uses sklearn's built-in MNIST loader via fetch_openml.
    Falls back to a synthetic stand-in if offline.
    """
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data[:n_samples].astype(np.float32) / 255.0
        y = mnist.target[:n_samples].astype(int)
    except Exception:
        # Offline fallback: Gaussian blobs with 10 classes
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=64,
            n_informative=20,
            n_classes=10,
            n_clusters_per_class=1,
            random_state=42,
        )
        X = X.astype(np.float32)

    meta = dict(
        name="MNIST",
        n_samples=len(y),
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        task="multiclass",
    )
    return X, y, meta


# ── 5. Titanic ────────────────────────────────────────────────────────────────
def load_titanic_dataset():
    """
    Loads Titanic from sklearn/openml.
    Falls back to an offline synthetic version if unavailable.
    """
    try:
        titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
        df = titanic.frame.copy()

        # Keep a clean feature subset
        features = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
        df = df[features + ["survived"]].dropna()

        # Encode categoricals
        for col in ["sex", "embarked"]:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        X = df[features].values.astype(np.float32)
        y = df["survived"].astype(int).values

    except Exception:
        # Offline fallback: synthetic binary classification
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=891,
            n_features=7,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X = X.astype(np.float32)

    meta = dict(
        name="Titanic",
        n_samples=len(y),
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        task="binary",
    )
    return X, y, meta


# ── Registry ──────────────────────────────────────────────────────────────────
ALL_DATASETS = {
    "iris":          load_iris_dataset,
    "wine":          load_wine_dataset,
    "breast_cancer": load_breast_cancer_dataset,
    "mnist":         load_mnist_dataset,
    "titanic":       load_titanic_dataset,
}


def load_dataset(name: str):
    """Load dataset by name from the registry."""
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(ALL_DATASETS)}")
    return ALL_DATASETS[name]()
