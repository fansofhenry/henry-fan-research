"""
models/registry.py
------------------
All five benchmark models with a common interface:

    get_model(name)  →  sklearn-compatible estimator

Models:
  1. logistic_regression
  2. random_forest
  3. svm
  4. gradient_boosting   (same conceptual role as XGBoost)
  5. neural_net          (sklearn MLPClassifier)

Each is wrapped with sensible defaults and a .name attribute.
"""

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _named(estimator, name: str):
    """Attach a human-readable .model_name attribute."""
    estimator.model_name = name
    return estimator


# ── 1. Logistic Regression ────────────────────────────────────────────────────

def get_logistic_regression():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    return _named(pipe, "Logistic Regression")


# ── 2. Random Forest ──────────────────────────────────────────────────────────

def get_random_forest():
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    return _named(clf, "Random Forest")


# ── 3. Support Vector Machine ─────────────────────────────────────────────────

def get_svm():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )),
    ])
    return _named(pipe, "SVM (RBF)")


# ── 4. Gradient Boosting (XGBoost analog) ────────────────────────────────────

def get_gradient_boosting():
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    return _named(clf, "Gradient Boosting")


# ── 5. Neural Network (MLP) ───────────────────────────────────────────────────

def get_neural_net():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    return _named(pipe, "Neural Network (MLP)")


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_MODELS = {
    "logistic_regression": get_logistic_regression,
    "random_forest":       get_random_forest,
    "svm":                 get_svm,
    "gradient_boosting":   get_gradient_boosting,
    "neural_net":          get_neural_net,
}


def get_model(name: str):
    """Instantiate a fresh model by name."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(ALL_MODELS)}")
    return ALL_MODELS[name]()
