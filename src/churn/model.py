"""Train the churn model and produce SHAP explanations.

Usage:  python -m churn.model
Output: artifacts/model.joblib (gitignored) + output/shap_summary.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_data():
    path = os.path.join(REPO_ROOT, "data", "curated", "saas_customers.csv")
    return pd.read_csv(path)


def main():
    from churn.synthesize import FEATURES

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["churned"], test_size=0.2, random_state=7, stratify=df["churned"]
    )

    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=4,
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    print(f"test AUC: {roc_auc_score(y_test, p):.4f}")
    print(f"test accuracy (p>0.5): {accuracy_score(y_test, (p > 0.5).astype(int)):.4f}")
    print(f"churn rate in test set: {y_test.mean():.3f}")

    # SHAP: which features drive the predictions?
    # (function-based explainer on a test sample: xgboost 3.x stores base_score
    # as a string, which trips up shap.TreeExplainer in some versions)
    import shap

    sample = X_test.sample(200, random_state=7)
    explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], sample)
    try:
        sv = np.asarray(explainer(sample).values)      # shap >= 0.46
    except AttributeError:
        sv = np.asarray(explainer.shap_values(sample))  # older shap
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = [FEATURES[i] for i in order]
    ax.barh(range(len(names)), mean_abs[order], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value| (test set)")
    ax.set_title("What drives churn predictions (XGBoost + SHAP)")
    fig.tight_layout()

    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)
    out = os.path.join(REPO_ROOT, "output", "shap_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    import joblib

    os.makedirs(os.path.join(REPO_ROOT, "artifacts"), exist_ok=True)
    joblib.dump(model, os.path.join(REPO_ROOT, "artifacts", "model.joblib"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
