"""Train the churn model and produce SHAP explanations.

Usage:  python -m churn.model
Output: artifacts/model.joblib (gitignored) + output/shap_summary.png

Pipeline: small deterministic hyperparameter grid (validation AUC), then a final
fit on all training rows, evaluated on the held-out test set. SHAP uses the
native TreeExplainer over the FULL test set (see make_tree_explainer for the
xgboost 3.x base_score workaround).
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_data():
    path = os.path.join(REPO_ROOT, "data", "curated", "saas_customers.csv")
    return pd.read_csv(path)


class _FixedBooster(xgb.Booster):
    """A Booster view that normalizes xgboost 3.x's base_score in the raw dump.

    xgboost >= 3 serializes (possibly estimated) base_score into the model as a
    stringified array - '[3.455E-1]' - and shap's XGBTreeModelLoader does a bare
    float() on it, which crashes TreeExplainer. We rewrite that one field in
    save_raw() to an equal-length plain scalar (UBJSON stores strings with an
    explicit length, so the byte count must not change). If xgboost/shap fix
    this upstream, the no-op path below kicks in automatically and this class
    can be deleted.
    """

    def __new__(cls, wrapped):
        obj = object.__new__(cls)
        obj._wrapped = wrapped
        cfg = json.loads(wrapped.save_config())
        raw_bs = cfg["learner"]["learner_model_param"].get("base_score")
        try:
            float(raw_bs)
            obj._fix = None  # already a plain scalar - nothing to do
        except (TypeError, ValueError):
            val = float(json.loads(raw_bs)[0])
            fixed = f"{val:.12f}"[: len(raw_bs)]
            assert len(fixed) == len(raw_bs) and re.fullmatch(r"[0-9.eE+-]+", fixed)
            obj._fix = (raw_bs.encode(), fixed.encode())
        return obj

    def __init__(self, wrapped):  # skip Booster's C++ init; we delegate everything
        pass

    def save_raw(self, raw_format="ubj"):
        raw = self._wrapped.save_raw(raw_format=raw_format)
        if self._fix is not None:
            old, new = self._fix
            assert old in raw, "base_score string not found in raw dump"
            raw = raw.replace(old, new, 1)
        return raw

    def __getattr__(self, name):
        return getattr(self.__dict__["_wrapped"], name)


def make_tree_explainer(model):
    """shap.TreeExplainer for a fitted XGBClassifier (native, full-dataset ready)."""
    import shap

    return shap.TreeExplainer(_FixedBooster(model.get_booster()))


def tune(X_train, y_train):
    """Small deterministic grid search; best config by validation AUC."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=7, stratify=y_train
    )
    best = None
    for depth in (3, 4, 6):
        for lr in (0.05, 0.1):
            m = XGBClassifier(
                n_estimators=400, learning_rate=lr, max_depth=depth,
                subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=4,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
            if best is None or auc > best[0]:
                best = (auc, depth, lr)
    print(f"tuned: max_depth={best[1]} learning_rate={best[2]} (val AUC {best[0]:.4f})")
    return dict(max_depth=best[1], learning_rate=best[2])


def main():
    from churn.synthesize import FEATURES

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["churned"], test_size=0.2, random_state=7, stratify=df["churned"]
    )

    params = tune(X_train, y_train)
    model = XGBClassifier(
        n_estimators=400, subsample=0.8, colsample_bytree=0.8,
        random_state=7, n_jobs=4, **params,
    )
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    print(f"test AUC: {roc_auc_score(y_test, p):.4f}")
    print(f"test accuracy (p>0.5): {accuracy_score(y_test, (p > 0.5).astype(int)):.4f}")
    print(f"churn rate in test set: {y_test.mean():.3f}")

    # SHAP over the FULL held-out set with the native TreeExplainer; if a future
    # xgboost/shap combination breaks even that, fall back to a function-based
    # explainer on a sample so the pipeline never hard-fails.
    import shap

    try:
        explainer = make_tree_explainer(model)
        sv = np.asarray(explainer.shap_values(X_test))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        scope = "full test set"
    except Exception as ex:
        print(f"TreeExplainer unavailable ({ex.__class__.__name__}); using function-based fallback")
        sample = X_test.sample(200, random_state=7)
        explainer = shap.Explainer(lambda x: model.predict_proba(x)[:, 1], sample)
        sv = np.asarray(explainer(sample).values)      # shap >= 0.46
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        scope = "200-row test sample"

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = [FEATURES[i] for i in order]
    ax.barh(range(len(names)), mean_abs[order], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean |SHAP value| ({scope}, n={len(sv)})")
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
