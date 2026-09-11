# saas-churn-radar

Predict which B2B SaaS accounts will churn at renewal - and explain *why*, in a way a non-technical VP can read (SHAP).

## What it does

1. **Dataset** (`src/churn/synthesize.py`): 5,000 customer accounts with realistic engagement, billing, and support features. Synthetic on purpose: fully reproducible from a seed, no license questions about someone else's data, and the churn drivers mirror what we actually see in B2B SaaS (usage decay, underused seats, payment friction, support load). Swap in your own customers via the same schema - nothing downstream cares where the rows came from.
2. **Model** (`src/churn/model.py`): XGBoost classifier, 80/20 stratified split, evaluated on held-out accounts.
3. **Explanations**: SHAP values show which features drive each prediction; the summary chart ranks them.

## Results (v1)

- Test AUC: **0.728** · test accuracy at p>0.5: 0.720 · churn rate in test set: 34.5%
- Top drivers of churn predictions (mean |SHAP|): `days_since_login`, `usage_trend_3m`, `seat_utilization` - i.e. engagement decay and underused seats, which matches operational intuition.

![shap summary](output/shap_summary.png)

## Layout

```
data/curated/   saas_customers.csv (committed, reproducible from seed 42)
src/churn/      synthesize.py (dataset), model.py (train + SHAP)
tests/          unit tests - run with pytest
artifacts/      trained model (gitignored)
output/         generated charts
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt pytest   # Windows; use bin/pip on Linux/macOS
set PYTHONPATH=src                                      # Windows (or export PYTHONPATH=src)

# regenerate the dataset if you ever change the generator
python -c "from churn.synthesize import generate; generate(5000, seed=42).to_csv('data/curated/saas_customers.csv', index=False)"

python -m churn.model     # trains, prints AUC, writes output/shap_summary.png
python -m pytest tests/
```

## Known simplifications (v1)

- SHAP is computed on a 200-row test sample (function-based explainer; xgboost 3.x stores base_score as a string, which trips up shap.TreeExplainer in some versions).
- Single renewal window per account; no time-series features yet. Next step: monthly usage panels and a survival-model comparison.
