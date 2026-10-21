"""Baseline MLP surrogate vs cross-validated LightGBM surrogate on UCI Airfoil Self-Noise.

Predicts sound pressure level (dB) from 5 geometry/flow parameters.
Test set is touched only once, at the end. Hyper-parameters are chosen by CV on the training set.
"""
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # headless backend for GitHub Actions
import matplotlib.pyplot as plt
import numpy as np
import requests
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]  # used for the randomized models (MLP init, LightGBM subsampling)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
HERE = os.path.dirname(os.path.abspath(__file__))


def download(url, retries=5):
    """GET with timeout=60; retry with exponential backoff on HTTP 429/5xx."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  download failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def fit_eval(model, Xtr, ytr, Xte, yte):
    """Fit on train, score on test. Returns (RMSE, R2, MAE)."""
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return (np.sqrt(mean_squared_error(yte, p)), r2_score(yte, p), mean_absolute_error(yte, p))


def make_mlp(seed):
    """BASELINE: standard MLPRegressor with fixed hidden layers; target standardised for training."""
    return TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed),
        transformer=StandardScaler())


def main():
    # 1. Load data (whitespace-separated, no header: 5 features + target)
    print("Downloading dataset...")
    data = np.loadtxt(io.StringIO(download(URL)))
    X, y = data[:, :5], data[:, 5]
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features")

    # 2. Single 80/20 split with fixed seed; scale inputs with training statistics only
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    print(f"Train {len(ytr)} / test {len(yte)}")

    # 3. IMPROVED: small grid search for LightGBM, scored by 5-fold CV on the training set only
    print("CV hyper-parameter search for LightGBM (training set only)...")
    grid = {"num_leaves": [15, 31], "learning_rate": [0.03, 0.1],
            "n_estimators": [300, 600], "min_child_samples": [5, 20]}
    lgbm_fixed = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    search = GridSearchCV(LGBMRegressor(random_state=SEED, **lgbm_fixed), grid,
                          cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring="neg_root_mean_squared_error", n_jobs=1)
    search.fit(Xtr_s, ytr)
    best = search.best_params_
    cv_rmse = -search.best_score_
    print(f"  best params {best} | CV RMSE {cv_rmse:.3f} dB")

    # 4. Test set used ONCE per model, for each seed
    print("Evaluating baseline MLP on test set...")
    base = np.array([fit_eval(make_mlp(s), Xtr_s, ytr, Xte_s, yte) for s in SEEDS])
    print("Evaluating improved LightGBM on test set...")
    imp = np.array([fit_eval(LGBMRegressor(random_state=s, **best, **lgbm_fixed),
                             Xtr_s, ytr, Xte_s, yte) for s in SEEDS])

    b_rmse, b_r2, b_mae = base.T
    i_rmse, i_r2, i_mae = imp.T
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(imp))):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    res = {
        "question": "Can a cross-validated gradient-boosting surrogate predict a design response "
                    "from geometry parameters more accurately than a standard MLP surrogate, "
                    "on a public stand-in dataset?",
        "method": "80/20 fixed-seed split; standardised inputs (training statistics); baseline MLPRegressor "
                  "vs LightGBM with 5-fold CV grid search on training data; test RMSE main metric; "
                  "seeds 42-44 for randomized models; test set used once.",
        "dataset": {
            "name": "UCI Airfoil Self-Noise",
            "source_url": URL,
            "n_rows": int(X.shape[0]),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "test RMSE (dB)",
        "higher_is_better": False,
        "baseline": "MLPRegressor (hidden layers 64-64, Adam, max_iter 2000) on standardised inputs and "
                    "standardised target; no hyper-parameter search.",
        "improved": "LightGBM gradient-boosting regressor on the same scaled inputs, with grid "
                    "search over num_leaves, learning_rate, n_estimators, min_child_samples by 5-fold CV "
                    "on the training set only.",
        "metrics": {
            "baseline_score": float(b_rmse.mean()),
            "baseline_score_std": float(b_rmse.std()),
            "improved_score": float(i_rmse.mean()),
            "improved_score_std": float(i_rmse.std()),
            "improvement": float(b_rmse.mean() - i_rmse.mean()),
            "baseline_r2": float(b_r2.mean()),
            "improved_r2": float(i_r2.mean()),
            "baseline_mae": float(b_mae.mean()),
            "improved_mae": float(i_mae.mean()),
            "improved_cv_rmse": float(cv_rmse),
        },
        "notes": "Single public stand-in dataset, not the paper's antenna data. One fixed 80/20 split; "
                 "seed variation covers only model randomness. The baseline has no tuning, so the comparison "
                 "confounds model family with tuning effort. Classification F1 does not apply to this "
                 "regression task, so R2 and MAE are reported instead. Hyper-parameters were chosen by CV on "
                 "training data only.",
    }
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Baseline RMSE {res['metrics']['baseline_score']:.3f} dB, "
          f"improved RMSE {res['metrics']['improved_score']:.3f} dB")

    # 5. Optional chart
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline MLP", "LightGBM (CV)"],
           [b_rmse.mean(), i_rmse.mean()],
           yerr=[b_rmse.std(), i_rmse.std()], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("Test RMSE (dB), lower is better")
    ax.set_title("Airfoil self-noise surrogate, mean over seeds 42-44")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "rmse_comparison.png"), dpi=120)
    print("Wrote results.json and figures/rmse_comparison.png")


if __name__ == "__main__":
    main()
