"""Fixed harness: data, splits, evaluation. Do not modify.

Downloads the TDC Caco2_Wang benchmark and defines the metric. train.py
imports run() from here and supplies the two things it is allowed to change.
"""

import resource
import sys
import time

import numpy as np
from tdc.benchmark_group import admet_group

TASK = "Caco2_Wang"
DATA_DIR = "data/"
SEEDS = (1, 2, 3, 4, 5)
TIMEOUT_SECONDS = 600


def _group():
    return admet_group(path=DATA_DIR)


def _check(X, n_rows, label):
    if X.ndim != 2 or X.shape[0] != n_rows:
        raise ValueError(f"{label}: expected ({n_rows}, n_features), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{label}: contains NaN or inf")
    if np.abs(X).max() > 1e30:
        raise ValueError(f"{label}: values too large for float32")


def _peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024  # Linux reports kilobytes
    return rss / 1024 ** 3


def run(featurize, make_model, split="valid"):
    """Score a featurizer/model pair over the five seeds.

    split="valid" during the loop. split="test" is the held-out set and is
    only for the final result, never for keep/discard decisions.
    """
    start = time.time()
    group = _group()
    benchmark = group.get(TASK)
    name = benchmark["name"]
    binary = set(benchmark["train_val"]["Y"].unique()) <= {0, 1}

    scores, train_scores = [], []
    for seed in SEEDS:
        train, valid = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        held = benchmark["test"] if split == "test" else valid

        X = featurize(list(train["Drug"]))
        _check(X, len(train), "train features")
        X_held = featurize(list(held["Drug"]))
        _check(X_held, len(held), f"{split} features")

        model = make_model(seed)
        model.fit(X, train["Y"].values)
        pred = model.predict(X_held)
        scores.append(float(np.mean(np.abs(pred - held["Y"].values))))
        train_scores.append(
            float(np.mean(np.abs(model.predict(X) - train["Y"].values)))
        )

    mean, std = float(np.mean(scores)), float(np.std(scores))
    train_mean = float(np.mean(train_scores))
    gap = mean - train_mean
    print("---")
    print(f"{split}_mae:         {mean:.6f}")
    print(f"{split}_mae_std:     {std:.6f}")
    print(f"train_mae:        {train_mean:.6f}")
    print(f"gap:              {gap:.6f}")
    # Diagnostic only. The gap is NOT a keep/discard criterion: minimising it
    # selects for underfitting (predicting the mean has a gap near zero). A
    # small gap alongside a very low error is the signature of leakage.
    if gap < 0.05 and mean < 0.30:
        print("WARNING:          small gap with low error - check for leakage")
    print(f"n_features:       {X.shape[1]}")
    print(f"peak_memory_gb:   {_peak_gb():.2f}")
    print(f"total_seconds:    {time.time() - start:.1f}")
    return mean, std
