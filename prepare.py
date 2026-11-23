"""Fixed harness: data, splits, metric. Do not modify.

run() scores a featurizer/model pair. It returns numbers and prints nothing;
experiment.py decides what to do with them.
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
    """Score over the five seeds.

    split="valid" drives keep/discard. split="test" is the audit set and is
    only ever called by experiment.py, which never shows the result.
    """
    start = time.time()
    group = admet_group(path=DATA_DIR)
    benchmark = group.get(TASK)
    name = benchmark["name"]

    held_scores, train_scores = [], []
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
        held_scores.append(
            float(np.mean(np.abs(model.predict(X_held) - held["Y"].values)))
        )
        train_scores.append(
            float(np.mean(np.abs(model.predict(X) - train["Y"].values)))
        )

    return {
        "split": split,
        "mae": float(np.mean(held_scores)),
        "mae_std": float(np.std(held_scores)),
        "train_mae": float(np.mean(train_scores)),
        "n_features": int(X.shape[1]),
        "peak_memory_gb": _peak_gb(),
        "seconds": time.time() - start,
    }


def format_summary(m):
    split = m["split"]
    return "\n".join([
        "---",
        f"{split}_mae:         {m['mae']:.6f}",
        f"{split}_mae_std:     {m['mae_std']:.6f}",
        f"train_mae:        {m['train_mae']:.6f}",
        f"n_features:       {m['n_features']}",
        f"peak_memory_gb:   {m['peak_memory_gb']:.2f}",
        f"total_seconds:    {m['seconds']:.1f}",
    ])
