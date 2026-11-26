"""Fixed harness: splits and metric. Do not modify.

Two entry points, deliberately separate:

    run_validation()  reads data/splits only. This is what the loop calls.
    run_audit()       reads data/holdout/test.csv. Only audit.py calls it.

run_validation names no path under data/holdout, so the validation process
never has test labels in memory. There is no flag that makes it load them.
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = Path("data/splits")
SEEDS = (1, 2, 3, 4, 5)
TIMEOUT_SECONDS = 600

# Named here only so audit.py can import it. Nothing in the validation path
# below refers to it.
HOLDOUT_TEST = Path("data/holdout/test.csv")


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


def _train(seed):
    path = SPLITS / f"seed{seed}" / "train.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: uv run setup.py")
    return pd.read_csv(path)


def _score(featurize, make_model, pairs, split):
    """pairs: [(seed, train_df, held_df), ...], one per seed."""
    start = time.time()
    held_scores, train_scores = [], []

    for seed, train, held in pairs:
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


def run_validation(featurize, make_model):
    """Score on the validation splits. Opens nothing under data/holdout."""
    pairs = []
    for seed in SEEDS:
        valid_path = SPLITS / f"seed{seed}" / "valid.csv"
        if not valid_path.exists():
            raise SystemExit(f"{valid_path} is missing. Run: uv run setup.py")
        pairs.append((seed, _train(seed), pd.read_csv(valid_path)))
    return _score(featurize, make_model, pairs, "valid")


def run_audit(featurize, make_model):
    """Score on the held-out test set. Called only by audit.py."""
    if not HOLDOUT_TEST.exists():
        raise SystemExit(f"{HOLDOUT_TEST} is missing. Run: uv run setup.py")
    test = pd.read_csv(HOLDOUT_TEST)
    pairs = [(seed, _train(seed), test) for seed in SEEDS]
    return _score(featurize, make_model, pairs, "test")


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
