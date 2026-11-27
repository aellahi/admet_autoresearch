"""Where does the variation in valid_mae come from?

    uv run variance.py

Three configurations over the same committed model, to separate the levers:

  splits only   five validation splits, one fixed model seed
  model only    one validation split, five model seeds
  both          what the campaign actually measures

Featurisation is deterministic, so it contributes nothing and is not varied.
Nothing here touches data/holdout.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

import prepare
import train

SEEDS = (1, 2, 3, 4, 5)


def _split(seed):
    d = prepare.SPLITS / f"seed{seed}"
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "valid.csv")


def _score(tr, va, model_seed, cache):
    """MAE for one split and one model seed. Features are cached per split."""
    key = id(tr)
    if key not in cache:
        cache[key] = (train.featurize(list(tr["Drug"])),
                      train.featurize(list(va["Drug"])))
    X, Xv = cache[key]
    m = train.make_model(model_seed)
    # Keep this analysis off the cores the campaign is using.
    if hasattr(m, "n_jobs"):
        m.n_jobs = 2
    m.fit(X, tr["Y"].values)
    return float(np.mean(np.abs(m.predict(Xv) - va["Y"].values)))


def main():
    cache = {}
    splits = {s: _split(s) for s in SEEDS}

    # 1. vary the split, hold the model seed
    a = [_score(*splits[s], 0, cache) for s in SEEDS]
    # 2. hold the split, vary the model seed
    b = [_score(*splits[1], k, cache) for k in range(5)]
    # 3. vary both together, as the campaign does
    c = [_score(*splits[s], s, cache) for s in SEEDS]

    rows = [
        ("splits only (model seed fixed)", a),
        ("model only (split 1 fixed)", b),
        ("both, as the campaign measures", c),
    ]
    print(f"{'configuration':34} {'mean':>8} {'std':>8} {'range':>8}")
    print("-" * 62)
    for name, v in rows:
        v = np.array(v)
        print(f"{name:34} {v.mean():8.4f} {v.std():8.4f} "
              f"{v.max() - v.min():8.4f}")

    print()
    print("per-seed values")
    for name, v in rows:
        print(f"  {name:34} {' '.join(f'{x:.4f}' for x in v)}")

    sd = np.std(c)
    print()
    print(f"standard error of the five-seed mean : {sd / np.sqrt(5):.4f}")
    print(f"95% interval on that mean            : +/-{1.96 * sd / np.sqrt(5):.4f}")
    print()
    print("The campaign's 0.03 bar compares two such means with a fixed")
    print("constant. Both are measured on the SAME five splits, so the")
    print("comparison is paired and its real resolution depends on how")
    print("correlated the two models' errors are -- which a constant cannot")
    print("know. prepare.py already computes mae_std; runner.py discards it.")


if __name__ == "__main__":
    main()
