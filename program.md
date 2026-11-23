# admet_autoresearch

Autonomous experimentation on TDC's Caco2_Wang permeability benchmark.

## Setup

1. **Agree on a run tag** with the user, based on today's date (e.g. `nov22`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**: `README.md`, `prepare.py`, `train.py`.
4. **Verify data exists**: check `data/` holds `admet_group/`. If not, the
   first run downloads it (1.5 MB).
5. **Initialize results.tsv** with just the header row.
6. **Confirm and go.**

## Experimentation

Run an experiment with `uv run train.py`. A run trains five models, one per
seed, and prints the mean validation MAE.

**What you CAN do:**
- Modify `train.py`. `featurize()` and `make_model()` are both fair game:
  any RDKit representation, any sklearn-compatible regressor.

**What you CANNOT do:**
- Modify `prepare.py`. It holds the data, the splits and the metric.
- Add dependencies beyond `pyproject.toml`.
- Evaluate on the test set during the loop. `--test` is for the final
  result only, after the user stops the run.

**The goal: lowest `valid_mae`.** Lower is better.

**Constraints:**
- `featurize()` must return a finite float array with one row per input,
  handle unparseable SMILES, and be deterministic. `prepare.py` checks this.
- Anything with fitted state (scaling, PCA, feature selection) must fit on
  train only. Fitting on the held-out split is leakage and invalidates the run.
- The machine has 4 cores and 8 GB RAM. Keep `n_jobs` at 4 or below. If a run
  exceeds 10 minutes, kill it and treat it as a failure.

**Noise floor**: the validation set is 91 molecules. A difference below
**0.03 MAE** on the five-seed mean is not meaningful. Do not keep a change
on a smaller margin.

**Simplicity criterion**: all else equal, simpler is better. A small gain
that adds ugly complexity is not worth it. Equal results from less code is
a win — keep it.

**The first run** establishes the baseline: run `train.py` unmodified.

## Output format

```
---
valid_mae:        0.436000
valid_mae_std:    0.021000
n_features:       2048
peak_memory_gb:   1.20
total_seconds:    38.4
```

Extract with `grep "^valid_mae:" run.log`.

## Logging results

Log each experiment to `results.tsv` (tab-separated). Header and 5 columns:

```
commit	valid_mae	memory_gb	status	description
a1b2c3d	0.436000	1.2	keep	baseline ECFP4 + random forest
b2c3d4e	0.363000	0.9	keep	switch to 6 physicochemical descriptors
c3d4e5f	0.441000	1.3	discard	radius 3 fingerprint
d4e5f6g	0.000000	0.0	crash	Ipc descriptor overflows float32
```

Use `0.000000` and `0.0` for crashes. Leave `results.tsv` untracked.

## The experiment loop

LOOP FOREVER:

1. Check the git state: current branch and commit.
2. Edit `train.py` with one experimental idea.
3. `git commit`
4. `uv run train.py > run.log 2>&1` (redirect; do not flood context)
5. `grep "^valid_mae:\|^peak_memory_gb:" run.log`
6. Empty output means a crash. `tail -n 50 run.log` for the trace, then fix
   or abandon after a few attempts.
7. Record the result in `results.tsv`.
8. If `valid_mae` improved by more than 0.03, keep the commit and advance.
9. Otherwise `git reset` back to where you started.

**Crashes**: fix typos and missing imports; skip ideas that are
fundamentally broken, log `crash`, move on.

**NEVER STOP**: once the loop has begun, do not pause to ask whether to
continue. The user may be asleep and expects you to work until manually
stopped. If you run out of ideas, re-read the in-scope files, combine
previous near-misses, or try a more radical change to the representation.
