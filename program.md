# admet_autoresearch

Autonomous experimentation on TDC's Caco2_Wang permeability benchmark.

## Setup

1. **Agree on a run tag** with the user, based on today's date (e.g. `nov22`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**: `README.md`, `prepare.py`, `train.py`.
4. **Verify data exists**: check `data/splits/seed1/` holds `train.csv` and
   `valid.csv`. If not, tell the human to run `uv run setup.py`.
5. **Initialize results.tsv** with just the header row.
6. **Confirm and go.**

## Experimentation

Run an experiment with:

```
uv run experiment.py "what this tried" > run.log 2>&1
```

It scores the current `train.py` over five seeds and then **commits or
reverts it for you**. You do not decide what to keep and you do not commit.

**What you CAN do:**
- Modify `train.py`. `featurize()` and `make_model()` are both fair game:
  any RDKit representation, any sklearn-compatible regressor.

**What you CANNOT do:**
- Modify `prepare.py`, `experiment.py` or `audit.py`. They hold the data,
  the metric and the keep/discard rule.
- Add dependencies beyond `pyproject.toml`.
- Run `audit.py`, or touch the test split in any other way. The audit set
  is the human's, scored outside this loop. Using it here would destroy
  the only unbiased measurement of your work.

**The goal: lowest `valid_mae`.** Lower is better.

**Constraints:**
- `featurize()` must return a finite float array with one row per input,
  handle unparseable SMILES, and be deterministic. `prepare.py` checks this.
- Anything with fitted state (scaling, PCA, feature selection) must fit on
  train only. Fitting on the held-out split is leakage and invalidates the run.
- The machine has 4 cores and 8 GB RAM. Keep `n_jobs` at 4 or below. If a run
  exceeds 10 minutes, kill it and treat it as a failure.

**Noise floor**: the validation set is 91 molecules. A difference below
**0.03 MAE** on the five-seed mean is not meaningful. `experiment.py`
enforces this; you do not need to apply it yourself.

**Simplicity criterion**: all else equal, simpler is better. A small gain
that adds ugly complexity is not worth it. Equal results from less code is
a win — keep it.

**The first run** establishes the baseline: run `train.py` unmodified.

## Output format

```
---
status:           DISCARD
valid_mae:        <float>
valid_mae_std:    <float>
train_mae:        <float>
n_features:       <int>
peak_memory_gb:   <float>
total_seconds:    <float>
```

Extract with `grep "^status:\|^valid_mae:" run.log`.

`status` is the decision, already applied. `KEEP` means the change is
committed; `DISCARD` means `train.py` has been reverted and the working tree
is back where it started.

**`train_mae` is a diagnostic, never a criterion.** It tells you what to try
next, not whether something worked:

- `train_mae` low, `valid_mae` high -> variance problem; regularise or simplify
- both high -> bias problem; richer features or more capacity
- both low and close together -> suspect leakage, not success

## Results log

`experiment.py` writes `results.tsv` for you. Do not edit it by hand.

```
commit	train_mae	valid_mae	status	description
<sha>	<float>	<float>	keep	<what you changed>
<sha>	<float>	<float>	discard	<what you changed>
```

Read it at the start of each experiment so you do not repeat yourself.

## The experiment loop

LOOP FOREVER:

1. Read `results.tsv` and the git log: what has been tried, what was kept.
2. Edit `train.py` with one experimental idea.
3. `uv run experiment.py "short description" > run.log 2>&1`
   (redirect; do not flood your context)
4. `grep "^status:\|^valid_mae:\|^train_mae:" run.log`
5. Empty output means a crash. `tail -n 50 run.log` for the trace, then fix
   or abandon after a few attempts.
6. Go to 1.

There is no step where you commit, revert, or decide. `experiment.py` has
already done it by the time you read the log. If `status` is `DISCARD` your
edit is gone; start the next idea from the current `train.py`.

**Crashes**: fix typos and missing imports; skip ideas that are
fundamentally broken and move on. A crash is logged automatically.

**NEVER STOP**: once the loop has begun, do not pause to ask whether to
continue. The user may be asleep and expects you to work until manually
stopped. If you run out of ideas, re-read the in-scope files, combine
previous near-misses, or try a more radical change to the representation.
