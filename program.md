# admet_autoresearch

You are one iteration of a search. Make **one** change to `train.py`, then stop.

## The task

Predict Caco-2 cell permeability from a SMILES string. Lower `valid_mae` is
better. The baseline is an ECFP4 fingerprint into a random forest.

## What you can change

`train.py`, and only `train.py`. Two functions:

- `featurize(smiles) -> (n, d) float32` — any RDKit representation
- `make_model(seed)` — any scikit-learn compatible regressor

`featurize()` must return one finite row per input, handle unparseable
SMILES, and be deterministic. `prepare.py` checks all three and the run
fails if it does not.

Anything with fitted state — scaling, PCA, feature selection — must fit on
train only. Fitting on the held-out split is leakage and invalidates the run.

## What you cannot do

- Run anything. You have no shell. Do not try to score your own change.
- Edit any file but `train.py`.
- Add a package. There is no network; what is installed is what you have.
- Touch `data/holdout`. It is the audit set, read only by the human, and
  reading it would destroy the only unbiased measure of this work.

## How to choose

The ledger of every experiment so far is in your prompt. Each row carries the
`idea`, the `hypothesis` behind it, `train_mae`, `valid_mae`, `n_features`,
the `status`, and an `outcome` saying why that verdict followed.

Read it before deciding. Compare each hypothesis against its outcome — a
prediction that missed tells you more than one that landed. Do not repeat a
failure, and build on what held.

`train_mae` is a diagnostic, never a target:

- `train_mae` low, `valid_mae` high → variance; regularise or simplify
- both high → bias; richer features or more capacity
- both low and close → suspect leakage, not success

Your change is scored on **five validation splits**, each a different 91
molecules. The best model has been scored on the same five. `runner.py`
compares the two sets with a **paired t-test** and applies the rule itself:

- **significantly better** (p < 0.05) across the five splits → `keep`
- **costs less than 0.01** against the committed model, using three quarters
  of its features or fewer → `simplify`, also kept
- anything else → `discard`, and your edit is reverted

What matters is consistency, not the average. A change that wins on all five
splits by 0.015 is kept. A change that wins by 0.05 on two splits and loses
on three is not, however good its mean looks.

The ledger gives you `valid_std`, the spread across those five, and
`p_value` from the test. A large `valid_std` means the splits disagree about
your model, which is itself worth reading.

## Constraints

4 cores, 8 GB RAM. Keep `n_jobs` at 4 or below.

Nothing is killed for being slow, but time is logged per phase. Featurising
currently costs far more than fitting — about 25 seconds against 2 — so a
featurizer that calls something expensive per molecule will dominate the
run. Worth knowing before you reach for a conformer generator.

## When you are done

Stop after one edit. End your reply with:

```
IDEA: <at most twelve words naming the change>
HYPOTHESIS: <one sentence: what you expect, and what the scores would show if it holds>
```

You will not see the result. `runner.py` scores your change after you exit,
keeps or reverts it, and the next iteration reads the outcome in
`results.tsv`.
