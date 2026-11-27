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

The validation set is 91 molecules, so a difference below **0.03 MAE** is
noise. `runner.py` enforces the rule; you do not apply it yourself:

- beats the best by **more than 0.03** → `keep`
- within 0.03 of the best, using **three quarters of the features or fewer**
  → `simplify`, also kept. Same accuracy from a smaller model is a win.
- anything else → `discard`, and your edit is reverted

So a change that merely ties is not enough. Either beat the bar, or hold the
score with markedly less model.

## Constraints

4 cores, 8 GB RAM. Keep `n_jobs` at 4 or below. A run over ten minutes is
killed and counted as a failure.

## When you are done

Stop after one edit. End your reply with:

```
IDEA: <at most twelve words naming the change>
HYPOTHESIS: <one sentence: what you expect, and what the scores would show if it holds>
```

You will not see the result. `runner.py` scores your change after you exit,
keeps or reverts it, and the next iteration reads the outcome in
`results.tsv`.
