# admet_autoresearch

An agent experiments on TDC's [Caco2_Wang](https://tdcommons.ai/benchmark/admet_group/01caco2/)
benchmark: predict Caco-2 cell permeability from a SMILES string. It edits
`train.py`, runs a five-seed evaluation, keeps the change if validation MAE
improved, and repeats.

Framework after [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Files

| file | role |
| --- | --- |
| `prepare.py` | Data, splits, metric. Fixed. |
| `train.py` | `featurize()` and `make_model()`. The agent edits this. |
| `program.md` | Instructions for the agent. You edit this. |
| `results.tsv` | Experiment log. Untracked. |

## Quick start

```bash
uv sync
uv run train.py
```

The first run downloads the benchmark (1.5 MB) into `data/`.

## Running the agent

Launch it with the restricted settings, so the limits apply to that session
and not to your own:

```bash
claude --settings agent-settings.json
```

Then ask it to read `program.md`. `agent-settings.json` turns on the Bash
sandbox, allows only the three domains the install needs, and denies reads
of `~/.claude` and the audit log. Check it took effect with `/sandbox`.

## Checking the audit set

Only you run this, and not often:

```bash
uv run audit.py
```

It scores the committed `train.py` on the held-out test split and appends to
`~/.cache/admet_autoresearch/audit.tsv`. Nothing in the agent's loop touches
the test split. Never feed the number back into the search.

## The metric

Mean absolute error on log10 Caco-2 permeability, averaged over five seeds,
lower better. Train, validation and test share no Murcko scaffold, so a run
measures generalization to unseen chemistry.

Reference points on the test set: predicting the training mean gives 0.565,
an ECFP4 random forest gives 0.427, and the leaderboard leader is near 0.28.

With 91 validation molecules, differences below 0.03 MAE are not meaningful.
