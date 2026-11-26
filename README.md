# admet_autoresearch

An agent experiments on TDC's [Caco2_Wang](https://tdcommons.ai/benchmark/admet_group/01caco2/)
benchmark: predict Caco-2 cell permeability from a SMILES string. It edits
`train.py`, runs a five-seed evaluation, keeps the change if validation MAE
improved, and repeats.

Framework after [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Files

| file | role | who runs it |
| --- | --- | --- |
| `setup.py` | Downloads the benchmark, writes the splits to disk | you, once |
| `prepare.py` | `run_validation()` and `run_audit()`. The metric. | imported |
| `train.py` | `featurize()` and `make_model()` | the agent edits |
| `experiment.py` | Scores validation, decides keep/discard, moves git | the agent |
| `audit.py` | Scores the held-out test set | you, rarely |
| `program.md` | Instructions for the agent | you edit |

## Quick start

```bash
uv sync
uv run setup.py
uv run train.py
```

`setup.py` downloads the benchmark (1.5 MB) and splits it into two trees:

```
data/splits/seed{1..5}/train.csv   the agent's world
data/splits/seed{1..5}/valid.csv
data/holdout/test.csv              the audit set
data/holdout/raw/                  raw download, holds test labels
```

`prepare.run_validation()` names no path under `data/holdout`, so the
validation process never has test labels in memory. Verified by setting
`data/holdout` to mode `000`: validation still scored 0.435680, and
`audit.py` failed with `PermissionError`.

## Running the agent

Launch it with the restricted settings, so the limits apply to that session
and not to your own:

```bash
claude --settings agent-settings.json
```

Then ask it to read `program.md`. `agent-settings.json` gives that session:

- **one shell command**, `uv run experiment.py` — everything else is refused
- **no network at all**, since the loop reads local CSVs and needs none
- **no reads** of `data/holdout`, the audit log or `~/.claude`
- **no edits** outside `train.py`

Verify with `/sandbox`. `failIfUnavailable` means the session refuses to
start rather than running unsandboxed, so if it launches, the sandbox is on.

Run `uv sync` and `setup.py` before this, in your own session — they are the
only steps that need network.

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
