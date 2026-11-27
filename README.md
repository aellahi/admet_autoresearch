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
| `runner.py` | The loop. Spawns the agent, scores, keeps or reverts | you |
| `prepare.py` | `run_validation()` and `run_audit()`. The metric. | imported |
| `train.py` | `featurize()` and `make_model()` | the agent edits |
| `audit.py` | Scores the held-out test set | you, rarely |
| `program.md` | The brief handed to each spawned agent | you edit |

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

## Running a campaign

```bash
uv run runner.py --iterations 20
```

`runner.py` owns the loop. Each iteration it spawns a headless agent that
edits `train.py` and exits; then, with the agent gone, it scores the change
on the five validation seeds, keeps it if `valid_mae` improved by more than
0.03, and otherwise reverts. The agent never scores, decides or commits, and
sees the outcome only as a row in `results.tsv` on its next turn.

Progress is saved in `.runs/state.json`, so a second invocation resumes the
campaign rather than restarting it.

```bash
uv run runner.py --iterations 2 --no-agent    # score train.py as it stands
```

**Before the first run**, the spawned agent needs a signed-in CLI.
`runner.py` looks at `$CLAUDE_BIN`, then `claude` on `PATH`, then the copy
inside Claude Desktop — which is not signed in on its own. Install the CLI
and sign in once, or point `CLAUDE_BIN` at one that is.

`agent-settings.json` gives the spawned agent:

- **no shell at all** — `Bash` is denied outright; it does not need one
- **no network**
- **no reads** of `data/holdout`, the audit log or `~/.claude`
- **no edits** outside `train.py`

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
