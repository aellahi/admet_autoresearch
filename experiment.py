"""Owns scoring, the keep/discard decision and git state. Do not modify.

    uv run experiment.py "what this experiment tried"

Scores the current train.py on the validation split, then commits it or
reverts it. The agent does not make this call; it only decides what to try.

Nothing here touches the test split. Scoring the audit set is audit.py,
which you run yourself.
"""

import json
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import prepare

# Measured noise floor: the smallest paired difference distinguishable on
# 91 validation molecules across five seeds is about 0.027 MAE.
KEEP_THRESHOLD = 0.03

RUNS_DIR = Path(".runs")
LOG = RUNS_DIR / "log.tsv"
STATE = RUNS_DIR / "state.json"
RESULTS = Path("results.tsv")


class Timeout(Exception):
    pass


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"best_valid_mae": None, "keeps": 0}


def _append(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("\t".join(header) + "\n")
    with path.open("a") as fh:
        fh.write("\t".join(str(c) for c in row) + "\n")


def main():
    description = " ".join(sys.argv[1:]).strip() or "(no description)"
    state = _load_state()
    best = state["best_valid_mae"]

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
    signal.alarm(prepare.TIMEOUT_SECONDS)
    try:
        import train
        m = prepare.run_validation(train.featurize, train.make_model)
    except Timeout:
        _finish(state, None, "crash", description,
                f"timeout after {prepare.TIMEOUT_SECONDS}s")
        return
    except Exception:
        traceback.print_exc()
        _finish(state, None, "crash", description, "exception")
        return
    finally:
        signal.alarm(0)

    if best is None:
        status = "keep"          # the first run is the baseline
    elif best - m["mae"] > KEEP_THRESHOLD:
        status = "keep"
    else:
        status = "discard"

    _finish(state, m, status, description, None)


def _finish(state, m, status, description, note):
    if status == "keep":
        _git("add", "-A", "train.py")
        _git("commit", "-q", "-m", description)
        state["best_valid_mae"] = m["mae"]
        state["keeps"] += 1
    else:
        _git("checkout", "--", "train.py")

    commit = _git("rev-parse", "--short", "HEAD")
    train_mae = f"{m['train_mae']:.6f}" if m else "0.000000"
    valid_mae = f"{m['mae']:.6f}" if m else "0.000000"

    for path, header, row in [
        (RESULTS,
         ["commit", "train_mae", "valid_mae", "status", "description"],
         [commit, train_mae, valid_mae, status, description]),
        (LOG,
         ["timestamp", "commit", "train_mae", "valid_mae", "best_valid_mae",
          "keeps", "status", "description", "note"],
         [datetime.now().isoformat(timespec="seconds"), commit, train_mae,
          valid_mae,
          f"{state['best_valid_mae']:.6f}" if state["best_valid_mae"] else "",
          state["keeps"], status, description, note or ""]),
    ]:
        _append(path, header, row)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))

    print("---")
    print(f"status:           {status.upper()}")
    if m:
        print(f"valid_mae:        {m['mae']:.6f}")
        print(f"valid_mae_std:    {m['mae_std']:.6f}")
        print(f"train_mae:        {m['train_mae']:.6f}")
        print(f"n_features:       {m['n_features']}")
        print(f"peak_memory_gb:   {m['peak_memory_gb']:.2f}")
        print(f"total_seconds:    {m['seconds']:.1f}")
    if note:
        print(f"note:             {note}")


if __name__ == "__main__":
    main()
