"""Drives the campaign. You run this; it runs the agent.

    uv run runner.py --iterations 20

Each iteration spawns a headless agent that edits train.py and exits. The
agent does not score, decide, commit or revert -- this process does, after
the agent is gone. It cannot see the result of its own change until the
next iteration reads results.tsv.

    --iterations N   how many experiments to run (default 10)
    --model NAME     sonnet (default) or haiku. Opus 5.5 refuses this prompt;
                     its safeguards flag drug-property work as bio-adjacent.
    --effort LEVEL   low | medium | high (default) | xhigh | max
    --no-agent       skip the spawn and score train.py as it stands, for
                     checking the harness without burning tokens
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import prepare

# Measured noise floor: the smallest paired difference distinguishable on
# 91 validation molecules across five seeds is about 0.027 MAE.
KEEP_THRESHOLD = 0.03
# A change that scores the same but uses this fraction of the features or
# fewer is kept as a simplification. Same accuracy from less model is a win,
# and it is not a claim of improvement, so it cannot bank noise.
SIMPLIFY_RATIO = 0.75
AGENT_TIMEOUT = 900

RUNS_DIR = Path(".runs")
LOG = RUNS_DIR / "log.tsv"
STATE = RUNS_DIR / "state.json"
RESULTS = Path("results.tsv")
SETTINGS = "agent-settings.json"

LEDGER_ROWS = 40

PROMPT = """Read program.md, then make ONE change to train.py.

{ledger}

You may edit train.py and nothing else. Do not run any command -- you have no
shell. Do not score your change; this process does that after you exit, and
the outcome reaches the next iteration as a row in the ledger above.

End your reply with exactly these two lines:
IDEA: <at most twelve words naming the change>
WHY: <one sentence: the hypothesis, and what the scores would show if it holds>"""


def _ledger():
    """The campaign so far, handed over rather than left to be found."""
    if not RESULTS.exists():
        return "No experiments yet. You are the first; establish something simple."
    rows = RESULTS.read_text().strip().splitlines()
    head, body = rows[0], rows[1:]
    if len(body) > LEDGER_ROWS:
        body = body[-LEDGER_ROWS:]
    return (
        "Every experiment so far, oldest first. Do not repeat a failure; "
        "build on what held.\n\n"
        + "\n".join([head, *body])
    )


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _claude_bin():
    """$CLAUDE_BIN, then PATH, then the Claude desktop app's bundled copy."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    found = shutil.which("claude")
    if found:
        return found
    bundles = sorted(
        (Path.home() / "Library/Application Support/Claude/claude-code").glob(
            "*/*/claude.app/Contents/MacOS/claude"
        )
    )
    return str(bundles[-1]) if bundles else None


def _spawn_agent(model, effort, iteration):
    """Let an agent edit train.py. Returns its one-line description."""
    binary = _claude_bin()
    if not binary:
        raise SystemExit(
            "No claude binary found. Install the CLI, or set CLAUDE_BIN to one."
        )
    result = subprocess.run(
        [binary, "-p", PROMPT.format(ledger=_ledger()), "--settings", SETTINGS,
         "--model", model, "--effort", effort],
        capture_output=True, text=True, timeout=AGENT_TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    out = (result.stdout or "") + (result.stderr or "")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"agent-{iteration:03d}.log").write_text(out)

    if "Please run /login" in out:
        raise SystemExit(
            f"{binary} is not logged in. Run it once interactively and sign in,\n"
            "or point CLAUDE_BIN at a CLI that is."
        )
    if "API Error" in out:
        # Surface it rather than reporting a silent "no change". Drug-property
        # work sometimes trips the bio safeguards; --model picks another.
        first = next(l for l in out.splitlines() if "API Error" in l)
        raise SystemExit(f"Agent call failed:\n  {first.strip()[:300]}")

    idea, why = "(no description given)", ""
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("WHY:") and not why:
            why = s.split("WHY:", 1)[1].strip()[:240]
        elif s.startswith("IDEA:"):
            idea = s.split("IDEA:", 1)[1].strip()[:90]
            break
    return idea, why


def _load_train():
    """Re-import train.py, which changed on disk since the last iteration."""
    sys.modules.pop("train", None)
    importlib.invalidate_caches()
    import train
    return train


def _load_state():
    """The best score survives between invocations, so a campaign resumes."""
    if STATE.exists():
        s = json.loads(STATE.read_text())
        return s.get("best_valid_mae"), s.get("keeps", 0), s.get("best_features")
    return None, 0, None


def _save_state(best, keeps, features):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"best_valid_mae": best, "keeps": keeps, "best_features": features},
        indent=2,
    ))


def _verdict(mae, features, best, best_features):
    """keep | simplify | discard. The only place this is decided."""
    if best is None or best - mae > KEEP_THRESHOLD:
        return "keep"
    same_score = abs(mae - best) <= KEEP_THRESHOLD
    if same_score and best_features and features <= SIMPLIFY_RATIO * best_features:
        return "simplify"
    return "discard"


def _append(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = "\t".join(header)
    if path.exists() and path.read_text().split("\n", 1)[0] != line:
        # Columns changed since this file was started. Keep it, start fresh.
        path.rename(path.with_suffix(path.suffix + ".bak"))
    if not path.exists():
        path.write_text(line + "\n")
    with path.open("a") as fh:
        fh.write("\t".join(str(c).replace("\t", " ") for c in row) + "\n")


def _record(m, status, idea, why, best, keeps, note):
    commit = _git("rev-parse", "--short", "HEAD")
    train_mae = f"{m['train_mae']:.6f}" if m else "0.000000"
    valid_mae = f"{m['mae']:.6f}" if m else "0.000000"
    feats = str(m["n_features"]) if m else "0"
    _append(
        RESULTS,
        ["train_mae", "valid_mae", "n_features", "status", "idea", "why"],
        [train_mae, valid_mae, feats, status, idea, why],
    )
    _append(
        LOG,
        ["timestamp", "commit", "train_mae", "valid_mae", "n_features",
         "best_valid_mae", "keeps", "status", "idea", "why", "note"],
        [datetime.now().isoformat(timespec="seconds"), commit, train_mae,
         valid_mae, feats, f"{best:.6f}" if best else "", keeps, status,
         idea, why, note or ""],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    best, keeps, best_features = _load_state()
    if best is not None:
        print(f"resuming — best valid_mae {best:.6f}, {keeps} kept so far")
    else:
        # Score train.py as it stands before any agent touches it, so the
        # bar is the baseline rather than whatever the first agent wrote.
        print("no state — scoring the baseline first", flush=True)
        train = _load_train()
        m = prepare.run_validation(train.featurize, train.make_model)
        best, keeps, best_features = m["mae"], 0, m["n_features"]
        _record(m, "baseline", "baseline as committed", "", best, keeps, None)
        _save_state(best, keeps, best_features)
        print(f"  BASELINE valid_mae {best:.4f}  train_mae {m['train_mae']:.4f}")
    kept_here = 0

    for i in range(1, args.iterations + 1):
        print(f"\n=== iteration {i} of {args.iterations} ===", flush=True)

        if args.no_agent:
            idea, why = "(no-agent: scoring train.py as it stands)", ""
        else:
            before = Path("train.py").read_text()
            idea, why = _spawn_agent(args.model, args.effort, i)
            if Path("train.py").read_text() == before:
                print(f"  no change to train.py -- {idea}")
                _record(None, "no-change", idea, why, best, keeps, None)
                continue
            print(f"  agent: {idea}")
            if why:
                print(f"     why: {why}")

        # Everything below here runs with the agent gone.
        try:
            train = _load_train()
            m = prepare.run_validation(train.featurize, train.make_model)
        except Exception:
            traceback.print_exc()
            _git("checkout", "--", "train.py")
            _record(None, "crash", idea, why, best, keeps, "exception")
            print("  CRASH -- reverted")
            continue

        status = _verdict(m["mae"], m["n_features"], best, best_features)
        if status in ("keep", "simplify"):
            best, best_features = m["mae"], m["n_features"]
            keeps, kept_here = keeps + 1, kept_here + 1
            _git("add", "-A", "train.py")
            _git("commit", "-q", "-m", idea)
        else:
            _git("checkout", "--", "train.py")

        _record(m, status, idea, why, best, keeps, None)
        _save_state(best, keeps, best_features)
        print(f"  {status.upper():8} valid_mae {m['mae']:.4f}  "
              f"train_mae {m['train_mae']:.4f}  "
              f"features {m['n_features']}  best {best:.4f}")

    if best is None:
        print("\nDone. Nothing scored.")
    else:
        print(f"\nDone. {kept_here} kept of {args.iterations} this run, "
              f"{keeps} in the campaign. Best valid_mae {best:.6f}")


if __name__ == "__main__":
    main()
