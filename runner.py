"""Drives the campaign. You run this; it runs the agent.

    uv run runner.py --iterations 20

Each iteration spawns a headless agent that edits train.py and exits. The
agent does not score, decide, commit or revert -- this process does, after
the agent is gone. It cannot see the result of its own change until the
next iteration reads results.tsv.

    --iterations N   how many experiments to run (default 10)
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
AGENT_TIMEOUT = 900

RUNS_DIR = Path(".runs")
LOG = RUNS_DIR / "log.tsv"
STATE = RUNS_DIR / "state.json"
RESULTS = Path("results.tsv")
SETTINGS = "agent-settings.json"

PROMPT = """Read program.md and results.tsv, then make ONE change to train.py.

You may edit train.py and nothing else. Do not run any command -- you have no
shell. Do not score your change; this process does that after you exit, and
you will see the outcome in results.tsv next time.

End your reply with a single line:
IDEA: <at most twelve words describing what you changed>"""


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


def _spawn_agent():
    """Let an agent edit train.py. Returns its one-line description."""
    binary = _claude_bin()
    if not binary:
        raise SystemExit(
            "No claude binary found. Install the CLI, or set CLAUDE_BIN to one."
        )
    result = subprocess.run(
        [binary, "-p", PROMPT, "--settings", SETTINGS],
        capture_output=True, text=True, timeout=AGENT_TIMEOUT,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if "Please run /login" in out:
        raise SystemExit(
            f"{binary} is not logged in. Run it once interactively and sign in,\n"
            "or point CLAUDE_BIN at a CLI that is."
        )
    for line in reversed(out.splitlines()):
        if line.strip().startswith("IDEA:"):
            return line.split("IDEA:", 1)[1].strip()[:90]
    return "(no description given)"


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
        return s.get("best_valid_mae"), s.get("keeps", 0)
    return None, 0


def _save_state(best, keeps):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"best_valid_mae": best, "keeps": keeps}, indent=2))


def _append(path, header, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("\t".join(header) + "\n")
    with path.open("a") as fh:
        fh.write("\t".join(str(c) for c in row) + "\n")


def _record(m, status, description, best, keeps, note):
    commit = _git("rev-parse", "--short", "HEAD")
    train_mae = f"{m['train_mae']:.6f}" if m else "0.000000"
    valid_mae = f"{m['mae']:.6f}" if m else "0.000000"
    _append(
        RESULTS,
        ["commit", "train_mae", "valid_mae", "status", "description"],
        [commit, train_mae, valid_mae, status, description],
    )
    _append(
        LOG,
        ["timestamp", "commit", "train_mae", "valid_mae", "best_valid_mae",
         "keeps", "status", "description", "note"],
        [datetime.now().isoformat(timespec="seconds"), commit, train_mae,
         valid_mae, f"{best:.6f}" if best else "", keeps, status,
         description, note or ""],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--no-agent", action="store_true")
    args = ap.parse_args()

    best, keeps = _load_state()
    if best is not None:
        print(f"resuming — best valid_mae {best:.6f}, {keeps} kept so far")
    kept_here = 0

    for i in range(1, args.iterations + 1):
        print(f"\n=== iteration {i} of {args.iterations} ===", flush=True)

        if args.no_agent:
            description = "(no-agent: scoring train.py as it stands)"
        else:
            before = Path("train.py").read_text()
            description = _spawn_agent()
            if Path("train.py").read_text() == before:
                print(f"  no change to train.py -- {description}")
                _record(None, "no-change", description, best, keeps, None)
                continue
            print(f"  agent: {description}")

        # Everything below here runs with the agent gone.
        try:
            train = _load_train()
            m = prepare.run_validation(train.featurize, train.make_model)
        except Exception:
            traceback.print_exc()
            _git("checkout", "--", "train.py")
            _record(None, "crash", description, best, keeps, "exception")
            print("  CRASH -- reverted")
            continue

        if best is None or best - m["mae"] > KEEP_THRESHOLD:
            status, best, keeps = "keep", m["mae"], keeps + 1
            kept_here += 1
            _git("add", "-A", "train.py")
            _git("commit", "-q", "-m", description)
        else:
            status = "discard"
            _git("checkout", "--", "train.py")

        _record(m, status, description, best, keeps, None)
        _save_state(best, keeps)
        print(f"  {status.upper():8} valid_mae {m['mae']:.4f}  "
              f"train_mae {m['train_mae']:.4f}  best {best:.4f}")

    if best is None:
        print("\nDone. Nothing scored.")
    else:
        print(f"\nDone. {kept_here} kept of {args.iterations} this run, "
              f"{keeps} in the campaign. Best valid_mae {best:.6f}")


if __name__ == "__main__":
    main()
