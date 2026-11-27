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
import time
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
HYPOTHESIS: <one sentence: what you expect and what the scores would show if it holds>"""


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
    # No timeout. A slow agent is a slow agent, not a failure.
    t0 = time.time()
    result = subprocess.run(
        [binary, "-p", PROMPT.format(ledger=_ledger()), "--settings", SETTINGS,
         "--model", model, "--effort", effort],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    agent_seconds = time.time() - t0
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

    idea, hypothesis = "(no description given)", ""
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("HYPOTHESIS:") and not hypothesis:
            hypothesis = s.split("HYPOTHESIS:", 1)[1].strip()[:240]
        elif s.startswith("IDEA:"):
            idea = s.split("IDEA:", 1)[1].strip()[:90]
            break
    return idea, hypothesis, agent_seconds


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


def _outcome(status, mae, features, best, best_features):
    """Why the verdict came out that way, in the ledger's own terms.

    Computed here rather than asked of the agent: the arithmetic is
    determinate, and an agent asked to explain its own rejection will
    sometimes invent a reason.
    """
    if status == "baseline":
        return "first measurement; the bar every later change must beat"
    if best is None:
        return "nothing to compare against"
    margin = best - mae
    if status == "keep":
        return (f"beat {best:.4f} by {margin:+.4f}, over the {KEEP_THRESHOLD} bar")
    if status == "simplify":
        return (f"within noise of {best:.4f} ({margin:+.4f}) using {features} "
                f"features against {best_features}; kept as a simplification")
    if margin > 0:
        return (f"better than {best:.4f} by {margin:+.4f}, but under the "
                f"{KEEP_THRESHOLD} bar, so indistinguishable from noise")
    return f"worse than {best:.4f} by {-margin:.4f}"


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


def _record(m, status, idea, hypothesis, outcome, best, keeps, agent_s):
    commit = _git("rev-parse", "--short", "HEAD")
    train_mae = f"{m['train_mae']:.6f}" if m else "0.000000"
    valid_mae = f"{m['mae']:.6f}" if m else "0.000000"
    feats = str(m["n_features"]) if m else "0"
    f = lambda k: f"{m[k]:.1f}" if m else "0.0"
    _append(
        RESULTS,
        ["train_mae", "valid_mae", "n_features", "status", "idea",
         "hypothesis", "outcome"],
        [train_mae, valid_mae, feats, status, idea, hypothesis, outcome],
    )
    _append(
        LOG,
        ["timestamp", "commit", "train_mae", "valid_mae", "n_features",
         "best_valid_mae", "keeps", "status", "agent_s", "featurize_s",
         "fit_s", "predict_s", "score_s", "iteration_s", "idea",
         "hypothesis", "outcome"],
        [datetime.now().isoformat(timespec="seconds"), commit, train_mae,
         valid_mae, feats, f"{best:.6f}" if best else "", keeps, status,
         f"{agent_s:.1f}", f("featurize_seconds"), f("fit_seconds"),
         f("predict_seconds"), f("seconds"),
         f"{agent_s + (m['seconds'] if m else 0):.1f}",
         idea, hypothesis, outcome],
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
        _record(m, "baseline", "baseline as committed", "",
                _outcome("baseline", best, best_features, None, None),
                best, keeps, 0.0)
        _save_state(best, keeps, best_features)
        print(f"  BASELINE valid_mae {best:.4f}  train_mae {m['train_mae']:.4f}")
    kept_here = 0

    for i in range(1, args.iterations + 1):
        print(f"\n=== iteration {i} of {args.iterations} ===", flush=True)

        if args.no_agent:
            idea, hypothesis, agent_s = "(no-agent: scoring as it stands)", "", 0.0
        else:
            before = Path("train.py").read_text()
            idea, hypothesis, agent_s = _spawn_agent(args.model, args.effort, i)
            if Path("train.py").read_text() == before:
                print(f"  no change to train.py -- {idea}")
                _record(None, "no-change", idea, hypothesis,
                        "the agent left train.py untouched", best, keeps, agent_s)
                continue
            print(f"  agent: {idea}  [{agent_s:.0f}s thinking]")
            if hypothesis:
                print(f"    expects: {hypothesis}")

        # Everything below here runs with the agent gone.
        try:
            train = _load_train()
            m = prepare.run_validation(train.featurize, train.make_model)
        except Exception as exc:
            traceback.print_exc()
            _git("checkout", "--", "train.py")
            _record(None, "crash", idea, hypothesis,
                    f"raised {type(exc).__name__}: {str(exc)[:120]}",
                    best, keeps, agent_s)
            print("  CRASH -- reverted")
            continue

        prev_best, prev_features = best, best_features
        status = _verdict(m["mae"], m["n_features"], best, best_features)
        if status in ("keep", "simplify"):
            # best only ever ratchets down. A simplify keeps the smaller
            # model but must not raise the bar, or a chain of them walks
            # the score back toward the baseline one noise step at a time.
            best = min(best, m["mae"]) if best is not None else m["mae"]
            best_features = m["n_features"]
            keeps, kept_here = keeps + 1, kept_here + 1
            _git("add", "-A", "train.py")
            _git("commit", "-q", "-m", idea)
        else:
            _git("checkout", "--", "train.py")

        outcome = _outcome(status, m["mae"], m["n_features"],
                           prev_best, prev_features)
        print(f"    {outcome}")
        _record(m, status, idea, hypothesis, outcome, best, keeps, agent_s)
        _save_state(best, keeps, best_features)
        print(f"  {status.upper():8} valid_mae {m['mae']:.4f}  "
              f"train_mae {m['train_mae']:.4f}  "
              f"features {m['n_features']}  best {best:.4f}")
        print(f"    {agent_s + m['seconds']:.0f}s total — "
              f"{agent_s:.0f}s agent, {m['featurize_seconds']:.0f}s featurize, "
              f"{m['fit_seconds']:.0f}s fit, {m['predict_seconds']:.0f}s predict")

    if best is None:
        print("\nDone. Nothing scored.")
    else:
        print(f"\nDone. {kept_here} kept of {args.iterations} this run, "
              f"{keeps} in the campaign. Best valid_mae {best:.6f}")


if __name__ == "__main__":
    main()
