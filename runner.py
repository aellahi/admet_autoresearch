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
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import prepare
import report

# Measured noise floor: the smallest paired difference distinguishable on
# 91 validation molecules across five seeds is about 0.027 MAE.
KEEP_THRESHOLD = 0.03
# A change that scores the same but uses this fraction of the features or
# fewer is kept as a simplification. Same accuracy from less model is a win,
# and it is not a claim of improvement, so it cannot bank noise.
SIMPLIFY_RATIO = 0.75
# How many simplifications may be accepted back to back. Each one is judged
# against the model before it, so a long chain can still travel a long way;
# a keep has to intervene before shrinking resumes.
MAX_SIMPLIFY_CHAIN = 3
# Whole-iteration budget: the agent's thinking and the scoring share it.
# Overrun is logged as a crash and the edit is reverted.
ITERATION_SECONDS = 3600

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


class Overran(Exception):
    """The iteration used its whole budget."""


class AgentFailed(Exception):
    """The agent call came back as an error. One iteration lost, not the run."""


# How many agent calls may fail back to back before the campaign gives up.
MAX_CONSECUTIVE_FAILURES = 8


def _alarm(signum, frame):
    raise Overran("scoring ran past the iteration budget")


def _spawn_agent(model, effort, iteration, budget):
    """Let an agent edit train.py. Returns idea, hypothesis, seconds."""
    binary = _claude_bin()
    if not binary:
        raise SystemExit(
            "No claude binary found. Install the CLI, or set CLAUDE_BIN to one."
        )
    t0 = time.time()
    try:
        result = subprocess.run(
            [binary, "-p", PROMPT.format(ledger=_ledger()), "--settings",
             SETTINGS, "--model", model, "--effort", effort],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        raise Overran(f"the agent ran past the {budget:.0f}s budget")
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
        # Recoverable. Drug-property work intermittently trips the bio
        # safeguards on every model we can use, so a flagged call costs one
        # iteration rather than the campaign.
        first = next(l for l in out.splitlines() if "API Error" in l)
        raise AgentFailed(first.strip()[:200])

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


DEFAULT_STATE = {
    "best_valid_mae": None,    # best ever kept: the bar a keep must beat
    "best_commit": None,       # the commit that achieved it, so it is findable
    "current_valid_mae": None, # what the committed model scores right now
    "current_features": None,
    "simplify_chain": 0,
    "keeps": 0,
}


def _load_state():
    """Campaign state survives between invocations, so a run resumes."""
    s = dict(DEFAULT_STATE)
    if STATE.exists():
        s.update(json.loads(STATE.read_text()))
    return s


def _save_state(s):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def _verdict(mae, features, s):
    """keep | simplify | discard. The only place this is decided."""
    best = s["best_valid_mae"]
    if best is None or best - mae > KEEP_THRESHOLD:
        return "keep"

    # A simplification is judged against the model that is actually
    # committed, not against a better score some earlier model set. Judged
    # against the pinned best, a chain of them walks the committed model
    # away from the bar it is still being measured by.
    current, current_features = s["current_valid_mae"], s["current_features"]
    if current is None or not current_features:
        return "discard"
    if s["simplify_chain"] >= MAX_SIMPLIFY_CHAIN:
        return "discard"
    if (abs(mae - current) <= KEEP_THRESHOLD
            and features <= SIMPLIFY_RATIO * current_features):
        return "simplify"
    return "discard"


def _outcome(status, mae, features, s):
    """Why the verdict came out that way, in the ledger's own terms.

    Computed here rather than asked of the agent: the arithmetic is
    determinate, and an agent asked to explain its own rejection will
    sometimes invent a reason.
    """
    if status == "baseline":
        return "first measurement; the bar every later change must beat"
    best, current = s["best_valid_mae"], s["current_valid_mae"]
    if best is None:
        return "nothing to compare against"
    margin = best - mae
    if status == "keep":
        return f"beat {best:.4f} by {margin:+.4f}, over the {KEEP_THRESHOLD} bar"
    if status == "simplify":
        return (f"within noise of the committed {current:.4f} "
                f"({current - mae:+.4f}) using {features} features against "
                f"{s['current_features']}; kept as a simplification")
    if s["simplify_chain"] >= MAX_SIMPLIFY_CHAIN and features < (s["current_features"] or 0):
        return (f"smaller, but {MAX_SIMPLIFY_CHAIN} simplifications have run "
                f"back to back; the next keep has to come from a better score")
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


def _record(m, status, idea, hypothesis, outcome, s, agent_s):
    commit = _git("rev-parse", "--short", "HEAD")
    train_mae = f"{m['train_mae']:.6f}" if m else "0.000000"
    valid_mae = f"{m['mae']:.6f}" if m else "0.000000"
    feats = str(m["n_features"]) if m else "0"
    best = s["best_valid_mae"]
    f = lambda k: f"{m[k]:.1f}" if m else "0.0"
    _append(
        RESULTS,
        ["commit", "train_mae", "valid_mae", "n_features", "status", "idea",
         "hypothesis", "outcome"],
        [commit, train_mae, valid_mae, feats, status, idea, hypothesis, outcome],
    )
    _append(
        LOG,
        ["timestamp", "commit", "train_mae", "valid_mae", "n_features",
         "best_valid_mae", "best_commit", "keeps", "status", "agent_s",
         "featurize_s", "fit_s", "predict_s", "score_s", "iteration_s",
         "idea", "hypothesis", "outcome"],
        [datetime.now().isoformat(timespec="seconds"), commit, train_mae,
         valid_mae, feats, f"{best:.6f}" if best else "",
         s["best_commit"] or "", s["keeps"], status,
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
    ap.add_argument("--time-limit", type=int, default=ITERATION_SECONDS,
                    help="seconds per iteration, agent and scoring together")
    args = ap.parse_args()

    s = _load_state()
    if s["best_valid_mae"] is not None:
        print(f"resuming — best {s['best_valid_mae']:.6f} "
              f"({s['best_commit'] or 'commit unrecorded'}), "
              f"committed model {s['current_valid_mae']:.6f} at "
              f"{s['current_features']} features, {s['keeps']} kept")
    else:
        # Score train.py as it stands before any agent touches it, so the
        # bar is the baseline rather than whatever the first agent wrote.
        print("no state — scoring the baseline first", flush=True)
        train = _load_train()
        m = prepare.run_validation(train.featurize, train.make_model)
        s.update(best_valid_mae=m["mae"],
                 best_commit=_git("rev-parse", "--short", "HEAD"),
                 current_valid_mae=m["mae"],
                 current_features=m["n_features"])
        _record(m, "baseline", "baseline as committed", "",
                _outcome("baseline", m["mae"], m["n_features"], s), s, 0.0)
        _save_state(s)
        print(f"  BASELINE valid_mae {m['mae']:.4f}  "
              f"train_mae {m['train_mae']:.4f}")
    kept_here, failures = 0, 0

    for i in range(1, args.iterations + 1):
        print(f"\n=== iteration {i} of {args.iterations} ===", flush=True)

        started = time.time()
        if args.no_agent:
            idea, hypothesis, agent_s = "(no-agent: scoring as it stands)", "", 0.0
        else:
            before = Path("train.py").read_text()
            try:
                idea, hypothesis, agent_s = _spawn_agent(
                    args.model, args.effort, i, args.time_limit
                )
                failures = 0
            except Overran as exc:
                _git("checkout", "--", "train.py")
                _record(None, "crash", "(agent overran)", "", str(exc),
                        s, args.time_limit)
                print(f"  OVERRAN -- {exc}; reverted")
                continue
            except AgentFailed as exc:
                failures += 1
                _git("checkout", "--", "train.py")
                _record(None, "api-error", "(agent call failed)", "",
                        str(exc), s, 0.0)
                print(f"  API ERROR ({failures} in a row) -- {exc}")
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nStopping: {failures} agent calls failed back to "
                          f"back. The campaign is resumable -- rerun when the "
                          f"calls go through again.")
                    break
                continue
            if Path("train.py").read_text() == before:
                print(f"  no change to train.py -- {idea}")
                _record(None, "no-change", idea, hypothesis,
                        "the agent left train.py untouched", s, agent_s)
                continue
            print(f"  agent: {idea}  [{agent_s:.0f}s thinking]")
            if hypothesis:
                print(f"    expects: {hypothesis}")

        # Everything below here runs with the agent gone. Whatever is left
        # of the iteration's budget is the scoring's to use.
        remaining = max(1, int(args.time_limit - (time.time() - started)))
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(remaining)
        try:
            train = _load_train()
            m = prepare.run_validation(train.featurize, train.make_model)
        except Exception as exc:
            traceback.print_exc()
            _git("checkout", "--", "train.py")
            _record(None, "crash", idea, hypothesis,
                    f"raised {type(exc).__name__}: {str(exc)[:120]}",
                    s, agent_s)
            print("  CRASH -- reverted")
            continue
        finally:
            signal.alarm(0)

        status = _verdict(m["mae"], m["n_features"], s)
        outcome = _outcome(status, m["mae"], m["n_features"], s)

        if status in ("keep", "simplify"):
            _git("add", "-A", "train.py")
            _git("commit", "-q", "-m", idea)
            commit = _git("rev-parse", "--short", "HEAD")
            # The bar ratchets down only. The committed model's own score
            # moves with every accepted change, and is what a later
            # simplification is judged against.
            if s["best_valid_mae"] is None or m["mae"] < s["best_valid_mae"]:
                s["best_valid_mae"], s["best_commit"] = m["mae"], commit
            s["current_valid_mae"] = m["mae"]
            s["current_features"] = m["n_features"]
            s["simplify_chain"] = s["simplify_chain"] + 1 if status == "simplify" else 0
            s["keeps"] += 1
            kept_here += 1
        else:
            _git("checkout", "--", "train.py")

        print(f"    {outcome}")
        _record(m, status, idea, hypothesis, outcome, s, agent_s)
        _save_state(s)
        print(f"  {status.upper():8} valid_mae {m['mae']:.4f}  "
              f"train_mae {m['train_mae']:.4f}  "
              f"features {m['n_features']}  "
              f"best {s['best_valid_mae']:.4f}  "
              f"committed {s['current_valid_mae']:.4f}")
        print(f"    {agent_s + m['seconds']:.0f}s total — "
              f"{agent_s:.0f}s agent, {m['featurize_seconds']:.0f}s featurize, "
              f"{m['fit_seconds']:.0f}s fit, {m['predict_seconds']:.0f}s predict")
        # Keep the dashboard's data file current as the campaign runs, so a
        # refresh is only a republish.
        try:
            report.build()
        except Exception as exc:
            print(f"    (report.json not rebuilt: {exc})")

    if s["best_valid_mae"] is None:
        print("\nDone. Nothing scored.")
    else:
        print(f"\nDone. {kept_here} kept of {args.iterations} this run, "
              f"{s['keeps']} in the campaign.\n"
              f"Best valid_mae {s['best_valid_mae']:.6f} at "
              f"{s['best_commit']}; committed model scores "
              f"{s['current_valid_mae']:.6f}.")


if __name__ == "__main__":
    main()
