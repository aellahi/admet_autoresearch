"""Turn the ledger into JSON for the progress dashboard.

    uv run report.py            # writes .runs/report.json
    uv run report.py --print    # and dumps it to stdout

Reads results.tsv, .runs/log.tsv and the audit log. Nothing here is part of
the campaign; it only summarises what already happened.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

RESULTS = Path("results.tsv")
LOG = Path(".runs/log.tsv")
AUDIT = Path.home() / ".cache" / "admet_autoresearch" / "audit.tsv"
OUT = Path(".runs/report.json")

KEPT = {"keep", "simplify", "baseline"}


def _rows(path):
    if not path.exists():
        return []
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    head = lines[0].split("\t")
    out = []
    for line in lines[1:]:
        cells = (line.split("\t") + [""] * len(head))[:len(head)]
        out.append(dict(zip(head, cells)))
    return out


def _f(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="echo")
    args = ap.parse_args()

    ledger, log, audit = _rows(RESULTS), _rows(LOG), _rows(AUDIT)

    experiments, best, running = [], None, []
    for i, r in enumerate(ledger, start=1):
        status = r.get("status", "")
        valid = _f(r.get("valid_mae"))
        train = _f(r.get("train_mae"))
        scored = status not in ("crash", "no-change")
        # The bar moves only on a kept change, and only downward -- the same
        # ratchet runner.py applies. A discarded run can score lower without
        # becoming the bar.
        if status in KEPT and valid is not None:
            best = valid if best is None else min(best, valid)
        running.append(best)
        timing = log[i - 1] if i - 1 < len(log) else {}
        experiments.append({
            "n": i,
            "status": status,
            "kept": status in KEPT,
            "train_mae": train if scored else None,
            "valid_mae": valid if scored else None,
            "best_so_far": best,
            "n_features": int(_f(r.get("n_features"), 0) or 0),
            "idea": r.get("idea", ""),
            "hypothesis": r.get("hypothesis", ""),
            "outcome": r.get("outcome", ""),
            "agent_s": _f(timing.get("agent_s"), 0.0),
            "featurize_s": _f(timing.get("featurize_s"), 0.0),
            "fit_s": _f(timing.get("fit_s"), 0.0),
            "predict_s": _f(timing.get("predict_s"), 0.0),
            "iteration_s": _f(timing.get("iteration_s"), 0.0),
        })

    audits = [{
        "test_mae": _f(a.get("test_mae")),
        "train_mae": _f(a.get("train_mae")),
        "n_features": int(_f(a.get("n_features"), 0) or 0),
        "iteration": int(_f(a.get("iteration"), 0) or 0),
        "at": a.get("timestamp", ""),
    } for a in audit if _f(a.get("test_mae")) is not None]

    scored_only = [e for e in experiments if e["valid_mae"] is not None]

    # Where the campaign started, on all three splits. The held-out figure
    # comes from an audit recorded at iteration 0; without one it stays None
    # rather than borrowing a later number.
    first = scored_only[0] if scored_only else {}
    pre = next((a for a in audits if a["iteration"] == 0), None)
    baseline = {
        "train_mae": first.get("train_mae"),
        "valid_mae": first.get("valid_mae"),
        "test_mae": pre["test_mae"] if pre else None,
        "n_features": first.get("n_features"),
    }

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "task": "Caco2_Wang",
        "iterations": len(experiments),
        "kept": sum(1 for e in experiments if e["kept"] and e["status"] != "baseline"),
        "crashes": sum(1 for e in experiments if e["status"] == "crash"),
        "baseline": baseline,
        "baseline_mae": baseline["valid_mae"],
        "best_mae": best,
        "lowest_seen": min((e["valid_mae"] for e in scored_only), default=None),
        "best_features": next(
            (e["n_features"] for e in reversed(experiments) if e["kept"]), None
        ),
        "leaderboard_mae": 0.28,
        "noise_floor": 0.03,
        "total_seconds": sum(e["iteration_s"] for e in experiments),
        "experiments": experiments,
        "audits": audits,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}  ({report['iterations']} iterations, "
          f"best {report['best_mae']:.4f})" if report["best_mae"]
          else f"wrote {OUT}")
    if args.echo:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
