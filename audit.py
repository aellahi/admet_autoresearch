"""Score the held-out test split. You run this, not the agent.

    uv run audit.py

Scores whatever train.py is currently committed, appends the result to
~/.cache/admet_autoresearch/audit.tsv, and prints it.

Keep this out of the agent's loop. Every time the test set is used to make
a decision it stops being held out, so run it occasionally -- every ten or
twenty keeps -- and never feed the number back into the search.
"""

from datetime import datetime
from pathlib import Path
import subprocess

import prepare
import train

AUDIT_DIR = Path.home() / ".cache" / "admet_autoresearch"
AUDIT_LOG = AUDIT_DIR / "audit.tsv"


def main():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    m = prepare.run(train.featurize, train.make_model, split="test")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG.exists():
        AUDIT_LOG.write_text(
            "timestamp\tcommit\ttrain_mae\ttest_mae\ttest_mae_std\tn_features\n"
        )
    with AUDIT_LOG.open("a") as fh:
        fh.write("\t".join([
            datetime.now().isoformat(timespec="seconds"), commit,
            f"{m['train_mae']:.6f}", f"{m['mae']:.6f}",
            f"{m['mae_std']:.6f}", str(m["n_features"]),
        ]) + "\n")

    print(prepare.format_summary(m))
    print(f"\nappended to {AUDIT_LOG}")


if __name__ == "__main__":
    main()
