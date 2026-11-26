"""One-time data preparation. You run this once; the agent never does.

    uv run setup.py

Downloads the TDC Caco2_Wang benchmark and writes the splits to disk as two
separate trees:

    data/splits/seed{1..5}/train.csv   the agent's world
    data/splits/seed{1..5}/valid.csv
    data/holdout/test.csv              the audit set
    data/holdout/raw/                  the raw download, which also holds
                                       test labels for all 22 tasks

Only this script and audit.py ever open data/holdout. The validation path in
prepare.py has no code that names it, and agent-settings.json denies reads of
the whole directory, so test labels cannot reach the agent's process.
"""

from pathlib import Path

from tdc.benchmark_group import admet_group

TASK = "Caco2_Wang"
SEEDS = (1, 2, 3, 4, 5)

SPLITS = Path("data/splits")
HOLDOUT = Path("data/holdout")
COLUMNS = ["Drug_ID", "Drug", "Y"]


def main():
    HOLDOUT.mkdir(parents=True, exist_ok=True)

    # The raw download lands inside the holdout tree, because it contains
    # test labels for every task in the benchmark group.
    group = admet_group(path=str(HOLDOUT / "raw"))
    benchmark = group.get(TASK)
    name = benchmark["name"]

    benchmark["test"][COLUMNS].to_csv(HOLDOUT / "test.csv", index=False)
    print(f"{HOLDOUT / 'test.csv'}: {len(benchmark['test'])} rows")

    for seed in SEEDS:
        train, valid = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        out = SPLITS / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        train[COLUMNS].to_csv(out / "train.csv", index=False)
        valid[COLUMNS].to_csv(out / "valid.csv", index=False)
        print(f"{out}: {len(train)} train, {len(valid)} valid")

    print("\nDone. The agent reads data/splits only.")


if __name__ == "__main__":
    main()
