"""The only file the agent edits.

featurize() turns SMILES into a feature matrix. make_model() returns an
unfitted sklearn-style regressor. Everything else is in prepare.py.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor

import prepare

RDLogger.DisableLog("rdApp.*")

# Skip the fr_* fragment counts (sparse) and the binned *VSA* surface-area
# histograms: polarity, lipophilicity and size are already covered by the
# whole-molecule descriptors.
_descriptors = [
    fn for name, fn in Descriptors.descList
    if not name.startswith("fr_") and "VSA" not in name
]


def _describe(mol, fn):
    try:
        return float(fn(mol))
    except Exception:
        return 0.0


def featurize(smiles):
    """list[str] -> (len(smiles), n_features) float32, finite, deterministic."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            np.zeros(len(_descriptors)) if mol is None
            else np.array([_describe(mol, fn) for fn in _descriptors])
        )
    X = np.nan_to_num(np.vstack(rows), nan=0.0, posinf=1e10, neginf=-1e10)
    # Some descriptors (e.g. Ipc) can exceed the float32 range.
    return np.clip(X, -1e10, 1e10).astype(np.float32)


def make_model(seed):
    return RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=seed)


if __name__ == "__main__":
    # A manual scoring run. Experiments go through experiment.py, which also
    # decides whether to keep the change.
    print(prepare.format_summary(prepare.run_validation(featurize, make_model)))
