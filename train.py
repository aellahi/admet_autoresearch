"""The only file the agent edits.

featurize() turns SMILES into a feature matrix. make_model() returns an
unfitted sklearn-style regressor. Everything else is in prepare.py.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

import prepare

RDLogger.DisableLog("rdApp.*")

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _descriptors(mol):
    """All RDKit 2D descriptors, made finite and safe to cast to float32."""
    values = np.array([fn(mol) for _, fn in Descriptors.descList], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(values, -1e10, 1e10).astype(np.float32)


def featurize(smiles):
    """list[str] -> (len(smiles), n_features) float32, finite, deterministic."""
    n_features = 2048 + len(Descriptors.descList)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_features, dtype=np.float32))
            continue
        fp = np.array(_fpgen.GetFingerprint(mol), dtype=np.float32)
        rows.append(np.concatenate([fp, _descriptors(mol)]))
    return np.vstack(rows)


def make_model(seed):
    return RandomForestRegressor(
        n_estimators=300, max_features=0.3, n_jobs=4, random_state=seed
    )


if __name__ == "__main__":
    # A manual scoring run. Experiments go through experiment.py, which also
    # decides whether to keep the change.
    print(prepare.format_summary(prepare.run_validation(featurize, make_model)))
