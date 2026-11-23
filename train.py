"""The only file the agent edits.

featurize() turns SMILES into a feature matrix. make_model() returns an
unfitted sklearn-style regressor. Everything else is in prepare.py.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

import prepare

RDLogger.DisableLog("rdApp.*")

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles):
    """list[str] -> (len(smiles), n_features) float32, finite, deterministic."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            np.zeros(2048, dtype=np.float32) if mol is None
            else np.array(_fpgen.GetFingerprint(mol), dtype=np.float32)
        )
    return np.vstack(rows)


def make_model(seed):
    return RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=seed)


if __name__ == "__main__":
    # A manual scoring run. Experiments go through experiment.py, which also
    # decides whether to keep the change.
    print(prepare.format_summary(prepare.run(featurize, make_model)))
