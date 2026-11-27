"""The only file the agent edits.

featurize() turns SMILES into a feature matrix. make_model() returns an
unfitted sklearn-style regressor. Everything else is in prepare.py.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

import prepare

RDLogger.DisableLog("rdApp.*")

# A compact hand-picked set covering size, lipophilicity, polarity and
# H-bonding.
_selected = {
    "MolWt", "MolLogP", "TPSA", "NumHDonors",
    "BCUT2D_LOGPHI",
}
_descriptors = [fn for name, fn in Descriptors.descList if name in _selected]


def _describe(mol, fn):
    try:
        return float(fn(mol))
    except Exception:
        return 0.0


# Groups likely to be charged near physiological pH. Charge hinders passive
# diffusion, and the whole-molecule descriptors do not encode it directly.
_acid_patterns = [
    Chem.MolFromSmarts(s) for s in (
        "[CX3](=O)[OX2H1,OX1-]",                        # carboxylic acid
        "[$([SX4](=O)(=O)[OX2H1,OX1-]),$([PX4](=O)[OX2H1,OX1-])]",  # S/P acids
        "[$(c1nnn[nH]1),$(c1nn[nH]n1),$(c1nnn[n-]1),$(c1nn[n-]n1)]",  # tetrazole
        "[SX4](=O)(=O)[NX3H1,NX2-][CX3]=O",             # acyl sulfonamide
    )
]
_base_patterns = [
    Chem.MolFromSmarts(s) for s in (
        # aliphatic amine (not amide, aniline, enamine, hydrazine, etc.)
        "[NX3;!$([N+]);!$(N-a);!$(N-C=C);!$(N-C#N);!$(N-[#7,#8,#16]);"
        "!$(N-[#6,#15]=[O,S,N])]",
        # amidine or guanidine, counted once per central carbon
        "[$([CX3](=[NX2;!$(N-[#6,#16]=[O,S]);!$(N-C#N);!$(N-[#7,#8])])"
        "[NX3;!$(N-[#6,#16]=[O,S]);!$(N-[#7,#8])])]",
    )
]
# Quaternary or already protonated nitrogen, excluding N-oxides.
_cation_pattern = Chem.MolFromSmarts("[$([NX4+]),$([nX3+]);!$([#7+]-[O-])]")
_N_IONIZATION = 1


def _ionization(mol):
    """Approximate net charge: basic and cationic groups minus acidic ones."""
    acids = sum(len(mol.GetSubstructMatches(p)) for p in _acid_patterns)
    bases = sum(len(mol.GetSubstructMatches(p)) for p in _base_patterns)
    cations = len(mol.GetSubstructMatches(_cation_pattern))
    return [bases + cations - acids]


def featurize(smiles):
    """list[str] -> (len(smiles), n_features) float32, finite, deterministic."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            np.zeros(len(_descriptors) + _N_IONIZATION) if mol is None
            else np.array(
                [_describe(mol, fn) for fn in _descriptors] + _ionization(mol)
            )
        )
    X = np.nan_to_num(np.vstack(rows), nan=0.0, posinf=1e10, neginf=-1e10)
    # Guard against descriptor values too large for float32.
    return np.clip(X, -1e10, 1e10).astype(np.float32)


class _LocalLinearForest(RegressorMixin, BaseEstimator):
    """Mean of a random forest and a ridge line fitted around each query.

    The local fit weights training molecules by how often they share a leaf
    with the query, so it can follow trends that leaf averages flatten.
    """

    def __init__(self, seed=0, ridge=0.1):
        self.seed = seed
        self.ridge = ridge

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        self.Z_ = (X - self.mean_) / self.scale_
        self.forest_ = RandomForestRegressor(
            n_estimators=300, n_jobs=4, random_state=self.seed
        ).fit(X, self.y_)
        self.leaves_ = self.forest_.apply(X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        penalty = self.ridge * np.eye(Z.shape[1])
        local = np.empty(len(X))
        for i, leaves in enumerate(self.forest_.apply(X)):
            match = self.leaves_ == leaves
            # Every leaf holds at least one training row, so no zero division.
            w = (match / match.sum(axis=0)).mean(axis=1)
            z_bar = w @ self.Z_
            y_bar = w @ self.y_
            D = self.Z_ - z_bar
            beta = np.linalg.solve(
                (D * w[:, None]).T @ D + penalty,
                D.T @ (w * (self.y_ - y_bar)),
            )
            local[i] = y_bar + (Z[i] - z_bar) @ beta
        return 0.5 * (self.forest_.predict(X) + local)


def make_model(seed):
    return _LocalLinearForest(seed=seed)


if __name__ == "__main__":
    # A manual scoring run. Experiments go through experiment.py, which also
    # decides whether to keep the change.
    print(prepare.format_summary(prepare.run_validation(featurize, make_model)))
