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

# A compact hand-picked set covering size, lipophilicity, polarity, H-bonding,
# flexibility, shape and charge distribution.
_selected = {
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "NumAromaticRings",
    "qed", "HallKierAlpha",
    "MinPartialCharge", "MinEStateIndex",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
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
_N_IONIZATION = 4


def _ionization(mol):
    """Acid and base counts, approximate net charge, zwitterion flag."""
    acids = sum(len(mol.GetSubstructMatches(p)) for p in _acid_patterns)
    bases = sum(len(mol.GetSubstructMatches(p)) for p in _base_patterns)
    cations = len(mol.GetSubstructMatches(_cation_pattern))
    positive = bases + cations
    return [acids, bases, positive - acids,
            float(acids > 0 and positive > 0)]


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


def make_model(seed):
    return RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=seed)


if __name__ == "__main__":
    # A manual scoring run. Experiments go through experiment.py, which also
    # decides whether to keep the change.
    print(prepare.format_summary(prepare.run_validation(featurize, make_model)))
