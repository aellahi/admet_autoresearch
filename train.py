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

# A compact hand-picked set covering size, lipophilicity, polarity and
# H-bonding.
_selected = {
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors",
    "BCUT2D_LOGPHI",
}


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


def _ionization(mol):
    """Approximate net charge."""
    acids = sum(len(mol.GetSubstructMatches(p)) for p in _acid_patterns)
    bases = sum(len(mol.GetSubstructMatches(p)) for p in _base_patterns)
    cations = len(mol.GetSubstructMatches(_cation_pattern))
    return [bases + cations - acids]


# Fixed, not fitted: extra polar-area penalty (square angstroms) per donor,
# and polar-area credit per unit of charge-penalised logP.
_DONOR_WEIGHT = 20.0
_LOGP_WEIGHT = 20.0
_N_COMPACT = 3


def _compact(mol):
    """Size, net polarity (polar area offset by logP), BCUT logP."""
    d = {fn_name: _describe(mol, fn)
         for fn_name, fn in Descriptors.descList if fn_name in _selected}
    charge = _ionization(mol)[0]
    polarity = d["TPSA"] + _DONOR_WEIGHT * d["NumHDonors"]
    lipophilicity = d["MolLogP"] - abs(charge)
    return [
        d["MolWt"],
        polarity - _LOGP_WEIGHT * lipophilicity,
        d["BCUT2D_LOGPHI"],
    ]


def featurize(smiles):
    """list[str] -> (len(smiles), n_features) float32, finite, deterministic."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            np.zeros(_N_COMPACT) if mol is None else np.array(_compact(mol))
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
