from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdchem import Mol
import sys
minor_version = sys.version_info.minor
if minor_version > 7:
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
from scipy.spatial.distance import jaccard


def maccs_fp(mol: Chem.rdchem.Mol) -> np.ndarray:
    return np.array(MACCSkeys.GenMACCSKeys(mol))


def morgan_fp(mol: Mol) -> np.ndarray:
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)
    vec1 = np.array(fp1)
    return vec1


def get_rxn_fp(rxn: str,
               molfp_type="Morgan",
               rxnfp_type: Literal['diff', 'sum', 'concatenate'] = 'diff',
               calculate_reverse: bool = False
               ) -> np.ndarray:
    reactant_str, product_str = rxn.split(">>")
    reactants = reactant_str.split(".")
    products = product_str.split(".")
    if not calculate_reverse:
        reactant_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants]
        product_mols = [Chem.MolFromSmiles(product) for product in products]
    else:
        product_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants]
        reactant_mols = [Chem.MolFromSmiles(product) for product in products]

    if molfp_type.lower() == "maccs":
        reactant_fp = np.sum(np.array([maccs_fp(mol) for mol in reactant_mols]), axis=0)
        product_fp = np.sum(np.array([maccs_fp(mol) for mol in product_mols]), axis=0)
    elif molfp_type.lower() == "morgan":
        reactant_fp = np.sum(np.array([morgan_fp(mol) for mol in reactant_mols]), axis=0)
        product_fp = np.sum(np.array([morgan_fp(mol) for mol in product_mols]), axis=0)
    else:
        raise KeyError(f"Fingerprint {molfp_type} is not yet supported. Choose between MACCS and Morgan")

    if rxnfp_type == 'diff':
        fp = product_fp - reactant_fp
    elif rxnfp_type == 'concatenate':
        fp = np.concatenate((reactant_fp, product_fp))
    else:
        fp = product_fp + reactant_fp

    return fp


def get_similarity(v1: np.ndarray, v2: np.ndarray):
    return 1 - jaccard(v1, v2)

