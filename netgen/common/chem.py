"""
A rdkit wrapper
"""
import os
from typing import Union, List
import inspect

from rdkit import Chem
from rdkit.Chem import rdChemReactions
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors

RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()


def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:
    """
    Converts a SMILES string to an RDKit molecule.

    :param string: The InChI or SMILES string.
    :param explicit_hydrogens: Whether to treat hydrogens explicitly.
    :return: The RDKit molecule.
    """
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)


def assign_map_num(smi, explicit_hydrogens: bool = True):
    """
    This function is to check and assign the atom map numbers

    Args:
        smi: the smiles string to be checked
        explicit_hydrogens: whether assign atom numbers of hydrogen numbers

    Returns: None

    """
    mol = str_to_mol(smi, explicit_hydrogens=explicit_hydrogens)
    # Check if smi has atom map numbers
    assign_num = False
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            assign_num = True
            break
    # if any atom without map_num
    if assign_num is True:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    else:
        return smi

    return Chem.MolToSmiles(mol)


def drop_map_num(smi: str):
    """
    Drop the atom map number to get the canonical smiles
    Args:
        smi: the molecule smiles

    Returns:

    """
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def drop_Hs_map_num(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def set_sym(symm: list, molecule: str) -> str:
    """
    Set atom-map numbers of symmetry molecules to same(the first map number in symmetry set)
    Args:
        symm: the symmetry list such as [{1, 2, 3}, {4, 5, 6}]
        molecule: A SMILES list suc as [A-mapped, B-mapped]
    Returns: atom-map molecule list

    """
    mol = str_to_mol(molecule)
    num_idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    for same_atoms in symm:
        same_atoms = list(same_atoms)
        for num in list(same_atoms)[1:]:
            mol.GetAtomWithIdx(num_idx_map[num]).SetAtomMapNum(same_atoms[0])
    new_smi = Chem.MolToSmiles(mol)

    return new_smi


class Molecule:
    """
        A molecule class
        """

    def __init__(self, smiles: str, explicit_hydrogens: bool = True):
        """
        Initialize the Molecule class
        Args:
            smiles:
            explicit_hydrogens:
        """
        self._explicit_hydrogens = explicit_hydrogens
        self._mol = str_to_mol(smiles, explicit_hydrogens=self._explicit_hydrogens)
        self._smiles = Chem.MolToSmiles(self._mol, allHsExplicit=self._explicit_hydrogens)

    def __hash__(self):

        return hash(self._smiles)

    def __str__(self):

        return self._smiles

    def __len__(self):

        return len(self._smiles.split("."))

    def __iter__(self):
        self._iter = self._smiles.split(".")
        self._iter_len = len(self._smiles.split("."))
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < self._iter_len:
            ret = self._iter[self._iter_idx]
            self._iter_idx += 1
            return Molecule(ret, self._explicit_hydrogens)
        else:
            raise StopIteration

    def assign_map_num(self):
        """
        Assign atom map number
        Returns: None
        """
        self._smiles = assign_map_num(self._smiles, self._explicit_hydrogens)
        self._mol = str_to_mol(self._smiles, explicit_hydrogens=self._explicit_hydrogens)
        return self

    def drop_map_num(self):
        """
        Drop atom map number
        Returns: None
        """
        self._smiles = drop_map_num(self._smiles)
        self._mol = str_to_mol(self._smiles, explicit_hydrogens=self._explicit_hydrogens)
        return self

    def drop_Hs_map_num(self):
        """
        Drop H atom map number
        Returns: None
        """
        self._smiles = drop_Hs_map_num(self._smiles)
        self._mol = str_to_mol(self._smiles, explicit_hydrogens=self._explicit_hydrogens)
        return self

    def get_symmetry(self) -> list:
        """
        Find symmetry of molecules
        Returns: a list if the molecules are symmetric
        """
        symmetric = []
        for ith, Mole in enumerate(self):
            mol = Mole.mol
            sym = mol.GetSubstructMatches(mol, uniquify=False)
            if len(sym) <= 1:
                continue

            mol_dict = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms()}
            """
            If the user set explicit hydrogen, all atoms are enumerated.
            If not, only heavy atoms are enumerated
            """
            """if 0 in mol_dict.values():
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx() + 1)
                    mol_dict[atom.GetIdx()] = atom.GetIdx() + 1"""
            if 0 in mol_dict.values():
                """Find atom without atom map. The symmetry cannot be clearly defined"""
                continue
            matched_dict = {}
            for idxs in sym:
                matching_dict = {atom.GetAtomMapNum(): mol_dict[idx] for atom, idx in zip(mol.GetAtoms(), idxs)}
                for key, value in matching_dict.items():
                    if key == value:
                        continue
                    else:
                        if key not in matched_dict.keys():
                            matched_dict[key] = [value]
                        else:
                            matched_dict[key].append(value)

            # match dict has been generated, check duplication
            if matched_dict == {}:
                continue
            else:
                results = []
                for init, match in matched_dict.items():
                    result = set(match)
                    result.add(init)
                    if len(result) > 1 and result not in results:
                        results.append(result)
                    else:
                        continue
                symmetric.extend(results)

        return symmetric

    @property
    def explicit_hydrogen(self) -> bool:

        return self._explicit_hydrogens

    @property
    def mol(self):

        return self._mol

    @property
    def smiles(self) -> str:

        return str(self._smiles)


def canonicalize_smarts(sma: str, explicit_hydrogens: bool = False):
    rsmis, psmis = sma.split('>>')
    rmol = str_to_mol(rsmis, explicit_hydrogens=explicit_hydrogens)
    pmol = str_to_mol(psmis, explicit_hydrogens=explicit_hydrogens)

    return f"{Chem.MolToSmiles(rmol)}>>{Chem.MolToSmiles(pmol)}"


