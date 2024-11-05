import os
from typing import Union, List
import json
from itertools import combinations

from rdkit import Chem
import rdkit.Chem.AllChem as AllChem

from netgen.common.chem import Molecule, str_to_mol, set_sym, drop_map_num


def read_temp(temp_path, reverse: bool = True):
    dic_path = temp_path
    file = open(dic_path, 'r')
    js = file.read()
    react_dic = json.loads(js)
    file.close()
    react_dic_reverse = {}
    if reverse:
        for key, value in react_dic.items():
            reac_temp = value['reaction_smarts']
            if 'reversible' in value.keys() and value['reversible'] == 'True':
                reversible = True
            else:
                reversible = False

            if reac_temp in [None, {}, ''] or not reversible:
                continue
            key_re = key + '_re'
            react_dic_reverse[key_re] = {}
            react_dic_reverse[key_re]['reaction_smarts'] = value['reaction_smarts'].split('>')[-1] + '>>' + \
                                                        value['reaction_smarts'].split('>')[0]
            react_dic_reverse[key_re]['react_temp_radical'] = value['prod_temp_radical']
            react_dic_reverse[key_re]['prod_temp_radical'] = value['react_temp_radical']

    return react_dic, react_dic_reverse


class RunReactions:
    """
    A class for run reactions
    This class is a wrapper of rdradical
    """

    def __init__(self,
                 explicit_hydrogens: bool = True):
        self._templates = dict({})
        self._explicit_hydrogens = explicit_hydrogens

    @staticmethod
    def check_templates(template: dict):
        """
        Check illegal templates
        Args:
            template:
        Returns: dict
        """
        for name, smarts_dict in template.items():
            assert type(smarts_dict).__name__ == "dict", "{}: reaction information should be a dictionary".format(name)
            keys = smarts_dict.keys()
            for key in ['reaction_smarts', 'react_temp_radical', 'prod_temp_radical']:
                if key not in keys:
                    raise KeyError("Missing {} of reaction {}".format(key, name))
            try:
                AllChem.ReactionFromSmarts(smarts_dict["reaction_smarts"])
            except Exception as e:
                raise Exception("{}::{}: Illegal reaction smarts".format(e, name))

    def load_template(self,
                      path: str = 'templates/rmg_template.txt',
                      template: dict = None,
                      reversible: bool = True):
        template_dict = {}
        if path:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            netgen_dir = os.path.dirname(curr_dir)
            new_path = os.path.join(netgen_dir, path)
            assert os.path.isfile(new_path), f"Unknown path: {new_path}"
            temp_dict, temp_dic_reverse = read_temp(new_path, reverse=reversible)
            template_dict.update(temp_dict)
            template_dict.update(temp_dic_reverse)
        if template:
            template_dict.update(template)
        if template_dict == {}:
            pass
        else:
            RunReactions.check_templates(template_dict)
            self._templates.update(template_dict)

    def add_template(self, template: dict):

        RunReactions.check_templates(template)
        self._templates.update(template)

    def drop_template(self, template_name: Union[str, list]):
        if type(template_name).__name__ == "str" and template_name in self._templates.keys():
            del self._templates[template_name]
        else:
            for temp_name in template_name:
                if temp_name in self._templates.keys():
                    del self._templates[template_name]

    def run(self,
            reactant: str,
            template_names: List = None,
            max_radical_electrons: int = 1,
            max_total_radical_electrons: int = 1,
            max_heavy_atoms: int = None,
            return_names: bool = False):
        """

        :param reactant: reactant SMILES string
        :param template_names: if None, enumerate all reaction templates
        :param max_carbene: the number of carbene larger than max_carbene would be filtered
        :param max_heavy_atoms: the number of atoms larger than max_heavy_atoms would be filtered
        :param return_names: return the reaction family names as same time
        :return:
        """
        ini_mols = Molecule(reactant, explicit_hydrogens=True)
        ini_mols.assign_map_num()
        sym = ini_mols.get_symmetry()  # get symmetry atoms
        all_smiles = [ini_mols.smiles.split('.')]
        if len(all_smiles[0]) == 2:
            # for reactant with two molecules, change the position to match the templates
            all_smiles.append([all_smiles[0][1], all_smiles[0][0]])

        if template_names is None:
            names = self._templates.keys()
        else:
            names = template_names

        prod_list = []
        for name in names:
            smarts_dict = self._templates[name]
            # run reactant by rdkit. react_temp: reaction SMARTS; rad_rt: dictionary; rad_pt:dictionary
            smarts = smarts_dict['reaction_smarts']
            rad_rt = smarts_dict['react_temp_radical']
            rad_pt = smarts_dict['prod_temp_radical']

            rxn = AllChem.ReactionFromSmarts(smarts)
            # set radical elec
            if rad_rt:
                for react in rxn.GetReactants():
                    for a in react.GetAtoms():
                        if str(a.GetAtomMapNum()) in rad_rt.keys():
                            a.SetNumRadicalElectrons(int(rad_rt[str(a.GetAtomMapNum())]))
            if rad_pt:
                for prod in rxn.GetProducts():
                    for a in prod.GetAtoms():
                        if str(a.GetAtomMapNum()) in rad_pt.keys():
                            a.SetNumRadicalElectrons(int(rad_pt[str(a.GetAtomMapNum())]))
            rxn.Initialize()
            # We can filter some templates by the number of reactants
            if len(ini_mols) != rxn.GetNumReactantTemplates():
                continue
            # or electronic numbers
            if Chem.Descriptors.NumRadicalElectrons(ini_mols.mol) < sum(rad_rt.values()):
                continue

            # get atom map to reactant map. For different template, this is different.
            atomMapToReactantMap = {}
            for ri in range(rxn.GetNumReactantTemplates()):
                rt = rxn.GetReactantTemplate(ri)
                for atom in rt.GetAtoms():
                    atomMapToReactantMap[atom.GetAtomMapNum()] = ri

            for smiles in all_smiles:  # enumerate all possible reactants
                new_mols = []
                num_idx_mapnum_dict = {}
                for i, smi in enumerate(smiles):
                    new_mol = str_to_mol(smi)
                    new_mols.append(new_mol)
                    num_idx_mapnum_dict[i] = {a.GetIdx(): a.GetAtomMapNum() for a in new_mol.GetAtoms()}
                prods = rxn.RunReactants(tuple(new_mols))

                for prod in prods:  # iter every product (a product might have two or more molecules)
                    ps = []
                    skip_by_radical_constrain = False
                    skip_by_atom_number_constrain = False
                    for p in prod:
                        try:
                            Chem.SanitizeMol(p)
                            for atom in p.GetAtoms():
                                if atom.HasProp('old_mapno'):
                                    r_idx = atomMapToReactantMap[int(atom.GetProp('old_mapno'))]
                                    map_num = num_idx_mapnum_dict[r_idx][int(atom.GetProp('react_atom_idx'))]
                                    atom.SetAtomMapNum(map_num)

                            # checking generated molecule
                            pmol = str_to_mol(Chem.MolToSmiles(p))
                            if max_heavy_atoms is not None:  # check heavy atoms
                                if pmol.GetNumHeavyAtoms() > max_heavy_atoms:
                                    skip_by_atom_number_constrain = True
                                    # print(f'molecule {Chem.MolToSmiles(pmol)} with {pmol.GetNumHeavyAtoms()} heavy atoms is skipped')
                            total_radical_electrons = 0
                            for pa in pmol.GetAtoms():
                                if pa.GetAtomMapNum() == 0:
                                    raise ValueError(f'Unknown atom {pa.GetSymbol()} with map number 0')
                                # check radicals of an atom
                                if pa.GetNumRadicalElectrons() > max_radical_electrons:
                                    skip_by_radical_constrain = True
                                # check total radicals
                                total_radical_electrons += pa.GetNumRadicalElectrons()
                                if total_radical_electrons > max_total_radical_electrons:
                                    skip_by_radical_constrain = True
                            ps.append(Chem.MolToSmiles(p))
                        except:
                            ps = []
                            break
                    if skip_by_atom_number_constrain or skip_by_radical_constrain:
                        continue

                    if ps:
                        psmi = '.'.join(ps)
                        try:
                            collected_map_num = []
                            for atom in str_to_mol(psmi).GetAtoms():
                                p_atom_map_num = atom.GetAtomMapNum()
                                if p_atom_map_num not in collected_map_num:
                                    collected_map_num.append(p_atom_map_num)
                                else:
                                    raise ValueError(
                                        f'Unknown duplicate atom {atom.GetSymbol()} with map number {p_atom_map_num}')
                            if return_names:
                                prod_list.append((psmi, name))
                            else:
                                prod_list.append(psmi)
                        except:
                            continue
        if not return_names:
            psmi_list = set([Chem.MolToSmiles(str_to_mol(s)) for s in prod_list])
            psmi_re_list = [set_sym(sym, smi) for smi in psmi_list]  # set similar atom with similar map numbers
            results = {}
            for p, p_re in zip(psmi_list, psmi_re_list):
                if p_re not in results.keys():  # if in dict, this molecule already exist, this can pure similar SMILES
                    results[p_re] = p
            final_results = {}
            for smi_re, smi in results.items():
                drop_num_smi = drop_map_num(smi_re)
                if drop_num_smi not in final_results.keys():
                    final_results[drop_num_smi] = [smi]
                else:
                    final_results[drop_num_smi].append(smi)
        else:
            psmi_list = set([(Chem.MolToSmiles(str_to_mol(s[0])), s[1]) for s in prod_list])
            psmi_re_list = [(set_sym(sym, s[0]), s[1]) for s in psmi_list]  # set similar atom with similar map numbers
            results = {}
            for p, p_re in zip(psmi_list, psmi_re_list):
                if p_re not in results.keys():  # if in dict, this molecule already exist, this can pure similar SMILES
                    results[p_re] = p
            final_results = {}
            for smi_re, smi in results.items():
                drop_num_smi = (drop_map_num(smi_re[0]), smi_re[1])
                if drop_num_smi not in final_results.keys():
                    final_results[drop_num_smi] = [smi]
                else:
                    final_results[drop_num_smi].append(smi)

        return ini_mols.smiles, final_results

    @property
    def templates(self):
        return self._templates

    @property
    def template_names(self):
        return self._templates.keys()


def get_reactants(unreacted: List[str], reacted: List[str] = None, canonicalize: bool = True):
    """
    Get reactants
    :param unreacted: smiles of unreacted species
    :param reacted: smile of reacted species
    :param canonicalize: canonicalize smiles
    :return: list of reactants
    """
    rs = []
    if canonicalize:
        rs.extend([Chem.MolToSmiles(Chem.MolFromSmiles(e)) for e in unreacted])
    else:
        rs.extend(unreacted)
    double_rs = []
    for r in combinations(unreacted, 2):
        double_rs.append('.'.join(r))
    for r in unreacted:
        double_rs.append(f'{r}.{r}')
        if reacted:
            for cs in reacted:
                double_rs.append(f'{r}.{cs}')
    if canonicalize:
        rs.extend([Chem.MolToSmiles(Chem.MolFromSmiles(d)) for d in double_rs])
    else:
        rs.extend(double_rs)

    return rs

