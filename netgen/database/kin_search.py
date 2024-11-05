import cantera as ct
import yaml
from netgen.common.data import register_id_key
import os
from typing import Dict
import sys
minor_version = sys.version_info.minor
if minor_version > 7:
    from typing import Literal
else:
    from typing_extensions import Literal
from rdkit import Chem
from netgen.common.rxn_similarity import get_similarity, get_rxn_fp


class Database:
    """
    A reaction database
    """

    def __init__(self,
                 data_path: str,
                 temperature: float,
                 pressure: float,
                 name: str = None):
        """
        load and parsing reactions
        :param data_path:
        :param load_input:
        :param name:
        """
        self.name = name  # reaction database name
        self.rxnfp_dict = {}
        self.smi2species_path = os.path.join(data_path, 'smiles_dictionary.yaml')
        self.input_path = os.path.join(data_path, 'chem.yaml')
        with open(self.smi2species_path, 'r') as f:
            self.species2smiles_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        gas = ct.Solution(self.input_path)
        gas.TP = temperature, pressure
        self.forward_rate_constants = gas.forward_rate_constants
        self.reverse_rate_constants = gas.reverse_rate_constants

        self.smarts2idx = {}
        self.id_key2smarts = {}
        self.smarts2info = {}
        for idx, reaction in enumerate(gas.reactions()):
            reactants = reaction.reactants
            products = reaction.products
            rsmis, psmis = [], []
            for s, num in reactants.items():
                rsmis.extend([self.species2smiles_dict[s]] * int(num))
            for s, num in products.items():
                psmis.extend([self.species2smiles_dict[s]] * int(num))
            smarts = f"{'.'.join(rsmis)}>>{'.'.join(psmis)}"
            if reaction.rate.type == 'Arrhenius' and '+ M' in reaction.equation:
                rate_type = 'three-body'
            else:
                rate_type = reaction.rate.type

            info_dict = {'rate_type': rate_type,
                         'mol_num': {'reactants': len(rsmis), 'products': len(psmis)}}
            info_dict.update({'input_data': reaction.rate.input_data})
            self.smarts2info[smarts] = info_dict
            self.smarts2idx[smarts] = idx
            self.id_key2smarts[register_id_key(smarts)] = smarts

    def cache_rxnfp(self,
                    mol_fp_type: Literal['Morgan', 'MACCS'] = 'Morgan',
                    rxnfp_type: Literal['diff', 'sum', 'concatenate'] = 'diff'):
        """
        Calculating the reaction fingerprint and cache the fingerprints to memory
        :return: None
        """
        for sma in self.smarts2idx.keys():
            self.rxnfp_dict[sma] = [get_rxn_fp(sma, molfp_type=mol_fp_type, rxnfp_type=rxnfp_type),
                                    get_rxn_fp(sma, molfp_type=mol_fp_type, rxnfp_type=rxnfp_type,
                                               calculate_reverse=True)]

    def search(self,
               smarts: str,
               max_num_rxns_by_black_search: int = 5,
               mol_fp_type: Literal['Morgan', 'MACCS'] = 'Morgan',
               rxnfp_type: Literal['diff', 'sum', 'concatenate'] = 'diff',
               only_get_arrhenius_form_kin: bool = True) -> [Dict, Dict]:
        """
        White or black search of reaction kinetics
        :param smarts: reaction smarts (in smiles)
        :param max_num_rxns_by_black_search: maximum number of searched reactions if use black search
        :param mol_fp_type:
        :param rxnfp_type:
        :param only_get_arrhenius_form_kin: Only get rate constant or kinetics from arrhenius form rate constants
        :return: dict of white search results, dict of black search results
        """
        selected_info_white_search = {}
        selected_info_black_search = {}

        # start white search
        id_keys = register_id_key(smarts)
        if id_keys in self.id_key2smarts.keys():
            sma = self.id_key2smarts[id_keys]
            s_rsmis, s_psmis = smarts.split('>>')
            t_rsmis, t_psmis = sma.split('>>')
            if Chem.MolToSmiles(Chem.MolFromSmiles(s_rsmis)) == Chem.MolToSmiles(Chem.MolFromSmiles(t_rsmis)):
                reverse = False
            else:
                reverse = True
            if self.smarts2info[sma]['rate_type'] in ['pressure-dependent-Arrhenius', 'Arrhenius']:
                if reverse:
                    reaction_rate_constant = float(self.reverse_rate_constants[self.smarts2idx[sma]])
                else:
                    reaction_rate_constant = float(self.forward_rate_constants[self.smarts2idx[sma]])
                selected_info_white_search.update({sma: {'reverse': reverse,
                                                         'rate_type': self.smarts2info[sma]['rate_type'],
                                                         'input_data': self.smarts2info[sma]['input_data'],
                                                         'reaction_rate_constant': reaction_rate_constant}})
            else:
                if only_get_arrhenius_form_kin:
                    selected_info_white_search = None
                else:
                    if reverse:
                        reaction_rate_constant = float(self.reverse_rate_constants[self.smarts2idx[sma]])
                    else:
                        reaction_rate_constant = float(self.forward_rate_constants[self.smarts2idx[sma]])
                    selected_info_white_search.update({sma: {'reverse': reverse,
                                                             'rate_type': self.smarts2info[sma]['rate_type'],
                                                             'reaction_rate_constant': reaction_rate_constant,
                                                             'input_data': self.smarts2info[sma]['input_data']}})

        else:
            selected_info_white_search = None

        # start black search
        sim_results = []
        fp1 = get_rxn_fp(smarts, molfp_type=mol_fp_type, rxnfp_type=rxnfp_type)
        if self.rxnfp_dict is not None:
            for sma, fp2 in self.rxnfp_dict.items():
                # [sma, is_target_re, similarity]
                sim_results.append([sma, False, get_similarity(fp1, fp2[0])])
                sim_results.append([sma, True, get_similarity(fp1, fp2[1])])
        else:
            # calculate similarity on-the-fly
            for sma, idx in self.smarts2idx:
                fp2 = [get_rxn_fp(sma, molfp_type=mol_fp_type, rxnfp_type=rxnfp_type),
                       get_rxn_fp(sma, molfp_type=mol_fp_type, rxnfp_type=rxnfp_type, calculate_reverse=True)]
                # [sma, is_target_re, similarity]
                sim_results.append([sma, False, get_similarity(fp1, fp2[0])])
                sim_results.append([sma, True, get_similarity(fp1, fp2[1])])
        # sort similarity results
        sorted_sim_results = sorted(sim_results, key=lambda x: x[-1], reverse=True)
        reacs, prods = smarts.split('>>')
        reacs_num, prods_num = len(reacs.split('.')), len(prods.split('.'))
        for r in sorted_sim_results:
            t_reacs, t_prods = r[0].split('>>')
            t_reacs_num, t_prods_num = len(t_reacs.split('.')), len(t_prods.split('.'))
            if r[1]:  # if get reverse reaction, get product molecule number
                t_num_r, t_num_p = t_prods_num, t_reacs_num
            else:
                t_num_r, t_num_p = t_reacs_num, t_prods_num
            if reacs_num != t_num_r or prods_num != t_num_p:  # assert same units
                continue
            # check reactive center



            # get reaction rate constants:
            if self.smarts2info[r[0]]['rate_type'] in ['pressure-dependent-Arrhenius', 'Arrhenius']:
                if r[1]:
                    reaction_rate_constant = float(self.reverse_rate_constants[self.smarts2idx[r[0]]])
                else:
                    reaction_rate_constant = float(self.forward_rate_constants[self.smarts2idx[r[0]]])
                selected_info_black_search.update({r[0]: {'reverse': r[1],
                                                          'rate_type': self.smarts2info[r[0]]['rate_type'],
                                                          'reaction_rate_constant': reaction_rate_constant, # kmol, m, s
                                                          'input_data': self.smarts2info[r[0]]['input_data'],
                                                          'similarity': r[-1]}})

            else:
                if not only_get_arrhenius_form_kin:
                    if r[1]:
                        reaction_rate_constant = float(self.reverse_rate_constants[self.smarts2idx[r[0]]])
                    else:
                        reaction_rate_constant = float(self.forward_rate_constants[self.smarts2idx[r[0]]])
                    selected_info_black_search.update({r[0]: {'reverse': r[1],
                                                              'rate_type': self.smarts2info[r[0]]['rate_type'],
                                                              'reaction_rate_constant': reaction_rate_constant,
                                                              # kmol, m, s
                                                              'input_data': self.smarts2info[r[0]]['input_data'],
                                                              'similarity': r[-1]}})

            if len(selected_info_black_search) > max_num_rxns_by_black_search:
                break

        return selected_info_white_search, selected_info_black_search

