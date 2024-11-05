#####################################################################
# A kinetic predictor using machine learning or reaction similarity #
# Here we mainly use two data-driven methods:                       #
# 1. ChemProp library for coefficients prediction (Deep Learning)   #
# 2. Reaction-similarity-based coefficients search from database    #
#####################################################################

from netgen.database.kin_search import Database
from netgen.thirdparty.chemprop_ng import chemprop
from typing import List
import sys

python_version = sys.version_info
if python_version.minor > 7:
    from typing import Literal
else:
    from typing_extensions import Literal

import os
from netgen.database.rmg_kinetics import get_all_kinetics_obj


def ml_pred(smiles: List,
            temperature: float,
            cp_path: str) -> dict:
    """
    A chemprop wrapper for predicting reaction rate coefficients
    :param smiles: A list of reaction represented by SMILES
    :param temperature: Reaction temperature
    :param cp_path: the checkpoint file path
    :return: predict results
    """
    if temperature is not None:
        features = [[temperature / 1000] for _ in range(len(smiles))]
    else:
        features = None
    pred_smiles = [[s] for s in smiles]

    arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--features_path', '/dev/null',
        '--checkpoint_dir', cp_path,
        '--uncertainty_method', 'evidential_total',
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)

    preds, uncs = chemprop.train.make_predictions(args=args,
                                                  smiles=pred_smiles,
                                                  return_uncertainty=True,
                                                  features=features)

    all_results = {}
    for smi, pred, unc in zip(pred_smiles, preds, uncs):
        all_results[smi[0]] = {'pred': pred[0], 'tot_unc': unc[0]}
    return all_results


def ml_preds(smiles: List,
             temperature: float,
             bi_cp_path: str = None,
             uni_cp_path: str = None):
    """
    Predicting the rate constants by unimolecule ML model and bimolecule model
    :param smiles: reaction smiles
    :param temperature: reaction temperature
    :param bi_cp_path: bimolecule ML model checkpoint file path
    :param uni_cp_path: unimolecule ML model checkpoint file path
    :return: None
    """
    curr_path = os.path.split(os.path.abspath(__file__))[0]
    net_gen_path = os.path.dirname(curr_path)
    if bi_cp_path is None:
        bi_cp_path = os.path.join(net_gen_path, 'thirdparty/chemprop_ng/trained_models/bimol01')
    if uni_cp_path is None:
        uni_cp_path = os.path.join(net_gen_path, 'thirdparty/chemprop_ng/trained_models/unimol001')

    uni_rxns, bi_rxns = [], []
    for s in smiles:
        reacs_num = len(s.split('>>')[0].split('.'))
        if reacs_num == 2:
            bi_rxns.append(s)
        elif reacs_num == 1:
            uni_rxns.append(s)
        else:
            raise KeyError(f'Reaction {s} is neither unimolecule nor bimolecule reaction')
    all_results = {}
    if uni_rxns:
        all_results.update(ml_pred(smiles=uni_rxns, temperature=temperature, cp_path=uni_cp_path))
    if bi_rxns:
        all_results.update(ml_pred(smiles=bi_rxns, temperature=temperature, cp_path=bi_cp_path))

    return all_results


class Predictor:
    """
    A kinetic predictor containing: machine learning predictor and databased-search predictor
    """

    def __init__(self,
                 temperature: float,
                 pressure: float,
                 checkpoint_file_name: str = 'ensemble_mve',
                 database_file_names: List[str] = None,
                 search_type: Literal['rmg_database', 'database'] = 'rmg_database'
                 ):
        self.temperature = temperature
        self.pressure = pressure
        current_file_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
        # database search
        if search_type == 'database':
            self.database_file_paths = []
            database_names = []
            if database_file_names is None:
                database_path = os.path.join(current_file_path, 'database')
                for file in os.listdir(database_path):
                    if os.path.exists(os.path.join(database_path, file, 'chem.yaml')):
                        database_names.append(file)
                        self.database_file_paths.append(os.path.join(database_path, file))
            else:
                database_path = os.path.join(current_file_path, 'database')
                for file in database_file_names:
                    if os.path.exists(os.path.join(database_path, file, 'chem.yaml')):
                        database_names.append(file)
                        self.database_file_paths.append(os.path.join(database_path, file))

            # load all database
            self.databases = []
            for name, database_file_path in zip(database_names, self.database_file_paths):
                db = Database(data_path=database_file_path,
                              temperature=self.temperature,
                              pressure=self.pressure,
                              name=name)
                db.cache_rxnfp()
                self.databases.append(db)

    def database_search(self, smarts: List, return_param_only: bool = False):
        """
        Get reaction rates from database using fuzzy(black) or clear(white) search
        """
        results = {}
        for sma in smarts:
            all_white_search_results = []
            all_black_search_results = []
            for db in self.databases:
                white_, black_ = db.search(sma, only_get_arrhenius_form_kin=False)
                if white_ is not None:
                    all_white_search_results.append(white_)
                if black_ is not None:
                    all_black_search_results.append(black_)

            # Now we have results from different datasets
            # 1. find the white_search results with
            #     a. Arrhenius/Pressure-dependent expression (forward) / rate (reverse)
            #     b. falloff rate constants with forward and reverse (only forward can be adopted)
            # 2. find the black_search results
            #     a. Arrhenius/Pressure-dependent expression (forward) / rate (reverse)
            #     b. falloff rate constants with forward and reverse (only forward can be adopted)
            def find_kin_parameters(result, search_by: Literal['black_search', 'white_search']):
                """
                get kinetic from a database
                :param result: the searched results in a database
                :param search_by: searching using black search or white search
                """
                for smas, res in result.items():
                    if res['rate_type'].lower() in ['arrhenius', 'pressure-dependent-arrhenius']:
                        if res['rate_type'].lower() == 'arrhenius' and not res['reverse']:
                            # For Arrhenius and reverse format, collect all kinetic parameters
                            reactant_num = len(smas.split('>>')[0].split('.'))  # get reactant number
                            input_rate = res['input_data']['rate-constant']
                            # A: 1/m^3/s/kmol -> 1/m^3/s/mol
                            # Ea: J/kmol -> J/mol
                            kin_p = {'A': input_rate['A'] / 1000 ** (reactant_num - 1),
                                     'Ea': input_rate['Ea'] / 1000,  # initial in J/kmol
                                     'b': input_rate['b']}
                            if search_by == 'black_search':
                                similarity_ = res['similarity']
                            else:
                                similarity_ = None
                            return kin_p, similarity_, 'Arrhenius', smas
                    if not res['reverse']:
                        reactant_num = len(smas.split('>>')[0].split('.'))  # get reactant number
                    else:
                        reactant_num = len(smas.split('>>')[-1].split('.'))
                    kin_p = {'A': res['reaction_rate_constant'] / 1000 ** (reactant_num - 1), 'Ea': 0, 'b': 0}
                    if search_by == 'black_search':
                        similarity_ = res['similarity']
                    else:
                        similarity_ = None
                    return kin_p, similarity_, 'Falloff', smas
                return None, None, None, None

            # find kinetic parameters using white search
            final_kin_param = []
            for white_search_result in all_white_search_results:
                kin_param, _, rate_type, sma_t = find_kin_parameters(white_search_result, search_by='white_search')
                if kin_param is not None:
                    if not final_kin_param:  # for first search, assign the parameters
                        final_kin_param = [kin_param, rate_type]
                    elif final_kin_param[1] == 'Falloff' and rate_type == 'Arrhenius':
                        # if find arrhenius format kinetics, replace
                        final_kin_param = [kin_param, rate_type]

            if not final_kin_param:  # if not get kinetic parameters by white search, searching by black search
                for black_search_result in all_black_search_results:
                    kin_param, similarity, _, sma_t = find_kin_parameters(black_search_result, search_by='black_search')
                    if kin_param is not None:
                        if not final_kin_param:
                            final_kin_param = [kin_param, similarity, sma_t]
                        else:
                            if final_kin_param[1] < similarity:
                                final_kin_param = [kin_param, similarity, sma_t]

            if return_param_only:
                results[sma] = final_kin_param[0]
            else:
                results[sma] = final_kin_param

        return results

    def rmg_search(self, smarts: List, rule='rate rules', forbidden_families: List[str] = None):
        """
        a rmg predictor for reaction property prediction
        """
        results = {}
        for sma in smarts:
            kins = get_all_kinetics_obj(sma)
            # filtering forbidden families
            filtered_kins = []
            if forbidden_families is not None:
                for all_info in kins:
                    # whether this kinetics in forbidden families
                    should_be_filtered = False
                    for ff in forbidden_families:
                        if ff in all_info[0]:
                            should_be_filtered = True
                            break
                    if not should_be_filtered:
                        filtered_kins.append(all_info)
            else:
                filtered_kins = kins

            # get kinetic information
            for kin, info, forward in filtered_kins:
                if rule in kin:
                    # rate = info.get_rate_coefficient(self.temperature, self.pressure)
                    kin_param = {'A': info.A.value_si, 'Ea': info.Ea.value_si, 'b': info.n.value_si}
                    # kin_param = {'A': rate, 'Ea': 0, 'b': 0}
                    results[sma] = kin_param

            if not results and len(filtered_kins) > 0:
                kin, info, forward = kins[0]
                if type(info).__name__ == "Arrhenius":
                    kin_param = {'A': info.A.value_si, 'Ea': info.Ea.value_si, 'b': info.n.value_si}
                elif type(info).__name__ in ["Troe", "Lindemann"]:
                    kin_param = {'A': info.arrheniusHigh.A.value_si,
                                 'Ea': info.arrheniusHigh.Ea.value_si,
                                 'b': info.arrheniusHigh.n.value_si}
                elif type(info).__name__ == "PDepArrhenius":
                    kin_param = {'A': info.arrhenius[-1].A.value_si,
                                 'Ea': info.arrhenius[-1].Ea.value_si,
                                 'b': info.arrhenius[-1].n.value_si}
                else:
                    rate = info.get_rate_coefficient(self.temperature, self.pressure)
                    kin_param = {'A': rate, 'Ea': 0, 'b': 0}
                """rate = info.get_rate_coefficient(self.temperature, self.pressure)
                kin_param = {'A': rate, 'Ea': 0, 'b': 0}"""
                results[sma] = kin_param

        return results


if __name__ == '__main__':
    smiles = ['[H:1].[O:2]>>[H:1][O:2]',
              '[C:1]([H:2])[H:3].[H:4][H:5]>>[C:1]([H:2])([H:3])[H:4].[H:5]',
              '[H:1].[H:2][O:3][O:4]>>[H:1][H:2].[O:3][O:4]']
    temperature = 850
    preds = ml_preds(smiles=smiles, temperature=temperature)

    print(preds)
