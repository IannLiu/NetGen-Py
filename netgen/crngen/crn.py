from netgen.crngen.reaction import RunReactions, get_reactants
from netgen.predictor.kin_predictor import ml_preds, Predictor
from rdkit import Chem
from typing import List
from netgen.common.data import Kinetics, Reaction, ReactionPool, Species, SpeciesPool, register_id_key
from netgen.common.chem import canonicalize_smarts
from netgen.common.utils import create_logger
import math
import numpy as np
from netgen.solver.base import solve_odes
import yaml
import sys
import os

minor_version = sys.version_info.minor
if minor_version > 7:
    from typing import Literal
else:
    from typing_extensions import Literal


class ReactionModel:
    """
    A reaction model for edge or pool
    """

    def __init__(self):
        self.species_pool = SpeciesPool()
        self.reaction_pool = ReactionPool()


class CoreEdgeReactionModel:
    """
    A core edge reaction model

    Compared with some package like RMG, this class is more flexible for data-driven reaction kinetics
    since 1. Every reaction in this class is SMILES-based and atom-mapped. This allows further
    cheminformatic analysis. 2. We developed a multi-source framework: reactions in this framework
    is firstly predicted using machine learning methods, and then important kinetics are calibrated
    using high-accuracy methods
    This allows the integration of different predictors with or without uncertainty estimations
    """

    def __init__(self,
                 temperature: float,
                 pressure: float,
                 simulation_time: float,
                 initial_concentration: str,
                 max_conversion: float = None,
                 kin_est_mode: Literal[
                     'rate_rule', 'database_search', 'ml', 'ml-rate_rule', 'ml-database_search'] = 'rate_rule',
                 tolerance: float = 0.1,
                 min_species_exist_iterations: int = 5,
                 explicit_hydrogens: bool = True,
                 eval_interval: float = 0.0001,
                 max_species_added_to_core_per_iter: int = 2,
                 min_species_in_core: int = 2,
                 log_save_dir: str = "NetGen.log",
                 uni_cp_path: str = None,
                 bi_cp_path: str = None,
                 max_heavy_atoms: int = None,
                 z_score: float = 1.64,
                 skip_unknown_reactions: bool = True,
                 forbidden_rmg_database_families: List[str] = None
                 ):
        """
        A class for employing the rate based rule
        :param temperature: reaction temperature
        :param pressure: reaction pressure
        :param simulation_time: the reactor simulation time
        :param max_conversion: the max conversion. if reached, simulation will stop
        :param initial_concentration: the species concentration
        :param kin_est_mode: how to estimate the kinetics of generated reactions
        :param tolerance: the edge to core tolerance
        :param min_species_exist_iterations:
        :param explicit_hydrogens:
        :param eval_interval: reactor simulation interval (interval to get simulation results)
        :param max_species_added_to_core_per_iter: the number of reactions added to core at every iteration
        :param min_species_in_core: at least add n species to core at initial stage
        :param forbidden_rmg_database_families: forbidden the specified RMG-database reaction families
        """
        self.rxn_generator = RunReactions(explicit_hydrogens=explicit_hydrogens)
        self.min_species_exist_iterations = min_species_exist_iterations
        self.tolerance = tolerance
        self.temperature = temperature
        # parsing species concentration dictionary
        self.ini_conc_dict = {}
        for sc in initial_concentration.split(','):
            s, c = sc.split(":")
            self.ini_conc_dict[Chem.MolToSmiles(Chem.MolFromSmiles(s.strip()))] = float(c.strip())
        self.global_reaction_id_key_pool = set({})
        self.core = ReactionModel()
        self.edge = ReactionModel()
        self.core.species_pool.add_species_list([Species(s) for s in self.ini_conc_dict.keys()])
        self.simulation_time = simulation_time
        self.max_conversion = max_conversion
        self.pressure = pressure
        self.eval_interval = eval_interval
        self.max_species_added_to_core_per_iter = max_species_added_to_core_per_iter
        if kin_est_mode in ['database_search', 'ml-database_search']:
            self.kin_predictor = Predictor(temperature=temperature, pressure=pressure,
                                           database_file_names=['JetSurF2.0', 'CurranPentane'])
        elif kin_est_mode in ['rate_rule', 'ml-rate_rule']:
            self.kin_predictor = Predictor(temperature=temperature, pressure=pressure,
                                           search_type='rmg_database')
        self.custom_kinetics = dict({})
        self.generated_rxn_num = 0
        self.calibrated_rxn_num = 0
        self.sim_results = dict({})
        self.kin_est_mode = kin_est_mode
        self.min_species_in_core = min_species_in_core
        self.finished = False
        self.logger = create_logger(name="NetGen", save_dir=log_save_dir)
        self.uni_cp_path = uni_cp_path
        self.bi_cp_path = bi_cp_path
        self.max_heavy_atoms = max_heavy_atoms
        self.z_score = z_score
        self.skip_unknown_reactions = skip_unknown_reactions
        self.forbidden_rmg_database_families = forbidden_rmg_database_families
        self.ith_iter = 0

    def load_template(self,
                      path: str = None,
                      template: dict = None,
                      reversible: bool = True
                      ):
        if path is None:
            netgen_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(netgen_path, "templates", 'rmg_template.txt')
        self.rxn_generator.load_template(path=path, template=template, reversible=reversible)
        self.logger.debug(f"Load template from {path}")

    def add_custom_kinetics(self, path: str):
        with open(path, 'r') as f:
            kinetics = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        for sma, kin in kinetics.items():
            self.custom_kinetics[canonicalize_smarts(sma)] = kin

    def run_one_step(self):
        # Start an iteration
        if self.finished:  # if the iteration already finished, return
            return

        # generating new species, add new reactions
        self.ith_iter += 1
        self.logger.debug("***************************************************************")
        self.logger.debug(f"*******          iteration {self.ith_iter}          *********")
        self.logger.debug("Starting reacting the unreacted core species with other species")
        self._react()

        # if core species and edge species is less than the minimal core size number
        # add all edge species to core directly
        edge_species = self.edge.species_pool.get_species()
        if len(self.core.species_pool.get_species()) + len(edge_species) < self.min_species_in_core:
            self._move_species_and_reactions_to_core(edge_species)
            return
        else:
            if self.min_species_in_core - len(self.core.species_pool.get_species()) > self.max_species_added_to_core_per_iter:
                # add self.max_species_added_to_core_per_iter species can not satisfy min_species_in_core
                max_species_added_to_core_per_iter = self.min_species_in_core - len(self.core.species_pool.get_species())
            else:
                max_species_added_to_core_per_iter = self.max_species_added_to_core_per_iter

        # Now, we simulate the simple reactor
        sim_results = self.simulate()
        time_points = sim_results['time']
        core_species2idx = sim_results['species']
        core_concentration = sim_results['concentration']
        character_rates = sim_results['character_rates']

        # get id_key of edge species
        edge_species2id_key = {s: self.edge.species_pool.get_species_reaction_id_keys(s) for s in edge_species}

        # get edge species rate constants
        # iterating the time points, calculating system character rate and edge species rate
        self.logger.info("-------------------------------------------------")
        self.logger.info("Starting find edge species to be added to core...")
        species_added_to_core = []
        calibrated_edge_species = []
        for iteration in range(len(time_points) - 1):
            # from 2nd interval to last, since all species concentration at 1st time point is 0
            species_c = core_concentration[:, iteration + 1]
            core_species2con = {s: species_c[idx] for s, idx in core_species2idx.items()}
            character_rate = character_rates[iteration]
            # Calculating edge species rate one by one and calibrating the rate constants
            while len(species_added_to_core) < max_species_added_to_core_per_iter:
                edge_species_rate_dict = {}
                for edge_species, id_keys in edge_species2id_key.items():
                    if edge_species not in species_added_to_core:
                        edge_species_rate = self.get_edge_species_rate(species=edge_species,
                                                                       id_keys=id_keys,
                                                                       get_max_rate=True,
                                                                       core_species2con=core_species2con)
                        edge_species_rate_dict[edge_species] = edge_species_rate
                if edge_species_rate_dict == {}:
                    break  # if there is no edge species can be added to core, finish iteration
                sorted_edge_species_rate = sorted(edge_species_rate_dict.items(), key=lambda x: x[1], reverse=True)

                if sorted_edge_species_rate[0][1] > character_rate:  # if maximum rate larger than character rate
                    if self.kin_est_mode in ['ml-rate_rule', 'ml-database_search'] and sorted_edge_species_rate[0][0] \
                            not in calibrated_edge_species:
                        # if we use calibration mode, the species with largest rate will be calibrated
                        if self.kin_est_mode == 'ml-rate_rule':
                            self.calibrate_edge_species_reactions(sorted_edge_species_rate[0][0],
                                                                  datasource='rate_rule')
                        else:
                            self.calibrate_edge_species_reactions(sorted_edge_species_rate[0][0],
                                                                  datasource='database')
                        calibrated_edge_species.append(sorted_edge_species_rate[0][0])
                    else:
                        # else, this species will be added to core
                        species_added_to_core.append(sorted_edge_species_rate[0][0])
                        if len(species_added_to_core) == self.max_species_added_to_core_per_iter:
                            if len(sorted_edge_species_rate) > 1:
                                if sorted_edge_species_rate[0][1] == sorted_edge_species_rate[1][1]:
                                    # at last iteration, whether the first two species come from same reactions?
                                    s1, s2 = sorted_edge_species_rate[0][0], sorted_edge_species_rate[1][0]
                                    s1_id_keys = self.edge.species_pool.get_species_reaction_id_keys(s1)
                                    s2_id_keys = self.edge.species_pool.get_species_reaction_id_keys(s2)
                                    if s1_id_keys == s2_id_keys:  # if true, add the second reaction as well
                                        species_added_to_core.append(s2)
                else:
                    break

            # if the conversion is larger than that of user-defined, break this iteration
            if self.max_conversion is not None:
                conversions = []
                for species_name, ini_con in self.ini_conc_dict.items():
                    con = core_species2con[species_name]
                    conversions.append(1 - con / ini_con)
                if max(conversions) > self.max_conversion:
                    break
            # if the species added to core were/was found, break this iteration as well
            if len(species_added_to_core) < self.max_species_added_to_core_per_iter:
                continue
            else:
                break

        self.logger.info(f"add {len(species_added_to_core)} edge species to core: ")
        for cs in species_added_to_core:
            self.logger.info(cs)
        if not species_added_to_core:
            self.finished = True
            self.logger.debug("The core is completed, finsh the model enlargement")
            return
        # move edge species and corresponding reactions to core
        self._move_species_and_reactions_to_core(species_added_to_core)

    def _react(self):
        """
        Generating new reactions, predicting reaction rates, adding new species and reactions to ege or core
        :return:
        """
        self.logger.debug("Now, we start generating new species and reactions...")
        # find reacted and unreacted reactants
        species_reactivity = self.core.species_pool.get_species_reactivity()
        unreacted_species = species_reactivity['unreacted']
        reacted_species = species_reactivity['reacted']
        # form reactants
        rs = get_reactants(reacted=reacted_species, unreacted=unreacted_species)
        # and then run reactants to generate products, collecting reaction(including reverse reactions),
        rxn_list_for_ml = []
        results = {}
        for r in rs:
            reacs, prods = self.rxn_generator.run(r, max_heavy_atoms=self.max_heavy_atoms)
            # reorganizing the data to generate {(id_key, unmapped_reactions): mapped_reactions} dict
            for unmapped_p, mapped_ps in prods.items():
                smarts = f'{r}>>{unmapped_p}'
                id_key = register_id_key(smarts)
                mapped_rxns = []
                for mapped_p in mapped_ps:
                    mapped_rxns.append([f'{reacs}>>{mapped_p}', f'{mapped_p}>>{reacs}'])
                    rxn_list_for_ml.extend([f'{reacs}>>{mapped_p}', f'{mapped_p}>>{reacs}'])
                results[(id_key, smarts)] = mapped_rxns
        self.generated_rxn_num += len(results)

        # select reaction mode
        if self.kin_est_mode in ['ml', 'ml-rate_rule']:
            ml_reaction_obj_dict = self._form_ml_reaction_objs(results, rxn_list_for_ml)
        else:
            ml_reaction_obj_dict = None

        # Since new reactions/species were generated, assigning reaction/new species to core or edge
        # if all species of a reaction are in core, this reaction will be added to core if not,
        # this species will be added to edge
        # Note all reactions in core are simulated to get species concentration
        core_species = set(self.core.species_pool.get_species())  # get core species for checking
        self.logger.info(f"Find {len(core_species)} core species: ")
        self.logger.info(core_species)

        new_edge_species_list, new_core_rxn_list, new_edge_rxn_list = [], [], []
        for id_key, unmapped_rxn in results.keys():
            rsmis, psmis = unmapped_rxn.split('>>')  # iterating every id_key

            if rsmis == psmis:  # if reactants equal to products, no contributions to ODEs, thus skip them
                continue

            if id_key in self.global_reaction_id_key_pool:
                # if id_key has been registered, this reaction has been processed
                # thus pass this reaction
                continue
            else:
                self.global_reaction_id_key_pool.add(id_key)  # register this reaction to global pool
                # check this new reaction should belong to core or edge
                # since reactant must be core species, we only check product smiles
                new_edge_species = []
                for psmi in psmis.split('.'):
                    if psmi not in core_species:
                        new_edge_species.append(psmi)  # if a species belongs to edge, this is an edge reaction
                    if psmi not in new_edge_species_list:
                        new_edge_species_list.append(psmi)
                # append reaction id_key to species
                if len(new_edge_species) == 0:  # if this is a core reaction, add reaction id_key to every species
                    smis_list = rsmis.split('.')
                    smis_list.extend(psmis.split('.'))
                    self.core.species_pool.add_reaction_id_key_to_species_list(smis_list, id_key)
                    new_core_rxn_list.append(id_key)
                else:  # if this is an edge reaction, add reaction id_key to product species (edge species)
                    # add new edge species first
                    self.edge.species_pool.add_species_list([Species(s) for s in new_edge_species])
                    new_edge_rxn_list.append(id_key)
                    for smi in psmis.split('.'):
                        if smi not in core_species:
                            self.edge.species_pool.add_reaction_id_key_to_species(species=smi,
                                                                                  reaction_id_key=id_key)

                # now, we should add this reaction to core reaction pool or edge reaction pool
                if len(new_edge_species) == 0:  # no edge species, add this reaction to core
                    # reactions moved to core must be calibrated
                    # get reaction object
                    if self.kin_est_mode in ['ml-rate_rule', 'rate_rule']:
                        reaction = self.get_reaction(smiles=unmapped_rxn, id_key=id_key, datasource='rate_rule')
                    elif self.kin_est_mode == 'database_search':
                        reaction = self.get_reaction(smiles=unmapped_rxn, id_key=id_key, datasource='database')
                    else:
                        reaction = ml_reaction_obj_dict[id_key]
                    self.calibrated_rxn_num += 1
                    self.core.reaction_pool.add_reaction(reaction)
                else:
                    # get reaction object
                    if self.kin_est_mode == 'rate_rule':
                        reaction = self.get_reaction(smiles=unmapped_rxn, id_key=id_key, datasource='rate_rule')
                    elif self.kin_est_mode == 'database_search':
                        reaction = self.get_reaction(smiles=unmapped_rxn, id_key=id_key, datasource='database')
                    else:
                        reaction = ml_reaction_obj_dict[id_key]
                    self.edge.reaction_pool.add_reaction(reaction)

        self.logger.info(f"add {len(new_edge_species_list)} new edge species")
        for s in new_edge_species_list:
            self.logger.info(s)
        self.logger.info(f"add {len(new_core_rxn_list)} new core reactions")
        for cr in new_core_rxn_list:
            self.logger.info(cr)
        self.logger.info(f"add {len(new_edge_rxn_list)} new edge reactions")
        for er in new_edge_rxn_list:
            self.logger.info(er)

        # set unreacted core species to reacted species after reacting with reacted core species
        for unreact in unreacted_species:
            self.core.species_pool.set_species_reactivity(unreact)
        self.logger.debug("Pre-reacting stage finished.")

    def _move_species_and_reactions_to_core(self, species_list: List[str]):
        """
        Move species and corresponding reactions from edge to core
        :param species_list: species move to core
        :return:
        """
        # add species and reactions to core
        # firstly, get all species that can appears in core reactions
        species_in_core_reactions = self.core.species_pool.get_species()
        species_in_core_reactions.extend(species_list)
        # move species and reactions from species to core
        for species in species_list:  # add species to core
            self.core.species_pool.add_species(Species(species))

        moved_reaction_id_keys = []
        for species in species_list:  # add reactions to core
            s_id_keys = self.edge.species_pool.get_species_reaction_id_keys(species)
            for id_key in s_id_keys:
                if id_key in moved_reaction_id_keys:
                    continue
                reaction = self.edge.reaction_pool.get_reaction_obj(id_key)
                # checking if all reaction species are in core_species, adding core reactions
                reaction_species = reaction.get_smis()
                if all(s in species_in_core_reactions for s in reaction_species):
                    self.core.reaction_pool.add_reaction(reaction)  # add reactions to core reaction pool
                    moved_reaction_id_keys.append(id_key)
                    # add reaction to all species
                    for smi in reaction.get_smis():
                        self.core.species_pool.add_reaction_id_key_to_species(smi, id_key)
        self.logger.info(f"Move {len(moved_reaction_id_keys)} reactions from edge to core:")
        for id_key in moved_reaction_id_keys:  # remove reactions from edge reaction pool
            self.edge.reaction_pool.remove_reaction_by_id_key(id_key)
            self.logger.info(id_key)
        self.edge.species_pool.remove_species(species_list)  # remove species from edge species pool

    def _parsing_k_y_matrix(self, parsing_core: bool = True, core_species2idx: dict = None):
        """
        Parsing core or edge species and reactions to species_index dict, k_array, y_array
        :param parsing_core:
        :return: species_idx dict, k_array, y_array
        """
        if parsing_core:
            reaction_model = self.core
            len_factor = 2  # forward and reverse reactions
        else:
            reaction_model = self.edge
            len_factor = 1  # forward reaction
        species_list = sorted(reaction_model.species_pool.get_species())
        species2idx = {s: idx for idx, s in enumerate(species_list)}  # assigning index to species
        id_keys = sorted(list(reaction_model.reaction_pool.id_key_pool))
        id_key2idx = {k: idx for idx, k in enumerate(id_keys)}  # assign index to reactions

        # dY / dt = K * Y1 * Y2
        k_array = np.zeros((len(species2idx), len_factor * len(id_key2idx)))  # species number * reaction number
        y1_array = -np.ones((len(species2idx), len_factor * len(id_key2idx)), dtype=np.int32)
        y2_array = -np.ones((len(species2idx), len_factor * len(id_key2idx)), dtype=np.int32)
        y_array = [y1_array, y2_array]  # two molecule reaction

        for species, s_idx in species2idx.items():
            reaction_id_keys = reaction_model.species_pool.get_species_reaction_id_keys(species=species)
            # print("*" * 40)
            # print(f'Start parsing species {species} with reaction id_keys {reaction_id_keys}')
            for id_key in reaction_id_keys:
                k_idx = id_key2idx[id_key] * len_factor
                reaction = reaction_model.reaction_pool.get_reaction_obj(id_key=id_key)  # get reaction object
                stoi_chem = reaction.get_stoichiometric_num
                # print(f'Start parsing reaction {reaction.smarts}, the stoi_chem is {stoi_chem}')

                if parsing_core:
                    # forward reactions
                    k_array[s_idx][k_idx] = reaction.get_coef() * stoi_chem[species]
                    # print(f'forward reaction k is {k_array[s_idx][k_idx]}')
                    for i, smi in enumerate(reaction.get_rsmis()):
                        y_array[i][s_idx][k_idx] = species2idx[smi]
                    # reverse_reactions
                    k_array[s_idx][k_idx + 1] = -reaction.get_coef(reverse=True) * stoi_chem[species]
                    # print(f'reverse reaction k is {k_array[s_idx][k_idx + 1]}')
                    for i, smi in enumerate(reaction.get_psmis()):
                        y_array[i][s_idx][k_idx + 1] = species2idx[smi]
                else:
                    # only parsing forward reactions for edge species
                    # reactants of edge reactions can only appear in core, for convenience,
                    # we parse the y_array of edge in terms of core species
                    k_array[s_idx][k_idx] = reaction.get_coef() * stoi_chem[species]
                    for i, smi in enumerate(reaction.get_rsmis()):
                        y_array[i][s_idx][k_idx] = core_species2idx[smi]  # get core species index

        return species2idx, k_array, y_array

    def calibrate_edge_species_reactions(self,
                                         species: str,
                                         id_keys=None,
                                         datasource: Literal['database', 'rate_rule'] = 'rate_rule'):
        """
        Calibrate edge reactions
        """
        if id_keys is None:
            id_keys = self.edge.species_pool.get_species_reaction_id_keys(species)
        for id_key in id_keys:
            if self.edge.reaction_pool.get_reaction_obj(id_key=id_key).calibrated:
                continue
            smarts = self.edge.reaction_pool.id_key2smarts[id_key]
            rsmi, psmi = smarts.split('>>')
            smarts_re = f'{psmi}>>{rsmi}'
            # firstly consider custom provided kinetics
            can_smarts = canonicalize_smarts(smarts)
            if can_smarts in self.custom_kinetics.keys():
                can_smarts_re = canonicalize_smarts(smarts_re)
                try:
                    kin_param = self.custom_kinetics[can_smarts]
                    kin_param_re = self.custom_kinetics[can_smarts_re]
                    new_reaction_obj = self.form_reaction_obj(smarts=can_smarts,
                                                              id_key=id_key,
                                                              kin_param=kin_param,
                                                              kin_param_re=kin_param_re)
                except:
                    raise KeyError('Both forward can reverse kinetics should be given')
            else:
                new_reaction_obj = self.get_reaction(id_key, datasource=datasource)

            self.edge.reaction_pool.reassign_reaction_obj(id_key, new_reaction_obj)

    def get_edge_species_rate(self, species: str, core_species2con: dict, id_keys=None, get_max_rate: bool = False):
        """
        Get edge species rate
        """
        if id_keys is None:
            id_keys = self.edge.species_pool.get_species_reaction_id_keys(species)

        edge_species_rate = 0
        for id_key in id_keys:
            # print(f'get reaction rate of {id_key}')
            reaction_obj = self.edge.reaction_pool.get_reaction_obj(id_key)
            stoi_chem = reaction_obj.get_stoichiometric_num
            if get_max_rate:
                rate_cst_i = stoi_chem[species] * reaction_obj.get_max_coef()
                # print(f'reaction {id_key} max rate is {rate_cst_i}')
            else:
                rate_cst_i = stoi_chem[species] * reaction_obj.get_coef()
                # print(f'reaction {id_key} rate is {rate_cst_i}')
            con_i = 1
            for rsmi in reaction_obj.get_rsmis():
                con_i *= core_species2con[rsmi]
            edge_species_rate += rate_cst_i * con_i  # rate = k * c
            # print(f'the rate is: {edge_species_rate} += {rate_cst_i} * {con_i}')
            # print(f'final rate is: {edge_species_rate}')
        # print('*'*20)

        return edge_species_rate

    def simulate(self, time=None):
        """
        Get reaction rate of core
        """
        core_species2idx, core_k_array, core_y_array = self._parsing_k_y_matrix(parsing_core=True)
        ini_state = np.zeros(len(core_species2idx))
        for s, con in self.ini_conc_dict.items():
            ini_state[core_species2idx[s]] = con
        # simulation across time points
        if time is None:
            simulate_time = self.simulation_time
        else:
            simulate_time = time
        results = solve_odes(t_span=(0, simulate_time),
                             ini_state=ini_state,
                             k_array=core_k_array,
                             y_array=core_y_array,
                             eval_interval=self.eval_interval)

        time_points = results.t
        core_concentration = results.y

        # get species rate at different t
        character_rates = []
        for iteration in range(len(time_points) - 1):
            species_c = core_concentration[:, iteration + 1]
            bv = np.append(species_c, 1)  # this is the core species concentration, 1 is a placeholder
            core_rates = np.sum(core_k_array * bv[core_y_array[0]] * bv[core_y_array[1]], axis=1)
            character_rates.append(np.sqrt(np.sum(core_rates ** 2)) * self.tolerance)
            # character_rates length equal to len(time_points) - 1 since the first time points is 0 and not  considered

        self.sim_results = {'time': time_points,
                            'concentration': core_concentration,
                            'character_rates': character_rates,
                            'species': core_species2idx}
        """print('core_species2idx: ', core_species2idx)
        print('core_k_array: ', core_k_array)
        print('core_y_array: ', core_y_array)"""

        return self.sim_results

    def get_reaction(self,
                     id_key: str,
                     smiles: str = None,
                     datasource: Literal['database', 'rate_rule'] = 'rate_rule') -> Reaction:
        """
        get a reaction object containing kinetic parameters from database or rmg_rate_rule
        :param id_key: id_key of a reaction
        :param smiles: reaction SMILES
        :param datasource: get reaction from database or rmg rate rule
        :return: a Reaction object
        """
        assert id_key or smiles, 'Please prove id_key or smiles at least'
        if smiles is None:
            if self.edge.reaction_pool.has_id_key(id_key):
                sma = self.edge.reaction_pool.get_reaction_obj(id_key).smarts
            elif self.edge.reaction_pool.has_id_key(id_key):
                sma = self.core.reaction_pool.get_reaction_obj(id_key).smarts
            else:
                raise KeyError(f'Can not find id_key {id_key}')
            rxn = sma
        else:
            rxn = smiles

        # for a reaction, get forward abd reverse smiles
        rsmi, psmi = rxn.split('>>')
        rxn_re = f'{psmi}>>{rsmi}'
        if rxn in self.custom_kinetics.keys():
            try:
                kin_param = self.custom_kinetics[rxn]
                kin_param_re = self.custom_kinetics[rxn_re]
            except:
                raise KeyError('Both forward and reverse kinetics should be given')
        else:
            if datasource == 'database':
                kin_parameters = self.kin_predictor.database_search([rxn, rxn_re], return_param_only=True)
            else:
                kin_parameters = self.kin_predictor.rmg_search([rxn, rxn_re],
                                                               forbidden_families=self.forbidden_rmg_database_families)
            kin_param, kin_param_re = kin_parameters[rxn], kin_parameters[rxn_re]
        # print(f'reaction id {id_key}, smiles: {rxn}, forward_param: {kin_param}, reverse_param: {kin_param_re}')

        return self.form_reaction_obj(rxn, id_key, kin_param, kin_param_re)

    def form_reaction_obj(self, smarts: str, id_key: str, kin_param: dict, kin_param_re: dict) -> Reaction:
        """
        From a reaction object from database or RMG-database rate rule
        :param smarts: reaction smiles
        :param id_key: reaction reverse smiles
        :param kin_param: kinetic parameters of forward reactions
        :param kin_param_re: kinetic parameter of reverse reactions
        :return: a Reaction obejct
        """
        new_kin_objs = [Kinetics(mapped_smarts=smarts,
                                 freq_factor=kin_param['A'],
                                 freq_factor_re=kin_param_re['A'],
                                 activation_energy=kin_param['Ea'],
                                 activation_energy_re=kin_param_re['Ea'],
                                 fitting_param=kin_param['b'],
                                 fitting_param_re=kin_param_re['b'],
                                 temperature=self.temperature)]
        new_reaction_obj = Reaction(smarts=smarts, id_key=id_key, kinetics_list=new_kin_objs, calibrated=True)

        return new_reaction_obj

    def _form_ml_reaction_objs(self, results: dict, all_mapped_smiles: List[str]):
        """
        Forming NetGen reaction objects of ML reactions

        :param results: {(id_key, unmapped_reactions): mapped_reactions}
                        mapped reaction is a list of [mapped_smarts, mapped_smarts_re]
        :return: None
        """
        reaction_obj_dict = {}
        pred_kin = ml_preds(smiles=all_mapped_smiles,
                            temperature=self.temperature,
                            uni_cp_path=self.uni_cp_path,
                            bi_cp_path=self.bi_cp_path)
        for (id_key, unmapped_reactions), mapped_reactions in results.items():
            kin_objects = []
            for s, s_re in mapped_reactions:  # for unmapped smarts, there may be several mapped smarts
                info = pred_kin[s]
                info_re = pred_kin[s_re]
                kin_objects.append(Kinetics(mapped_smarts=s,
                                            freq_factor=math.exp(info['pred']),
                                            freq_factor_re=math.exp(info_re['pred']),
                                            freq_factor_unc=math.exp(math.sqrt(info['tot_unc']) * self.z_score),
                                            freq_factor_unc_re=math.exp(math.sqrt(info_re['tot_unc']) * self.z_score),
                                            temperature=self.temperature))
            reaction_obj = Reaction(smarts=unmapped_reactions, id_key=id_key, kinetics_list=kin_objects)
            reaction_obj_dict[id_key] = reaction_obj

        return reaction_obj_dict

    def save_model(self, path: str):
        save_model(model=self, pool_path=path)

    def load_model_info(self, pool_path: str):
        """
        Load reaction and species pool to core and edge
        :param pool_path: the yaml file path
        :return:
        """
        with open(pool_path, 'r') as f:
            pools = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        core_species_pool, core_reaction_pool = _create_reaction_model(species_pool_dict=pools["core_species_pool"],
                                                                       reaction_pool_list=pools["core_reaction_pool"])
        edge_species_pool, edge_reaction_pool = _create_reaction_model(species_pool_dict=pools["edge_species_pool"],
                                                                       reaction_pool_list=pools["edge_reaction_pool"])
        self.core.species_pool = core_species_pool
        self.core.reaction_pool = core_reaction_pool
        self.edge.species_pool = edge_species_pool
        self.edge.reaction_pool = edge_reaction_pool


def _get_reaction_model_data(model: ReactionModel):
    # 1. get species pool
    species_pool_dict = {"filter_iter_number": model.species_pool.filter_iter_num,
                         "species_pool": {}}
    for s, s_obj in model.species_pool.pool.items():
        species_dict = {"smiles": s_obj.smiles,
                        "exist_iterations": s_obj.exist_iterations,
                        "reaction_id_key": list(s_obj.reaction_id_keys),
                        "is_reacted": s_obj.is_reacted}
        species_pool_dict["species_pool"][s] = species_dict

    # 2. get reaction pool. Reaction pool can be saved by a list
    reaction_pool_list = []
    for r_obj in model.reaction_pool.id_key2reaction.values():
        reaction_dict = {"smarts": r_obj.smarts,
                         "calibrated": r_obj.calibrated,
                         "id_key": r_obj.id_key,
                         }
        kin_dict_list = []
        for kin_obj in r_obj.kinetics:
            kin_dict = {"mapped_smarts": kin_obj.mapped_smarts,
                        "temperature": kin_obj.T,
                        "A": kin_obj.A,
                        "Ea": kin_obj.Ea,
                        "b": kin_obj.b,
                        "A_re": kin_obj.A_re,
                        "Ea_re": kin_obj.Ea_re,
                        "b_re": kin_obj.b_re,
                        "A_unc": kin_obj.A_unc,
                        "Ea_unc": kin_obj.Ea_unc,
                        "A_unc_re": kin_obj.A_unc_re,
                        "Ea_unc_re": kin_obj.Ea_unc_re}
            kin_dict_list.append(kin_dict)
        reaction_dict["kinetics"] = kin_dict_list
        reaction_pool_list.append(reaction_dict)

    return species_pool_dict, reaction_pool_list


def save_model(pool_path: str, model: CoreEdgeReactionModel):
    """
    Saving a reaction model
    for a reaction model, we have core and edge
    for core or edge, we have species pool that contains the species objects and some mapping dicts
    we also have reaction pool that contains reaction objects and some mapping dicts
    All of these should be saved

    :param pool_path: save path, should be a yaml file
    :param model: the model to be saved
    :return: None
    """
    model_dict = {}
    core, edge = model.core, model.edge
    core_species_pool, core_reaction_pool = _get_reaction_model_data(core)
    edge_species_pool, edge_reaction_pool = _get_reaction_model_data(edge)
    model_dict["core_reaction_pool"] = core_reaction_pool
    model_dict["core_species_pool"] = core_species_pool
    model_dict["edge_reaction_pool"] = edge_reaction_pool
    model_dict["edge_species_pool"] = edge_species_pool

    with open(pool_path, 'w') as f:
        yaml.dump(model_dict, f)
        f.close()


def _create_reaction_model(species_pool_dict: dict, reaction_pool_list: list):
    """
    create a reaction model
    :param species_pool_dict: the species pool dictionary
    :param reaction_pool_list: the reaction pool list
    :return:
    """
    if species_pool_dict["filter_iter_number"] is None:
        filter_iter_num = None
    else:
        filter_iter_num = int(species_pool_dict["filter_iter_number"])
    species_pool = SpeciesPool(filter_iter_num=filter_iter_num)
    for s, species_dict in species_pool_dict['species_pool'].items():
        species_obj = Species(smiles=species_dict['smiles'],
                              species_exist_iterations=species_dict["exist_iterations"],
                              reaction_id_key=species_dict["reaction_id_key"],
                              is_reacted=species_dict["is_reacted"])
        species_pool.add_species(species_obj)

    reaction_pool = ReactionPool()
    for reaction_dict in reaction_pool_list:
        kin_objs = []
        for kinetic_dict in reaction_dict["kinetics"]:
            kin_obj = Kinetics(mapped_smarts=kinetic_dict["mapped_smarts"],
                               temperature=kinetic_dict["temperature"],
                               freq_factor=kinetic_dict["A"],
                               freq_factor_re=kinetic_dict["A_re"],
                               fitting_param=kinetic_dict["b"],
                               fitting_param_re=kinetic_dict["b_re"],
                               activation_energy=kinetic_dict["Ea"],
                               activation_energy_re=kinetic_dict["Ea_re"],
                               freq_factor_unc=kinetic_dict["A_unc"],
                               freq_factor_unc_re=kinetic_dict["A_unc_re"],
                               activation_energy_unc=kinetic_dict["Ea_unc"],
                               activation_energy_unc_re=kinetic_dict["Ea_unc_re"])
            kin_objs.append(kin_obj)

        reaction_obj = Reaction(smarts=reaction_dict["smarts"],
                                calibrated=reaction_dict["calibrated"],
                                id_key=reaction_dict["id_key"],
                                kinetics_list=kin_objs)

        reaction_pool.add_reaction(reaction_obj)

    return species_pool, reaction_pool

