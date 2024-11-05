"""
This is the basic data type of species and species pool for Edge Species !!!
Edge species is decoupled from reaction networks, and should not be considered
during solving ODEs. The users must determine whether the species should be
wrapper by Species object and/or added to the Species pool
"""

from typing import List, Dict
from netgen.common import constants
from rdkit import Chem
import numpy as np
from typing import Union


class Kinetics:
    """
    A class of kinetics, current only developing for modified Arrhenius function

    k = A * T ** (b) * exp(-Ea / R / T)
    in which A is the frequency factor, T is temperature, Ea is the activation energy,
    R is the gas law constant, b is the fitting parameter
    All parameters should be in SI. J, m, s
    """
    def __init__(self,
                 mapped_smarts: str,
                 freq_factor: float = 0,
                 freq_factor_re: float = 0,
                 fitting_param: float = 0,
                 fitting_param_re: float = 0,
                 activation_energy: float = 0,
                 activation_energy_re: float = 0,
                 freq_factor_unc: float = 1,
                 freq_factor_unc_re: float = 1,
                 activation_energy_unc: float = 0,
                 activation_energy_unc_re: float = 0,
                 temperature: float = 278.15):
        self._mapped_smarts = mapped_smarts
        self.Ea = activation_energy
        self.Ea_re = activation_energy_re
        self.b = fitting_param
        self.b_re = fitting_param_re
        self.T = temperature
        self.A = freq_factor
        self.A_re = freq_factor_re
        self.A_unc = freq_factor_unc
        self.A_unc_re = freq_factor_unc_re
        self.Ea_unc = activation_energy_unc
        self.Ea_unc_re = activation_energy_unc_re

    def __str__(self):
        return self._mapped_smarts

    def __hash__(self):
        return hash(self._mapped_smarts)

    def __repr__(self):

        return self._mapped_smarts

    def get_rate_coeff(self, temperature: float = None, reverse: bool = False):
        if temperature is not None:
            self.T = temperature
        if not reverse:
            return self.A * (self.T ** self.b) * np.exp(-self.Ea / constants.R / self.T)
        else:
            return self.A_re * (self.T ** self.b_re) * np.exp(-self.Ea_re / constants.R / self.T)

    def get_max_rate_coeff(self,  temperature: float = None, reverse: bool = False):
        if temperature is not None:
            self.T = temperature
        if not reverse:
            return (self.A * self.A_unc) * (self.T ** self.b) * np.exp((-self.Ea + self.Ea_unc) / constants.R / self.T)
        else:
            return (self.A_re * self.A_unc_re) * (self.T ** self.b_re) * np.exp((-self.Ea_re + self.Ea_unc_re) / constants.R / self.T)

    @property
    def mapped_smarts(self):
        return self._mapped_smarts


class Reaction:
    """
    A reaction class contains the unmapped reaction smiles and corresponding reactions
    """

    def __init__(self,
                 smarts: str,
                 kinetics_list: List[Kinetics],
                 id_key: str,
                 in_core: bool = False,
                 calibrated: bool = False):
        """
        :param smarts: unmapped reaction smarts
        :param kinetics_list: a list of kinetics object containing the mapped smiles and kinetic parameters
        """
        self.smarts = smarts
        self._kinetics = kinetics_list
        self.in_core = in_core
        self.id_key = id_key
        self.calibrated = calibrated

    def __hash__(self):
        return hash(self.smarts)

    def __str__(self):
        return self.smarts

    def __repr__(self):
        return self.smarts

    @property
    def mapped_smarts(self):
        return [kin.mapped_smarts for kin in self._kinetics]

    @property
    def kinetics(self):
        return self._kinetics

    def get_coef(self, reverse: bool = False):
        """
        Get rate coefficient

        :param reverse: return reverse coefficients
        :return:
        """
        coeff = 0
        for kin in self.kinetics:
            coeff += kin.get_rate_coeff(reverse=reverse)
        return coeff

    def get_max_coef(self, reverse: bool = False):
        """

        """
        coeff = 0
        for kin in self.kinetics:
            coeff += kin.get_max_rate_coeff(reverse=reverse)
        return coeff

    def get_rate(self, concentration: Dict[str, float], reverse: bool = False):
        """
        Get reaction rate
        :param concentration: a dictionary contains species concentration
        :param reverse: return reverse rate

        :return: None
        """
        reactants = self.smarts.split('>>')[0].split('.')
        rate = 1
        coeff = self.get_coef(reverse=reverse)
        for s in reactants:
            rate *= concentration[s]

        return rate * coeff

    @property
    def get_stoichiometric_num(self) -> Dict[str, int]:

        """
        Get stoichiometric num for every species
        Note that if a species appears in both reactant and product side once
        its stoichiometric number is 0
        :return:
        """
        stoi = {}
        reacs, prods = self.smarts.split('>>')
        for smi in reacs.split('.'):
            stoi[smi] = stoi.get(smi, 0) - 1

        for smi in prods.split('.'):
            stoi[smi] = stoi.get(smi, 0) + 1

        return stoi

    def get_rsmis(self):
        reacs, prods = self.smarts.split('>>')
        return reacs.split('.')

    def get_psmis(self):
        reacs, prods = self.smarts.split('>>')
        return prods.split('.')

    def get_smis(self):
        reacs, prods = self.smarts.split('>>')
        rsmis_list = reacs.split('.')
        psmis_list = prods.split('.')
        rsmis_list.extend(psmis_list)
        return rsmis_list


def register_id_key(smarts: str, canonicalize=True):
    """
    Generate a unique reaction id_key
    :param smarts: reaction smarts in smiles
    :param canonicalize:
    :return:
    """
    rsmis, psmis = smarts.split('>>')
    if canonicalize:
        rsmis_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in rsmis.split('.')]
        psmis_list = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in psmis.split('.')]
    else:
        rsmis_list, psmis_list = rsmis.split('.'), psmis.split('.')

    sorted_rsmis = sorted(rsmis_list)
    sorted_psmis = sorted(psmis_list)
    sorted_smarts = sorted(['.'.join(sorted_rsmis), '.'.join(sorted_psmis)])

    return '<=>'.join(sorted_smarts)


class ReactionPool:
    """
    A pool recording reactions
    Every reaction added to ReactionPool will be assigned a unique id_key
    """
    def __init__(self):
        self.id_key_pool = set({})
        self.id_key2reaction: Dict[str, Reaction] = {}
        self.counter = 0
        self.id_key2smarts = dict({})

    def get_reaction_obj(self, id_key: str) -> Reaction:

        return self.id_key2reaction[id_key]

    def add_reaction(self, reaction: Reaction):
        """
        :param reaction:
        :return:
        """
        self.id_key_pool.add(reaction.id_key)
        self.id_key2reaction[reaction.id_key] = reaction
        self.id_key2smarts[reaction.id_key] = reaction.smarts

    def remove_reaction_by_id_key(self, id_key):
        """

        :param id_key: the reaction id_key
        :return:
        """
        self.id_key_pool.remove(id_key)
        del self.id_key2reaction[id_key]
        del self.id_key2smarts[id_key]

    def remove_reaction(self, reaction: Union[Reaction, str]):
        if type(reaction).__name__ == 'str':
            id_key = reaction
        else:
            id_key = reaction.id_key
        self.id_key_pool.remove(id_key)
        del self.id_key2reaction[id_key]
        del self.id_key2smarts[id_key]

    def has_id_key(self, id_key: str):
        if id_key in self.id_key_pool:
            return True
        else:
            return False

    def reassign_reaction_obj(self, id_key: str, reaction: Reaction):
        """
        Reassign a reaction
        """
        self.id_key2reaction[id_key] = reaction
        self.id_key2smarts[id_key] = reaction.smarts

    @property
    def reaction_objs(self) -> List[Reaction]:
        """
        Return all reaction objects
        :return:
        """
        return list(self.id_key2reaction.values())


class Species:
    """

    """
    def __init__(self,
                 smiles: str,
                 reaction_id_key: List[str] = None,
                 species_exist_iterations: int = 0,
                 is_reacted: bool = False):
        """
        Initializing a species
        Args:
            smiles: species smiles
            reaction_id_key: the list of reaction id_skeys
            species_exist_iterations: iterations the species exists in.
        """
        self._smiles = smiles
        self._exist_iterations = species_exist_iterations
        self._is_reacted = is_reacted
        if reaction_id_key is None:
            self._reaction_id_keys = set({})  # for fast query
        else:
            self._reaction_id_keys = set(reaction_id_key)

    def __hash__(self):
        return hash(self._smiles)

    def add_reaction_id_key(self, id_key: str):
        """
        Add an id_key to species
        :param id_key:
        :return:
        """
        self._reaction_id_keys.add(id_key)

    def remove_reaction(self, id_key: str):
        pass

    def add_iteration(self, iters: int = 1):
        self._exist_iterations += iters

    def set_reacted(self, is_reacted: bool):
        self._is_reacted = is_reacted

    @property
    def smiles(self):
        return self._smiles

    @property
    def reaction_id_keys(self):
        """
        :return: {rxns: Kinetics}
        """
        return self._reaction_id_keys

    @property
    def exist_iterations(self):
        return self._exist_iterations

    @property
    def is_reacted(self):
        return self._is_reacted


class SpeciesPool:
    """
    An edge species pool contains Species objects
    """
    def __init__(self,
                 species_list: List[Species] = None,
                 filter_iter_num: int = None):
        self.pool: Dict[str, Species] = {}
        if species_list is not None:
            for species in species_list:
                self.pool.update({species.smiles: species})
        self.filter_iter_num = filter_iter_num

    def __str__(self):
        return ', '.join(list(self.pool.keys()))

    def __repr__(self):
        return ', '.join(list(self.pool.keys()))

    def has_species(self, species: str):
        """
        Check species
        :param species: The SMILES of species
        :return: bool
        """
        if species in self.pool:
            return True
        else:
            return False

    def get_species(self):
        """
        Return species smiles list
        :return:
        """
        return list(self.pool.keys())

    def get_species_reaction_id_keys(self, species):

        return self.pool[species].reaction_id_keys

    def add_species(self, species: Species):
        """

        :param species: A Species object
        :return: None
        """
        if species.smiles not in self.pool:
            self.pool[species.smiles] = species

    def add_species_list(self, species_list: List[Species]):
        for species in species_list:
            self.add_species(species)

    def remove_species(self, species_list: List[str]):
        """
        Remove species
        :param species_list: the species to be removed from pool
        :return: None
        """
        for s in species_list:
            if self.has_species(s):
                del self.pool[s]

    def add_reaction_id_key_to_species_list(self, species: List[str], reaction_id_key: str):
        """
        Add Rxn object to different species
        :param species: a SMILES list
        :param reaction_id_key: Rxn object to be added to species
        :return: None
        """
        for s in species:
            self.pool[s].add_reaction_id_key(reaction_id_key)

    def set_species_reactivity(self, species: str, reactivity: bool = True):

        self.pool[species].set_reacted(reactivity)

    def add_reaction_id_key_to_species(self, species: str, reaction_id_key: str):
        """
        Add Rxn object to different species
        :param species: a SMILES list
        :param reaction_id_key: Rxn object to be added to species
        :return: None
        """
        if species in self.pool:
            self.pool[species].add_reaction_id_key(reaction_id_key)
        else:
            self.pool[species] = Species(smiles=species, reaction_id_key=[reaction_id_key])

    def get_species_rate(self, concentration: Dict[str, float]) -> Dict:
        """
        Get edge species rate
        :param concentration: Species concentration
        :return: A dictionary {species: rate}
        """
        pass

    def get_species_reactivity(self):
        species_reactivity = {'reacted': [], 'unreacted': []}
        for n, s in self.pool.items():
            if s.is_reacted:
                species_reactivity['reacted'].append(n)
            else:
                species_reactivity['unreacted'].append(n)
        return species_reactivity

    def add_iter_num(self):
        for k, spec in self.pool.items():
            spec.add_iteration()

    def filter_by_iter_num(self):
        if self.filter_iter_num:
            filtered_spec = []
            for k, spec in self.pool.items():
                if spec.exist_iterations > self.filter_iter_num:
                    filtered_spec.append(k)

            for k in filtered_spec:
                del self.pool[k]

