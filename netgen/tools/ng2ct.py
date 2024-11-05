# Generating cantera input files from netgen output files
from netgen.database.rmg_transport import get_cantera_transport_input
from netgen.database.rmg_thermo import get_cantera_thermo_input
from netgen.common.chem import str_to_mol
import yaml


def ng2ct(path: str,
          save_path: str,
          additional_species_smiles: list = None,
          phase_name: str = 'gas',
          thermo: str = 'ideal-gas',
          kinetic_model: str = 'gas',
          transport_models: str = 'mixture-averaged',
          temperature: float = 1000,
          pressure: float = 101325,
          init_composition: dict = None
          ):
    """
    Convert netgen model to cantera model
    Note that the temperature, pressure, and composition can be reset when doing cantera simulation

    see https://cantera.org/tutorials/yaml/phases.html for more cantera input file information
    :param path: the path of netgen output file
    :param save_path: the path of saving converted file
    :param additional_species_smiles: add additional species to the output file.
    :param phase_name: the name of cantera phase
    :param thermo: the thermodynamic model ['ideal-gas', 'ideal-molal-solution'...]
    :param kinetic_model: the type of kinetic model ['gas', 'surface', 'edge']
    :param transport_models: the transport model ['high-pressure', 'mixture-averaged']
    :param temperature: the reaction temperature, K
    :param pressure: the reaction system pressure, Pa
    :param init_composition: the reaction system composition, like {O2: 1.0, N2: 3.76} in mole fraction
    :return:
    """
    with open(path, 'r') as f:
        netgen_output = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    species = list(netgen_output['core_species_pool']['species_pool'].keys())
    if additional_species_smiles is not None:
        species.extend(additional_species_smiles)

    # parsing reactions
    reactions = []
    for rxn_dict in netgen_output['core_reaction_pool']:
        cantera_rxn_dict = {}
        cantera_rxn_dict_re = {}
        sma, kinetics = rxn_dict['smarts'], rxn_dict['kinetics'][0]
        # get forward reaction equation
        reacs, prods = sma.split('>>')
        rsmis = [s.strip() for s in reacs.split('.')]
        psmis = [s.strip() for s in prods.split('.')]
        # reactions should be set as irreversible reaction to prevent the cantera to estimate
        # reverse rate based on the equilibrium constant
        cantera_rxn_dict['equation'] = f"{' + '.join(rsmis)} => {' + '.join(psmis)}"  # irreversible reaction
        cantera_rxn_dict_re['equation'] = f"{' + '.join(psmis)} => {' + '.join(rsmis)}"  # irreversible reaction
        cantera_rxn_dict['rate-constant'] = {'A': kinetics['A'], 'b': kinetics['b'], 'Ea': kinetics['Ea']}
        cantera_rxn_dict_re['rate-constant'] = {'A': kinetics['A_re'], 'b': kinetics['b_re'], 'Ea': kinetics['Ea_re']}
        reactions.append(cantera_rxn_dict)
        reactions.append(cantera_rxn_dict_re)

    # parsing speceis
    element_set = set({})
    species_dicts = []
    for s in species:
        # get composition
        species_dict = {'name': s}
        mol = str_to_mol(s)
        composition = {}
        for atom in mol.GetAtoms():
            composition[atom.GetSymbol()] = composition.get(atom.GetSymbol(), 0) + 1
        element_set.update(list(composition.keys()))
        species_dict['composition'] = composition
        # get thermo
        species_dict['thermo'] = get_cantera_thermo_input(s)
        # get transport
        species_dict['transport'] = get_cantera_transport_input(s)
        species_dicts.append(species_dict)

    # generate state
    state = {}
    if temperature is not None:
        state['T'] = f'{temperature} K'
    if pressure is not None:
        state['P'] = f'{pressure} Pa'
    if init_composition is not None:
        state['X'] = init_composition

    if not state:
        raise KeyError('the temperature, pressure, (composition) of reaction system should be defined')
    # Generate phase dict
    phase_dict = {'name': phase_name,
                  'thermo': thermo,
                  'elements': list(element_set),
                  'species': species,
                  'kinetics': kinetic_model,
                  'transport': transport_models,
                  'state': state}

    unit_dict = {'length': 'm', 'time': 's', 'quantity': 'mol', 'activation-energy': 'J/mol'}
    cantera_dict = {'units': unit_dict, 'phases': phase_dict, 'species': species_dicts, 'reactions': reactions}
    with open(save_path, 'w') as f:
        yaml.dump(cantera_dict, f, sort_keys=False)
        f.close()

