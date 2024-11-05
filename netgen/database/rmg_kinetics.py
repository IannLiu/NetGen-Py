import rmgpy
from rmgpy.species import Species

from rmgpy.kinetics import Arrhenius, ArrheniusEP, ArrheniusBM, KineticsData
from netgen.database.rmg_tools import database, generateReactions, generateSpeciesThermo, reactionHasReactants

from rmgpy.data.kinetics import TemplateReaction, LibraryReaction
from rmgpy.data.kinetics.depository import DepositoryReaction

global database

database.load('kinetics')
database.load('thermo')
database_path = rmgpy.settings['database.directory']


def sel_kinetics(kinetic_data_list, required_rxn_type_list):
    # Selecting the required kinetics
    kinetics = None
    kin_info = None
    for kinetic_data in kinetic_data_list:
        # print(kinetic_data)
        kin_info = kinetic_data[0].split('-')
        is_forward = kinetic_data[2]
        if kin_info[0] in required_rxn_type_list:
            if is_forward:
                kinetics = kinetic_data[1]
                break
            else:
                kinetics = kinetic_data[1]
                break
    return kinetics, kin_info


def get_all_kinetics(r_list, p_list, T, P, required_rxn_type):
    """
    Get all of the kinetics from the RMG databse
    """

    reactant_list = [Species().from_smiles(smi.strip()) for smi in r_list if smi]
    product_list = [Species().from_smiles(smi.strip()) for smi in p_list if smi]
    reaction_list = generateReactions(database, reactant_list, product_list, resonance=True)

    # Go through database and group additivity kinetics entries
    kinetics_data_list = []
    for reaction in reaction_list:
        # Generate the thermo data for the species involved
        for reactant in reaction.reactants:
            generateSpeciesThermo(reactant, database)
        for product in reaction.products:
            generateSpeciesThermo(product, database)

        if isinstance(reaction.kinetics, (ArrheniusEP, ArrheniusBM)):
            reaction.kinetics = reaction.kinetics.to_arrhenius(reaction.get_enthalpy_of_reaction(298))

        is_forward = reactionHasReactants(reaction, reactant_list)

        # find the reaction source to determine which reaction should be selected
        if isinstance(reaction, TemplateReaction):
            source = reaction.estimator + '-' + reaction.family
        elif isinstance(reaction, DepositoryReaction):
            source = "depository" + '-' + reaction.depository.name
        elif isinstance(reaction, LibraryReaction):
            source = "library" + '-' + reaction.library.name

        forward_kinetics = reaction.kinetics
        # The reverse form kinetics should be transformed
        if is_forward:

            kinetics_data_list.append([source, forward_kinetics, is_forward])
        else:
            try:
                reverse_kinetics = reaction.generate_reverse_rate_coefficient()
            except:
                # The method does not support `generate_reverse_rate_coefficient`
                reverse_kinetics = None
            else:
                reverse_kinetics.Tmin = forward_kinetics.Tmin
                reverse_kinetics.Tmax = forward_kinetics.Tmax
                reverse_kinetics.Pmin = forward_kinetics.Pmin
                reverse_kinetics.Pmax = forward_kinetics.Pmax
            finally:
                kinetics_data_list.append([source, reverse_kinetics, is_forward])

    # Selecting the required kineitcs
    kinetics, kin_info = sel_kinetics(kinetics_data_list, [required_rxn_type])

    """
    if kinetics is None:
        kinetics, kin_info = selec_kinetics(kinetics_data_list, ['group additivity', 'rate rules'])
    if kinetics is None:
        kinetics, kin_info = selec_kinetics(kinetics_data_list, ['depository'])
    if kinetics is None:
        kinetics, kin_info = selec_kinetics(kinetics_data_list, ['library'])
    """

    # if all of the kinetics are None
    if kinetics is None:
        raise Exception(" No kinetics data is obtianed")
    else:
        rate = kinetics.get_rate_coefficient(T, P)

    return rate, kin_info[0]


def get_all_kinetics_obj(s: str):
    """
    Get all of the kinetics from the RMG databse
    """
    rsmis, psmis = s.split('>>')
    reactant_list = [Species().from_smiles(smi.strip()) for smi in rsmis.split('.') if smi]
    product_list = [Species().from_smiles(smi.strip()) for smi in psmis.split('.') if smi]
    reaction_list = generateReactions(database, reactant_list, product_list, resonance=True)  # generate a reaction

    # Go through database and group additivity kinetics entries
    kinetics_data_list = []
    for reaction in reaction_list:
        # Generate the thermo data for the species involved
        for reactant in reaction.reactants:
            generateSpeciesThermo(reactant, database)
        for product in reaction.products:
            generateSpeciesThermo(product, database)

        if isinstance(reaction.kinetics, (ArrheniusEP, ArrheniusBM)):
            reaction.kinetics = reaction.kinetics.to_arrhenius(reaction.get_enthalpy_of_reaction(298))

        is_forward = reactionHasReactants(reaction, reactant_list)

        # find the reaction source to determine which reaction should be selected
        if isinstance(reaction, TemplateReaction):
            source = reaction.estimator + '-' + reaction.family
        elif isinstance(reaction, DepositoryReaction):
            source = "depository" + '-' + reaction.depository.name
        elif isinstance(reaction, LibraryReaction):
            source = "library" + '-' + reaction.library.name

        forward_kinetics = reaction.kinetics
        # The reverse form kinetics should be transformed
        if is_forward:
            kinetics_data_list.append([source, forward_kinetics, is_forward])
        else:
            try:
                reverse_kinetics = reaction.generate_reverse_rate_coefficient()
            except:
                # The method does not support `generate_reverse_rate_coefficient`
                reverse_kinetics = None
            else:
                reverse_kinetics.Tmin = forward_kinetics.Tmin
                reverse_kinetics.Tmax = forward_kinetics.Tmax
                reverse_kinetics.Pmin = forward_kinetics.Pmin
                reverse_kinetics.Pmax = forward_kinetics.Pmax
            finally:
                kinetics_data_list.append([source, reverse_kinetics, is_forward])

    return kinetics_data_list

