from rmgpy.species import Species
from netgen.database.rmg_tools import database, generateSpeciesThermo
import rmgpy

global database

database.load('thermo')
database_path = rmgpy.settings['database.directory']


def get_cantera_thermo_input(smiles: str, data_source: str = None):
    """
    Get canetra input thermo data of given SMILES.
    :param smiles: the species to get thermo data
    :param data_source: which source of the thermo inpout comes from
    :return:
    """
    species = Species().from_smiles(smiles.strip())
    all_data = []
    for data, library, entry in database.thermo.get_all_thermo_data(species):
        source, data_type = data.comment, type(data).__name__
        all_data.append([source, data_type, data])

    selected_data = None
    if data_source is not None:
        for data in all_data:
            if data[0] == data_source:
                selected_data = data
                break
    else:
        # get NASA type data first match found
        for data in all_data:
            if data[1] == 'NASA':
                selected_data = data
                break
        # else, get other type data
        if selected_data is None:
            selected_data = all_data[0]

    if selected_data is None:
        # if still no data object were found,return None
        return None
    else:
        if selected_data[1] == 'NASA':
            therm_nasa = selected_data[-1]
        else:
            therm_nasa = selected_data[-1].to_nasa(Tmin=100, Tmax=5000, Tint=1000)
        ct_obj = therm_nasa.to_cantera()
        return ct_obj.input_data

