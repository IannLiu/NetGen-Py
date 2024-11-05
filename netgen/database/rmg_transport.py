from rmgpy.species import Species
from netgen.database.rmg_tools import database
import rmgpy

global database

database.load('transport')
database_path = rmgpy.settings['database.directory']


def get_cantera_transport_input(smiles: str, data_source: str = None):
    transport_data_list = []
    species = Species().from_smiles(smiles.strip())
    for d, library, entry in database.transport.get_all_transport_properties(species):
        if library is None:
            source = 'Group additivity'
        elif library in database.transport.libraries.values():
            source = library.label
        else:
            source = None
        # data = d.to_cantera()
        transport_data_list.append([source, d, entry])

    data_cantera = None
    # find the transport property of given data_source
    if data_source is not None:
        for transport_data in transport_data_list:
            if transport_data[0] == data_source:
                data_cantera = transport_data[1].to_cantera()
                return data_cantera.input_data
        if data_cantera is None:  # cannot find the data of given data_source
            return None
    else:
        # get the first obj
        if transport_data_list:  # if we got data object
            data_cantera = transport_data_list[0][1].to_cantera()
            return data_cantera.input_data
        else:
            return None

