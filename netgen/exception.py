class IchemeException(Exception):
    """Base iChemE exception"""


# --------------------- Calculation exceptions --------------------------------
class AtomMappingException(IchemeException):
    """Incorrect atom mapping between reactants and products"""

