import os
from rmgpy.data.kinetics import KineticsDatabase, TemplateReaction
from rmgpy.data.kinetics.depository import DepositoryReaction
from rmgpy.data.rmg import RMGDatabase, SolvationDatabase, StatmechDatabase
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.transport import TransportDatabase
from rmgpy.reaction import same_species_lists
from rmgpy import settings


db_path = settings['database.directory']


class RMGWebDatabase(object):
    """Wrapper class for RMGDatabase that provides loading functionality."""

    def __init__(self):
        self.database = RMGDatabase()
        self.database.kinetics = KineticsDatabase()
        self.database.thermo = ThermoDatabase()
        self.database.transport = TransportDatabase()
        self.database.statmech = StatmechDatabase()
        self.database.solvation = SolvationDatabase()
        self.database.load_forbidden_structures(
            os.path.join(db_path, 'forbiddenStructures.py')
            )
        self.timestamps = {}

    @property
    def kinetics(self):
        """
        Get the kinetics database.
        """
        return self.database.kinetics

    @property
    def thermo(self):
        """
        Get the thermo database.
        """
        return self.database.thermo

    @property
    def transport(self):
        """
        Get the transport database.
        """
        return self.database.transport

    @property
    def statmech(self):
        """
        Get the statmech database.
        """
        return self.database.statmech

    @property
    def solvation(self):
        """
        Get the solvation database.
        """
        return self.database.solvation

    def reset_timestamp(self, path):
        """
        Reset the files timestamp in the dictionary of timestamps.
        """
        mtime = os.stat(path).st_mtime
        self.timestamps[path] = mtime

    def reset_dir_timestamps(self, dirpath):
        """
        Walk the directory tree from dirpath, calling reset_timestamp(file) on each file.
        """
        print("Resetting 'last loaded' timestamps for {0} in process {1}".format(dirpath, os.getpid()))
        for root, dirs, files in os.walk(dirpath):
            for name in files:
                self.reset_timestamp(os.path.join(root, name))

    def is_file_modified(self, path):
        """
        Return True if the file at `path` has been modified since `reset_timestamp(path)` was last called.
        """
        # If path doesn't denote a file and were previously
        # tracking it, then it has been removed or the file type
        # has changed, so return True.
        if not os.path.isfile(path):
            return path in self.timestamps

        # If path wasn't being tracked then it's new, so return True
        elif path not in self.timestamps:
            return True

        # Force restart when modification time has changed, even
        # if time now older, as that could indicate older file
        # has been restored.
        elif os.stat(path).st_mtime != self.timestamps[path]:
            return True

        # All the checks have been passed, so the file was not modified
        else:
            return False

    def is_dir_modified(self, dirpath):
        """
        Returns True if anything in the directory at dirpath has been modified since reset_dir_timestamps(dirpath).
        """
        to_check = set([path for path in self.timestamps if path.startswith(dirpath)])
        for root, dirs, files in os.walk(dirpath):
            for name in files:
                path = os.path.join(root, name)
                if self.is_file_modified(path):
                    return True
                to_check.remove(path)
        # If there's anything left in to_check, it's probably now gone and this will return True:
        for path in to_check:
            if self.is_file_modified(path):
                return True
        # Passed all tests.
        return False

    ################################################################################

    def load(self, component='', section=''):
        """
        Load the requested `component` of the RMG database if modified since last loaded.
        """
        if component in ['thermo', '']:
            if section in ['depository', '']:
                dirpath = os.path.join(db_path, 'thermo', 'depository')
                if self.is_dir_modified(dirpath):
                    self.database.thermo.load_depository(dirpath)
                    self.reset_dir_timestamps(dirpath)
            if section in ['libraries', '']:
                dirpath = os.path.join(db_path, 'thermo', 'libraries')
                if self.is_dir_modified(dirpath):
                    self.database.thermo.load_libraries(dirpath)
                    # put them in our preferred order, so that when we look up thermo in order to estimate kinetics,
                    # we use our favorite values first.
                    preferred_order = [
                        'primaryThermoLibrary',
                        'DFT_QCI_thermo',
                        'GRI-Mech3.0',
                        'CBS_QB3_1dHR',
                        'KlippensteinH2O2',
                    ]
                    new_order = [i for i in preferred_order if i in self.database.thermo.library_order]
                    for i in self.database.thermo.library_order:
                        if i not in new_order:
                            new_order.append(i)
                    self.database.thermo.library_order = new_order
                    self.reset_dir_timestamps(dirpath)
            if section in ['groups', '']:
                dirpath = os.path.join(db_path, 'thermo', 'groups')
                if self.is_dir_modified(dirpath):
                    self.database.thermo.load_groups(dirpath)
                    self.reset_dir_timestamps(dirpath)
            # Load metal database if necessary
            if section in ['surface', '']:
                dirpath = os.path.join(db_path, 'surface')
                if self.is_dir_modified(dirpath):
                    self.database.thermo.load_surface()
                    self.reset_dir_timestamps(dirpath)

        if component in ['transport', '']:
            if section in ['libraries', '']:
                dirpath = os.path.join(db_path, 'transport', 'libraries')
                if self.is_dir_modified(dirpath):
                    self.database.transport.load_libraries(dirpath)
                    self.reset_dir_timestamps(dirpath)
            if section in ['groups', '']:
                dirpath = os.path.join(db_path, 'transport', 'groups')
                if self.is_dir_modified(dirpath):
                    self.database.transport.load_groups(dirpath)
                    self.reset_dir_timestamps(dirpath)

        if component in ['solvation', '']:
            dirpath = os.path.join(db_path, 'solvation')
            if self.is_dir_modified(dirpath):
                self.database.solvation.load(dirpath)
                self.reset_dir_timestamps(dirpath)

        if component in ['kinetics', '']:
            if section in ['libraries', '']:
                dirpath = os.path.join(db_path, 'kinetics', 'libraries')
                if self.is_dir_modified(dirpath):
                    self.database.kinetics.load_libraries(dirpath)
                    self.reset_dir_timestamps(dirpath)
            if section in ['families', '']:
                dirpath = os.path.join(db_path, 'kinetics', 'families')
                if self.is_dir_modified(dirpath):
                    self.database.kinetics.load_families(dirpath, families='all', depositories='all')
                    self.reset_dir_timestamps(dirpath)

                    # Make sure to load the entire thermo database prior to adding training values to the rules
                    self.load('thermo', '')
                    for family in self.database.kinetics.families.values():
                        old_entries = len(family.rules.entries)
                        family.add_rules_from_training(thermo_database=self.database.thermo)
                        new_entries = len(family.rules.entries)
                        if new_entries != old_entries:
                            print('{0} new entries added to {1} family after adding rules '
                                  'from training set.'.format(new_entries - old_entries, family.label))
                        # Filling in rate rules in kinetics families by averaging...
                        family.fill_rules_by_averaging_up()

        if component in ['statmech', '']:
            dirpath = os.path.join(db_path, 'statmech')
            if self.is_dir_modified(dirpath):
                self.database.statmech.load(dirpath)
                self.reset_dir_timestamps(dirpath)

    def get_transport_database(self, section, subsection):
        """
        Return the component of the transport database corresponding to the
        given `section` and `subsection`. If either of these is invalid, a
        :class:`ValueError` is raised.
        """
        try:
            if section == 'libraries':
                db = self.database.transport.libraries[subsection]
            elif section == 'groups':
                db = self.database.transport.groups[subsection]
            else:
                raise ValueError('Invalid value "%s" for section parameter.' % section)
        except KeyError:
            raise ValueError('Invalid value "%s" for subsection parameter.' % subsection)

        return db

    def get_solvation_database(self, section, subsection):
        """
        Return the component of the solvation database corresponding to the
        given `section` and `subsection`. If either of these is invalid, a
        :class:`ValueError` is raised.
        """
        try:
            if section == '':
                db = self.database.solvation  # return general SolvationDatabase
            elif section == 'libraries':
                db = self.database.solvation.libraries[subsection]
            elif section == 'groups':
                db = self.database.solvation.groups[subsection]
            else:
                raise ValueError('Invalid value "%s" for section parameter.' % section)
        except KeyError:
            raise ValueError('Invalid value "%s" for subsection parameter.' % subsection)

        return db

    def get_statmech_database(self, section, subsection):
        """
        Return the component of the statmech database corresponding to the
        given `section` and `subsection`. If either of these is invalid, a
        :class:`ValueError` is raised.
        """
        try:
            if section == 'depository':
                db = self.database.statmech.depository[subsection]
            elif section == 'libraries':
                db = self.database.statmech.libraries[subsection]
            elif section == 'groups':
                db = self.database.statmech.groups[subsection]
            else:
                raise ValueError('Invalid value "%s" for section parameter.' % section)
        except KeyError:
            raise ValueError('Invalid value "%s" for subsection parameter.' % subsection)

        return db

    def get_thermo_database(self, section, subsection):
        """
        Return the component of the thermodynamics database corresponding to the
        given `section` and `subsection`. If either of these is invalid, a
        :class:`ValueError` is raised.
        """
        try:
            if section == 'depository':
                db = self.database.thermo.depository[subsection]
            elif section == 'libraries':
                db = self.database.thermo.libraries[subsection]
            elif section == 'groups':
                db = self.database.thermo.groups[subsection]
            else:
                raise ValueError('Invalid value "%s" for section parameter.' % section)
        except KeyError:
            raise ValueError('Invalid value "%s" for subsection parameter.' % subsection)

        return db

    def get_kinetics_database(self, section, subsection):
        """
        Return the component of the kinetics database corresponding to the
        given `section` and `subsection`. If either of these is invalid, a
        :class:`ValueError` is raised.
        """
        db = None
        try:
            if section == 'libraries':
                db = self.database.kinetics.libraries[subsection]
            elif section == 'families':
                subsection = subsection.split('/')
                if subsection[0] != '' and len(subsection) == 2:
                    family = self.database.kinetics.families[subsection[0]]
                    if subsection[1] == 'groups':
                        db = family.groups
                    elif subsection[1] == 'rules':
                        db = family.rules
                    else:
                        label = '{0}/{1}'.format(family.label, subsection[1])
                        db = next((d for d in family.depositories if d.label == label))
            else:
                raise ValueError('Invalid value "%s" for section parameter.' % section)
        except (KeyError, StopIteration):
            raise ValueError('Invalid value "%s" for subsection parameter.' % subsection)
        return db


def generateSpeciesThermo(species, database):
    """
    Generate the thermodynamics data for a given :class:`Species` object
    `species` using the provided `database`.
    """
    species.generate_resonance_structures()
    species.thermo = database.thermo.get_thermo_data(species)

################################################################################


def generateReactions(database, reactants, products=None, only_families=None, resonance=True):
    """
    Generate the reactions (and associated kinetics) for a given set of
    `reactants` and an optional set of `products`. A list of reactions is
    returned, with a reaction for each matching kinetics entry in any part of
    the database. This means that the same reaction may appear multiple times
    with different kinetics in the output.

    If `only_families` is a list of strings, only those labeled families are
    used: no libraries and no RMG-Java kinetics are returned.
    """
    from rmgpy.rmg.model import get_family_library_object
    # get RMG-py reactions
    reaction_list = database.kinetics.generate_reactions(
        reactants, products, only_families=only_families, resonance=resonance)
    if len(reactants) == 1:
        # if only one reactant, react it with itself bimolecularly, with RMG-py
        # the java version already does this (it includes A+A reactions when you react A)
        reactants2 = [reactants[0], reactants[0]]
        reaction_list.extend(database.kinetics.generate_reactions(
            reactants2, products, only_families=only_families, resonance=resonance))

    # get RMG-py kinetics
    reaction_data_list = []
    template_reactions = []
    for reaction in reaction_list:
        # If the reaction already has kinetics (e.g. from a library),
        # assume the kinetics are satisfactory
        if reaction.kinetics is not None:
            reaction_data_list.append(reaction)
        else:
            # Set the reaction kinetics
            # Only reactions from families should be missing kinetics
            assert isinstance(reaction, TemplateReaction)

            # Determine if we've already processed an isomorphic reaction with a different template
            duplicate = False
            for t_rxn in template_reactions:
                if reaction.is_isomorphic(t_rxn):
                    assert set(reaction.template) != set(t_rxn.template), 'There should not be duplicate reactions with identical templates.'
                    duplicate = True
                    break
            else:
                # We haven't encountered this reaction yet, so add it to the list
                template_reactions.append(reaction)

            # Get all of the kinetics for the reaction
            family = get_family_library_object(reaction.family)
            kinetics_list = family.get_kinetics(reaction, template_labels=reaction.template, degeneracy=reaction.degeneracy, return_all_kinetics=True)
            if family.own_reverse and hasattr(reaction, 'reverse'):
                kinetics_list_rev = family.get_kinetics(reaction.reverse, template_labels=reaction.reverse.template, degeneracy=reaction.reverse.degeneracy, return_all_kinetics=True)
                for kinetics, source, entry, is_forward in kinetics_list_rev:
                    for kinetics0, source0, entry0, is_forward0 in kinetics_list:
                        if (source0 is not None) and (source is not None) and (entry0 is entry) and (is_forward != is_forward0):
                            # We already have this estimate from the forward direction, so don't duplicate it in the results
                            break
                    else:
                        kinetics_list.append([kinetics, source, entry, not is_forward])
                # We're done with the "reverse" attribute, so delete it to save a bit of memory
                delattr(reaction, 'reverse')
            # Make a new reaction object for each kinetics result
            for kinetics, source, entry, is_forward in kinetics_list:
                if duplicate and source != 'rate rules':
                    # We've already processed this reaction with a different template,
                    # so we only need the new rate rule estimates
                    continue

                if is_forward:
                    reactant_species = reaction.reactants[:]
                    product_species = reaction.products[:]
                else:
                    reactant_species = reaction.products[:]
                    product_species = reaction.reactants[:]

                if source == 'rate rules' or source == 'group additivity':
                    rxn = TemplateReaction(
                        reactants=reactant_species,
                        products=product_species,
                        kinetics=kinetics,
                        degeneracy=reaction.degeneracy,
                        reversible=reaction.reversible,
                        family=reaction.family,
                        estimator=source,
                        template=reaction.template,
                    )
                else:
                    rxn = DepositoryReaction(
                        reactants=reactant_species,
                        products=product_species,
                        kinetics=kinetics,
                        degeneracy=reaction.degeneracy,
                        reversible=reaction.reversible,
                        depository=source,
                        family=reaction.family,
                        entry=entry,
                    )

                reaction_data_list.append(rxn)

    return reaction_data_list

################################################################################


def reactionHasReactants(reaction, reactants):
    """
    Return ``True`` if the given `reaction` has all of the specified
    `reactants` (and no others), or ``False if not.
    """
    return same_species_lists(reaction.reactants, reactants, strict=False)


################################################################################

# Initialize module level database instance
database = RMGWebDatabase()
