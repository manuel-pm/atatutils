from __future__ import print_function

import os

import numpy as np

from six import string_types
import yaml

from atatutils.clusterexpansion import is_int
from atatutils.file_parsers import ATATLattice


def make_has_file(base_folder, property_file):
    def has_file(folder):
        return os.path.isfile(os.path.join(base_folder, folder, property_file))
    return has_file


class ATATFolderParser(object):
    """Class to parse ATAT folders.

    Attributes
    ----------
    base_dir : str
        Reference path for folder names.
    condentrations : dict of dict from structure to triplet
        For each folder, for each structure, a triplet containing
        the total number of atoms, the number of atoms of each species
        and the concentration of each species.
    elements : dict of lists of pairs (str, str)
        Dictionary with the elements for each folder and their folder.
    folders : list of str
        List with all the parsed folders.
    lattice : dict of str
        Dictionary with the lattice file for each folder.
    structures: dict of list
        Dictionary with all the structures in each folder.

    Parameters
    ----------
    base_dir : str
        Reference path for folder names.

    Notes
    -----
    Schematically, the structure of an ATAT folder is the following:

    Material
        |
        |_ YAML descriptor file (optional)
        |_ lat.in
        |_ structure ID1
        |            |
        |            |_ str.out
        |            |_ property file 1
        |            |_ property file 2
        |            |_ ...
        |            |_ property folder 1
        |            |             |
        |            |             |_ property file 1
        |            |             |_ property file 2
        |            |             |_ ...
        |            |_ ...
        |
        |_ structure ID2
        |            |
        |            |_ str.out
        |            |_ property_file 1
        |            |_ property_file 2
        |            |_ ...
        |_ ...

    """
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
        self.concentrations = {}
        self.elements = {}
        self.folders = []
        self.lattice = {}
        self.structures = {}

    def add_folder(self, folder, descriptor_file=None, filters=[is_int]):
        """Adds a new folder and parses it.

        Parameters
        ----------
        folder : str
            Name of folder to add.
        descriptor_file : str, optional
            Name of YAML descriptor file with folder properties.
        filters : list of callables
            Filters that the structure folders must satisfy.

        Returns
        -------

        """
        self.folders.append(folder)
        if descriptor_file is not None:
            folder_properties = yaml.load(open(os.path.join(self.base_dir, folder, descriptor_file)))
            self.elements[folder] = zip(folder_properties['alloy_elements'],
                                        folder_properties['pure_element_folders'])
            if 'lattice_file' in folder_properties.keys():
                lattice_file = folder_properties['lattice_file']
            else:
                lattice_file = 'lat.in'
        else:
            self.elements[folder] = []
            structures = ['0', '1']  # FIXME: Assumes binary alloy
            for structure in structures:
                str_out = open(os.path.join(self.base_dir, folder, structure, 'str.out'))
                for line in str_out.readlines():
                    v = line.split()
                    if len(v) == 4:
                        elem = v[-1]
                str_out.close()
                self.elements[folder].append((elem, structure))

            lattice_file = 'lat.in'

        self.lattice[folder] = ATATLattice(base_dir=folder, lattice_file=lattice_file)

        self.structures[folder]  = [d for d in os.listdir(os.path.join(self.base_dir, folder)) if
                                    os.path.isdir(os.path.join(self.base_dir, folder, d))
                                    and all([f(d) for f in filters])]
        self.structures[folder].sort(key=int)

        self.concentrations[folder] = {}
        for structure in self.structures[folder]:
            str_out = open(os.path.join(self.base_dir, folder, structure, 'str.out'))
            n_atoms = 0.0
            n_atoms_s = np.zeros(len(self.elements[folder]), dtype=float)
            for line in str_out.readlines():
                v = line.split()
                if len(v) == 4:
                    n_atoms += 1
                    for j, elem in enumerate(self.elements[folder]):
                        if v[-1] == elem[0]:
                            n_atoms_s[j] += 1.
                            break
            str_out.close()
            self.concentrations[folder][structure] = (n_atoms, n_atoms_s, n_atoms_s/n_atoms)

    def structures_with_property(self, properties, folders=None, structures=None):
        """Returns a subset of all parsed structures containing the requested properties.

        Parameters
        ----------
        properties : str of list of str
            Properties requested.
        folders : list of str, optional
            Folders whose structures will be included.
             If not specified, use all parsed folders.
        structures : dict of list of str, optional
            Dictionary with all the structures for at least the
            requested folders. If not specified, use all parsed structures.

        Returns
        -------
        filtered_structures : dict of list of str
            Dictionary containing, for each folder, all the structures with
            the desired property files.

        """
        if folders is None:
            folders = self.folders
        if structures is None:
            structures = self.structures
        filtered_structures = {}
        if isinstance(properties, string_types):
            property = [properties]
        for folder in folders:
            filters = []
            for p in property:
                filters.append(make_has_file(os.path.join(self.base_dir, folder), p))
                filtered_structures[folder] = []
            for structure in structures[folder]:
                if all([f(structure) for f in filters]):
                    filtered_structures[folder].append(structure)

        return filtered_structures

    def as_list(self, folders=None, structures=None):
        """Returns the path to all structures in the folders parsed as a single list.

        Parameters
        ----------
        folders : list of str, optional
            Folders whose structures will be included.
             If not specified, use all parsed folders.
        structures : dict of list of str, optional
            Dictionary with all the structures for at least the
            requested folders. If not specified, use all parsed structures.

        Returns
        -------
        all_structures : list of str
            List with all the structures (full path) in all the folders.

        """
        if folders is None:
            folders = self.folders
        if structures is None:
            structures = self.structures
        all_structures = []
        for folder in folders:
            all_structures += [os.path.join(self.base_dir, folder, s) for s in structures[folder]]
        return all_structures

    def get_property(self, property, formation=False, intensive=False, folders=None, structures=None):
        """Get the given property and the structures which have it.

        Parameters
        ----------
        property : str
            Name of the file containing the property to load.
        formation : bool
            Whether the formation value of the property should be returned.
        intensive : bool
            Whether the property is intensive.
        folders : list of str, optional
            Folders whose properties will be included.
             If not specified, use all parsed folders.
        structures : dict of list of str, optional
            Dictionary with all the structures for at least the
            requested folders. If not specified, use all parsed structures
            with the desired property.

        Returns
        -------
        X : 2-D np.ndarray
            Array with the structures with the desired property. One row per
            structure in the same order as the folders and structures given.
        Y : 2-D np.ndarray
            Array with the desired property. One row per structure in the same
            order as the folders and structures given.

        Notes
        -----
        This functions treats each file as the property, i.e., if there are several values
        per file all of them are associated to the same structure. If each value corresponds
        to a different parameter (e.g., temperature) then you should use get_parametric_property.

        See Also
        --------
        get_parametric_property

        """
        if folders is None:
            folders = self.folders
        if structures is None:
            structures = self.structures
        filtered_structures = self.structures_with_property(property, folders, structures)
        Y = []
        for folder in folders:
            if formation:
                Y_pures = []
                for e, ef in self.elements[folder]:
                    n = self.concentrations[folder][ef][0]
                    data = np.loadtxt(os.path.join(self.base_dir, folder, ef, property))
                    if not intensive:
                        data /= n
                    Y_pures.append(data)
            for structure in filtered_structures[folder]:
                data = np.loadtxt(os.path.join(self.base_dir, folder, structure, property))
                if not intensive:
                    n = self.concentrations[folder][structure][0]
                    data /= n
                if formation:
                    x = self.concentrations[folder][structure][2]
                    Y_ref = x[0] * Y_pures[0] + x[1] * Y_pures[1]
                    data -= Y_ref
                Y.append(data)

        Y = np.array(Y)
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]

        X = self.as_list(folders, filtered_structures)
        X = np.array(X, dtype=str)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        return X, Y

    def get_parametric_property(self, property, parameter, formation=False,
                                intensive=True, folders=None, structures=None):
        """Get the given property and the structures which have it.

        Parameters
        ----------
        property : str
            Name of the file containing the property to load.
        parameter : str
            Name of the parameter (e.g. `temperature`).
        formation : bool
            Whether the formation value of the property should be returned.
        intensive : bool
            Whether the property is intensive.
        folders : list of str, optional
            Folders whose properties will be included.
             If not specified, use all parsed folders.
        structures : dict of list of str, optional
            Dictionary with all the structures for at least the
            requested folders. If not specified, use all parsed structures
            with the desired property.

        Returns
        -------
        X : 2-D np.ndarray
            Array with the structures with the desired property. One row per
            structure in the same order as the folders and structures given.
        Y : 2-D np.ndarray
            Array with the desired property. One row per structure in the same
            order as the folders and structures given.

        Notes
        -----
        This functions treats each file as a parameter dependent property, i.e.,
        if there are several values per file each of them is associated to a different
        input. If each value corresponds to the same parameter (e.g., elastic constants)
        then you should use get_property.

        """
        if folders is None:
            folders = self.folders
        if structures is None:
            structures = self.structures
        filtered_structures = self.structures_with_property(property, folders, structures)
        Y = []
        X = []
        for folder in folders:
            t = self.load_parameter(parameter, folder)
            if formation:
                Y_pures = []
                for e, ef in self.elements[folder]:
                    n = self.concentrations[folder][ef][0]
                    data = np.loadtxt(os.path.join(self.base_dir, folder, ef, property))
                    if not intensive:
                        data /= n
                    Y_pures.append(data)
            for structure in filtered_structures[folder]:
                data = np.loadtxt(os.path.join(self.base_dir, folder, structure, property))
                if not intensive:
                    n = self.concentrations[folder][structure][0]
                    data /= n
                if formation:
                    x = self.concentrations[folder][structure][2]
                    Y_ref = x[0] * Y_pures[0] + x[1] * Y_pures[1]
                    data -= Y_ref
                for i, t_i in enumerate(t):
                    X.append((os.path.join(self.base_dir, folder, structure), t_i))
                    Y.append(data[i])

        Y = np.array(Y)
        if len(Y.shape) == 1:
            Y = Y[:, np.newaxis]

        X = np.array(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        return X, Y

    def load_parameter(self, parameter, folder):
        """Load the parameter description file and returns the generated values.

        Parameters
        ----------
        parameter : str
            Name of the parameter to load.
        folder : str
            Folder to look for the parameter file.

        Returns
        -------
        1-D np.ndarray
            Values of the parameter as described by the parameter file.

        """
        if parameter.lower() in ['t', 'temperature']:
            t_param = np.loadtxt(os.path.join(self.base_dir, folder, 'Trange.in'))
            t_max = t_param[0]
            n_t = t_param[1]
            t = np.linspace(0, t_max, n_t)
        else:
            print('Error: unknown parameter file for {}'.format(parameter))
        return t

    def split_in_subsets(self, fractions, folders=None, structures=None, replace=False, force_pures_in_first=False):
        """Split the structures for the selected folders in groups with sizes
         calculated from fractions.

        Parameters
        ----------
        fractions : list of float > 0, sum(fractions) <= 1
            List with the fraction of data to be included in each subset.
        folders : list of str, optional
            Folders whose properties will be included.
             If not specified, use all parsed folders.
        structures : dict of list of str, optional
            Dictionary with all the structures for at least the
            requested folders. If not specified, use all parsed structures
            with the desired property.
        replace : bool
            Whether the split is done with or without replacement.
        force_pures_in_first : bool
            Whether to force the inclusion of the pure elements in the
            first subset.

        Returns
        -------
        list of dict of list of str
            List with the subset structures.
        """
        if folders is None:
            folders = self.folders
        if structures is None:
            structures = self.structures
        assert (0. < np.sum(fractions) <= 1.)

        n_sets = len(fractions)
        list_of_structures = []
        for i in range(n_sets):
            list_of_structures.append({})

        n_structures = {}
        for folder in folders:
            n_structures[folder] = [int(f * len(structures[folder])) for f in fractions]
            if force_pures_in_first:
                pure_element_index = []
                for e, f in self.elements[folder]:
                    where_in_structures = structures[folder].index(f)
                    pure_element_index.append(where_in_structures)
            if not replace:
                indices = np.random.permutation(np.arange(len(structures[folder])))
                if force_pures_in_first:
                    indices = list(indices)
                    for pi in pure_element_index:
                        indices.remove(pi)
                    indices = pure_element_index + indices
                start = 0
                print(n_structures[folder])
                for i, f in enumerate(n_structures[folder]):
                    idx = np.sort(indices[start: start + f])
                    list_of_structures[i][folder] = list(np.array(structures[folder])[idx])
                    start = start + f
            else:
                for i, f in enumerate(n_structures[folder]):
                    idx = np.sort(np.random.permutation(np.arange(len(structures[folder])))[: f])
                    list_of_structures[i][folder] = list(np.array(structures[folder])[idx])
                    if i == 0 and force_pures_in_first:
                        for pi in pure_element_index:
                            if structures[folder][pi] not in list_of_structures[i][folder]:
                                list_of_structures[i][folder] = structures[folder][pi] + list_of_structures[i][folder]

        return list_of_structures
