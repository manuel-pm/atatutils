from __future__ import print_function

import math
import os
import sys

import numpy as np

import ase
import ase.build
import ase.data
from ase import Atoms


class ATATCluster(object):
    """
    Parses and prints info about "clusters.out"
    files from ATAT.

    Notes
    -----
    A "cluster orbit" (also called a cluster family)
    is simply a collection of symmetrically equivalent
    clusters. This is really what the clusters.out file
    lists: a prototype cluster from each cluster orbit.

    The file contains a series of blocks representing
    one cluster each. The format of a cluster block is
    the following:

    multiplicity
    diameter
    n_sites
    x_1 y_1 z_1
    x_2 y_2 z_2
    ...
    x_{n_sites} z_{n_sites} z_{n_sites}


    Author/Date: Jesper Kristensen; Summer 2015
    From: http://www.jespertoftkristensen.com/JTK/Software.html
    """
    @staticmethod
    def _comment(msg):
        """
        Print a comment to the user.
        """
        print('IIII {}'.format(msg))

    @staticmethod
    def _errorquit(msg):
        """
        Prints msg and exits.
        """
        print('EEEE {}'.format(msg))
        sys.exit(1)

    def __init__(self, clusters_out='clusters.out'):
        """
        Parses a clusters.out file in path
        "clusters_out".
        """
        if not os.path.isfile(clusters_out):
            self._errorquit('please provide a valid clusters.out file')
        self._clusters_out = clusters_out
        self._cluster_info = {}

        self._all_lines_in_clusters_out = None
        self._all_cluster_blocks = []

    def __len__(self):
        return len(self._all_cluster_blocks)

    def __getitem__(self, index):
        return self._all_cluster_blocks[index]

    @staticmethod
    def _parse_single_site_coords(line_with_site_details=None):
        """
        Line which contains details of site in cluster.
        Only parses (x,y,z) coordinates of site.
        """
        line_details = line_with_site_details.split()
        return map(float, line_details[:3])

    @staticmethod
    def _parse_single_site_all(line_with_site_details=None):
        """
        Line which contains details of site in cluster.
        Parses all line (not just coordinates of cluster).
        """
        return line_with_site_details.split()

    def _parse_a_single_cluster_block(self, starting_at=None):
        """
        Parses a single cluster block starting at index
        "starting_at" in the clusters.out file.

        Returns a dictionary containing the cluster block
        information (multiplicity, size of the cluster, etc.)
        """
        all_lines = self._all_lines_in_clusters_out

        block_start = starting_at + 1

        _cluster_block = {}

        clus_mult = int(all_lines[block_start])
        clus_size = float(all_lines[block_start + 1])
        num_sites = int(all_lines[block_start + 2])

        all_coords = []
        for j in range(num_sites):
            # coordinates of this site in the cluster
            entire_line = self._parse_single_site_coords(
                                                all_lines[block_start + 3 + j])
            all_coords.append(entire_line)

        _cluster_block['multiplicity'] = clus_mult
        _cluster_block['diameter'] = clus_size
        _cluster_block['n_points'] = num_sites
        _cluster_block['coordinates'] = np.array(all_coords)
        # Note that except for the coordinates, the information is common for
        #  all clusters in the orbit. We just store the coordinates of the
        #  prototype cluster present in clusters.out
        # To get the coordinates of any other clusters in the orbit you need to
        #  apply the space group symmetry operations (not part of clusters.out)
        #  to the coordinates in "all_coords".

        if str(num_sites) in self._cluster_info:
            self._cluster_info[str(num_sites)] += 1
        else:
            self._cluster_info[str(num_sites)] = 1

        return _cluster_block

    def parse(self):
        """
        Parse the clusters.out file.
        This is a brute-force approach.
        """
        with open(self._clusters_out, 'r') as fd:
            self._all_lines_in_clusters_out = fd.readlines()
            if self._all_lines_in_clusters_out[0].rstrip('\n'):
                # let us create an empty line to treat the first
                # block the same as the rest
                newline = ['\n']
                newline.extend(self._all_lines_in_clusters_out)
                self._all_lines_in_clusters_out = newline

            # clean the end of the file for newlines:
            k = -1
            while not self._all_lines_in_clusters_out[k].rstrip('\n'):
                k -= 1
            if k < -1:
                self._all_lines_in_clusters_out = \
                    self._all_lines_in_clusters_out[:k + 1]

            self._all_cluster_blocks = []

            for i, line_ in enumerate(self._all_lines_in_clusters_out):
                # go through all lines in clusters.out file
                line_no_newline = line_.rstrip('\n')

                # is this part of the file a new "cluster block"?
                if not line_no_newline or i == 0:
                    # yes, so parse the block and put in a dictionary:
                    cluster_block = \
                        self._parse_a_single_cluster_block(starting_at=i)
                    self._all_cluster_blocks.append(cluster_block)

            if len(self._all_cluster_blocks) == 0:
                self._errorquit("No clusters found in the file?")

    def maximum_cluster_size(self, n_points):
        """
        Returns the maximum size of the included clusters with n_points points.
        """
        max_size = -1.
        for b_ in self._all_cluster_blocks:
            if b_['n_points'] == n_points and b_['diameter'] > max_size:
                    max_size = b_['diameter']
        return max_size

    def maximum_number_of_sites(self):
        """
        Returns the maximum number of sites in any included cluster.
        """
        return int(sorted(self._cluster_info.keys())[-1])

    def multiplicities_in_cluster_orbits(self):
        """
        Returns a list containing as the ith element the multiplicities
        of the clusters in the orbit (number of cluster in the cluster orbit).
        """
        return [block_['multiplicity'] for block_ in self._all_cluster_blocks]

    def number_of_sites_in_cluster_orbits(self):
        """
        Returns a list containing as the ith element the number of sites
        in cluster orbit i.
        """
        return [block_['n_points'] for block_ in self._all_cluster_blocks]

    def sizes_of_cluster_orbits(self):
        """
        Returns list containing as the ith element the size of clusters
        in cluster orbit i.
        """
        return [block_['diameter'] for block_ in self._all_cluster_blocks]

    def site_coordinates(self):
        """
        Returns the xyz-coordinates of the sites in the prototype cluster
        from each cluster orbit.
        """
        return [block_['coordinates'] for block_ in self._all_cluster_blocks]

    def size(self):
        """
        Returns the number of clusters
        """
        return len(self._all_cluster_blocks)

    def cluster_info(self):
        """
        Print some info about the clusters file.
        """
        from monty.pprint import pprint_table

        tab_ = []
        print('There are {} clusters:'.format(self.size()))
        for points, number in sorted(self._cluster_info.items()):
            singular = int(number) == 1
            col1 = 'There {}:'.format('is' if singular else 'are')
            col2 = '{}'.format(number)
            col3 = '{}-point cluster{}'.format(points,
                                               ' ' if singular else 's')
            tab_.append([col1, col2, col3])

        pprint_table(tab_, out=sys.stdout)

    def pickle_clusters(self, filename='cfp.pkl'):
        """
        Saves the cluster information to the Python pickle format.
        """
        import six

        six.moves.cPickle.dump(self._cluster_info, open(filename, 'w'))


class ATATLattice(object):
    r"""Class containing data for an alloy lattice.

    Parameters
    ----------
    lattice_file : str
        Name of the file containing the lattice description
    base_dir : str
        Base directory to look for the lattice file

    Notes
    -----
    At the moment it only accepts ATAT formatted files.
    For the alloy lattice, an example from ATAT manual is
    (https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node21.html):

    3.1 3.1 5.062 90 90 120	        (Coordinate system: :math:`a\; b\; c\; \alpha\; \beta\; \gamma` notation)
    1 0 0	                        (Primitive unit cell: one vector per line
    0 1 0	                         expressed in multiples of the above coordinate
    0 0 1	                         system vectors)
    0 0 0 Al,Ti	                    (Atoms in the lattice)
    0.6666666 0.3333333 0.5 Al,Ti

    The class can also parse the random lattice input files used
    to find special quasi-random structures. The difference is the
    specification of the partial occupation of the sites. An example
    from the manual is
    (https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node46.html):

     0.707 0.707 6.928 90 90 120
     0.3333  0.6667 0.3333
    -0.6667 -0.3333 0.3333
     0.3333 -0.3333 0.3333
     0       0      0       Li=0.75,Vac=0.25
     0.3333  0.6667 0.0833  O
     0.6667  0.3333 0.1667  Co=0.25,Ni=0.25,Al=0.5
     0       0      0.25    O

     The partial occupations are used by this class to generate
     random realizations of the alloy.

    """
    def __init__(self, lattice_file='lat.in', base_dir='.'):
        self._atomic_numbers = None
        self._chemical_symbols = None
        self._coordinate_system = np.empty((3, 3), dtype=float)
        self._positions = None
        self._occupation_probabilities = None
        self._unit_cell = np.empty((3, 3), dtype=float)
        self._number_of_sites = None
        self._vacancy_id = -1

        self.lattice_file = os.path.join(base_dir, lattice_file)
        self.parse_lattice_file(self.lattice_file)

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @atomic_numbers.setter
    def atomic_numbers(self, value):
        assert len(value) == self.n_sites
        self._atomic_numbers = value

    @property
    def chemical_symbols(self):
        return self._chemical_symbols

    @chemical_symbols.setter
    def chemical_symbols(self, value):
        assert len(value) == self.n_sites
        self._chemical_symbols = value

    @property
    def coordinate_system(self):
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value):
        if value.shape == (3, 3):
            np.copyto(self._coordinate_system, value)
        elif value.ravel().shape == (6,):
            a, b, c, alpha, beta, gamma = value.ravel()
            alpha, beta, gamma = [angle * np.pi / 180.
                                  for angle in [alpha, beta, gamma]]
            (ca, sa) = (math.cos(alpha), math.sin(alpha))
            (cb, sb) = (math.cos(beta), math.sin(beta))
            (cg, sg) = (math.cos(gamma), math.sin(gamma))
            # v_unit is a volume of unit cell with a = b = c = 1
            v_unit = math.sqrt(1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg)
            # from the reciprocal lattice
            ar = sa / (a * v_unit)
            cgr = (ca * cb - cg) / (sa * sb)
            sgr = math.sqrt(1.0 - cgr ** 2)
            self._coordinate_system[0, :] = np.array([1.0 / ar, -cgr / sgr / ar, cb * a])
            self._coordinate_system[1, :] = np.array([0.0, b * sa, b * ca])
            self._coordinate_system[2, :] = np.array([0.0, 0.0, c])
        else:
            print('ERROR: Unknown format for coordinate system. Valid formats are:'
                  '3x3 array. Example:'
                  'np.array([[  5.53127,   0,   0],'
                  '          [  0,   5.53127,   0],'
                  '          [  0,   0,   5.53127]])'
                  '1x6 (or 6x1 or 6) array. Example:'
                  'np.array([5.53127 5.53127 5.53127 90. 90. 90.])')
            self._coordinate_system = None

    @property
    def n_sites(self):
        return self._number_of_sites

    @n_sites.setter
    def n_sites(self, value):
        self._number_of_sites = value

    @property
    def occupation_probabilities(self):
        return self._occupation_probabilities

    @occupation_probabilities.setter
    def occupation_probabilities(self, value):
        assert len(value) == self.n_sites
        self._occupation_probabilities = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        assert (len(value.shape) == 2) and (value.shape[1] == 3)
        self._positions = value

    @property
    def unit_cell(self):
        return self._unit_cell

    @unit_cell.setter
    def unit_cell(self, value):
        assert value.shape == (3, 3)
        np.copyto(self._unit_cell, value)

    def get_scaled_positions(self):
        """Get the scaled positions, i.e., the positions in the system specified
         by self.unit_cell.

        Returns
        -------
        2-D np.ndarray
        Scaled positions of the lattice sites.

        """
        return np.dot(self.positions, np.linalg.inv(self.unit_cell))

    def get_transformed_positions(self):
        """Get the transformed positions, i.e., the positions in the system specified
         by self.coordinate_system.

        Returns
        -------
        2-D np.ndarray
            Transformed positions of the lattice sites.

        """
        return np.dot(self.positions, np.linalg.inv(self.coordinate_system))

    def get_transformed_unit_cell(self):
        """Get the transformed cell vectors, i.e., the cell vectors in the system specified
         by self.coordinate_system.

        Returns
        -------
        2-D np.ndarray
            Trnasformed cell vectors of the lattice.

        """
        return np.dot(self.unit_cell, np.linalg.inv(self.coordinate_system))

    def create_realizations(self, size=(1, 1, 1), n=None, pbc=(1, 1, 1), niggli=False):
        """Creates n realizations of the alloy lattice with given size.

        Parameters
        ----------
        size : triplet, optional
            Triplet with the size in the three directions of the unit cell vectors.
            Non-periodic directions are ignored.
        n : int > 0, optional
            Number of realizations to generate. If not specified, generate 1.
        pbc : triplet, optional
            Triplet determining the nature of the boundary conditions (periodic or
            non-periodic) in each lattice direction.
        niggli : bool, optional
            Whether to return the Niggli reduced cell of the generated realizations.
            Only works for unit cells periodic in the 3 dimensions.

        Returns
        -------
        realizations : ase.Atoms or list of ase.Atoms
            List with the realizations of the alloy lattice. If n is not
            specified, return the only realization.

        """
        realizations = []
        for repeat in range(max(n, 1)):
            positions = self.positions
            atomic_numbers = np.empty(self.n_sites)
            for i, a in enumerate(self.atomic_numbers):
                atomic_numbers[i] = np.random.choice(a)
            cell = np.dot(np.diag(size), self.unit_cell)
            for i in range(3):
                layer = np.copy(positions)
                for j in range(1, size[i] * int(bool(pbc[i]))):
                    positions = np.vstack((positions, layer + j * self.unit_cell[i]))
            atomic_numbers = []
            for i in range(positions.shape[0]/self.n_sites):
                for i_a, a in enumerate(self.atomic_numbers):
                    atomic_numbers.append(np.random.choice(a, p=self.occupation_probabilities[i_a]))
            atomic_numbers = np.array(atomic_numbers)
            # Create atoms object
            not_vacancies = np.where(atomic_numbers != self._vacancy_id)
            atoms = Atoms(symbols=atomic_numbers[not_vacancies],
                          positions=positions[not_vacancies],
                          cell=cell,
                          pbc=pbc)
            if all(pbc) and niggli:
                ase.build.niggli_reduce(atoms)
            realizations.append(atoms)
        if n is None:
            return realizations[0]
        return realizations

    def parse_lattice_file(self, filename):
        """Parse the given lattice file and updates the object.

        Parameters
        ----------
        filename : str
            Name of the file containing the alloy lattice description.

        """
        with open(filename, 'r') as ifile:
            # Read coordinate system
            coordinate_system = np.empty((3, 3), dtype=float)
            first_line = ifile.readline()
            items = [float(i) for i in first_line.split()]
            if len(items) == 3:
                coordinate_system[0, :] = np.array(items)
                coordinate_system[1, :] = np.array(ifile.readline().split())
                coordinate_system[2, :] = np.array(ifile.readline().split())
            else:
                coordinate_system = np.array(items, dtype=float)
            self.coordinate_system = coordinate_system

            # Read unit cell
            unit_cell = np.zeros((3, 3), dtype=float)
            for i in range(3):
                unit_cell[i, :] = np.array(ifile.readline().split())

            self.unit_cell = np.dot(unit_cell, self.coordinate_system)

            # Read atoms positions
            rest = ifile.readlines()
            positions = []
            chemical_symbols = []
            for line in rest:
                split = line.split()
                positions.append(np.dot(np.array(split[:3], dtype=float), self.coordinate_system))
                chemical_symbols.append(split[3].split(','))

            atomic_numbers = []
            occupations = []
            for i_cs, cs in enumerate(chemical_symbols):
                atomic_numbers.append([])
                occupations.append([])
                for i_e, e in enumerate(cs):
                    ep = e.split('=')
                    chemical_symbols[i_cs][i_e] = ep[0]
                    if ep[0] == 'Vac':
                        atomic_numbers[-1].append(self._vacancy_id)
                    else:
                        atomic_numbers[-1].append(ase.data.atomic_numbers[ep[0]])
                    if len(ep) == 2:
                        occupations[-1].append(float(ep[1]))
                    elif len(ep) == 1:
                        occupations[-1].append(1./len(cs))
                    else:
                        print('ERROR: lattice file misformed {}'.format(ep))

            self.n_sites = len(positions)
            self.positions = np.array(positions)
            self.chemical_symbols = chemical_symbols
            self.atomic_numbers = atomic_numbers
            self.occupation_probabilities = occupations

    def write_lattice_file(self, filename, base_dir='.', overwrite=False):
        """Writes the lattice to a file in ATAT format.

        Parameters
        ----------
        filename : str
            Name of the file to write the structure file to.
        base_dir : str
            Name of the folder to write the output file to.
        overwrite : bool
            Whether to overwrite an existing file of the same name.

        """
        if os.path.isfile(os.path.join(base_dir, filename)) and not overwrite:
            print('WARNING: File {0} already exists. Writing to {0}.new'.format(os.path.join(base_dir, filename)))
            filename += '.new'
        with open(os.path.join(base_dir, filename), 'wb') as ofile:
            ofile.write('1.0 1.0 1.0 90. 90. 90.\n')
            for i in range(3):
                ofile.write(' '.join("{:.12f}".format(cell) for cell in self.unit_cell[i]))
                ofile.write('\n')
            for i in range(self.n_sites):
                ofile.write(' '.join("{:.12f}".format(p) for p in self.positions[i])
                            + ' '
                            + ','.join(self.chemical_symbols[i]))
                ofile.write('\n')


class ATATStructure(object):
    r"""Class containing data for an alloy structure.

    Parameters
    ----------
    structure_file : str
        Name of the file containing the structure description.
    base_dir : str
        Base directory to look for the structure file.
    atoms : ase.Atoms
        Atoms object of the structure.

    Notes
    -----
    At the moment it only accepts ATAT str.out formatted files.
    Example from ATAT manual (https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node21.html):

    3.1 3.1 5.062 90 90 120	        (Coordinate system: :math:`a\; b\; c\; \alpha\; \beta\; \gamma` notation)
    1 0 0	                        (Primitive unit cell: one vector per line
    0 1 0	                         expressed in multiples of the above coordinate
    0 0 1	                         system vectors)
    0 0 0 Ti	                    (Atoms in the structure)
    0.6666666 0.3333333 0.5 Al

    """
    def __init__(self, structure_file='str.out', base_dir='.', atoms=None):
        self._atomic_numbers = None
        self._chemical_symbols = None
        self._coordinate_system = np.empty((3, 3), dtype=float)
        self._positions = None
        self._unit_cell = np.empty((3, 3), dtype=float)
        self._number_of_sites = None
        self._vacancy_id = -1

        if atoms is not None:
            self.structure_file = None
        else:
            self.structure_file = os.path.join(base_dir, structure_file)
            self.parse_structure_file(self.structure_file)

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @atomic_numbers.setter
    def atomic_numbers(self, value):
        assert len(value) == self.n_sites
        self._atomic_numbers = value

    @property
    def chemical_symbols(self):
        return self._chemical_symbols

    @chemical_symbols.setter
    def chemical_symbols(self, value):
        assert len(value) == self.n_sites
        self._chemical_symbols = value

    @property
    def coordinate_system(self):
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value):
        if value.shape == (3, 3):
            np.copyto(self._coordinate_system, value)
        elif value.ravel().shape == (6,):
            a, b, c, alpha, beta, gamma = value.ravel()
            alpha, beta, gamma = [angle * np.pi / 180.
                                  for angle in [alpha, beta, gamma]]
            (ca, sa) = (math.cos(alpha), math.sin(alpha))
            (cb, sb) = (math.cos(beta), math.sin(beta))
            (cg, sg) = (math.cos(gamma), math.sin(gamma))
            # v_unit is a volume of unit cell with a = b = c = 1
            v_unit = math.sqrt(1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg)
            # from the reciprocal lattice
            ar = sa / (a * v_unit)
            cgr = (ca * cb - cg) / (sa * sb)
            sgr = math.sqrt(1.0 - cgr ** 2)
            self._coordinate_system[0, :] = np.array([1.0 / ar, -cgr / sgr / ar, cb * a])
            self._coordinate_system[1, :] = np.array([0.0, b * sa, b * ca])
            self._coordinate_system[2, :] = np.array([0.0, 0.0, c])
        else:
            print('ERROR: Unknown format for coordinate system. Valid formats are:'
                  '3x3 array. Example:'
                  'np.array([[  5.53127,   0,   0],'
                  '          [  0,   5.53127,   0],'
                  '          [  0,   0,   5.53127]])'
                  '1x6 (or 6x1 or 6) array. Example:'
                  'np.array([5.53127 5.53127 5.53127 90. 90. 90.])')
            self._coordinate_system = None

    @property
    def n_sites(self):
        return self._number_of_sites

    @n_sites.setter
    def n_sites(self, value):
        self._number_of_sites = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        assert (len(value.shape) == 2) and (value.shape[1] == 3)
        self._positions = value

    @property
    def unit_cell(self):
        return self._unit_cell

    @unit_cell.setter
    def unit_cell(self, value):
        assert value.shape == (3, 3)
        np.copyto(self._unit_cell, value)

    def get_scaled_positions(self):
        """Get the scaled positions, i.e., the positions in the system specified
         by self.coordinate_system.

        Returns
        -------
        2-D np.ndarray
        Scaled positions of the atoms.

        """
        return np.dot(self.positions, np.linalg.inv(self.coordinate_system))

    def get_scaled_unit_cell(self):
        """Get the scaled cell vectors, i.e., the cell vectors in the system specified
         by self.coordinate_system.

        Returns
        -------
        2-D np.ndarray
            Scaled cell vectors of the lattice.

        """
        return np.dot(self.unit_cell, np.linalg.inv(self.coordinate_system))

    def parse_structure_file(self, filename):
        """Parse the given structure file and updates the object.

        Parameters
        ----------
        filename : str
            Name of the file containing the alloy lattice description.

        """
        with open(filename, 'r') as ifile:
            # Read coordinate system
            coordinate_system = np.empty((3, 3), dtype=float)
            first_line = ifile.readline()
            items = [float(i) for i in first_line.split()]
            if len(items) == 3:
                coordinate_system[0, :] = np.array(items)
                coordinate_system[1, :] = np.array(ifile.readline().split())
                coordinate_system[2, :] = np.array(ifile.readline().split())
            else:
                coordinate_system = np.array(items, dtype=float)
            self.coordinate_system = coordinate_system

            # Read unit cell
            unit_cell = np.zeros((3, 3), dtype=float)
            for i in range(3):
                unit_cell[i, :] = np.array(ifile.readline().split())

            self.unit_cell = np.dot(unit_cell, self.coordinate_system)

            # Read atoms positions
            rest = ifile.readlines()
            positions = []
            chemical_symbols = []
            atomic_numbers = []
            for line in rest:
                split = line.split()
                positions.append(np.dot(np.array(split[:3], dtype=float), self.coordinate_system))
                chemical_symbols.append(split[3].split('=')[0])
                if chemical_symbols[-1] == 'Vac':
                    atomic_numbers.append(self._vacancy_id)
                else:
                    atomic_numbers.append(ase.data.atomic_numbers[chemical_symbols[-1]])
            self.n_sites = len(positions)
            self.positions = np.array(positions)
            self.chemical_symbols = chemical_symbols
            self.atomic_numbers = atomic_numbers

    def write_structure_file(self, filename, base_dir='.', overwrite=False):
        """Writes the structure to a file in ATAT format.

        Parameters
        ----------
        filename : str
            Name of the file to write the structure file to.
        base_dir : str
            Name of the folder to write the output file to.
        overwrite : bool
            Whether to overwrite an existing file of the same name.

        """
        if os.path.isfile(os.path.join(base_dir, filename)) and not overwrite:
            print('WARNING: File {0} already exists. Writing to {0}.new'.format(os.path.join(base_dir, filename)))
            filename += '.new'
        with open(os.path.join(base_dir, filename), 'wb') as ofile:
            ofile.write('1.0 1.0 1.0 90. 90. 90.\n')
            for i in range(3):
                ofile.write(' '.join("{:.12f}".format(cell) for cell in self.unit_cell[i]))
                ofile.write('\n')
            for i, atom in enumerate(self.chemical_symbols):
                ofile.write(' '.join("{:.12f}".format(p) for p in self.positions[i]) + ' ' + atom)
                ofile.write('\n')

    def as_atoms(self, niggli=False):
        """Returns the structure as an ase.Atoms object.

        Parameters
        ----------
        niggli : bool, optional
            Whether to return the Niggli reduced cell of the generated realizations.
            Only works for unit cells periodic in the 3 dimensions.

        Returns
        -------
        ase.Atoms
            Structure represented by the structure file as an ase.Atoms object.

        """
        atomic_numbers = np.array(self.atomic_numbers)
        not_vacancies = np.where(atomic_numbers != self._vacancy_id)
        atoms = Atoms(symbols=atomic_numbers[not_vacancies],
                      positions=self.positions[not_vacancies],
                      cell=self.unit_cell,
                      pbc=True)

        if niggli:
            ase.build.niggli_reduce(atoms)

        return atoms

    def from_atoms(self, atoms):
        """Update the structure from an atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to update the structure.

        """
        self.atomic_numbers = atoms.get_atomic_numbers()
        self.chemical_symbols = atoms.get_chemical_symbols()
        self.coordinate_system = np.eye(3, dtype=float)
        self.positions = atoms.get_positions()
        self.unit_cell = atoms.get_cell()
        self.n_sites = atoms.get_positions().shape[0]
