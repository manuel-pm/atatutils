from __future__ import print_function

import os
import sys
import numpy as np


class ClustersFileParser(object):
    """
    Parses and prints info about "clusters.out"
    files from ATAT.

    Notes
    -----
    A "cluster orbit" (also called a cluster family)
    is simply a collection of symmetrically equivalent
    clusters. This is really what the clusters.out file
    lists: a prototype cluster from each cluster orbit.

    Author/Date: Jesper Kristensen; Summer 2015
    From: http://www.jespertoftkristensen.com/JTK/Software.html
    """

    def _comment(self, msg):
        """
        Print a comment to the user.
        """
        print('IIII {}'.format(msg))

    def _errorquit(self, msg):
        """
        Prints msg and exits.
        """
        print('EEEE {}'.format(msg))
        sys.exit(1)

    def __init__(self, clusters_out=None):
        """
        Parses a clusters.out file in path
        "clusters_out".
        """
        clusters_out = 'clusters.out'
        if not os.path.isfile(clusters_out):
            self._errorquit('please provide a valid clusters.out file')
        self._clusters_out = clusters_out
        self._cluster_info = {}

    def __len__(self):
        return len(self._all_cluster_blocks)

    def __getitem__(self, index):
        return self._all_cluster_blocks[index]

    def _parse_single_site_coords(self, line_with_site_details=None):
        """
        Line which contains details of site in cluster.
        Only parses (x,y,z) coordinates of site.
        """
        line_details = line_with_site_details.split()
        return map(float, line_details[:3])

    def _parse_single_site_all(self, line_with_site_details=None):
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
        import cPickle as pickle
        pickle.dump(self._cluster_info, open(filename, 'w'))

