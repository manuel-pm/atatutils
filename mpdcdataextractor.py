"""Analyze MC data.

Author:
    Ilias Bilionis

Date:
    1/24/2013

Modified:
    Manuel Aldegunde

Date:
    05/2016

"""
from __future__ import print_function

import glob
import os
import re
import shutil
import warnings

import numpy as np

from atatutils.clusterfileparser import ClustersFileParser


def split_path(path):
    """ Splits a path into its components """
    dirname = path
    path_split = []
    while True:
        dirname, leaf = os.path.split(dirname.rstrip('/\\'))
        if leaf:
            # Adds one element, at the beginning of the list
            path_split = [leaf] + path_split
        else:
            # Uncomment the following line to have also the drive, e.g. "Z:\"
            # path_split = [dirname] + path_split
            break
    return path_split


def extract_from_dirname(dirname, varname):
    """ Extract the value corresponding to varname from dirname in a string
    of the form "xxxvarname1=..._xxxvarname2=..._...varnameN=..." """
    i0 = dirname.find(varname)
    i1 = i0 + dirname[i0:].find('=')
    i2 = dirname[i1:].find('_')
    if i2 != -1:
        return float(dirname[i1 + 1:i1 + i2])
    else:
        return float(dirname[i1 + 1:])


class MpdcDataExtractor:
    """ Extracts data (energy, temperature, concentration) from the results
    of the MPDC code """
    def __init__(self, prefix, output=None, verbosity=0):
        """ Initializes class variables and gets parameters from the
        simulation output file """
        warnings.filterwarnings('error')
        # self.nproc = nproc
        self.prefix = prefix
        self.parent_folder = split_path(self.prefix)[0]
        self.folder_prefix = split_path(self.prefix)[-1]
        self.verbosity = verbosity
        self.data = None
        self.restart_folders = ['empty']
        clusters = ClustersFileParser()
        clusters.parse()
        self.nclusters = clusters.size()
        supercel_size_line = 'Simple supercell created'
        supercel_size_line2 = 'MC simulation cell size'
        ofilename = 'output.log'
        if output is not None:
            ofilename = output
        # file = open('jobid_' + str(nproc), "r")
        # for line in file:
        #     continue
        # jobid = line.rstrip()
        # file = open('slurm-' + jobid + '.out', "r")
        ofile = open(ofilename, 'r')
        for line in ofile:
            if re.search(supercel_size_line, line):
                break
        ofile.close()
        size = line.split()[-3:]
        try:
            self.size = [int(s) for s in size]
        except:
            ofile = open(ofilename, 'r')
            for line in ofile:
                if re.search(supercel_size_line2, line):
                    break
            ofile.close()
            size = line.split()[-3:]
            self.size = [int(s) for s in size]

        self.natoms_p_supercell = self.size[0] * self.size[1] * self.size[2]
        self.nfolders = 0
        for d in glob.glob(prefix + '_beta=*_mu=*'):
            if (not os.path.isdir(d) or
                    not os.path.isfile(os.path.join(d, 'phi.dat'))):
                continue
            self.nfolders += 1

    def archive(self, filename=None):
        """ Archives the simulation directory """
        ofilename = split_path(self.parent_folder)[-1]
        if filename is not None:
            ofilename = filename
        shutil.make_archive(ofilename, 'gztar', base_dir=self.parent_folder)

    def dump_correlation(self, which=2):
        """ Writes the \emph{which}-correlation from the design matrix phi """
        if isinstance(which, int):
            which = [which]
        rootdir = os.path.abspath('.')
        for d in glob.glob(self.prefix + '_beta=*_mu=*'):
            os.chdir(d)
            if not os.path.isfile('phi.dat'):
                continue
            correlations = np.loadtxt('phi.dat')
            for i in which:
                np.savetxt('{}_correlation.dat'.format(i),
                           correlations[i::self.nclusters])
            os.chdir(rootdir)

    def get_data(self):
        """ Returns the extracted data as a numpy array """
        if self.data is not None:
            return self.data
        self.extract_data()
        return self.data

    def extract_data(self):
        """ Extracts the data from the simulation """
        self.data = np.empty((self.nfolders, 10))
        idx = 0
        max_beta = -1.e16
        # print(self.prefix)
        for d in glob.glob(self.prefix + '_beta=*_mu=*'):
            if not os.path.isdir(d):
                continue
            if self.verbosity == 1:
                print('Entering folder ' + d)
            beta = extract_from_dirname(d, 'beta')
            mu = extract_from_dirname(d, 'mu')
            if beta > max_beta and \
               os.path.isfile(os.path.join(d, 'x.dat')) and \
               os.path.isfile(os.path.join(d, 'energy.dat')) and \
               os.path.isfile(os.path.join(d, 'weights.dat')):
                all_particles = True
                for p in range(64):
                    if not os.path.isfile(
                            os.path.join(d, 'particle_{}.dat.gz'.format(str(
                                p)))):
                        all_particles = False
                        break
                if all_particles:
                    max_beta = beta
                    self.restart_folders.append(d)
            try:
                if not os.path.isfile(os.path.join(d, 'phi.dat')):
                    continue
                if self.verbosity == 1:
                    print('Extracting in foder ' + d)
                energy = np.loadtxt(os.path.join(d, 'energy.dat'))
                log_w = np.loadtxt(os.path.join(d, 'weights.dat'))
                pair = np.loadtxt(os.path.join(d,
                                               'phi.dat'))[2::self.nclusters]
            except:
                if self.verbosity == 1:
                    print('Skipping foder ' + d)
                continue

            w = np.exp(log_w)
            mean_energy = np.dot(energy, w)
            mean_energy2 = np.dot(energy ** 2, w)
            var_energy = mean_energy2 - mean_energy ** 2

            mean_pair = np.dot(pair, w)
            mean_pair2 = np.dot(pair*pair, w)
            var_pair = mean_pair2 - mean_pair ** 2

            x = np.loadtxt(os.path.join(d, 'x.dat'))
            mean_x = np.dot(x, w)

            mean_x2 = np.dot(x ** 2, w)
            var_x = mean_x2 - mean_x ** 2

            # Quantities of interest
            k_b = 8.6173325e-5  # 8.6173324(78)x10-5 eV/K
            temp = 1. / (k_b * beta)
            c_p = beta * beta * k_b * var_energy / self.natoms_p_supercell
            # eV K^-1 atom^-1
            c_p *= 1.60218e-16 * 6.02214129e23  # mJ mol^-1 K^-1

            if self.verbosity == 2:
                print('{0:1.12f} {1:2.12f} {2:2.12} {3:2.12} {4:2.12} '
                      '{5:2.12} {6:2.12} {7:2.12} {8:1.12f} {9:1.12f}'.format(
                          beta, mu, mean_energy, var_energy, mean_x, var_x,
                          mean_pair, var_pair, temp, c_p))

            self.data[idx, 0] = beta
            self.data[idx, 1] = mu
            self.data[idx, 2] = mean_energy / self.natoms_p_supercell
            self.data[idx, 3] = var_energy / self.natoms_p_supercell
            self.data[idx, 4] = (mean_x / self.natoms_p_supercell + 1.) / 2.
            self.data[idx, 5] = var_x / (self.natoms_p_supercell * 2.0)**2
            self.data[idx, 6] = mean_pair
            self.data[idx, 7] = var_pair
            self.data[idx, 8] = temp
            self.data[idx, 9] = c_p
            idx += 1

        # Sort in order of increasing temperature
        self.restart_folders.sort()
        self.data = self.data[self.data[:, 8].argsort()]
        # np.savetxt(prefix + '.dat', self.data)
        temp_max = self.data[self.data[:, 9].argmax(), 8]
        print("Maximum specific heat at T={} K".format(temp_max))
        print("Minimum temperature T={} K (beta={})".format(self.data[0, 8],
                                                            self.data[0, 0]))
        print("Last restart point: {}".format(self.restart_folders[-1]))

    def get_restart_folder(self, which=-1):
        """ Returns the latest folder than can be used for restarting a MC
        simulation """
        return self.restart_folders[which]

    def save_data(self, filename=None):
        """ Saves extracted data to a file """
        if self.data is None:
            print('WARNING: extracting data...')
            self.extract_data()
        ofilename = self.folder_prefix + self.parent_folder[6:] + '.dat'
        if filename is not None:
            ofilename = filename
        np.savetxt(ofilename, self.data)

    def load_data(self, filename=None):
        """ Loads the simulation data from a file """
        ifilename = self.folder_prefix + self.parent_folder[6:] + '.dat'
        # ifilename = self.folder_prefix + '.dat'
        if filename is not None:
            ifilename = filename
        self.data = np.loadtxt(ifilename)
