"""

"""
from __future__ import print_function

import collections
import hashlib
import os
import subprocess
import six
import sys
import time

import numpy as np

from atatutils.clusterfileparser import ClustersFileParser
from utils.bcolors import print_warning, print_success
from BayesianLinearRegression.rvm import RelevanceVectorMachine as RVM
import BayesianLinearRegression.myBayesianLinearRegression as BLR
from BayesianLinearRegression.RJMCMC import BL_RJMCMC


def get_train_test_sets(base_dir, test_samples=None, test_sample_size=None, prng=np.random):
    sigma_all= [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and is_int(d)]
    sigma_all = [f for f in sigma_all if os.path.isfile(os.path.join(base_dir, f, 'energy'))]
    sigma_all.sort(key=int)

    sigma_test = []
    try:
        sigma_test = [s for s in test_samples if s in sigma_all]
    except:
        if test_sample_size < 1.0:
            n_test = int(test_sample_size*len(sigma_all))
        else:
            n_test = int(test_sample_size)
        sigma_test = list(prng.choice(sigma_all[2:], n_test, replace=False))
    sigma_test.sort(key=int)

    if not len(sigma_test):
        print_warning('No test structures selected')
    else:
        print_success('{}/{} structures selected for testing:\n{}'.format(len(sigma_test), len(sigma_all), sigma_test))
    sigma_train = list(set(sigma_all) - set(sigma_test))
    sigma_train.sort(key=int)

    return sigma_train, sigma_test


def is_int(string):
    v = str(string).strip()
    return (v == '0' or
            (v if v.find('..') > -1 else
             v.lstrip('-+').rstrip('0').rstrip('.')).isdigit())

# returns list of tuples with: [x_elem1, x_elem2, ...], [n_elem1, ...], n_atoms
def configuration_density(configurations, elements):
    if isinstance(configurations, six.string_types):
        configurations = [configurations]
    if isinstance(configurations, collections.Iterable):
        xs = []
        ns = []
        n_atoms = []
        for i, sigma in enumerate(configurations):
            str_out = open(os.path.join(sigma, 'str.out'))
            xs.append(np.zeros(len(elements)))
            ns.append(np.zeros(len(elements)))
            n_atoms.append(0.0)
            for line in str_out.readlines():
                v = line.split()
                if len(v) == 4:
                    n_atoms[-1] += 1.
                    for j, elem in enumerate(elements):
                        if v[-1] == elem:
                            ns[-1][j] += 1.
                            break
            str_out.close()
            xs[-1] = ns[-1]/n_atoms[-1]
        return xs, ns, n_atoms
    else:
        raise TypeError('configurations must be string or iterable of strings')


class ClusterExpansion:
    def __init__(self, diameters=None, base_dir='.', base_lattice='lat.in',
                 data_filename='energy', two_d=False, use_only=None,
                 ce_type='standard', m_vbce=None, intensive=False):
        self.design_vectors = {}
        self.concentration = {}
        self.two_d = two_d
        self.ce_type = ce_type
        self.base_dir = base_dir
        self.folders = next(os.walk(base_dir))[1]
        self.folders = [f for f in self.folders if is_int(f)]
        self.folders.sort(key=int)
        self.exp_folders = [f for f in self.folders if
                            os.path.isfile(os.path.join(base_dir,
                                                        f, data_filename))]
        self.exp_folders.sort(key=int)
        if use_only is not None:
            if isinstance(use_only[0], (int, long)):
                self.exp_folders = list(np.asarray(self.exp_folders)[use_only])
            elif isinstance(use_only[0], six.string_types):
                self.exp_folders = [os.path.join(base_dir,
                                                 f) for f in use_only]
        self.diameters = diameters
        self.input_lattice = os.path.join(base_dir, base_lattice)
        ilattice = open(self.input_lattice)
        x_t = 0.0
        for line in ilattice.readlines():
            v = line.split()
            if len(v) == 4:
                x_t += 1.
        ilattice.close()
        self.n_at_basis = x_t

        if not os.path.isdir(os.path.join(base_dir, '0')):
            self.make_new_structures(N=1)

        self.update_clusters()
        self.maximum_cluster_size_vbce = [0]*len(self.clusters)
        if m_vbce is not None:
            self.m_vbce = m_vbce
            for i, j, k in m_vbce:
                counter = 0
                for n, cluster in enumerate(self.clusters):
                    if cluster['n_points'] == i and counter == j:
                        self.maximum_cluster_size_vbce[n] = k
                        break
                    if cluster['n_points'] == i:
                        counter += 1

        self.n_data = len(self.exp_folders)
        self.n_basis = len(self.clusters) + np.sum(self.maximum_cluster_size_vbce)
        self.data_errors = np.empty(self.n_data)
        self.design_matrix = np.empty((self.n_data, self.n_basis))

        if not os.path.isfile(os.path.join(base_dir, '0', data_filename)):
            print('WARNING: no %s file in folder 0' % data_filename)
            self.run_new_structure(['0'])

        folder = '0'
        energy = np.loadtxt(os.path.join(base_dir, folder, data_filename))
        if energy.shape == ():
            energy = energy.reshape((1, ))
        self.E_pure_2 = energy
        str_out = open(os.path.join(base_dir, folder, 'str.out'))
        x_t = 0.0
        for line in str_out.readlines():
            v = line.split()
            if len(v) == 4:
                x_t += 1.
                self.elem_2 = v[-1]
        str_out.close()
        if not intensive:
            self.E_pure_2 = self.E_pure_2/x_t
        self.concentration[folder] = (0, 0, x_t)

        if not os.path.isdir(os.path.join(base_dir, '1')):
            self.make_new_structures(N=1)
            self.run_new_structure(['1'])
        elif not os.path.isfile(os.path.join(base_dir, '1', data_filename)):
            print('WARNING: no %s file in folder 1' % data_filename)
            self.run_new_structure(['1'])

        folder = '1'
        energy = np.loadtxt(os.path.join(base_dir, folder, data_filename))
        if energy.shape == ():
            energy = energy.reshape((1, ))
        self.E_pure_1 = energy
        str_out = open(os.path.join(base_dir, folder, 'str.out'))
        x_t = 0.0
        for line in str_out.readlines():
            v = line.split()
            if len(v) == 4:
                x_t += 1.
                self.elem_1 = v[-1]
        str_out.close()
        if not intensive:
            self.E_pure_1 = self.E_pure_1/x_t
        self.concentration[folder] = (1, x_t, x_t)

        self.data_values = np.empty((self.n_data, energy.shape[0]))
        self.design_matrix_built = False
        self.fitted = False
        self.predicted_energies = {}
        self.experimental_energies = {}
        self.data_filename = data_filename
        self.intensive = intensive

    def update_folders(self, pattern=is_int, path=None):
        if path is None:
            self.folders = next(os.walk(self.base_dir))[1]
        else:
            all_folders = next(os.walk(path))
            root = os.path.relpath(all_folders[0], self.base_dir)
            self.folders = all_folders[1]
        self.folders = [os.path.join(root, f) for f in self.folders if pattern(f)]
        self.folders.sort()

    def make_new_structures(self, N=10):
        if self.two_d:
            two_d = "-2d"
        else:
            two_d = ""
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        p = subprocess.Popen(["maps", "-d", two_d])
        counter = 0
        while counter < N:
            if not os.path.isfile('ready'):
                os.mknod('ready')
                print("structure %d/%d" % (counter, N))
                counter += 1
            time.sleep(3)
        p.terminate()
        os.remove('maps_is_running')
        os.chdir(original_dir)
        self.folders = next(os.walk(self.base_dir))[1]
        self.folders = [f for f in self.folders if is_int(f)]
        self.folders.sort(key=int)

    def run_new_structure(self, folders):
        original_dir = os.getcwd()
        os.chdir(self.base_dir)
        new_folders = []
        for i, folder in enumerate(folders):
            if os.path.isfile(os.path.join(folder, self.data_filename)):
                print("WARNING: structure in", folder, "already simulated")
                continue
            else:
                new_folders.append(folder)
                os.chdir(folder)
                subprocess.call(["python", os.path.join('..', "str2gpaw.py"),
                                 "str.out"])
                os.chdir(self.base_dir)
        os.chdir(original_dir)
        self.exp_folders += new_folders
        self.exp_folders.sort(key=int)
        self.design_matrix_built = False

    def get_design_vector(self, base_folder, x=0.0, x_1=0., x_t=0.,
                          force_update=False, save=False, load=False,
                          str_file=None):
        if (not force_update) and load:
            str_dia = '_'.join(map(str, self.diameters))
            fdesignv = 'phi' + str_dia + '.dat'
            if os.path.isfile(os.path.join(base_folder, fdesignv)):
                design_vector = np.loadtxt(os.path.join(base_folder,
                                                        fdesignv))
                self.design_vectors[base_folder] = design_vector
                return design_vector
        if (not force_update) and (base_folder in self.design_vectors):
            return self.design_vectors[base_folder]
        multiplicities = [m['multiplicity'] for m in self.clusters]
        multiplicities = np.array(multiplicities)
        if str_file is None:
            output_lattice = os.path.join(base_folder, 'str.out')
        else:
            output_lattice = os.path.join(base_folder, str_file)
        if self.ce_type == 'vbce':
            x_sigma = 2.*x - 1.
            corrdump = subprocess.check_output(["corrdump",
                                                "-l", self.input_lattice,
                                                "-2="+str(self.diameters[0]),
                                                "-3="+str(self.diameters[1]),
                                                "-4="+str(self.diameters[2]),
                                                "-5="+str(self.diameters[3]),
                                                "-6="+str(self.diameters[4]),
                                                "-s", output_lattice,
                                                "-crf=vbce",
                                                "-corrconc="+str(x_sigma)])
            corrdump = np.array(corrdump.split(), dtype=float)
            corrdump *= multiplicities
            n_points = np.array([m['n_points'] for m in self.clusters])
            corrdump *= np.power(1 - x_sigma*x_sigma, n_points/2)
            if abs(x_sigma) == 1.:
                corrdump[1:] = 0.
            for i, m in enumerate(self.maximum_cluster_size_vbce):
                xp = 1.
                for j in range(1, m + 1):
                    xp = xp*x_sigma
                    corrdump = np.hstack((corrdump, corrdump[i]*xp))
        else:
            corrdump = subprocess.check_output(["corrdump",
                                                "-l", self.input_lattice,
                                                "-2="+str(self.diameters[0]),
                                                "-3="+str(self.diameters[1]),
                                                "-4="+str(self.diameters[2]),
                                                "-5="+str(self.diameters[3]),
                                                "-6="+str(self.diameters[4]),
                                                "-s", output_lattice])
            corrdump = np.array(corrdump.split(), dtype=float)
            corrdump *= multiplicities

        design_vector = corrdump/self.n_at_basis
        if save:
            str_dia = '_'.join(map(str, self.diameters))
            np.savetxt(os.path.join(base_folder, 'phi' + str_dia + '.dat'),
                       design_vector)
        self.design_vectors[base_folder] = design_vector
        return design_vector

    def get_concentration(self, base_folder):
        if base_folder in self.concentration.keys():
            return self.concentration[base_folder]
        else:
            str_out = open(os.path.join(base_folder, 'str.out'))
            x_1 = 0.0
            x_t = 0.0
            for line in str_out.readlines():
                v = line.split()
                if len(v) == 4:
                    x_t += 1.
                    if v[-1] == self.elem_1:
                        x_1 += 1.
            str_out.close()
            x = x_1/x_t
            self.concentration[base_folder] = (x, x_1, x_t)
            return self.concentration[base_folder]

    def build_design_matrix(self, force_update=False, use_only=None, fit_formation=True,
                            save_design=False, load_design=False):
        if use_only is None:
            exp_folders = self.exp_folders
        else:
            exp_folders = list(np.asarray(self.exp_folders)[use_only])
        for i, folder in enumerate(exp_folders):
            base_folder = os.path.join(self.base_dir, folder)
            x, x_1, x_t = self.get_concentration(base_folder)
            print('\rcorrdump {}/{}: {}'.format(i, len(exp_folders), base_folder), end='')
            self.design_matrix[i, :] = self.get_design_vector(base_folder,
                                                              x, x_1, x_t,
                                                              force_update,
                                                              save=save_design, load=load_design)
            energy = np.loadtxt(os.path.join(base_folder, self.data_filename))
            if energy.shape == ():
                energy = energy.reshape((1, ))
            E = energy
            if not self.intensive:
                E = E/x_t
            # print(E.shape, self.E_pure_1.shape, self.E_pure_2.shape)
            if fit_formation:
                self.data_values[i, :] = (E - x*self.E_pure_1 -
                                          (1.-x)*self.E_pure_2)
            else:
                self.data_values[i, :] = E
            self.experimental_energies[folder] = (x, self.data_values[i, :])
            # print("{0:.6f} {1: .6f} {2}".
            #       format(x, data_values[i], folder))
        self.design_matrix_built = True

    def fit(self, mode='RVM', use_only=None, fit_formation=True,
            keep_for_testing=None, save_design=False, load_design=False):
        self.fit_formation = fit_formation
        self.fitter = []
        n_train = self.data_values.shape[0]
        if keep_for_testing is not None:
            if isinstance(keep_for_testing, int):
                self.test_idx = np.random.choice(self.n_data, size=3,
                                                 replace=False)
                self.test_idx.sort()
            else:
                self.test_idx = np.array(list(keep_for_testing))
                self.test_idx.sort()
            self.train_idx = np.array(list(set(range(self.n_data)) -
                                           set(self.test_idx)))
            n_train = self.train_idx.shape[0]
        else:
            self.test_idx = []
            self.train_idx = None
        if not self.design_matrix_built:
            self.build_design_matrix(use_only=use_only,
                                     fit_formation=fit_formation,
                                     save_design=save_design, load_design=load_design)
        # B = np.empty((self.design_matrix.shape[1], self.data_values.shape[1]))
        if mode == 'RVM':
            for i in range(self.data_values.shape[1]):
                if (self.data_values[:, i] == 0.).all():
                    self.fitter.append(RVM(verbosity=2,
                                           Phi=self.design_matrix))
                    continue
                self.fitter.append(RVM(verbosity=2))
                self.fitter[-1].regression(
                    self.data_values[:, i][self.train_idx].reshape(n_train, 1),
                    design_matrix=np.squeeze(self.design_matrix[self.train_idx]))
                # B[:, i] = self.fitter[-1].m.flatten()
        elif mode == 'BRR':
            for i in range(self.data_values.shape[1]):
                self.fitter.append(BLR.BLR(verbosity=1))
                self.fitter[-1].regression(
                    self.data_values[self.train_idx, i, np.newaxis],
                    design_matrix=self.design_matrix[self.train_idx])
                # B[:, i] = self.fitter[-1].m.flatten()
        elif mode == 'BL-RJMCMC':
            for i in range(self.data_values.shape[1]):
                mcmc = BL_RJMCMC(self.design_matrix[self.train_idx, :],
                                 self.data_values[self.train_idx, i],
                                 self.design_matrix[self.test_idx, :],
                                 self.data_values[self.test_idx, i],
                                 10000000, 0.0012, 1000,
                                 100000, 100000, 'PLOT',
                                 outdir='OUT_rjmcmc')
                self.fitter.append(mcmc)
                self.fitter[-1].sample(False)
                burn = mcmc.trace('test_error').shape[0] / 2
                self.fitter[-1].m = np.median(mcmc.trace('B')[burn:, :],
                                              axis=0)
                # B[:, i] = np.median(mcmc.trace('B')[burn:, :], axis=0)
                print("Error (test) of LASSO CV: {} eV".format(
                    mcmc.ERR_LASSO_TEST))
                print("Error (test) of Least Squares: {} eV".format(
                    mcmc.ERR_LST_TEST))

        self.eci_mean = np.empty((self.data_values.shape[1], self.n_basis))
        self.eci_std = np.empty((self.data_values.shape[1], self.n_basis))
        for i in range(self.data_values.shape[1]):
            self.eci_mean[i, :] = self.fitter[i].m.flatten()
            self.eci_std[i, :] = np.sqrt(np.diag(self.fitter[i].SN)).flatten()
        self.fitted = True
        if keep_for_testing is not None:
            ERR_TEST = self.eval_error(self.eci_mean.T,
                                       self.design_matrix[self.test_idx, :],
                                       self.data_values[self.test_idx])
            print("Error (test) of median prediction: {} eV".format(ERR_TEST))

    def mean_eci(self):
        return self.eci_mean

    def sample_eci(self, size=None, seed=None):
        scoeff = np.empty((self.data_values.shape[1], self.n_basis))
        if seed is None:
            for i in range(self.data_values.shape[1]):
                scoeff[i, :] = self.fitter[i].sample_coefficients(size=size)
            return scoeff
        else:
            for i in range(self.data_values.shape[1]):
                scoeff[i, :] = self.fitter[i].sample_coefficients_from_seed(
                                                                 size=size,
                                                                 seed=seed)
            return scoeff

    def sample_eci_with_state(self, size=None, seed=None):
        scoeff = np.empty((self.data_values.shape[1], self.n_basis))
        if seed is None:
            for i in range(self.data_values.shape[1]):
                scoeff[i, :], st = \
                    self.fitter[i].sample_coefficients_with_state(size=size)
            return scoeff, st[0]
        else:
            for i in range(self.data_values.shape[1]):
                scoeff[i, :], st = \
                    self.fitter[i].sample_coefficients_from_seed_with_state(
                                                                    size=size,
                                                                    seed=seed)
            return scoeff, st[0]

    def dump_eci(self, size=None, seed=None, mean=False):
        if mean:
            np.savetxt('%s_mean.eci' % self.data_filename, self.eci_mean,
                       delimiter='\n', newline='\n\n')
        else:
            if size is None:
                size = 1
            for i in range(size):
                c, s = self.sample_eci_with_state(size=1, seed=seed)
                np.savetxt('%s_%s.eci' % (self.data_filename, s), c,
                           delimiter='\n', newline='\n\n')

    def prediction(self, folders, force_update=False, save_design=False, load_design=False):
        if not self.fitted:
            self.fit()
        b = set(self.predicted_energies.keys())
        new_folders = [(i, f) for i, f in enumerate(folders) if f not in b]
        design_matrix = np.empty((len(folders), self.n_basis))
        for i, folder in enumerate(folders):
            base_folder = os.path.join(self.base_dir, folder)
            x, x_1, x_t = self.get_concentration(base_folder)
            print('\rcorrdump {}/{}: {}'.format(i, len(folders), base_folder), end='')
            design_matrix[i, :] = self.get_design_vector(base_folder,
                                                         x, x_1, x_t,
                                                         force_update,
                                                         save=save_design, load=load_design)
        mean = np.empty((design_matrix.shape[0], self.data_values.shape[1]))
        error = np.empty((design_matrix.shape[0], self.data_values.shape[1]))
        for j in range(self.data_values.shape[1]):
            m_tmp, e_tmp = self.fitter[j].eval_regression(
                                                   design_matrix=design_matrix)
            mean[:, j] = m_tmp.ravel()
            error[:, j] = e_tmp.ravel()
        for i, folder in new_folders:
            base_folder = os.path.join(self.base_dir, folder)
            x, x_1, x_t = self.get_concentration(base_folder)
            self.predicted_energies[folder] = (x, mean[i], error[i])

        return mean, error

    def eval_error(self, B, design_matrix, data_values):
        PRED_TEST = np.dot(design_matrix, B)
        ERR_TEST = np.sqrt(np.sum(np.square(
            np.subtract(PRED_TEST, data_values)), axis=0) /
            float(len(PRED_TEST)))
        return ERR_TEST

    def update_clusters(self):
        output_lattice = os.path.join(self.base_dir, '0/str.out')
        subprocess.check_output(["corrdump",
                                 "-l", self.input_lattice,
                                 "-2="+str(self.diameters[0]),
                                 "-3="+str(self.diameters[1]),
                                 "-4="+str(self.diameters[2]),
                                 "-5="+str(self.diameters[3]),
                                 "-6="+str(self.diameters[4]),
                                 "-s", output_lattice])

        # clustersf = open(os.path.join(self.base_dir, 'clusters.out'))
        self.clusters = ClustersFileParser('clusters.out')
        self.clusters.parse()
        self.clusters.cluster_info()
        # clustersf = open('clusters.out')
        # self.clusters = []
        # m = True
        # r = False
        # n = False
        # a = False
        # for i, line in enumerate(clustersf.readlines()):
        #     if m:
        #         multiplicity = int(line)
        #         self.clusters.append({'multiplicity': multiplicity})
        #         self.clusters[-1]['coordinates'] = []
        #         m = False
        #         r = True
        #         continue
        #     elif r:
        #         radius = float(line)
        #         self.clusters[-1]['diameter'] = radius
        #         r = False
        #         n = True
        #         continue
        #     elif n:
        #         natoms = int(line)
        #         self.clusters[-1]['n_points'] = natoms
        #         n = False
        #         a = True
        #         continue
        #     elif a:
        #         if natoms == 0:
        #             a = False
        #             m = True
        #             continue
        #         positions = np.array(line.split()[:3], dtype=float)
        #         self.clusters[-1]['coordinates'].append(positions)
        #         natoms -= 1
        #         continue
        # clustersf.close()
        # for c in self.clusters:
        #     c['coordinates'] = np.array(c['coordinates'])

    def set_maximum_cluster_size_vbce(self, sizes):
        self.maximum_cluster_size_vbce = sizes

    def plot_eci(self, which=0):
        if not self.fitted:
            self.fit()
        import matplotlib.pyplot as plt
        groups = ['pair', 'triplet', 'quartet', 'quintet']
        labels = []
        max_atoms = -1
        offset = 15
        plt.figure(figsize=(10, 8))
        for i, cluster in enumerate(self.clusters):
            if cluster['n_points'] < 2:
                continue
            if cluster['n_points'] > max_atoms:
                max_atoms = cluster['n_points']
            x = cluster['diameter'] + offset*(cluster['n_points'] - 2)
            y = self.eci_mean[which, i]
            y_std = self.eci_std[which, i]
            plt.errorbar(x, y*1.e3, yerr=1.96e3*y_std,
                         fmt='o', color='r', markersize=7,
                         ecolor='r', elinewidth=3, mew=3)
        if self.ce_type == 'vbce':
            counter = 0
            for i, j in enumerate(self.maximum_cluster_size_vbce):
                for k in range(1, j+1):
                    cluster = self.clusters[i]
                    if cluster['n_points'] < 2:
                        counter += 1
                        continue
                    x = cluster['diameter'] + offset*(cluster['n_points'] - 2)
                    y = self.eci_mean[which, len(self.clusters) + counter]
                    y_std = self.eci_std[which, len(self.clusters) + counter]
                    plt.errorbar(x, y*1.e3, yerr=1.96e3*y_std,
                                 fmt='x', color='b', markersize=7,
                                 ecolor='b', elinewidth=2, mew=3)
                    if abs(y)*1.e3 > 1.:
                        plt.text(x + 0.2, y*1.e3, str(k), fontsize=20)
                    counter += 1

        plt.xlim([0.0, (max_atoms - 1)*offset])
        xticks = plt.xticks()[0]
        labels = [str(x % offset) for x in xticks]
        i = 0
        for j, l in enumerate(labels):
            if l == '0.0':
                labels[j] = groups[i]
                if i > max_atoms - 2:
                    labels[j] = str(offset)
                i += 1

        for i in range(max_atoms-2):
            plt.axvline((i + 1)*offset, linestyle='--', color='k')
        plt.xticks(xticks, labels)
        plt.xlabel("Cluster diameter (A)", fontsize=28)
        plt.ylabel("ECI strength [meV]", fontsize=28)
        plt.tick_params(labelsize=24)
        plt.axhline(0.0, color='k')
        plt.tight_layout()

        plt.figure(figsize=(10, 8))
        current_size = 0
        i_init = 0
        y_arr = np.min(self.eci_mean[which, :])*1.e3
        for i, cluster in enumerate(self.clusters):
            if cluster['n_points'] < 2:
                current_size = cluster['n_points']
                continue
            if cluster['n_points'] != current_size and cluster['n_points'] > 2:
                plt.axvline(i - 0.5, linestyle='--', color='k')
                plt.annotate('', xy=(i_init, y_arr), xytext=(i - 0.5, y_arr),
                         # xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='left',
                         verticalalignment='bottom')
                i_init = i
            x = i
            plt.errorbar(x, self.eci_mean[which, i]*1.e3,
                         yerr=1.96e3*self.eci_std[which, i],
                         fmt='o', color='r', markersize=7,
                         ecolor='r', elinewidth=3, mew=3)
            current_size = cluster['n_points']

        xticks = plt.xticks()[0]
        labels = ['' for x in xticks]
        plt.xticks(xticks, labels)
        plt.xlabel("Points in cluster family", fontsize=28)
        plt.ylabel("ECI strength [meV]", fontsize=28)
        plt.tick_params(labelsize=24)
        plt.axhline(0.0, color='k')
        plt.tight_layout()
        plt.show()  # block=False)

    def plot_vbce_eci(self, points=0, which=0):
        import matplotlib.pyplot as plt
        x = np.linspace(-1., 1., 100)
        y = []
        n_points = {}
        for i, c in enumerate(self.clusters):
            n_atoms = c['n_points']
            y.append((1-x*x)**(n_atoms/2)*self.eci_mean[which, i])
            if n_atoms in n_points:
                n_points[n_atoms] += 1
            else:
                n_points[n_atoms] = 1
        counter = 0
        for i, j in enumerate(self.maximum_cluster_size_vbce):
            for k in range(1, j+1):
                y[i] += ((1-x*x)**(self.clusters[i]['n_points']/2) *
                         x**k *
                         self.eci_mean[0, len(self.clusters) + counter])
                counter += 1
        y_0 = np.array(y)[:, 50]
        plt.figure(figsize=(10, 8))
        start = 0
        finish = 1
        for i in range(points):
            start += n_points[i]
            finish = start + n_points[i + 1]
        for i in range(start, finish):
            plt.plot(x, y[i]*1.e3, lw=2)
            plt.text(0.0, y_0[i]*1.05e3, str(i-start), fontsize=20)

        plt.xlabel("x_{} - x_{}".format(self.elem_1, self.elem_2),
                   fontsize=28)
        plt.ylabel("ECI strength [meV]", fontsize=28)
        plt.tick_params(labelsize=24)
        plt.tight_layout()
        plt.show()

    def get_exp_ground_state_line(self, which=0):
        exp_points = np.empty((self.n_data, 2))
        for i, folder in enumerate(self.exp_folders):
            exp_points[i, 0] = self.experimental_energies[folder][0]
            exp_points[i, 1] = self.experimental_energies[folder][1][which]

        from scipy.spatial import ConvexHull
        p_exp_points = exp_points[exp_points[:, 1] <= 0.0]
        hull = ConvexHull(p_exp_points)
        vertices = np.roll(hull.vertices, -1)
        gsl = p_exp_points[vertices]
        gsl_folders = np.array(self.exp_folders)[exp_points[:, 1] <= 0.0][vertices]

        return gsl, gsl_folders

    def get_ground_state_line(self, folders=None, force_update=False,
                              random=False, ECI=None, save_phi=False,
                              load_phi=False, which=0, dump_rnd_eci_to=''):
        if folders is None:
            folders = self.folders
        predicted_energies = {}
        if random:
            new_folders = folders
            leci, prng_state = self.sample_eci_with_state(size=1)
            if dump_rnd_eci_to:
                ofile = '{}_{}.eci'.format(self.data_filename, prng_state)
                ofile = os.path.join(dump_rnd_eci_to, ofile)
                self.last_prng_state = prng_state
                np.savetxt(ofile, leci, delimiter='\n', newline='\n\n')
        else:
            predicted_energies.update(self.predicted_energies)
            b = set(predicted_energies.keys())
            new_folders = [f for f in folders if f not in b]
        for i, folder in enumerate(new_folders):
            base_folder = os.path.join(self.base_dir, folder)
            x, x_1, x_t = self.get_concentration(base_folder)
            phi = self.get_design_vector(base_folder,
                                         x, x_1, x_t,
                                         force_update,
                                         save=save_phi,
                                         load=load_phi).reshape(1,
                                                                self.n_basis)
            if ECI is not None:
                mean = np.dot(phi, ECI)[0]
                if self.fit_formation:
                    predicted_energies[folder] = (x, mean)
                else:
                    predicted_energies[folder] = (x, mean - x*self.E_pure_1 -
                                                  (1.-x)*self.E_pure_2)
            if random:
                # leci, prng_state = self.sample_eci_with_state(size=1)
                # if dump_rnd_eci_to:
                #     ofile = '{}_{}.eci'.format(self.data_filename, prng_state)
                #     ofile = os.path.join(dump_rnd_eci_to, ofile)
                #     self.last_prng_state = prng_state
                #     np.savetxt(ofile, leci,
                #                delimiter='\n', newline='\n\n')
                mean = np.dot(phi, leci.T)[0]
                if self.fit_formation:
                    predicted_energies[folder] = (x, mean)
                else:
                    predicted_energies[folder] = (x, mean - x*self.E_pure_1 -
                                                  (1.-x)*self.E_pure_2)
            else:
                mean, error = self.fitter[which].eval_regression(
                                                             design_matrix=phi)
                mean = np.dot(phi, self.mean_eci().T)[0]
                predicted_energies[folder] = (x, mean, error[0])

        if not random:
            self.predicted_energies.update(predicted_energies)

        points = np.empty((len(folders), 3))
        for i, folder in enumerate(folders):
            points[i, 0] = predicted_energies[folder][0]
            points[i, 1] = predicted_energies[folder][1][which]
            if not random:
                points[i, 2] = 0.0  # predicted_energies[folder][2][which]
        points[0, 1] = 0.0
        points[0, 2] = 0.0
        points[1, 1] = 0.0
        points[1, 2] = 0.0

        energies = points[:, :2]
        energy_filter = energies[:, 1] <= 0.
        energies = energies[energy_filter]
        folders = np.array(folders)[energy_filter]
        from scipy.spatial import ConvexHull
        hull = ConvexHull(energies)
        vertices = np.roll(hull.vertices, -1)
        gsl = energies[vertices]
        gsl_folders = folders[vertices]

        return gsl, gsl_folders

    def plot_ground_state_line(self, folders=None, force_update=False,
                               convex_hull=True, which=0):
        if folders is None:
            folders = self.folders
        b = set(self.predicted_energies.keys())
        new_folders = [f for f in folders if f not in b]
        for i, folder in enumerate(new_folders):
            base_folder = os.path.join(self.base_dir, folder)
            x, x_1, x_t = self.get_concentration(base_folder)
            phi = self.get_design_vector(base_folder,
                                         x, x_1, x_t,
                                         force_update).reshape(1, self.n_basis)
            mean, error = self.fitter[which].eval_regression(design_matrix=phi)
            self.predicted_energies[folder] = (x, mean[0], error[0])

        points = np.empty((len(folders), 3))
        exp_points = np.empty((self.n_data, 2))
        for i, folder in enumerate(folders):
            points[i, 0] = self.predicted_energies[folder][0]
            points[i, 1] = self.predicted_energies[folder][1]
            points[i, 2] = self.predicted_energies[folder][2]
        points[0, 1] = 0.0
        points[0, 2] = 0.0
        points[1, 1] = 0.0
        points[1, 2] = 0.0
        for i, folder in enumerate(self.exp_folders):
            exp_points[i, 0] = self.experimental_energies[folder][0]
            exp_points[i, 1] = self.experimental_energies[folder][1][which]
        if convex_hull:
            from scipy.spatial import ConvexHull
            p_points = points[points[:, 1] <= 0.]
            hull = ConvexHull(p_points[:, :2])
            p_exp_points = exp_points[exp_points[:, 1] <= 0.]
            exp_hull = ConvexHull(p_exp_points)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.scatter(exp_points[:, 0],
                    exp_points[:, 1],
                    marker='o', c='k', s=50)
        plt.scatter(points[:, 0],
                    points[:, 1],
                    marker='x', c='b', s=50)
        if convex_hull:
            vertices = np.roll(hull.vertices, -1)
            plt.plot(p_points[vertices, 0],
                     p_points[vertices, 1],
                     'b--', lw=2)
            plt.fill_between(p_points[vertices, 0],
                             p_points[vertices, 1] - 1.96*p_points[vertices, 2],
                             p_points[vertices, 1] + 1.96*p_points[vertices, 2],
                             facecolor='red', alpha=0.2)
            for simplex in exp_hull.simplices:
                plt.plot(p_exp_points[simplex, 0],
                         p_exp_points[simplex, 1],
                         'k-')
        plt.xlim([0.0, 1.0])
        plt.xlabel("Percent " + self.elem_1 + " in " + self.elem_1 +
                   "-" + self.elem_2, fontsize=28)
        plt.ylabel("Formation energy [eV]", fontsize=28)
        plt.tick_params(labelsize=24)
        plt.tight_layout()
        plt.show(block=False)

    def plot_regression(self, folders=None, which=0, scale=1.e3, units='meV',
                        xlabel=None, ylabel=None):
        import matplotlib.pyplot as plt
        if folders is None:
            folders = self.exp_folders
        if xlabel is None:
            xlabel = 'Experimental data [meV]'
        if ylabel is None:
            ylabel = 'Fitted data [meV]'
        fitted, errors = self.prediction(folders)
        plt.figure(figsize=(10, 8))
        plt.errorbar(self.data_values*scale, fitted*scale, yerr=errors*scale,
                     fmt='o', color='r', markersize=11)
        plt.plot(self.data_values*scale, self.data_values*scale,
                 linewidth=2, color='k')
        error = self.fitter[which].validation_error()[0]
        plt.annotate("RMS error: {:.2f} [{}]".format(error*scale, units),
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=28)
        plt.xlabel(xlabel, fontsize=32)
        plt.ylabel(ylabel, fontsize=32)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    cluster_diameters = [6.5, 3.0, 0.0, 0.0, 0.0]
    ce = ClusterExpansion(diameters=cluster_diameters)
