from __future__ import print_function

import argparse
import os
import time
import sys

try:
    import matplotlib.pyplot as plt
except:
    plt = None
import numpy as np
try:
    import scipy.stats as scipy_stats
except:
    scipy_stats = None

# from ase import Atoms
import ase.db
from ase import units

from matscipy import elasticity as el
import spglib

from atatutils.str2emt import ATAT2EMT
from atatutils.str2gpaw import read_atat_input
import BayesianLinearRegression.myBayesianLinearRegression as BLR
import BayesianLinearRegression.myExpansionBasis as EB


graphics = True
if plt is None:
    graphics = False


def extract_free_energy(filename):
    start_line = 'Free energy\n'
    end_line = 'end\n'
    fe = []
    with open(filename) as svsl_log:
        for line in svsl_log:
            if line == start_line:
                # Free energy block
                fe.append([])
                while True:
                    ll = next(svsl_log)
                    if ll == end_line:
                        break
                    temperature, strain, free_energy = ll.strip().split()
                    fe[-1].append([float(free_energy), float(strain), float(temperature)])
    return np.array(fe)


class SJEOS(object):
    sjeos_m = np.array([[1, 1, 1, 1],
                        [3, 2, 1, 0],
                        [18, 10, 4, 0],
                        [108, 50, 16, 0]
                        ])

    def __init__(self, nsamples=1000):
        self.nsamples = nsamples
        self.v0s = []
        self.e0s = []
        self.Bs = []
        self.B1s = []
        self.blr = BLR.BLR()

    def fit(self, volume, energy, beta=1e6):
        self.blr.regression(energy.reshape(len(energy), 1),
                            X=volume.reshape(len(volume), 1)**(1./3),
                            basis=EB.Basis('inverse_monomial', 4),
                            alpha_mode='scalar', beta=beta)
        self.v0s.append(np.zeros(self.nsamples))
        self.e0s.append(np.zeros(self.nsamples))
        self.Bs.append(np.zeros(self.nsamples))
        self.B1s.append(np.zeros(self.nsamples))
        rc = np.random.multivariate_normal(self.blr.m.reshape(4),
                                           self.blr.SN,
                                           self.nsamples)
        for i in range(self.nsamples):
            c = rc[i, ::-1]

            t = (-c[1] + np.sqrt(c[1]*c[1] - 3*c[2]*c[0]))/(3*c[0])
            v0 = t**-3
            self.v0s[-1][i] = v0
            if t != t:
                continue
            d = np.array([v0**(-1), v0**(-2./3), v0**(-1./3), 1])
            d = np.diag(d)
            o = np.dot(np.dot(self.sjeos_m, d), c)
            self.e0s[-1][i] = -o[0]  # fit0(t)
            self.Bs[-1][i] = o[2] / v0 / 9  # (t**5 * fit2(t) / 9)
            self.Bs[-1][i] = self.Bs[-1][i] / ase.units.kJ * 1.e24
            self.B1s[-1][i] = o[3] / o[2] / 3

    def bulk_modulus(self, i=None):
        if i is not None:
            return self.Bs[i]
        return self.Bs

    def bulk_modulus_derivative(self, i=None):
        if i is not None:
            return self.B1s[i]
        return self.B1s

    def cohesive_energy(self, i=None):
        if i is not None:
            return self.e0s[i]
        return self.e0s

    def equilibrium_volume(self, i=None):
        if i is not None:
            return self.v0s[i]
        return self.v0s


# Get V(T)
def get_V_T(samples=None, plot=False):
    tmp_path = os.path.join('properties', 'tdep', 'T_batch.dat')
    vol_path = os.path.join('properties', 'tdep', 'V_batch.dat')
    tmp_batch = np.loadtxt(tmp_path)
    vol_batch = np.loadtxt(vol_path)

    tmp_mean = np.mean(tmp_batch, axis=1)
    tmp_std = np.std(tmp_batch, axis=1)
    vol_mean = np.mean(vol_batch, axis=1)
    vol_std = np.std(vol_batch, axis=1)

    n_tmp = tmp_batch.shape[0]
    if samples is not None:
        tmp_samples = np.empty((n_tmp, samples))
        vol_samples = np.empty((n_tmp, samples))
        for i in range(n_tmp):
            tmp_samples[i, :] = np.random.normal(tmp_mean[i], tmp_std[i], samples)
            vol_samples[i, :] = np.random.normal(vol_mean[i], vol_std[i], samples)
    else:
        tmp_samples = tmp_mean.reshape((n_tmp, 1))
        vol_samples = vol_mean.reshape((n_tmp, 1))

    return vol_samples, tmp_samples


def get_V_T_from_free_energy(fc_samples=1, plot=False):
    str_relax = read_atat_input('str_relax.out')
    vol_relax = np.linalg.det(str_relax.cell)
    n_samples = 1000
    F_V_T = extract_free_energy(os.path.join('svsls', 'svsl0.log'))
    n_T = F_V_T.shape[1]
    all_V0 = np.empty((n_T, n_samples*fc_samples))
    for fc_sample in range(fc_samples):
        F_V_T = extract_free_energy(os.path.join('svsls', 'svsl{}.log'.format(fc_sample)))
        volumes = vol_relax * F_V_T[:, 0, 1]**3

        sjeos = SJEOS(nsamples=n_samples)
        for i in range(F_V_T.shape[1]):
            F_V = F_V_T[:, i, 0]
            sjeos.fit(volumes, F_V, beta=1e6)

        V0s = sjeos.equilibrium_volume()
        all_V0[:, n_samples*fc_sample: n_samples*(fc_sample + 1)] = np.array(V0s)[:, :]
        V0_means = np.array([np.mean(V0s[i]) for i in range(len(V0s))])
        V0_stds = np.array([np.std(V0s[i]) for i in range(len(V0s))])
        V0_std = np.around(np.mean(V0_stds), decimals=2)

        # print('Equilibrium volume: {} [A^3]'.format(np.array_str(V0_means,
        #                                                    precision=2)))

        with open('eqvolume{}'.format(fc_sample), 'w') as bfile:
            for i in range(len(V0_means)):
                V0 = V0_means[i]
                std = V0_std
                bfile.write('{:.2f}\n'.format(V0))

        with open('eqvolume_err', 'w') as bfile:
            bfile.write('{:.2f}'.format(V0_std))

    # Plot last force constant sample
    if plot and (plt is not None):
        labels = [""] * len(V0s)
        for i in range(0, len(labels), 3):
            labels[i] = np.asarray(F_V_T[0, :, 2], dtype=int)[i]
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(V0s) + 1), V0_means)
        plt.boxplot(V0s, labels=labels)
        plt.ylabel("Equilibrium volume [A^3]", fontsize=34)
        plt.xlabel("Temperature [K]", fontsize=34)
        plt.tick_params('both', labelsize=30)
        plt.gcf().set_tight_layout(True)
        plt.show(block=False)
    return all_V0, F_V_T[0, :, 2]


def do_fit(index1, index2, stress, strain, patt, plot=False):
    """
    if scipy_stats is not None:
        cijFitted, intercept, r, tt, stderr = scipy_stats.linregress(
                                                             strain[:, index2],
                                                             stress[:, index1])
    else:
        fit = np.polyfit(strain[:, index2], stress[:, index1], 1)
        cijFitted, intercept = fit[0], fit[1]
        r, tt, stderr = 0.0, None, 0.0
    """

    ndata = strain[:, index1].shape[0]
    fitter = BLR.BLR()  # RVM(verbosity=2)
    fitter.regression(stress[:, index1].reshape(ndata, 1),
                      X=strain[:, index2].reshape(ndata, 1),
                      basis=EB.Basis('monomial', 3),
                      beta=1.e8)
    intercept, cijFitted, _ = fitter.coefficients()
    stderr = fitter.SN[1, 1]
    if plot:
        # position this plot in a 6x6 grid
        sp = plt.subplot(6, 6, 6*index1 + (index2 + 1))
        sp.set_axis_on()
        # change the labels on the axes
        xlabels = sp.get_xticklabels()
        plt.setp(xlabels, 'rotation', 90, fontsize=20)
        ylabels = sp.get_yticklabels()
        plt.setp(ylabels, fontsize=20)
        # colour the plot depending on the strain pattern
        colourDict = {0: '#BAD0EF', 1: '#FFCECE', 2: '#BDF4CB',
                      3: '#EEF093', 4: '#FFA4FF', 5: '#75ECFD'}
        sp.set_axis_bgcolor(colourDict[patt])
        # plot the data
        # plt.plot([strain[0, index2], strain[-1, index2]],
        #          [(cijFitted*strain[0, index2] + intercept)/units.GPa,
        #           (cijFitted*strain[-1, index2] + intercept)/units.GPa])
        plt.plot(strain[:, index2], stress[:, index1]/units.GPa, 'ro')
        strain_interp = np.linspace(strain[0, index2], strain[-1, index2], 20)
        T, S = fitter.eval_regression(strain_interp[:, np.newaxis])
        plt.plot(strain_interp, T[:, 0] / units.GPa)
        plt.fill_between(strain_interp,
                         (T[:, 0] - 1.0 * S[:, 0])/units.GPa,
                         (T[:, 0] + 1.0 * S[:, 0])/units.GPa,
                         facecolor='red', alpha=0.2)
        plt.xticks(strain[:, index2])
    return cijFitted, stderr


class ElasticConstants(object):
    def __init__(self, structure='str_relax.out', delta=0.01, n_steps=5, n_volumes=1, max_iso_strain=0.01,
                 plot=False, verbosity=0):
        # relaxed structure
        self.structure = ATAT2EMT(structure, verbosity=0)
        self.structure.atoms.wrap()
        isgn = spglib.get_symmetry_dataset(self.structure.atoms, symprec=1e-3)['number']
        self.symmetry = el.crystal_system(isgn)
        print("Crystal symmetry: {}".format(self.symmetry))
        self.delta = delta
        self.n_steps = n_steps
        self.n_volumes = n_volumes
        if n_volumes > 1:
            self.delta_v = max_iso_strain/(n_volumes - 1)
        else:
            self.delta_v = 0.
        self.base_dir = os.getcwd()
        self.verbosity = verbosity
        self.plot = plot

        self.A_u = {}
        self.B = {}
        self.C = {}
        self.C_err = {}
        self.G = {}
        self.T = {}

    def generate_strains(self):
        for ivol in range(self.n_volumes):
            vdname = 'v_{0:+}'.format(self.delta_v * ivol)
            if not os.path.exists(vdname):
                os.makedirs(vdname)
            os.chdir(vdname)
            T = np.eye(3) * (1 + self.delta_v * ivol)
            iso_strained_atoms = self.structure.atoms.copy()
            iso_strained_atoms.set_cell(np.dot(T, iso_strained_atoms.cell.T).T, scale_atoms=False)
            iso_strained_atoms.positions[:] = np.dot(T, iso_strained_atoms.positions.T).T

            strained_configs = el.generate_strained_configs(iso_strained_atoms,
                                                            symmetry=self.symmetry,
                                                            N_steps=self.n_steps,
                                                            delta=self.delta)

            # generate one folder for each configuration. Relax atoms but not cells!
            # dump strain from atoms.info['strain'] transformed to Voigt

            atom_types = self.structure.atoms.get_chemical_symbols()
            for i, c in enumerate(strained_configs):
                dname = 's_{0:+}_{1}'.format(-self.delta * (self.n_steps / 2 - i % self.n_steps),
                                             i / self.n_steps)
                if not os.path.exists(dname):
                    os.makedirs(dname)
                if not os.path.isfile(os.path.join(dname, 'str.out')):
                    with open(os.path.join(dname, 'str.out'), 'w') as strfile:
                        strfile.write('1. 1. 1. 90. 90. 90.\n')
                        for l in c.get_cell():
                            strfile.write('{0} {1} {2}\n'.format(l[0], l[1], l[2]))
                        for j, l in enumerate(c.get_positions()):
                            strfile.write('{0} {1} {2} {3}\n'.format(l[0], l[1], l[2],
                                                                     atom_types[j]))
                if not os.path.isfile(os.path.join(dname, 'strain.in')):
                    np.savetxt(os.path.join(dname, 'strain.in'),
                               el.full_3x3_to_Voigt_6_strain(c.info['strain']))
            os.chdir(self.base_dir)

    def calculate_stresses(self, calc, **kwargs):
        for ivol in range(self.n_volumes):
            vdname = 'v_{0:+}'.format(self.delta_v * ivol)
            if not os.path.exists(vdname):
                os.makedirs(vdname)
            os.chdir(vdname)
            vdir = os.getcwd()
            directories = [d for d in os.listdir('.') if os.path.isdir(d) and
                           d[:2] == 's_']
            for sdname in directories:
                if not os.path.isfile(os.path.join(sdname, 'stress.out')):
                    if self.verbosity:
                        print('{}/{}'.format(vdname, sdname))
                    os.chdir(sdname)
                    structure = ATAT2EMT('str.out', calc)
                    structure.optimise_positions(**kwargs)
                    structure.dump_outputs()
                    os.chdir(vdir)

            os.chdir(self.base_dir)

    def get_elastic_constants(self, save_at='.'):
        # analyse results: load stress from each folder
        # stress/strain[pattern_index, strain_level, stress/strain]
        # For each pattern, fit stress-strain for independent components
        # Try plotting E - strain as well -> 2nd derivative

        # Stiffness matrix and error
        # relax only atom positions, not unit call stress.
        for ivol in range(self.n_volumes):
            volume_str = '{}'.format(self.delta_v * ivol)
            if save_at is not None:
                if not os.path.exists(save_at):
                    os.makedirs(save_at)
                elas_name = os.path.join(save_at, 'elas{}'.format(volume_str))
                elas_err_name = os.path.join(save_at, 'elas{}_err'.format(volume_str))

                if os.path.isfile(elas_name) and os.path.isfile(elas_err_name):
                    C = np.loadtxt(elas_name).reshape((3, 3, 3, 3))
                    C_err = np.loadtxt(elas_err_name).reshape((3, 3, 3, 3))
                    self.C[volume_str] = el.full_3x3x3x3_to_Voigt_6x6(C)
                    self.C_err[volume_str] = el.full_3x3x3x3_to_Voigt_6x6(C_err)
                    continue

            vdname = 'v_{0:+}'.format(self.delta_v * ivol)
            os.chdir(vdname)
            # There are 21 independent elastic constants
            Cijs = {}
            Cij_err = {}

            # Construct mapping from (i,j) to index into Cijs in range 1..21
            # (upper triangle only to start with)
            Cij_map = {}
            Cij_map_sym = {}
            for i in range(6):
                for j in range(i, 6):
                    Cij_map[(i, j)] = el.Cij_symmetry[None][i, j]
                    Cij_map_sym[(i, j)] = el.Cij_symmetry[self.symmetry][i, j]

            # Reverse mapping, index 1..21 -> tuple (i,j) with i, j in range 0..5
            Cij_rev_map = dict(zip(Cij_map.values(), Cij_map.keys()))

            # Add the lower triangle to Cij_map, e.g. C21 = C12
            for (i1, i2) in Cij_map.copy().keys():
                Cij_map[(i2, i1)] = Cij_map[(i1, i2)]
                Cij_map_sym[(i2, i1)] = Cij_map_sym[(i1, i2)]

            N_patterns = len(el.strain_patterns[self.symmetry])
            strain = np.zeros((N_patterns, self.n_steps, 6))
            stress = np.zeros((N_patterns, self.n_steps, 6))

            directories = [d for d in os.listdir('.') if os.path.isdir(d) and
                           d[:2] == 's_']
            for d in directories:
                if self.verbosity > 1:
                    print("processing {}/{}".format(vdname, d))
                step = int(round(float(d.split('_')[1]) / self.delta + self.n_steps / 2))
                pattern_index = int(d.split('_')[2])
                stress_d = np.loadtxt(os.path.join(d, 'stress.out'))
                strain_d = np.loadtxt(os.path.join(d, 'strain.in'))
                stress[pattern_index, step, :] = el.full_3x3_to_Voigt_6_stress(stress_d)
                strain[pattern_index, step, :] = strain_d

            if self.plot:
                plt.figure(figsize=(18, 15))
            for pattern_index, (pattern, fit_pairs) in enumerate(el.strain_patterns[self.symmetry]):
                for (index1, index2) in fit_pairs:
                    fitted, err = do_fit(index1, index2,
                                         stress[pattern_index, :, :],
                                         strain[pattern_index, :, :],
                                         pattern_index, plot=self.plot)

                    index = abs(Cij_map_sym[(index1, index2)])

                    if index not in Cijs:
                        Cijs[index] = [fitted]
                        Cij_err[index] = [err]
                    else:
                        Cijs[index].append(fitted)
                        Cij_err[index].append(err)

            if self.plot:
                plt.show()

            C = np.zeros((6, 6))
            C_err = np.zeros((6, 6))
            C_labels = np.zeros((6, 6), dtype='S4')
            C_labels[:] = '    '

            # Convert lists to mean
            for k in Cijs:
                Cijs[k] = np.mean(Cijs[k])

            # Combine statistical errors
            for k, v in Cij_err.items():
                Cij_err[k] = np.sqrt(np.sum(np.array(v) ** 2)) / np.sqrt(len(v))

            if self.symmetry.startswith('trigonal'):
                # Special case for trigonal lattice: C66 = (C11 - C12)/2
                Cijs[Cij_map[(5, 5)]] = 0.5 * (Cijs[Cij_map[(0, 0)]] -
                                               Cijs[Cij_map[(0, 1)]])
                Cij_err[Cij_map[(5, 5)]] = np.sqrt(Cij_err[Cij_map[(0, 0)]] ** 2 +
                                                   Cij_err[Cij_map[(0, 1)]] ** 2)

            # Generate the 6x6 matrix of elastic constants
            # - negative values signify a symmetry relation
            for i in range(6):
                for j in range(6):
                    index = el.Cij_symmetry[self.symmetry][i, j]
                    if index > 0:
                        C[i, j] = Cijs[index]
                        C_err[i, j] = Cij_err[index]
                        ii, jj = Cij_rev_map[index]
                        C_labels[i, j] = ' C%d%d' % (ii + 1, jj + 1)
                        C_err[i, j] = Cij_err[index]
                    elif index < 0:
                        C[i, j] = -Cijs[-index]
                        C_err[i, j] = Cij_err[-index]
                        ii, jj = Cij_rev_map[-index]
                        C_labels[i, j] = '-C%d%d' % (ii + 1, jj + 1)
            os.chdir(self.base_dir)
            self.C[volume_str] = C / units.GPa
            self.C_err[volume_str] = C_err / units.GPa
            if save_at is not None:
                np.savetxt(elas_name,
                           el.Voigt_6x6_to_full_3x3x3x3(C / units.GPa).ravel())
                np.savetxt(elas_err_name,
                           el.Voigt_6x6_to_full_3x3x3x3(C_err / units.GPa).ravel())

        if self.verbosity > 1:
            np.set_printoptions(precision=4)
            print("C [GPa] = ")
            print(C / units.GPa)
            print("+/-")
            print(C_err / units.GPa)
            np.set_printoptions(precision=8)

        if self.plot:
            plt.figure(figsize=(10, 8))
            x = np.zeros(self.n_volumes)
            y = np.zeros(self.n_volumes)
            for ivol in range(self.n_volumes):
                x[ivol] = self.delta_v * ivol
                y[ivol] = self.C['{}'.format(x[ivol])][0, 0]
            plt.plot(x, y)
            plt.show()

        return self.C

    @staticmethod
    def get_elastic_moduli(C, verbosity=0):
        E, nu, G, B, K = el.elastic_moduli(C)
        if verbosity > 1:
            np.set_printoptions(precision=4)
            print("Young\'s modulus E = \n {}".format(E / units.GPa))
            print("Poisson\'s ratio \\nu = \n {}".format(nu))
            print("Shear modulus G = \n {}".format(G / units.GPa))
            print("Bulk modulus B = \n {}".format(B / units.GPa))
            print("Bulk modulus tensor K = \n {}".format(K / units.GPa))
            np.set_printoptions(precision=8)
        return E / units.GPa, nu, G / units.GPa, B / units.GPa, K / units.GPa

    def get_bulk_modulus(self, average='voigt', save_at='.'):
        """

        Parameters
        ----------
        average : str
            Type of average for bulk modulus: 'voigt', 'reuss', 'vrh'

        Returns
        -------
        B : dictionary
            Bulk modulus average for each volume

        """
        B = {}
        for volume_str in self.C.keys():
            if save_at is not None:
                if not os.path.exists(save_at):
                    os.makedirs(save_at)
                b_name = os.path.join(save_at, 'bulk_modulus_{}{}'.format(average, volume_str))
                if os.path.isfile(b_name):
                    B[volume_str] = np.loadtxt(b_name)
                    continue
            C = self.C[volume_str]
            if average == 'voigt':
                B[volume_str] = np.sum(C[:3, :3]) / 9.
            elif average == 'reuss':
                S = np.linalg.inv(C)
                B[volume_str] = 1. / np.sum(S[:3, :3])
            elif average == 'vrh':
                S = np.linalg.inv(C)
                B[volume_str] = (np.sum(C[:3, :3]) / 9. + 1. / np.sum(S[:3, :3])) / 2.
            else:
                print('ERROR: average {} not implemented'.format(average))
                sys.exit(1)
            if save_at is not None:
                np.savetxt(b_name, np.asarray([B[volume_str]]))
        self.B = B
        return B

    def get_shear_modulus(self, average='voigt', save_at='.'):
        """

        Parameters
        ----------
        average : str
            Type of average for shear modulus: 'voigt', 'reuss', 'vrh'

        Returns
        -------
        G : dictionary
            Shear modulus average for each volume

        """
        G = {}
        for volume_str in self.C.keys():
            if save_at is not None:
                if not os.path.exists(save_at):
                    os.makedirs(save_at)
                g_name = os.path.join(save_at, 'shear_modulus_{}{}'.format(average, volume_str))
                if os.path.isfile(g_name):
                    G[volume_str] = np.loadtxt(g_name)
                    continue
            C = self.C[volume_str]
            if average == 'voigt':
                G[volume_str] = (C[0, 0] + C[1, 1] + C[2, 2]
                                 - (C[0, 1] + C[1, 2] + C[0, 2])
                                 + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.
            elif average == 'reuss':
                S = np.linalg.inv(C)
                G[volume_str] = 15. / (4 * (S[0, 0] + S[1, 1] + S[2, 2])
                                       - 4 * (S[0, 1] + S[1, 2] + S[0, 2])
                                       + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
            elif average == 'vrh':
                S = np.linalg.inv(C)
                G_v = (C[0, 0] + C[1, 1] + C[2, 2]
                       - (C[0, 1] + C[1, 2] + C[0, 2])
                       + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.
                G_r = 15. / (4 * (S[0, 0] + S[1, 1] + S[2, 2])
                             - 4 * (S[0, 1] + S[1, 2] + S[0, 2])
                             + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
                G[volume_str] = (G_v + G_r) / 2.
            else:
                print('ERROR: average {} not implemented'.format(average))
                sys.exit(1)
            if save_at is not None:
                np.savetxt(g_name, np.asarray([G[volume_str]]))
        self.G = G
        return G

    def get_universal_anisotropy(self, save_at='.'):
        """Get universal anisotropy as defined in [1]_
        
        Returns
        -------
        A_u : dictionary
            Universal anisotropy for each volume
            
        References
        ----------
        .. [1] S. I. Ranganathan and M. Ostoja-Starzewski, *Universal Elastic Anisotropy Index*.
        Physical Review Letters, 101, 055504 (2008). http://dx.doi.org/10.1103/PhysRevLett.101.055504
        
        """
        B_v = self.get_bulk_modulus(average='voigt', save_at=save_at)
        B_r = self.get_bulk_modulus(average='reuss', save_at=save_at)
        G_v = self.get_shear_modulus(average='voigt', save_at=save_at)
        G_r = self.get_shear_modulus(average='reuss', save_at=save_at)

        A_u = {}
        for volume_str in self.C.keys():
            if save_at is not None:
                au_name = os.path.join(save_at, 'universal_anisotropy{}'.format(volume_str))
                if os.path.isfile(au_name):
                    A_u[volume_str] = np.loadtxt(au_name)
                    continue
            A_u[volume_str] = 5. * (G_v[volume_str] / G_r[volume_str]) + B_v[volume_str] / B_r[volume_str] - 6.
            if save_at is not None:
                np.savetxt(au_name, np.asarray([A_u[volume_str]]))
        self.A_u = A_u
        return A_u

    def get_bulk_modulus_samples(self, samples=1, average='voigt'):
        """
        
        Parameters
        ----------
        samples : int
            Number of samples of the bulk modulus
        average : str
            Type of average for bulk modulus: 'voigt', 'reuss', 'vgh'

        Returns
        -------
        Bs : 2-D np.ndarray
            Array with samples of the selected bulk modulus average for each volume

        """
        C_samples = samples
        Vs = np.empty(self.n_volumes)
        Bs = np.empty((self.n_volumes, C_samples))
        for ivol in range(self.n_volumes):
            volume_str = '{}'.format(self.delta_v * ivol)
            V = np.linalg.det(self.structure.atoms.cell)
            C = self.C[volume_str]
            Cerr = self.C_err[volume_str]
            Vs[ivol] = V * (1 + self.delta_v * ivol) ** 3
            csamples = np.random.normal(C, Cerr, size=C_samples)
            for i, csample in enumerate(csamples):
                if average == 'voigt':
                    Bs[ivol, i] = np.sum(csample[:3, :3]) / 9.
                elif average == 'reuss':
                    ssample = np.linalg.inv(csample)
                    Bs[ivol, i] = 1. / np.sum(ssample[:3, :3])
                elif average == 'vrh':
                    ssample = np.linalg.inv(csample)
                    Bs[ivol, i] = (np.sum(csample[:3, :3]) / 9. + 1. / np.sum(ssample[:3, :3])) / 2.
        return Bs

    def get_temperature(self, samples=None):
        V0s, T0s = get_V_T(samples)
        n_samples = T0s.shape[1]
        T = {}
        # Volumes to interpolate at
        Vs = np.empty(self.n_volumes)
        for ivol in range(self.n_volumes):
            volume_str = '{}'.format(self.delta_v * ivol)
            T[volume_str] = np.empty(n_samples)
            V = np.linalg.det(self.structure.atoms.cell)
            Vs[ivol] = V * (1 + self.delta_v * ivol) ** 3
        # For each V-T sample, interpolate at Vs
        for i in range(n_samples):
            lT0 = T0s[:, i]
            lV0 = V0s[:, i]
            for ivol in range(self.n_volumes):
                volume_str = '{}'.format(self.delta_v * ivol)
                T[volume_str][i] = np.interp(Vs[ivol], lV0, lT0)
        self.T = T

        # print(Vs, y_t)
