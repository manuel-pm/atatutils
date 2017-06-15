from __future__ import print_function

import collections
import os

import matplotlib.pyplot as plt
import numpy as np

from ase import units
from ase.dft import monkhorst_pack
from ase.phonons import Phonons
from ase.thermochemistry import CrystalThermo

from atatutils.str2emt import ATAT2EMT

import BayesianLinearRegression.myBayesianLinearRegression as BLR
import BayesianLinearRegression.myExpansionBasis as EB

from utils.finite_differences import FiniteDifference as FD


class AP2EOS(object):
    """Adapted Polynomial (order 2) Equation of state as described in [1]_

    References
    ----------
    .. [1] W. B. Holzapfel, *Equations of state for solids under strong compression*,
        High Pressure Research, 16 (2), 81-126. http://dx.doi.org/10.1080/08957959808200283

    """
    def __init__(self):
        self.V0_mean = []
        self.E0_mean = []
        self.B0_mean = []
        self.B1_mean = []

        self.V0s = []
        self.E0s = []
        self.B0s = []
        self.B1s = []

    def clean(self):
        """Reset all model fits.

        """
        self.V0_mean = []
        self.E0_mean = []
        self.B0_mean = []
        self.B1_mean = []

        self.V0s = []
        self.E0s = []
        self.B0s = []
        self.B1s = []

    def fit(self, volume, energy, beta=1e7, samples=1):
        """Fit the SJEOS to the given volume energy data.

        Parameters
        ----------
        volume : np.ndarray
            Volumes to use for the fit.
        energy : np.ndarray
            Energies to use for the fit.
        beta :
            Precision of the energy data.
        samples :
            Number of samples from the fitted model.

        """
        pass

    def get_bulk_modulus(self, i=None, mean=False):
        """Return the bulk modulus.

        Parameters
        ----------
        i : int > 0
            Sample for which the bulk modulus is returned.
        mean : bool
            Whether to return the bulk modulus for the mean model.

        Returns
        -------
        B0s : (list of) float
            Bulk modulus for the selected models.

        """
        if i is not None:
            if mean:
                return self.B0_mean[i]
            return self.B0s[i]
        if mean:
            return self.B0_mean
        return self.B0s

    def get_bulk_modulus_derivative(self, i=None, mean=False):
        """Return the bulk modulus derivative.

        Parameters
        ----------
        i : int > 0
            Sample for which the bulk modulus derivative is returned.
        mean : bool
            Whether to return the bulk modulus derivative for the mean model.

        Returns
        -------
        B0s : (list of) float
            Bulk modulus derivative for the selected models.

        """
        if i is not None:
            if mean:
                return self.B1_mean[i]
            return self.B1s[i]
        if mean:
            return self.B1_mean
        return self.B1s

    def get_cohesive_energy(self, i=None, mean=False):
        """Return the cohesive energy.

        Parameters
        ----------
        i : int > 0
            Sample for which the cohesive energy is returned.
        mean : bool
            Whether to return the cohesive energy for the mean model.

        Returns
        -------
        B0s : (list of) float
            Cohesive energy for the selected models.

        """
        if i is not None:
            if mean:
                return self.E0_mean[i]
            return self.E0s[i]
        if mean:
            return self.E0_mean
        return self.E0s

    def get_equilibrium_volume(self, i=None, mean=False):
        """Return the equilibrium volume.

        Parameters
        ----------
        i : int > 0
            Sample for which the equilibrium volume is returned.
        mean : bool
            Whether to return the equilibrium volume for the mean model.

        Returns
        -------
        B0s : (list of) float
            Equilibrium volume for the selected models.

        """
        if i is not None:
            if mean:
                return self.V0_mean[i]
            return self.V0s[i]
        if mean:
            return self.V0_mean
        return self.V0s

    def plot(self, block=False):
        """Plot the last fitted SJEOS.

        Parameters
        ----------
        block : bool
            Whether the plot is blocking.

        """
        pass


class SJEOS(object):
    """Stabilized Jellium Equation of State as described in [1]_

    Attributes
    ----------
    sjeos_m : np.ndarray
        Coefficients to transform between model and physical parameters.
    V0_mean : list of float
        Equilibrium volume from the mean model.
    E0_mean : list of float
        Cohesive energy from the mean model.
    B0_mean : list of float
        Bulk modulus from the mean model.
    B1_mean : list of float
        Pressure derivative of the bulk modulus from the mean model.
    V0s : list of list of float
        Samples of the equilibrium volume.
    E0s : list of list of float
        Samples of the cohesive energy.
    B0s : list of list of float
        Samples of the bulk modulus.
    B1s : list of list of float
        Samples of the pressure derivative of the bulk modulus.
    blr : myBayesianLinearRegression
        Bayesian lineal regression object for the fit.
    fit_volumes : list of float
        Last used volumes for the fit.
    fit_energy : list of float
        Last used energies for the fit.

    References
    ----------
    .. [1] A. B. Alchagirov, J. P. Perdew, J. C. Boettger, R. C. Albers, and C. Fiolhais,
        *Energy and pressure versus volume: Equations of state motivated by the stabilized jellium model*.
        Physical Review B, 63, 224115 (2001). http://dx.doi.org/10.1103/PhysRevB.63.224115

    """
    sjeos_m = np.array([[1, 1, 1, 1],
                        [3, 2, 1, 0],
                        [18, 10, 4, 0],
                        [108, 50, 16, 0]
                        ])

    def __init__(self):
        self.V0_mean = []
        self.E0_mean = []
        self.B0_mean = []
        self.B1_mean = []

        self.V0s = []
        self.E0s = []
        self.B0s = []
        self.B1s = []
        self.blr = BLR.BLR()

    def clean(self):
        """Reset all model fits.

        """
        self.V0_mean = []
        self.E0_mean = []
        self.B0_mean = []
        self.B1_mean = []

        self.V0s = []
        self.E0s = []
        self.B0s = []
        self.B1s = []

    def fit(self, volume, energy, beta=1e7, samples=1):
        """Fit the SJEOS to the given volume energy data.

        Parameters
        ----------
        volume : np.ndarray
            Volumes to use for the fit.
        energy : np.ndarray
            Energies to use for the fit.
        beta :
            Precision of the energy data.
        samples :
            Number of samples from the fitted model.

        """
        self.blr.regression(energy.reshape(len(energy), 1),
                            X=volume.reshape(len(volume), 1)**(1./3),
                            basis=EB.Basis('inverse_monomial', 4),
                            alpha_mode='scalar', beta=beta)
        self.fit_volumes = volume
        self.fit_energy = energy
        properties = self._fit_to_properties(self.blr.m.reshape(4))
        if properties is not None:
            V0, E0, B0, B1 = properties
            self.V0_mean.append(V0)
            self.E0_mean.append(E0)
            self.B0_mean.append(B0)
            self.B1_mean.append(B1)
        else:
            print('ERROR fitting SJEOS:\nVolumes: {}\nEnergies: {}'.format(volume, energy))
            print('Mean coefficients: {}'.format(self.blr.m.reshape(4)))
            self.plot(True)

        self.V0s.append([])
        self.E0s.append([])
        self.B0s.append([])
        self.B1s.append([])

        valid = 0
        while valid < samples:
            rc = np.random.multivariate_normal(self.blr.m.reshape(4),
                                               self.blr.SN,
                                               samples - valid)
            for i in range(samples - valid):
                rproperties = self._fit_to_properties(rc[i])
                if rproperties is None:
                    continue
                valid += 1
                V0, E0, B0, B1 = rproperties
                self.V0s[-1].append(V0)
                self.E0s[-1].append(E0)
                self.B0s[-1].append(B0)
                self.B1s[-1].append(B1)

        assert len(self.V0s[-1]) == samples

        """
        c = rc[i, ::-1]
        t = (-c[1] + np.sqrt(c[1]*c[1] - 3*c[2]*c[0]))/(3*c[0])
        v0 = t**-3
        self.V0s[-1][i] = v0
        if t != t:
            continue
        d = np.array([v0**(-1), v0**(-2./3), v0**(-1./3), 1])
        d = np.diag(d)
        o = np.dot(np.dot(self.sjeos_m, d), c)
        self.E0s[-1][i] = -o[0]  # fit0(t)
        self.B0s[-1][i] = o[2] / v0 / 9  # (t**5 * fit2(t) / 9)
        self.B0s[-1][i] = self.B0s[-1][i] / units.kJ * 1.e24
        self.B1s[-1][i] = o[3] / o[2] / 3
        """

    def _fit_to_properties(self, fit_coeffs):
        """ Tranforms the fit coefficients into the physical properties.

        Parameters
        ----------
        fit_coeffs : np.ndarray
            Coefficients from a fit to the SJEOS.

        """
        c = fit_coeffs[::-1]
        t = (-c[1] + np.sqrt(c[1] * c[1] - 3 * c[2] * c[0])) / (3 * c[0])
        v0 = t ** -3
        V0 = v0
        if t != t:
            return None
        d = np.array([v0 ** (-1), v0 ** (-2. / 3), v0 ** (-1. / 3), 1])
        d = np.diag(d)
        o = np.dot(np.dot(self.sjeos_m, d), c)
        E0 = -o[0]  # fit0(t)
        B0 = o[2] / v0 / 9  # (t**5 * fit2(t) / 9)
        B0 = B0 / units.kJ * 1.e24
        B1 = o[3] / o[2] / 3

        return V0, E0, B0, B1

    def get_bulk_modulus(self, i=None, mean=False):
        """Return the bulk modulus.

        Parameters
        ----------
        i : int > 0
            Sample for which the bulk modulus is returned.
        mean : bool
            Whether to return the bulk modulus for the mean model.

        Returns
        -------
        B0s : (list of) float
            Bulk modulus for the selected models.

        """
        if i is not None:
            if mean:
                return self.B0_mean[i]
            return self.B0s[i]
        if mean:
            return self.B0_mean
        return self.B0s

    def get_bulk_modulus_derivative(self, i=None, mean=False):
        """Return the bulk modulus derivative.

        Parameters
        ----------
        i : int > 0
            Sample for which the bulk modulus derivative is returned.
        mean : bool
            Whether to return the bulk modulus derivative for the mean model.

        Returns
        -------
        B0s : (list of) float
            Bulk modulus derivative for the selected models.

        """
        if i is not None:
            if mean:
                return self.B1_mean[i]
            return self.B1s[i]
        if mean:
            return self.B1_mean
        return self.B1s

    def get_cohesive_energy(self, i=None, mean=False):
        """Return the cohesive energy.

        Parameters
        ----------
        i : int > 0
            Sample for which the cohesive energy is returned.
        mean : bool
            Whether to return the cohesive energy for the mean model.

        Returns
        -------
        B0s : (list of) float
            Cohesive energy for the selected models.

        """
        if i is not None:
            if mean:
                return self.E0_mean[i]
            return self.E0s[i]
        if mean:
            return self.E0_mean
        return self.E0s

    def get_equilibrium_volume(self, i=None, mean=False):
        """Return the equilibrium volume.

        Parameters
        ----------
        i : int > 0
            Sample for which the equilibrium volume is returned.
        mean : bool
            Whether to return the equilibrium volume for the mean model.

        Returns
        -------
        B0s : (list of) float
            Equilibrium volume for the selected models.

        """
        if i is not None:
            if mean:
                return self.V0_mean[i]
            return self.V0s[i]
        if mean:
            return self.V0_mean
        return self.V0s

    def plot(self, block=False):
        """Plot the last fitted SJEOS.

        Parameters
        ----------
        block : bool
            Whether the plot is blocking.

        """
        xrange = [np.min(self.fit_volumes), np.max(self.fit_volumes)]
        xs = np.linspace(xrange[0], xrange[1], 50)
        ys, stdys = self.blr.eval_regression(xs[:, np.newaxis]**(1./3))
        plt.figure()
        plt.fill_between(xs,
                         ys.ravel() - 1.96 * stdys.ravel(),
                         ys.ravel() + 1.96 * stdys.ravel(),
                         alpha=0.2)
        plt.scatter(self.fit_volumes, self.fit_energy)
        plt.xlabel(r'Volume [$\AA^3$]', fontsize=30)
        plt.ylabel('Energy [eV]', fontsize=30)
        plt.gcf().set_tight_layout(True)
        plt.plot(xs, ys.ravel())
        plt.show(block=block)


class ThermalProperties(object):
    """Class with methods to calculate thermal properties of a structure within the QHA

    Attributes
    ----------
    name : str
        Base name for the files used for the phonon calculations.
    structure : ATAT2EMT
        ATAT structure file describing the primitive cell.
    atoms : ase.Atoms
        Atoms object with the primitive cell.
    calc : ase.Calculator
        Callable returning a Calculator to use for all energy and force calculations.
    n_atoms : int
        Number of atoms in the primitive cell.
    temperature : Iterable of float
        Temperatures at which to calculate temperature dependent properties.
    strains : Iterable of float
        Strains to apply to the atoms for volume dependent properties.
    strained_thermo : list of ThermalProperties
        List with classes for the thermal properties of strained versions of the atoms object.
    sjeos : SJEOS
        Class to fit the equation of state at constant temperature.
    base_dir : str
        Base directory for the calculations. (WARNING: not fully implemented)
    verbosity : int >= 0
        Verbosity level.
    do_plotting : bool
        Determines if the plottings are activated.
    supercell_size : tuple of 3 int
        Size of the supercell to do the phonon calculations.
    thermo : ase.CrystalThermo
        Class to delegate the calculation of some thermodynamic properties.
    phonons : ase.Phonons
        Class to perform phonon calculations using the supercell approach.
    phonon_kpts_mp : (N, 3) np.ndarray
        Monkhorst-Pack k-point grid.
    phonon_energy_mp : (N,) np.ndarray
        Energies of the corresponding MP k-points.
    phonon_energy : np.ndarray
        Energies to calculate the Phonon density of states.
    phonon_dos : np.ndarray
        Phonon density of states at given energies.

    Parameters
    ----------
    atoms :
    calc :
    supercell_size :
    atat_structure :
    plot :
    verbosity :
    name :

    """
    def __init__(self, atoms=None, calc=None, supercell_size=5,
                 atat_structure=None, plot=False, verbosity=0,
                 name='thermo'):
        # relaxed structure
        self.name = name
        self.structure = None
        self.atoms = None
        self.calc = None
        self.n_atoms = 0
        self.temperature = None
        self.strains = None
        self.strained_thermo = []
        if atoms is not None and atat_structure is not None:
            print('ERROR: only atoms OR atat_structure can be specified')
            return
        if atoms is not None:
            self.atoms = atoms
            self.n_atoms = len(self.atoms)
            if self.atoms.calc is None:
                assert calc is not None
                self.atoms.set_calculator(calc())
                self.calc = calc
            else:
                self.calc = atoms.calc
        elif atat_structure is not None:
            assert calc is not None
            self.calc = calc
            self.structure = ATAT2EMT(atat_structure, calc(), to_niggli=True, verbosity=verbosity)
            self.structure.atoms.wrap()
            self.atoms = self.structure.atoms
            self.n_atoms = len(self.atoms)
        # isgn = spglib.get_symmetry_dataset(self.atoms, symprec=1e-3)['number']
        # self.symmetry = el.crystal_system(isgn)
        self.sjeos = SJEOS()

        self.base_dir = os.getcwd()
        self.verbosity = verbosity
        self.do_plotting = plot

        if isinstance(supercell_size, int):
            self.supercell_size = (supercell_size, supercell_size, supercell_size)
        else:
            assert len(supercell_size) == 3
            self.supercell_size = supercell_size

        self.get_phonons()

        self.thermo = CrystalThermo(phonon_energies=self.phonon_energy,
                                    phonon_DOS=self.phonon_dos,
                                    potentialenergy=self.atoms.get_potential_energy(),
                                    formula_units=self.n_atoms)

    def set_temperature(self, temperature, save_at='.'):
        """Set the temperature grid.

        Parameters
        ----------
        temperature : iterable of float
            Iterable containing the temperatures at which to calculate the properties.
        save_at : string
            Path (relative or absolute) in which to store the value.

        """
        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'T.dat')
            np.savetxt(save_name, temperature)
        self.temperature = temperature

    def get_phonons(self, kpts=(50, 50, 50), npts=5000):
        """Calculate the phonon spectrum and DOS.

        Parameters
        ----------
        kpts : tuple
            Number of points in each directions of the k-space grid.
        npts : int
            Number of energy points to calculate the DOS at.

        """
        self.phonons = Phonons(self.atoms, self.calc(),
                               supercell=self.supercell_size, delta=0.05,
                               name=self.name)
        self.phonons.run()
        # Read forces and assemble the dynamical matrix
        self.phonons.read(acoustic=True)
        self.phonon_kpts_mp = monkhorst_pack(kpts)
        self.phonon_energy_mp = self.phonons.band_structure(self.phonon_kpts_mp)
        self.phonon_energy, self.phonon_dos = \
            self.phonons.dos(kpts=kpts, npts=npts, delta=5e-4)

    def get_volume_phonons(self, nvolumes=5, max_strain=0.02):
        """Calculate the volume dependent phonons.

        Parameters
        ----------
        nvolumes : int > 0
            Number of volumes to calculate the phonon spectrum.
        max_strain : float > 0
            Maximum (isotropic) strain used to deform equilibrium volume.

        """
        strains = np.linspace(-max_strain, max_strain, nvolumes)
        load = False
        if self.strains is None:
            self.strains = strains
        else:
            if not (strains == self.strains).all():
                self.strains = strains
                self.strained_thermo = []
            else:
                load = True
        strain_matrices = [np.eye(3) * (1 + s) for s in self.strains]
        atoms = self.atoms
        cell = atoms.cell
        for i, s in enumerate(strain_matrices):
            satoms = atoms.copy()
            satoms.set_cell(np.dot(cell, s.T), scale_atoms=True)
            if load:
                pass
            else:
                satoms.set_calculator(None)
                sthermo = ThermalProperties(satoms, self.calc, name='thermo_{:.2f}'.format(self.strains[i]))
                self.strained_thermo.append(sthermo)

    def get_volume_energy(self, temperature=None, nvolumes=5, max_strain=0.02):
        """Return the volume dependent (Helmholtz) energy.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the volume-energy curve is calculated.
        nvolumes : int > 0
            Number of volumes to calculate the energy at.
        max_strain : float > 0
            Maximum (isotropic) strain used to deform equilibrium volume.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        volume : list of double
            Volumes at which the entropy was calculated.
        energy : list of double
            Helmholtz energy for each of the volumes.

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if isinstance(temperature, collections.Iterable):
            volume_energy = [self.get_volume_energy(T, nvolumes, max_strain) for T in temperature]
            return volume_energy

        self.get_volume_phonons(nvolumes, max_strain)

        energy = []
        volume = []
        for sthermo in self.strained_thermo:
            energy.append(sthermo.get_helmholtz_energy(temperature, save_at=None))
            volume.append(sthermo.atoms.get_volume())

        return volume, energy

    def get_volume_entropy(self, temperature=None, nvolumes=5, max_strain=0.02):
        """Return the volume dependent entropy.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the volume-entropy curve is calculated.
        nvolumes : int > 0
            Number of volumes to calculate the entropy at.
        max_strain : float > 0
            Maximum (isotropic) strain used to deform equilibrium volume.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        volume : list of double
            Volumes at which the entropy was calculated.
        entropy : list of double
            Entropy for each of the volumes.

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if isinstance(temperature, collections.Iterable):
            volume_entropy = [self.get_volume_entropy(T, nvolumes, max_strain) for T in temperature]
            return volume_entropy

        self.get_volume_phonons(nvolumes, max_strain)

        entropy = []
        volume = []
        for sthermo in self.strained_thermo:
            entropy.append(sthermo.get_entropy(temperature, save_at=None))
            volume.append(sthermo.atoms.get_volume())

        return volume, entropy

    def get_entropy(self, temperature=None, save_at='.'):
        """Return entropy per atom in eV / atom.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the Helmholtz energy is calculated.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        entropy : float
            Entropy in eV / atom

        Notes
        -----
        To convert to SI units, divide by units.J.
        At the moment only vibrational entropy is included. Electronic entropy can
        be included if the calculator provides the electronic DOS.

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'S.dat')

        if isinstance(temperature, collections.Iterable):
            vib_entropy = [self.get_entropy(T, save_at=None) for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, vib_entropy)
            return np.array(vib_entropy)
        if temperature == 0.:
            if save_at is not None:
                np.savetxt(save_name, np.asarray([0.]))
            return 0.

        vib_entropy = self.thermo.get_entropy(temperature, self.verbosity)
        if save_at is not None:
            np.savetxt(save_name, vib_entropy)
        return vib_entropy

    def get_helmholtz_energy(self, temperature=None, save_at='.'):
        """Return Helmholtz energy per atom in eV / atom.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the Helmholtz energy is calculated.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        helmholtz_energy : float
            Helmholtz energy in eV / atom

        Notes
        -----
        To convert to SI units, divide by units.J.

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'F.dat')

        if isinstance(temperature, collections.Iterable):
            helmholtz_energy = [self.get_helmholtz_energy(T, save_at=None) for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, helmholtz_energy)
            return np.array(helmholtz_energy)
        if temperature == 0.:
            helmholtz_energy = self.get_zero_point_energy() + self.thermo.potentialenergy
            if save_at is not None:
                np.savetxt(save_name, helmholtz_energy)
            return helmholtz_energy

        helmholtz_energy = self.thermo.get_helmholtz_energy(temperature, self.verbosity)
        if save_at is not None:
            np.savetxt(save_name, helmholtz_energy)
        return helmholtz_energy

    def get_internal_energy(self, temperature=None, save_at='.'):
        """Return internal energy per atom in eV / atom.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the internal energy is calculated.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        internal_energy : float
            Internal energy in eV / atom

        Notes
        -----
        To convert to SI units, divide by units.J.

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'U.dat')

        if isinstance(temperature, collections.Iterable):
            internal_energy = [self.get_internal_energy(T, save_at=None) for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, internal_energy)
            return np.array(internal_energy)
        if temperature == 0.:
            internal_energy = self.get_zero_point_energy() + self.thermo.potentialenergy
            if save_at is not None:
                np.savetxt(save_name, internal_energy)
            return internal_energy

        internal_energy = self.thermo.get_internal_energy(temperature, self.verbosity)

        if save_at is not None:
            np.savetxt(save_name, internal_energy)

        return internal_energy

    def get_zero_point_energy(self):
        """Return the Zero Point Energy in eV / atom.

        Returns
        -------
        zpe: float
            Zero point energy in eV / atom.

        """
        zpe_list = self.phonon_energy / 2.
        zpe = np.trapz(zpe_list * self.phonon_dos, self.phonon_energy) / self.n_atoms
        return zpe

    def get_specific_heat(self, temperature=None, save_at='.'):
        """Return heat capacity per atom in eV / atom K.
        
        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the specific heat is calculated.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        C_V : float
            Specific heat in eV / atom K
            
        Notes
        -----
        To convert to SI units, multiply by (units.mol / units.J).

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'Cv.dat')

        if isinstance(temperature, collections.Iterable):
            C_V = [self.get_specific_heat(T, save_at=None) for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, C_V)
            return np.array(C_V)

        if temperature == 0.:
            if save_at is not None:
                np.savetxt(save_name, np.asarray([0.]))
            return 0.

        if self.phonon_energy[0] == 0.:
            self.phonon_energy = np.delete(self.phonon_energy, 0)
            self.phonon_dos = np.delete(self.phonon_dos, 0)
        i2kT = 1. / (2. * units.kB * temperature)
        arg = self.phonon_energy * i2kT
        C_v = units.kB * arg**2 / np.sinh(arg)**2
        C_V = np.trapz(C_v * self.phonon_dos, self.phonon_energy) / self.n_atoms
        if save_at is not None:
            np.savetxt(save_name, np.asarray([C_V]))
        return C_V

    def get_thermal_expansion(self, temperature=None, exp_norm_temp=None,
                              nvolumes=5, max_strain=0.02,
                              ntemperatures=5, delta_t=1.,
                              save_at='.'):
        """Return the isotropic volumetric thermal expansion in K^-1.
        
        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the expansion coefficient is calculated.
        exp_norm_temp : float > 0
            Temperature for the normalization of the thermal expansion (usually to compare with experiment).
        nvolumes : int > 0
            Number of volumes to fit the equation of state to extract equilibrium volumes.
        max_strain : float > 0
            Maximum strain used to fit the equation of state to extract equilibrium volumes.
        ntemperatures : int > 0
            Number of temperatures to approximate the temperature derivative of the volume.
        delta_t : float >0
            Temperature step to approximate the temperature derivative of the volume.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
            : double
            Isotropic volumetric thermal expansion in K^-1

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'thermal_expansion.dat')

        if isinstance(temperature, collections.Iterable):
            alpha_v = [self.get_thermal_expansion(T, exp_norm_temp,
                                                  nvolumes, max_strain,
                                                  ntemperatures, delta_t,
                                                  save_at=None)
                       for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, alpha_v)
            return np.array(alpha_v)

        max_delta_t = (ntemperatures - 1) * delta_t
        if temperature - max_delta_t / 2. > 0.:
            temperatures = np.linspace(temperature - max_delta_t / 2., temperature + max_delta_t / 2., ntemperatures)
            t0 = (ntemperatures - 1) / 2
            mode = 'c'
        else:
            ntemperatures = (ntemperatures + 2) / 2
            temperatures = np.linspace(temperature, temperature + max_delta_t / 2., ntemperatures)
            t0 = 0
            mode = 'f'
        print(temperatures, ntemperatures)
        # 1.- Get V-F points
        Vs = []
        Fs = []
        for T in temperatures:
            V, F = self.get_volume_energy(T, nvolumes, max_strain)
            Vs.append(V)
            Fs.append(F)
        # 2.- Fit EOS to V-F points for each T
        self.sjeos.clean()
        for i in range(ntemperatures):
            V = np.asarray(Vs[i])
            F = np.asarray(Fs[i])
            self.sjeos.fit(V, F)
        # 3.- Numerical derivative dV/dT
        V0s = self.sjeos.get_equilibrium_volume(mean=True)
        fd = FD(temperatures)
        dV_dT = fd.derivative(1, t0, V0s, acc_order=2, mode=mode)
        if self.do_plotting:
            plt.plot(temperatures, V0s)
        # 4a.- Normalize by volume at temperature (same as derivative)
        if exp_norm_temp is None:
            return dV_dT / V0s[(ntemperatures - 1) / 2]
        # 4b.- Normalize by volume at some give reference temperature (different from derivative)
        V, F = self.get_volume_energy(exp_norm_temp, nvolumes, max_strain)
        self.sjeos.clean()
        self.sjeos.fit(np.asarray(V), np.asarray(F))
        V_norm = self.sjeos.get_equilibrium_volume(mean=True)
        alpha_v = dV_dT / V_norm
        if save_at is not None:
            np.savetxt(save_name, alpha_v)
        return alpha_v

    def get_gruneisen(self, temperature=None, nvolumes=5, max_strain=0.02, save_at='.'):
        r"""Return the Gr\"uneisen parameter.

        Parameters
        ----------
        temperature : float > 0
            Temeprature at which the expansion coefficient is calculated.
        nvolumes : int > 0
            Number of volumes to fit the equation of state to extract equilibrium volumes.
        max_strain : float > 0
            Maximum strain used to fit the equation of state to extract equilibrium volumes.
        save_at : string
            Path (relative or absolute) in which to store the value.

        Returns
        -------
        gruneisen : double
            Gr\"uneisen parameter.

        Notes
        -----
        The Gr\"uneisen parameter is calculated as

        .. math ::
            \gamma=\frac{C_v}{V}\left.\frac{\partial S}{\partial V}\right|_T

        """
        if temperature is None and self.temperature is None:
            print('ERROR. You nee to specify a temperature for the calculations.')
            return
        elif temperature is None:
            temperature = self.temperature

        if save_at is not None:
            if not os.path.exists(save_at):
                os.makedirs(save_at)
            save_name = os.path.join(save_at, 'gruneisen.dat')

        if isinstance(temperature, collections.Iterable):
            gruneisen = [self.get_gruneisen(T, nvolumes, max_strain, save_at=None) for T in temperature]
            if save_at is not None:
                np.savetxt(save_name, gruneisen)
            return gruneisen

        self.get_volume_phonons(nvolumes, max_strain)
        C_V = self.get_specific_heat(temperature)

        V, F = self.get_volume_energy(temperature, nvolumes, max_strain)
        self.sjeos.clean()
        self.sjeos.fit(np.asarray(V), np.asarray(F))
        V_0 = self.sjeos.get_equilibrium_volume(mean=True)[0]

        V, S = self.get_volume_entropy(temperature, nvolumes, max_strain)
        fd = FD(V)
        dS_dV = fd.derivative(1, nvolumes / 2, S, acc_order=2, mode='c')
        gruneisen = dS_dV * V_0 / C_V

        """
        phonon_energy_mp = []
        volumes = []
        hw_V = [[] for i in range(len(self.phonon_energy_mp.ravel()))]
        for sthermo in self.strained_thermo:
            volumes.append(sthermo.atoms.get_volume())
            phonon_energy_mp.append(sthermo.phonon_energy_mp.ravel())
            for j, hw in enumerate(phonon_energy_mp[-1]):
                hw_V[j].append(hw)

        fd = FD(volumes)
        gruneisen_i = np.empty_like(phonon_energy_mp[0])
        for i, hw in enumerate(hw_V):
            dhw_dV = fd.derivative(1, nvolumes/2, hw, acc_order=2, mode='c')
            # print(dhw_dV, hw[nvolumes/2], dhw_dV * volumes[nvolumes/2] / hw[nvolumes/2])
            gruneisen_i[i]\
                = - dhw_dV * volumes[nvolumes/2] / hw[nvolumes/2]

        self.hw_V = hw_V
        self.volumes = volumes

        i2kT = 1. / (2. * units.kB * temperature)
        arg = phonon_energy_mp[nvolumes/2] * i2kT
        C_v = units.kB * arg ** 2 / np.sinh(arg) ** 2

        # print(C_V, C_v, gruneisen_i, volumes)
        # gruneisen = np.trapz(C_v * gruneisen_i * self.phonon_dos, self.phonon_energy) / C_V
        gruneisen = np.sum(C_v * gruneisen_i) / np.sum(C_v)

        print(gruneisen, gruneisen_S)
        plt.scatter(temperature, gruneisen, color='r')
        plt.scatter(temperature, gruneisen_S, color='g')
        """
        if save_at is not None:
            np.savetxt(save_name, gruneisen)

        return gruneisen
