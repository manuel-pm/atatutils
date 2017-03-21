"""
 Utility class and functions to read/write ATAT structure files and setup
 an Atoms object that can be used to run a DFT simulation of the structure.
 The class ATAT2GPAW assumes that the k-points for the calculator can be set as
  self.calc.set(kpts={'size': kpts, 'gamma': True})
 where kpts is a tuple with three elements containing the number of k points
 in the x, y and z directions.

 Example usage:

    from gpaw import GPAW, PW, Mixer, FermiDirac
    from gpaw.eigensolvers import CG
    from gpaw.poisson import PoissonSolver

    xcf = 'PBE'
    # xcf = 'mBEEF-WCPM'
    fmm = True
    calc = GPAW(mode=PW(400),
                h=0.10,
                xc=xcf,
                occupations=FermiDirac(0.05, fixmagmom=fmm),
                eigensolver=CG(niter=5, rtol=0.15),
                poissonsolver=PoissonSolver(nn=3, relax='J', eps=1e-12),
                convergence={'energy': 0.0005,
                             'bands': 'all',
                             'density': 1.e-4,
                             'eigenstates': 1.e-4
                             },
                mixer=Mixer(0.055, 3, 50))

    cv = ATAT2GPAW('str.out', calc)
    cv.atoms.get_potential_energy()
    opt = BFGS(cv.atoms)
    opt.run(0.05)

    ofile = open(os.path.join('.', 'energy'), 'wb')

    ofile.write(str(cv.atoms.get_potential_energy()))
    ofile.close()
"""

from __future__ import print_function

import copy
import math
import os
from shutil import copyfile

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Parameters
from ase.constraints import UnitCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE as ASEFIRE
from ase.optimize.lbfgs import LBFGS as ASELBFGS
from ase.parallel import barrier, parprint, paropen, rank
from ase.optimize.precon import Exp, PreconLBFGS, PreconFIRE
# import ase.utils.geometry as ase_geometry
import ase.build as ase_geometry


def read_lattice_file(filename='lat.in', pbc=(1, 1, 1), verbosity=0,
                    minimize_tilt=False, niggli_reduce=False):
    """ Reads an ATAT lattice file (lat.in) and returns the cell, positions and
    atom types """
    if verbosity > 1:
        parprint("read_lattice_file called with options:\n\t filename: {}\n"
                 "\t pbc: {} \n\t verbosity: {} \n\t minimize_tilt: {}\n"
                 "\t niggli_reduce: {}".format(filename, pbc, verbosity,
                                               minimize_tilt, niggli_reduce))
    ifile = open(filename, 'rb')
    # Read coordinate system
    cs = np.zeros((3, 3), dtype=float)
    l1 = ifile.readline()
    items = [float(i) for i in l1.split()]
    if len(items) == 3:
        cs[0, :] = np.array(items)
        cs[1, :] = np.array(ifile.readline().split())
        cs[2, :] = np.array(ifile.readline().split())
    else:
        #print("WARNING: [a, b, c, alpha, beta, gamma] format" +
        #      " not well tested")
        a, b, c, alpha, beta, gamma = items
        alpha, beta, gamma = [angle*np.pi/180.
                              for angle in [alpha, beta, gamma]]
        (ca, sa) = (math.cos(alpha), math.sin(alpha))
        (cb, sb) = (math.cos(beta),  math.sin(beta))
        (cg, sg) = (math.cos(gamma), math.sin(gamma))
        # Vunit is a volume of unit cell with a=b=c=1
        Vunit = math.sqrt(1.0 + 2.0*ca*cb*cg - ca*ca - cb*cb - cg*cg)
        # from the reciprocal lattice
        ar = sa/(a*Vunit)
        cgr = (ca*cb - cg)/(sa*sb)
        sgr = math.sqrt(1.0 - cgr**2)
        cs[0, :] = np.array([1.0/ar, -cgr/sgr/ar, cb*a])
        cs[1, :] = np.array([0.0, b*sa, b*ca])
        cs[2, :] = np.array([0.0, 0.0, c])

    if verbosity > 0:
        parprint("Coordinate system:\n {}".format(cs))

    # Read unit cell
    cell = np.zeros((3, 3), dtype=float)
    for i in range(3):
        cell[i, :] = np.array(ifile.readline().split())
    if verbosity > 1:
        parprint("Initial cell:\n {}".format(cell))

    cell = np.dot(cell, cs)
    if verbosity > 0:
        parprint("Cell:\n {}".format(cell))
        parprint("Cell volume: {}".format(np.linalg.det(cell)))

    # Read atoms positions
    rest = ifile.readlines()
    ifile.close()
    positions = []
    atom_symbols = []
    for line in rest:
        split = line.split()
        positions.append(np.dot(np.array(split[:3], dtype=float), cs))
        atom_symbols.append(split[3])

    if verbosity > 0:
        parprint("Positions:\n {}".format(positions))

    return cell, positions, atom_symbols


def write_atat_input(atoms, filename='str_last.out'):
    """ Writes the ATAT structure file (str.out) corresponding to the given
    ASE Atoms object """
    with paropen(os.path.join('.', filename), 'wb') as ofile:
        ofile.write('1.0 1.0 1.0 90. 90. 90.\n')
        for i in range(3):
            ofile.write(' '.join("{:.12f}".format(cell) for
                                 cell in atoms.get_cell()[i]))
            ofile.write('\n')
        for i, atom in enumerate(atoms.get_chemical_symbols()):
            ofile.write(' '.join("{:.12f}".format(p) for
                                 p in atoms.get_positions()[i]) + ' ' + atom)
            ofile.write('\n')


def read_atat_input(filename='str.out', pbc=(1, 1, 1), verbosity=0,
                    minimize_tilt=False, niggli_reduce=False):
    """ Reads an ATAT structure file (str.out) and returns the
    corresponding ASE Atoms object """
    if verbosity > 1:
        parprint("read_atat_input called with options:\n\t filename: {}\n"
                 "\t pbc: {} \n\t verbosity: {} \n\t minimize_tilt: {}\n"
                 "\t niggli_reduce: {}".format(filename, pbc, verbosity,
                                               minimize_tilt, niggli_reduce))
    ifile = open(filename, 'rb')
    # Read coordinate system
    cs = np.zeros((3, 3), dtype=float)
    l1 = ifile.readline()
    items = [float(i) for i in l1.split()]
    if len(items) == 3:
        cs[0, :] = np.array(items)
        cs[1, :] = np.array(ifile.readline().split())
        cs[2, :] = np.array(ifile.readline().split())
    else:
        #print("WARNING: [a, b, c, alpha, beta, gamma] format" +
        #      " not well tested")
        a, b, c, alpha, beta, gamma = items
        alpha, beta, gamma = [angle*np.pi/180.
                              for angle in [alpha, beta, gamma]]
        (ca, sa) = (math.cos(alpha), math.sin(alpha))
        (cb, sb) = (math.cos(beta),  math.sin(beta))
        (cg, sg) = (math.cos(gamma), math.sin(gamma))
        # Vunit is a volume of unit cell with a=b=c=1
        Vunit = math.sqrt(1.0 + 2.0*ca*cb*cg - ca*ca - cb*cb - cg*cg)
        # from the reciprocal lattice
        ar = sa/(a*Vunit)
        cgr = (ca*cb - cg)/(sa*sb)
        sgr = math.sqrt(1.0 - cgr**2)
        cs[0, :] = np.array([1.0/ar, -cgr/sgr/ar, cb*a])
        cs[1, :] = np.array([0.0, b*sa, b*ca])
        cs[2, :] = np.array([0.0, 0.0, c])

    if verbosity > 0:
        parprint("Coordinate system:\n {}".format(cs))

    # Read unit cell
    cell = np.zeros((3, 3), dtype=float)
    for i in range(3):
        cell[i, :] = np.array(ifile.readline().split())
    if verbosity > 1:
        parprint("Initial cell:\n {}".format(cell))

    cell = np.dot(cell, cs)
    if verbosity > 0:
        parprint("Cell:\n {}".format(cell))
        parprint("Cell volume: {}".format(np.linalg.det(cell)))

    # Read atoms positions
    rest = ifile.readlines()
    ifile.close()
    positions = []
    atom_symbols = []
    for line in rest:
        split = line.split()
        positions.append(np.dot(np.array(split[:3], dtype=float), cs))
        atom_symbols.append(split[3])

    if verbosity > 0:
        parprint("Positions:\n {}".format(positions))

    # Create atoms object
    atoms = Atoms(symbols=atom_symbols,
                  positions=positions,
                  cell=cell,
                  pbc=pbc)

    # Modify cell to the maximally-reduced Niggli unit cell or minimize tilt
    #    angle between cell axes. Niggli takes precedence.
    if niggli_reduce:
        ase_geometry.niggli_reduce(atoms)
        if verbosity > 0:
            parprint("Niggli cell:\n {}".format(atoms.get_cell()))
            parprint("N. cell volume: {}".format(np.linalg.det(atoms.get_cell())))
        write_atat_input(atoms, filename='str_niggli.out')
        barrier()
        if rank == 0:
            if not os.path.isfile('str_atat.out'):
                copyfile('str.out', 'str_atat.out')
            copyfile('str_niggli.out', 'str.out')
        barrier()
    elif minimize_tilt:
        ase_geometry.minimize_tilt(atoms)
        if verbosity > 0:
            parprint("Minimum tilt cell:\n {}".format(atoms.cell))
            parprint("M. T. cell volume: {}".format(np.linalg.det(atoms.cell)))
        write_atat_input(atoms, filename='str_mintilt.out')
        barrier()
        if rank == 0:
            if not os.path.isfile('str_atat.out'):
                copyfile('str.out', 'str_atat.out')
            copyfile('str_mintilt.out', 'str.out')
        barrier()

    return atoms


class ATAT2GPAW:
    default_parameters = {'to_niggli': False,
                          'to_min_tilt': False,
                          'use_precon': True,
                          'use_armijo': True,
                          'optimizer': 'LBFGS'}
    'Default parameters'

    def __init__(self, structure=None, calc=None, verbosity=0, **kwargs):
        """ Initialize variables and set structure and calculator if given """
        self.atoms = None
        self.calc = calc
        self.verbosity = verbosity
        self.parameters = self.get_default_parameters()
        self.set(**kwargs)
        if structure is not None:
            self.set_atoms(structure)
            if calc is not None:
                self.set_calculator(self.calc)

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not (value == oldvalue):
                if isinstance(oldvalue, dict):
                    # Special treatment for dictionary parameters:
                    for name in value:
                        if name not in oldvalue:
                            raise KeyError(
                                'Unknown subparameter "{}" in '
                                'dictionary parameter "{}"'.format(name, key))
                    oldvalue.update(value)
                    value = oldvalue
                changed_parameters[key] = value
                self.parameters[key] = value

        return changed_parameters

    def set_atoms(self, structure=None):
        """ Sets the atoms from a given ATAT structure file """
        if structure is not None:
            to_min_tilt = self.parameters.to_min_tilt
            to_niggli = self.parameters.to_niggli
            self.atoms = read_atat_input(structure,
                                         pbc=(1, 1, 1),
                                         verbosity=self.verbosity,
                                         minimize_tilt=to_min_tilt,
                                         niggli_reduce=to_niggli)
        else:
            parprint("ERROR: No ATAT structure file given")

    def set_calculator(self, calc=None, nkpts=4096):
        """ Sets the calculator and resets the k-points according to the cell
        shape """
        if calc is None and self.calc is None:
            parprint("ERROR: no calculator provided")
            return
        elif calc is not None:
            self.calc = calc
        kcell = self.atoms.get_reciprocal_cell()
        # Vcell = np.abs(np.dot(kcell[0, :],
        #                       np.cross(kcell[1, :], kcell[2, :])))
        k12 = np.cross(kcell[0, :], kcell[1, :])
        k23 = np.cross(kcell[1, :], kcell[2, :])
        k31 = np.cross(kcell[2, :], kcell[0, :])
        n1 = 1./np.linalg.norm(k23)
        n2 = 1./np.linalg.norm(k31)
        n3 = 1./np.linalg.norm(k12)
        tot_kpts = nkpts/self.atoms.get_number_of_atoms()
        av_kpts = (tot_kpts/(n1*n2*n3))**(1./3)
        kpts1 = int(round(n1*av_kpts, 0))
        kpts2 = int(round(n2*av_kpts, 0))
        kpts3 = int(round(n3*av_kpts, 0))
        kpts = (np.max((kpts1, 1)), np.max((kpts2, 1)), np.max((kpts3, 1)))
        if self.verbosity > 0:
            parprint("k-points:", kpts)
        self.calc.set(kpts={'size': kpts, 'gamma': True})
        self.atoms.set_calculator(calc)

    def get_atoms(self):
        """ Returns the member atoms object """
        return self.atoms

    def static_run(self):
        """ Runs static simulation (no position relaxation) """
        return self.atoms.get_potential_energy()

    def optimise_cell(self, fmax=0.01, use_precon=None, use_armijo=None):
        """ Relax cell to a given force/stress threshold """
        if use_precon is None:
            use_precon = self.parameters.use_precon
        if use_armijo is None:
            use_armijo = self.parameters.use_armijo
        if use_precon:
            precon = Exp(A=3, use_pyamg=False)
        else:
            precon = None
        uf = UnitCellFilter(self.atoms)
        if self.parameters.optimizer == 'BFGS':
            relax = BFGS(uf)
        elif self.parameters.optimizer == 'FIRE':
            relax = PreconFIRE(uf, precon=precon)
        elif self.parameters.optimizer == 'ase-FIRE':
            relax = ASEFIRE(uf)
        elif self.parameters.optimizer == 'LBFGS':
            relax = PreconLBFGS(uf, precon=precon, use_armijo=use_armijo)
        elif self.parameters.optimizer == 'ase-LBFGS':
            relax = ASELBFGS(uf)
        else:
            parprint("ERROR: unknown optimizer {}. "
                     "Reverting to BFGS".format(self.parameters.optimizer))
            relax = BFGS(self.atoms)
        name = self.atoms.get_chemical_formula()
        relax.attach(lambda: self.atoms.calc.write(name + '_relax.gpw',
                                                   mode='all'))
        relax.attach(lambda: write_atat_input(self.atoms, 'str_last.out'))
        relax.run(fmax=fmax, steps=100)
        if not relax.converged():
            relax = BFGS(uf)
            relax.run(fmax=fmax, steps=100)
            if not relax.converged():
                max_force = self.atoms.get_forces()
                max_force = np.sqrt((max_force**2).sum(axis=1).max())
                print('WARNING: optimisation not converged.' +
                      ' Maximum force: %.4f' % max_force)

    def optimise_positions(self, fmax=0.01, use_precon=None, use_armijo=None):
        """ Relax atoms positions with the fixed cell to a given force
        threshold """
        if use_precon is None:
            use_precon = self.parameters.use_precon
        if use_armijo is None:
            use_armijo = self.parameters.use_armijo

        if use_precon:
            precon = Exp(A=3, use_pyamg=False)
        else:
            precon = None
        if self.parameters.optimizer == 'BFGS':
            relax = BFGS(self.atoms)
        elif self.parameters.optimizer == 'FIRE':
            relax = PreconFIRE(self.atoms, precon=precon)
        elif self.parameters.optimizer == 'ase-FIRE':
            relax = ASEFIRE(self.atoms)
        elif self.parameters.optimizer == 'LBFGS':
            relax = PreconLBFGS(self.atoms, precon=precon, use_armijo=use_armijo)
        elif self.parameters.optimizer == 'ase-LBFGS':
            relax = ASELBFGS(self.atoms)
        else:
            parprint("ERROR: unknown optimizer {}. "
                     "Reverting to BFGS".format(self.parameters.optimizer))
            relax = BFGS(self.atoms)
        name = self.atoms.get_chemical_formula()
        relax.attach(lambda: self.atoms.calc.write(name + '_relax.gpw',
                                                   mode='all'))
        relax.attach(lambda: write_atat_input(self.atoms, 'str_last.out'))
        relax.run(fmax=fmax, steps=100)
        if not relax.converged():
            relax = LBFGS(self.atoms)
            relax.run(fmax=fmax, steps=100)
            if not relax.converged():
                max_force = self.atoms.get_forces()
                max_force = np.sqrt((max_force**2).sum(axis=1).max())
                print('WARNING: optimisation not converged.' +
                      ' Maximum force: %.4f' % max_force)
