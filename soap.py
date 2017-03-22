from __future__ import print_function

import copy
import itertools
import numbers
import os
import sys
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.special as sp

import GPy
from GPy.kern import Kern
from GPy import Param
from paramz.caching import Cache_this
from paramz.transformations import Logexp, Logistic

import ase
import ase.neighborlist

from atatutils.str2gpaw import ATAT2GPAW
from exp_spherical_in import exp_spherical_in, exp_spherical_in_test, cexp_spherical_in


def gregory_weights(n_nodes, h, order):
    from scipy.linalg import cholesky, pascal, toeplitz
    # http://www.colorado.edu/amath/sites/default/files/attached-files/gregory.pdf
    # Create Gregory coefficients
    r = 1./np.arange(1, order + 1)
    tp = toeplitz(r[:-1], r[0] * np.eye(1, order - 1))
    gc = np.linalg.solve(tp, r[1:])
    # create the weights
    w = h * np.ones(n_nodes)
    # Generate the Pascal Cholesky factor (up to signs).
    # http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/elmat/pascal.m
    P = np.diag((-1)**np.arange(order - 1, dtype=int))
    P[:, 0] = np.ones(order - 1)
    for j in range(1, order - 2):
        for i in range(j + 1, order - 1):
            P[i, j] = P[i - 1, j] - P[i - 1, j - 1]
    # Update the endpoint weights
    w_updates = np.sum(h * np.repeat(gc.reshape(order - 1, 1), order -1, 1) * P, axis=0)
    w[:order - 1] -= w_updates
    w[:-order:-1] -= w_updates
    return w


def gram_schmidt(vectors, inner_product):
    basis = []
    for v in vectors:
        w = v - np.sum(inner_product(v, b) * b  for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.sqrt(inner_product(w, w)))
    return np.array(basis)


def r_basis(n, delta_r, alpha, x):
    return np.exp(-alpha*(x-n*delta_r)**2)


def dr_basis_dalpha(n, delta_r, alpha, x):
    return (x - n*delta_r)**2


def phi_mat(n, delta_r, alpha, x):
    phi = np.empty((len(n), len(x)))
    for i in range(len(n)):
        for j in range(len(x)):
            phi[i, j] = r_basis(n[i], delta_r, alpha, x[j])
    return phi


def basis_overlap(nmax, rcut, alpha):
    overlap = np.empty((nmax, nmax))
    c1 = np.sqrt(2*alpha**3)/np.sqrt(128*alpha**5)
    c2 = np.sqrt(np.pi)*alpha/np.sqrt(128*alpha**5)
    for i in range(nmax):
        for j in range(i + 1):
            ri = i*(rcut/nmax)
            rj = j*(rcut/nmax)
            T1 = c1 * np.exp(-alpha*(ri**2 + rj**2)) * (ri + rj)
            T1 -= c1 * np.exp(-alpha*(2*rcut**2 + ri**2 + rj**2 - 2*rcut*(ri+rj))) * (2*rcut + ri + rj)
            T2 = c2 * np.exp(-alpha/2.*(ri-rj)**2) * (1. + alpha*(ri + rj)**2) * \
                (sp.erf(np.sqrt(alpha/2) * (2*rcut - ri - rj)) + sp.erf(np.sqrt(alpha/2) * (ri + rj)))
            # print(ri, rj, T1, T2)
            overlap[i, j] = overlap[j, i] = T1 + T2
    return overlap

def sympy_basis_overlap(nmax, rcut):
    from sympy import Matrix, sqrt, pi, exp, erf
    from sympy.abc import a
    overlap = [0] * nmax
    for i in range(len(overlap)):
        overlap[i] = [0] * nmax

    c1 = sqrt(2 * a ** 3) / sqrt(128 * a ** 5)
    c2 = sqrt(pi) * a / sqrt(128 * a ** 5)
    for i in range(nmax):
        for j in range(i + 1):
            ri = i * (rcut / nmax)
            rj = j * (rcut / nmax)
            T1 = c1 * exp(-a * (ri ** 2 + rj ** 2)) * (ri + rj)
            T1 -= c1 * exp(-a * (2 * rcut ** 2 + ri ** 2 + rj ** 2 - 2 * rcut * (ri + rj))) * (
            2 * rcut + ri + rj)
            T2 = c2 * exp(-a / 2. * (ri - rj) ** 2) * (1. + a * (ri + rj) ** 2) * \
                 (erf(sqrt(a / 2) * (2 * rcut - ri - rj)) + erf(sqrt(a / 2) * (ri + rj)))
            # print(ri, rj, T1, T2)
            overlap[i][j] = overlap[j][i] = T1 + T2
    overlap = Matrix(overlap)
    return overlap

def sympy_dG_dalpha(n_max, delta_r, alpha, x, r_cut):
    from sympy.abc import a
    n = range(n_max)

    S = sympy_basis_overlap(n_max, r_cut)
    dS = np.array(S.diff().evalf(subs={a:alpha})).astype(float)
    L = scipy.linalg.cholesky(np.array(S).astype(float), lower=True)
    L_1 = np.linalg.inv(L)
    theta = np.tril(np.dot(np.dot(L_1, dS), L_1.T))
    theta[np.diag_indices(3)] *= 0.5
    dL_T = -np.dot(theta, L_1).T

    Phi = phi_mat(n, delta_r, alpha, x)
    dPhi = np.zeros_like(Phi)
    for i in range(Phi.shape[0]):
        for j in range(Phi.shape[1]):
            dPhi[i, j] = -Phi[i, j] * dr_basis_dalpha(n[j], delta_r, alpha, x[i])

    dG_dalpha = np.dot(dPhi.T, L_1.T).T + np.dot(Phi.T, dL_T)
    return dG_dalpha


def fcut(rcut, rdelta, r):
    if r <= rcut-rdelta:
        return 1.
    elif r <= rcut:
        return 0.5*(1+np.cos(np.pi*(r-rcut+rdelta)/rdelta))
    return 0.


def nearest_neighbour_distance(atoms, which=0, largest=False):
    # Neighbour list of size given by the cell. Always contains some atom
    nl = ase.neighborlist.NeighborList([np.min(np.linalg.norm(atoms.cell, axis=1))] * atoms.positions.shape[0],
                                       self_interaction=False)
    nl.update(atoms)
    if not largest:
        # Consider only atom 'which'
        ds = []
        indices, offsets = nl.get_neighbors(which)
        for i, o in zip(indices, offsets):
            ds.append(np.linalg.norm(atoms.positions[i] + np.dot(o, atoms.get_cell()) - atoms.positions[which]))
        nnd = np.min(ds)
    else:
        nnds = []
        for which in range(atoms.positions.shape[0]):
            ds = []
            indices, offsets = nl.get_neighbors(which)
            for i, o in zip(indices, offsets):
                ds.append(np.linalg.norm(atoms.positions[i] + np.dot(o, atoms.get_cell()) - atoms.positions[which]))
            nnds.append(np.min(ds))
        nnd = np.max(nnds)
    return nnd


def iv(n, x):
    try:
        return sp.spherical_in(n, x)
    except:
        x += 1.e-15
        return sp.iv(n + 0.5, x) * np.sqrt(np.pi / (2 * x))


def exp_iv(y, n, x):
    # return np.asarray(cexp_spherical_in(y, n, x))
    return exp_spherical_in(y, n, x)

def exp_iv_test(y, n, x):
    return exp_spherical_in_test(y, n, x, 1)
    #return exp_spherical_in(y, n, x)


def c_ilm(l, m, alpha, ri, thetai, phii, x, derivative=False):
    #I_01 = 4 * np.pi * np.exp(-alpha * (x*x + ri*ri)) * \
    #    iv(l, 2 * alpha * x * ri) * \
    #    np.conj(sp.sph_harm(m, l, thetai, phii))
    I_01 = 4 * np.pi * exp_iv(-alpha * (x*x + ri*ri), l, 2 * alpha * x * ri) * np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        dI_01 = I_01 * (l / alpha - (x*x + ri*ri))
        dI_01 += 8 * np.pi * x * ri * exp_iv(-alpha * (x*x + ri*ri), l + 1, 2 * alpha * x * ri) * \
                 np.conj(sp.sph_harm(m, l, thetai, phii))
        # numerical
        #delta = 0.001
        #Ip = 4 * np.pi * exp_iv(-(alpha + delta) * (x*x + ri*ri), l, 2 * (alpha + delta) * x * ri) * \
        #np.conj(sp.sph_harm(m, l, thetai, phii))
        #Im = 4 * np.pi * exp_iv(-(alpha - delta) * (x*x + ri*ri), l, 2 * (alpha - delta) * x * ri) * \
        #np.conj(sp.sph_harm(m, l, thetai, phii))
        #dI_01n = (Ip - Im)/(2*delta)
        #idx = 2
        #if I_01[idx].real > 1.:
        #    print('exp_iv: ', alpha, dI_01[idx].real, dI_01n[idx].real, Ip[idx].real, Im[idx].real, I_01[idx].real)
    # print np.conj(sp.sph_harm(m, l, thetai, phii))
    # print( 'prefactor iv = ',  -alpha * (x[-1]**2 + ri*ri), l, 2 * alpha * x[-1] * ri, np.exp(-alpha * (x[-1]**2 + ri*ri))*iv(l, 2 * alpha * x[-1] * ri), exp_iv(-alpha * (x[-1]**2 + ri*ri), l, 2 * alpha * x[-1] * ri) )
    #print(I_01[-1].real, I_01_new[-1].real, 100.*np.max((I_01.real-I_01_new.real)/(I_01.real+1.e-15)))
    if derivative:
        return I_01, dI_01
    return I_01


def c_ilm2(l, m, alpha, ar2g2, thetai, phii, arg, derivative=False):
    I_01 = 4 * np.pi * exp_iv(-ar2g2, l, arg) * np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        dI_01 = I_01 * ((l - ar2g2) / alpha)
        dI_01 += 4 * np.pi * arg / alpha * exp_iv(-ar2g2, l + 1, arg) * \
                 np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        return I_01, dI_01
    return I_01


def sum_squares_odd_integers(n):
    return n * (2 * n + 1) * (2 * n - 1) / 3


def cart2sph(coords):
    r = np.linalg.norm(coords, axis=1)
    coords_hat = np.zeros_like(coords)
    theta = np.zeros(r.shape[0])
    phi = np.zeros_like(theta)
    mask = r > 0
    coords_hat[mask] = (coords[mask].T / r[mask]).T
    theta[mask] = np.arccos(coords_hat[mask, 2])
    phi[mask] = np.arctan2(coords[mask, 1], coords[mask, 0])
    return r, theta, phi


def sph2cart(r, theta, phi):
    rsin_theta = r * np.sin(theta)
    x = rsin_theta * np.cos(phi)
    y = rsin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def dirac_delta(i, j):
    if i == j:
        return 1
    return 0


def partition1d(ndata, rank, size):
    base = ndata / size
    leftover = ndata % size
    chunksizes = np.ones(size, dtype=int) * base
    chunksizes[:leftover] += 1
    offsets = np.zeros(size, dtype=int)
    offsets[1:] = np.cumsum(chunksizes)[:-1]
    # Local update
    lchunk = [offsets[rank], offsets[rank] + chunksizes[rank]]
    return lchunk, chunksizes, offsets


def partition_ltri_rows(nrows, rank, size):
    ndata = nrows * (nrows + 1) / 2
    base = ndata / size
    offsets = np.zeros(size, dtype=int)

    for i in range(1, size):
        inside = False
        prev = offsets[i - 1]
        for j in range(prev + 1, nrows):
            if j * (j + 1) / 2 - prev * (prev + 1) / 2 > base:
                offsets[i] = j
                inside = True
                break
        if not inside:
            offsets[i] = nrows
            print('WARNING: too many processes for the size of the matrix')
    chunksizes = np.ones(size, dtype=int)
    chunksizes[:-1] = offsets[1:] - offsets[:-1]
    chunksizes[-1] = nrows - offsets[-1]
    # Local update
    lchunk = [offsets[rank], offsets[rank] + chunksizes[rank]]
    return lchunk, chunksizes, offsets


def partition_utri_rows(nrows, rank, size):
    ndata = nrows * (nrows + 1) / 2
    base = ndata / size
    offsets = np.zeros(size, dtype=int)
    for i in range(1, size):
        prev = offsets[i - 1]
        for j in range(0, nrows):
            if (j * (nrows - prev) - j * (j - 1) / 2) > base:
                offsets[i] = prev + j
                break
    chunksizes = np.ones(size, dtype=int)
    chunksizes[:-1] = offsets[1:] - offsets[:-1]
    chunksizes[-1] = nrows - offsets[-1]
    # Local update
    lchunk = [offsets[rank], offsets[rank] + chunksizes[rank]]
    return lchunk, chunksizes, offsets


class miniAtoms(object):
    def __init__(self, atoms=None, positions=None, numbers=None):
        if atoms is not None:
            self.positions = np.copy(atoms.positions)
            self.numbers = np.copy(atoms.numbers)
        elif positions is not None:
            assert len(positions.shape) == 2, 'Positions array must be 2-dimensional'
            assert positions.shape[1] == 3, 'Each coordinate must be 3-dimensional'
            self.positions = np.copy(positions)
            if numbers is not None:
                self.numbers = np.copy(numbers)
            else:
                self.numbers = np.array(positions.shape[0]*[-1])
        else:
            raise NotImplementedError('No initialization for miniAtoms using given inputs implemented')

    def __eq__(self, other):
        return (self.positions == other.positions).all()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return miniAtoms(positions=self.positions[i:i+1, :], numbers=self.numbers[i:i+1])
        elif isinstance(i, list) and len(i) > 0:
            # Make sure a list of booleans will work correctly and not be
            # interpreted at 0 and 1 indices.
            i = np.array(i)
            return miniAtoms(positions=self.positions[i, :], numbers=self.numbers[i, :])
        else:
            raise IndexError('Invalid index.')

    def __delitem__(self, i):
        li = range(len(self))
        if isinstance(i, int):
            i = np.array([i])
        if isinstance(i, list) and len(i) > 0:
            # Make sure a list of booleans will work correctly and not be
            # interpreted at 0 and 1 indices.
            i = np.array(i)
        mask = np.ones(len(self), bool)
        mask[i] = False
        self.positions = self.positions[mask]
        self.numbers = self.numbers[mask]


class SOAP(Kern):
    def __init__(self, input_dim, sigma=1., r_cut=5., l_max=10, n_max=10, exponent=1, r_grid_points=None,
                 similarity=None, multi_atom=False, verbosity=0, structure_file='str_relax.out',
                 parallel=None, optimize_sigma=True, optimize_exponent=False, use_pss_buffer=True,
                 quadrature_order=2, quadrature_type='gauss-legendre', materials=None, elements=None):
        super(SOAP, self).__init__(input_dim, [0],  'SOAP')
        self.alpha = 1. / (2 * sigma**2)  # alpha = 2./(0.5**2) ? Check
        self.r_cut = r_cut
        self.l_max = l_max
        self.n_max = n_max
        self.delta_r = r_cut/n_max
        self.quad_order = quadrature_order
        self.quad_type = quadrature_type
        self.set_r_grid(r_grid_points)
        if multi_atom and similarity is not None:
            self.similarity = similarity
        else:
            self.similarity = dirac_delta
        self.multi_atom = multi_atom
        self.materials = materials
        self.elements = elements
        if self.elements is not None:
            for k in self.elements.keys():
                self.elements[k].sort()
        self.verbosity = verbosity
        self.pss_buffer = []
        self.use_pss_buffer = use_pss_buffer
        self.Kdiag_buffer = {}
        self.Kcross_buffer = {}
        self.structure_file = structure_file
        self.soap_input_dim = input_dim
        self.r_grid_points = r_grid_points
        self.n_eval = 0
        self.parallel_data = False
        self.parallel_cnlm = False
        if parallel == 'data':
            self.parallel_data = True
        elif parallel == 'cnlm':
            self.parallel_cnlm = True
        self.parallel = self.parallel_data or self.parallel_cnlm
        if self.parallel:
            from utils.parprint import parprint
            self.print = parprint
        else:
            self.print = print

        self.optimize_sigma = optimize_sigma
        self.derivative = False

        sigma = np.array(sigma)
        self.sigma = Param('sigma', sigma)  #, Logistic(0.2, 2.2))  # Logexp())
        self.link_parameter(self.sigma)
        self.sigma.set_prior(GPy.priors.Gamma.from_EV(0.7, 0.2), warning=False)  # Prior mean: 0.7 A, prior variance: 0.2 A^2
        if not optimize_sigma:
            self.sigma.fix()

        self.optimize_exponent = optimize_exponent
        if optimize_exponent:
            exponent = np.array(exponent)
            self.exponent = Param('exponent', exponent, Logistic(0.5, 5))  # Logexp())
            self.link_parameter(self.exponent)
        else:
            self.exponent = exponent

        # Timings
        self.kernel_times = []
        self.power_spectrum_times = []
        self.reduction_times_X_X = []
        self.reduction_times_X_X2 = []

    def __del__(self):
        self.print('{} evaluations of kernel performed'.format(self.n_eval))

    def set_r_grid(self, n_points=None):
        if n_points is None:
            self.r_grid = np.array([self.delta_r*i for i in range(self.n_max)])
        else:
            self.r_grid = np.linspace(0, self.r_cut, n_points)
        self.update_G()

    def update_G(self):
        order = self.quad_order  # for 'gregory' only
        inner_product_mode = self.quad_type  # 'gauss-legendre', 'romberg', 'gregory'
        # n_max Gaussian basis functions evaluated at r_grid
        # alpha = self.alpha
        sigma = self.delta_r / 3.
        alpha = 1. / (2. * sigma**2)
        # S = basis_overlap(self.n_max, self.r_cut, alpha)
        # Weights for higher accuracy integration,
        # \int_0^{r_c} \g_n(r)f(r)r^2\,dr \approx \sum_i \omega_i \g_{n, i} f_i r_i^2 \Delta r
        if inner_product_mode == 'gauss-legendre':
            r_grid, self.iweights = np.polynomial.legendre.leggauss(self.r_grid.shape[0])
            self.r_grid = 0.5 * self.r_cut * (r_grid + 1)
        elif inner_product_mode == 'gregory':
            self.iweights = self.compute_weights(self.r_grid.shape[0], 1, order, 'gregory')  # spacing included in r2dr
        else:
            pass

        # \Phi_{nm} = \phi_n(x_m)
        Phi = phi_mat(range(self.n_max), self.delta_r, alpha, self.r_grid)
        L = scipy.linalg.cholesky(basis_overlap(self.n_max, self.r_cut, alpha), lower=True)
        # g_n(x_m) = \sum_k L^{-T}_{kn} \phi_k(x_m)
        G = np.dot(Phi.T, np.linalg.inv(L).T).T
        if False:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()

            ax1.plot(self.r_grid, G.T)
            plt.xlabel('r [$\AA$]', fontsize=24)
            plt.ylabel('g$_n$(r)', fontsize=24)
            plt.tick_params(labelsize=24)
            # plt.gcf().set_tight_layout(True)
            fig.subplots_adjust(bottom=0.15, left=0.12, right=0.97, top=0.95)
            left, bottom, width, height = [0.38, 0.45, 0.55, 0.45]
            ax2 = fig.add_axes([left, bottom, width, height])
            ax2.plot(self.r_grid[20:], G[:, 20:].T)
            ax2.set_xlim([1, 5])
            ax2.set_ylim([-.5, 2])
            plt.savefig('radial_basis_functions.pdf')
            #plt.show(block=True)


        # Define Gr2dr as g_n(r)*r^2 evaluated at r_grid such that
        # np.dot(Gr2dr[n, :], f[:]) approximates \int_0^{r_c}r^2 \g_n(r)f(r)\,dr
        # For Romberg, scipy.integrate.romb(self.Gr2dr[n, :] * f) approximates \int_0^{r_c}r^2 \g_n(r)f(r)\,dr
        delta_r = self.r_grid[1] - self.r_grid[0]
        if inner_product_mode == 'gauss-legendre':
            self.r2dr = self.r_grid * self.r_grid * self.iweights * 0.5 * self.r_cut
        elif inner_product_mode == 'gregory':
            self.r2dr = self.r_grid * self.r_grid * self.iweights * delta_r
        elif inner_product_mode == 'romberg':
            self.r2dr = self.r_grid * self.r_grid * delta_r
        else:
            pass
        self.Gr2dr = np.einsum('ij, j -> ij', G, self.r2dr)

        # Using QR factorisation on numerical approximation of the basis overlap
        # multiply by r to orthonomalise in radial coordinates
        # \int_{0}^{r_{c}} \phi_n(r)\phi_{n'}(r)r^2\,dr -> \int_0^{r_c} \g_n(r)\g_{n'}(r)r^2\,dr = \delta{nn'}
        # Phir = np.einsum('ij, j -> ij', Phi, self.r_grid)
        # q, r = np.linalg.qr(Phir.T)
        # Contains g_n(r)*r evaluated at r_grid such that np.dot(self.Gr[i, :], self.Gr[j, :]) = \delta{ij'}
        # i.e., np.dot(Gr[i, :], Gr[j, :]) approximates \int_0^{r_c} \g_n(r)\g_{n'}(r)r^2\,dr
        # Gr = q.T
        # self.Gr2dr = np.einsum('ij, j -> ij', Gr, self.r_grid) * np.sqrt(delta_r)

    def compute_weights(self, r_nodes, delta_r, order, type='gregory'):
        if type=='gregory':
            w = gregory_weights(r_nodes, delta_r, order)
        else:
            w = np.ones(r_nodes.shape[0])
        return w

    def inner_product(self, u, v, romberg=False):
        # Given weigths and r grid, \int r^2 u(r) v(r) \,dr \approx \Delta r \sum_i w_i r_i^2 u_i v_i
        if romberg:
            return scipy.integrate.romb(self.r2dr * u * v)
        return np.dot(self.r2dr * u, v)

    def inner_product_Gn(self, n, v, romberg=False):
        # Given weigths and r grid, \int r^2 u(r) v(r) \,dr \approx \Delta r \sum_i w_i r_i^2 u_i v_i
        if romberg:
            return scipy.integrate.romb(self.Gr2dr[n] * v)
        return np.dot(self.Gr2dr[n], v)

    def get_cnlm(self, atoms, derivative=False, alpha=None):

        if alpha is None:
            alpha=self.alpha
        c_nlm = np.zeros(self.n_max * self.l_max * self.l_max, dtype=complex)
        if derivative:
            dc_nlm = np.zeros(self.n_max * self.l_max * self.l_max, dtype=complex)
        r_cartesian = np.copy(atoms.positions)
        r, theta, phi = cart2sph(r_cartesian)

        r_grid2 = self.r_grid * self.r_grid
        ar2g2 = np.empty((r.shape[0], r_grid2.shape[0]))
        arg = np.empty((r.shape[0], r_grid2.shape[0]))
        for a in range(r.shape[0]):
            ar2g2[a] = alpha * (r[a] * r[a] + r_grid2)
            arg[a] = 2. * alpha * r[a] * self.r_grid

        if self.parallel_cnlm:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            # lchunk, chunksizes, offsets = partition1d(self.n_max*self.l_max*self.l_max, rank, size)

            chunksize = (self.n_max*self.l_max*self.l_max) / size
            leftover = (self.n_max*self.l_max*self.l_max) % size
            chunksizes = np.ones(size, dtype=int)*chunksize
            chunksizes[:leftover] += 1
            offsets = np.zeros(size, dtype=int)
            offsets[1:] = np.cumsum(chunksizes)[:-1]
            # Local update
            lchunk = [offsets[rank], offsets[rank] + chunksizes[rank]]

            for idx in range(lchunk[0], lchunk[1]):
                n = idx / (self.l_max*self.l_max)
                nidx = idx % (self.l_max*self.l_max)
                l = int(np.sqrt(nidx))
                m = nidx - (l * (l + 1))
                for a in range(r.shape[0]):
                    if derivative:
                        I_01, dI_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
                    else:
                        I_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a])
                    c_nlm[idx] += np.dot(self.Gr2dr[n], I_01)  # \sum_i\int r^2 g_n(r) c^i_{lm}(r)\,dr
                    if derivative:
                        dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_01)  # \sum_i\int r^2 g_n(r) c^i_{lm}(r)'\,dr

            comm.Allgatherv(c_nlm[lchunk[0]: lchunk[1]],
                            [c_nlm, chunksizes, offsets, MPI.DOUBLE_COMPLEX])
            if derivative:
                comm.Allgatherv(dc_nlm[lchunk[0]: lchunk[1]],
                                [dc_nlm, chunksizes, offsets, MPI.DOUBLE_COMPLEX])

        else:
            for a, n, l in itertools.product(range(r.shape[0]), range(self.n_max), range(self.l_max)):
                for m in range(-l, l + 1):
                    if derivative:
                        I_01, dI_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
                    else:
                        I_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a])
                    idx = n * self.l_max * self.l_max + l * l + (m + l)
                    c_nlm[idx] += np.dot(self.Gr2dr[n], I_01)
                    if derivative:
                        dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_01)

        if derivative:
            return c_nlm, dc_nlm
        return c_nlm

    def get_power_spectrum(self, atoms, species=None, derivative=False):
        if species is not None:
            n_species = len(species)
        else:
            n_species = 1
        pspectrum = np.empty((n_species, n_species, self.n_max, self.n_max, self.l_max), dtype=complex)
        c_nlm = np.empty((n_species, self.n_max, self.l_max*self.l_max), dtype=complex)
        if derivative:
            dc_nlm = np.ones((n_species, self.n_max, self.l_max * self.l_max), dtype=complex)
            dpspectrum = np.empty((n_species, n_species, n_species, self.n_max, self.n_max, self.l_max), dtype=complex) # FIXME: multi-alpha
        for i in range(n_species):
            satoms = copy.deepcopy(atoms)
            del satoms[[n for (n, atom) in enumerate(satoms) if atom.numbers[0]!=species[i]]]
            # TODO: Choose atom dependent alpha here
            # FIXME: multi-alpha
            alpha = self.alpha[i]
            # alpha = self.alpha
            if derivative:
                c, dc =  self.get_cnlm(satoms, derivative, alpha=alpha)
                dc_nlm[i] = dc.reshape((self.n_max, self.l_max*self.l_max))
            else:
                c = self.get_cnlm(satoms, alpha=alpha)
            c_nlm[i] = c.reshape((self.n_max, self.l_max*self.l_max))

        for s1 in range(n_species):
            for s2 in range(n_species):
                for i in range(self.n_max):
                    for j in range(self.n_max):
                        for l in range(self.l_max):
                            pspectrum[s1, s2, i, j, l] = \
                                    np.vdot(c_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                            c_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1)
        if derivative:
            """
            for s1 in range(n_species):
                for s2 in range(n_species):
                    for i in range(self.n_max):
                        for j in range(self.n_max):
                            for l in range(self.l_max):
                                dpspectrum[s1, s2, i, j, l] = \
                                    np.vdot(c_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                            dc_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1) + \
                                    np.vdot(dc_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                            c_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1)
                                #dpspectrum[s1, s2, i, j, l] = \
                                #    np.dot(dc_nlm[s1, i, l * l: l * l + 2 * l + 1],
                                #           np.conj(c_nlm[s2, j, l * l: l * l + 2 * l + 1])) / np.sqrt(2 * l + 1) + \
                                #    np.dot(c_nlm[s1, i, l * l: l * l + 2 * l + 1],
                                #           np.conj(dc_nlm[s2, j, l * l: l * l + 2 * l + 1])) / np.sqrt(2 * l + 1)
            """
            # FIXME: multi-alpha

            for s0 in range(n_species):
                for s1 in range(n_species):
                    for s2 in range(n_species):
                        for i in range(self.n_max):
                            for j in range(self.n_max):
                                for l in range(self.l_max):
                                    if s0 == s1 == s2:
                                        dpspectrum[s0, s1, s2, i, j, l] = \
                                            np.vdot(c_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                                    dc_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1) + \
                                            np.vdot(dc_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                                    c_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1)
                                    elif s0 == s1:
                                        dpspectrum[s0, s1, s2, i, j, l] = \
                                            np.vdot(c_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                                    dc_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1)
                                    elif s0 == s2:
                                        dpspectrum[s0, s1, s2, i, j, l] = \
                                            np.vdot(dc_nlm[s2, j, l * l: l * l + 2 * l + 1],
                                                    c_nlm[s1, i, l * l: l * l + 2 * l + 1]) / np.sqrt(2 * l + 1)
                                    else:
                                        dpspectrum[s0, s1, s2, i, j, l] = 0.


            # print(c_nlm[:, :, 0], dc_nlm[:, :, 0])
            #return (pspectrum.reshape((n_species, n_species, self.n_max*self.n_max*self.l_max)) * np.sqrt(8*np.pi**2),
            #        dpspectrum.reshape((n_species, n_species, self.n_max*self.n_max*self.l_max)) * np.sqrt(8*np.pi**2))
            # FIXME: multi-alpha
            return (pspectrum.reshape((n_species, n_species, self.n_max * self.n_max * self.l_max)) * \
                    np.sqrt(8 * np.pi ** 2),
                    dpspectrum.reshape((n_species, n_species, n_species, self.n_max * self.n_max * self.l_max)) * \
                    np.sqrt(8 * np.pi ** 2))
        return pspectrum.reshape((n_species, n_species, self.n_max*self.n_max*self.l_max)) * np.sqrt(8*np.pi**2)


    def get_approx_density(self, atoms, r):
        # atoms has positions of atoms, (natoms x 3)
        # r has the points where we want the density, (npoints x 3)
        # print(r.shape, '\n', atoms.positions)
        # TODO: Add contributions from each atom type
        if len(r.shape) == 1:
            r = r[np.newaxis, :]
        d = np.linalg.norm(atoms[:, :, np.newaxis] - r.T, axis=1)
        return np.exp(-self.alpha*d*d).sum(axis=0)
 
    def I(self, atoms0, atoms1):
        r0_cartesian = atoms0.positions
        r1_cartesian = atoms1.positions

        r0, theta0, phi0 = cart2sph(r0_cartesian)
        r1, theta1, phi1 = cart2sph(r1_cartesian)

        l_max = self.l_max
        I = np.zeros(sum_squares_odd_integers(l_max + 1), dtype=complex)

        idx = 0
        for l in range(0, l_max):
            for m0 in range(2 * l + 1):
                for m1 in range(2 * l + 1):
                    for i in range(r0.shape[0]):
                        for j in range(r1.shape[0]):
                            I_01 = 4 * np.pi * (np.pi/(2*self.alpha))**(3./2) * \
                                   np.exp(-self.alpha *
                                          (r0[i] * r0[i] + r1[j] * r1[j]) /
                                          2) * \
                                   iv(l, self.alpha * r0[i] * r1[j]) * \
                                   sp.sph_harm(m0 - l, l, theta0[i], phi0[i])\
                                   * \
                                   np.conj(sp.sph_harm(m1 - l, l, theta1[j],
                                                   phi1[j]))
                            
                            I[idx] += I_01 / np.sqrt(2*l + 1)
                    idx += 1

        return I * np.sqrt(8*np.pi**2)

    def I2(self, atoms0, atoms1):
        I = self.I(atoms0, atoms1)
        return np.dot(np.conj(I),  I).real

    def k(self, atoms0, atoms1):
        p0 = self.get_power_spectrum(atoms0)
        p1 = self.get_power_spectrum(atoms1)
        k01 = np.dot(p0, np.conj(p1)).real
        return k01

    def kold(self, atoms0, atoms1):
        k01 = self.I2(atoms0, atoms1)
        return k01

    def K_reduction(self, K):
        Kred = K.sum()
        return Kred

    def get_all_power_spectrums(self, atoms, nl, species, derivative=False):
        start = timer()
        if self.multi_atom:
            n_species = len(species)
        else:
            n_species = 1

        n = atoms.positions.shape[0]
        if nl is None:
            nl = ase.neighborlist.NeighborList(n*[self.r_cut/1.99], skin=0., self_interaction=False, bothways=True)
            nl.update(atoms)
        if derivative:
            # dp = np.empty((n, n_species, n_species, self.n_max * self.n_max * self.l_max), dtype=complex)
            dp = np.empty((n, n_species, n_species, n_species, self.n_max * self.n_max * self.l_max), dtype=complex) # FIXME: multi-alpha
        p = np.empty((n, n_species, n_species, self.n_max * self.n_max * self.l_max), dtype=complex)

        for i in range(n):
            indices, offsets = nl.get_neighbors(i)
            positions = np.empty((len(indices) + 1, 3))
            numbers = np.empty(len(indices) + 1)
            positions[0, :] = np.copy(atoms.positions[i])
            numbers[0] = atoms.numbers[i]
            for n, (idx, offset) in enumerate(zip(indices, offsets)):
                positions[n + 1, :] = atoms.positions[idx] + np.dot(offset, atoms.get_cell())
                numbers[n + 1] = atoms.numbers[idx]
            tmp = miniAtoms(positions=positions, numbers=numbers)
            tmp.positions -= atoms.positions[i]
            if derivative:
                p[i], dp[i] = self.get_power_spectrum(tmp, species, derivative)
            else:
                p[i] = self.get_power_spectrum(tmp, species)
        self.power_spectrum_times.append(timer() - start)
        if derivative:
            return p, dp
        return p

    @Cache_this(limit=3, ignore_args=(0,))
    def K(self, X, X2):
        start = timer()
        X_shape = X.shape[0]
        if X2 is not None:
            X2_shape = X2.shape[0]
        else:
            X2_shape = X.shape[0]
        K = np.empty((X_shape, X2_shape))

        if self.parallel_data:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            chunk, chunksizes, offsets = partition1d(X.shape[0], rank, size)

            X = X[chunk[0]: chunk[1], :]

        load_X = []
        load_X2 = []
        # if self.do_pss_buffer:
        # Mark pss for saving for new inputs. Mark other inputs for loading pss.
        for j, X_i in enumerate(X[:, 0]):
            save_X = True
            for i, (X_old, pss_old) in enumerate(self.pss_buffer):
                if X_i == X_old:
                    load_X.append((j, i))
                    save_X = False
                    break
            if save_X:
                self.pss_buffer.append([np.copy(X_i), j])

        # TODO: TEST buffering of X2
        if X2 is not None:
            for j, X2_i in enumerate(X2[:, 0]):
                save_X = True
                for i, (X_old, pss_old) in enumerate(self.pss_buffer):
                    if X2_i == X_old:
                        # X will always be visited before X2, so it is OK if the buffer is not in the list yet
                        load_X2.append((j, i))
                        save_X = False
                        break
                # Need checking if in X so that it does not need saving (it will be saved from its occurrence in X)
                ## FIXME: Also, we can load from the calculation for X (always done before X2)
                for i, X_i in enumerate(X[:, 0]):
                    if X_i == X2_i:
                        # All X_i are either marked for saving or loading, so no need for X2 to save.
                        # In either case they will be available at the time of processing X2.
                        ## FIXME If X_i == X2_i but X2_i not in pss_buffer then X_i wasn't in pss_buffer, is marked for saving
                        ##for i_pss, pss in enumerate(self.pss_buffer):
                        ##    if X_i == pss[0]:  # pss[1] == i:
                        ##        load_X2.append((j, i_pss))
                        ##        break
                        save_X = False
                        break
                if save_X:
                    self.pss_buffer.append([np.copy(X2_i), j + X.shape[0]])

        self.folder2idx = {}
        self.idx2folder = {}
        self.idx2folderX2 = {}
        material_id = []
        material_id2 = []
        if X.dtype.kind == 'f':
            X = np.asarray(abs(np.asarray(X, dtype=int)), dtype=str)
        if X.dtype.kind == 'S':
            tmp = np.empty(X.shape, dtype=ase.Atoms)
            for i, folder in enumerate(X[:, 0]):
                tmp[i, 0] = ATAT2GPAW(os.path.join(folder, self.structure_file)).get_atoms()
                self.folder2idx[folder] = i
                self.idx2folder[i] = folder
                material_id.append('/'.join(folder.split('/')[0:-1]))
            X = tmp

        if X2 is not None:
            if X2.dtype.kind == 'f':
                X2 = np.asarray(abs(np.asarray(X2, dtype=int)), dtype=str)
            if X2.dtype.kind == 'S':
                tmp = np.empty(X2.shape, dtype=ase.Atoms)
                for i, folder in enumerate(X2[:, 0]):
                    tmp[i, 0] = ATAT2GPAW(os.path.join(folder, self.structure_file)).get_atoms()
                    self.idx2folderX2[i] = folder
                    material_id2.append('/'.join(folder.split('/')[0:-1]))
                X2 = tmp

        if False:  # Plot some pseudo-densities
            from mpl_toolkits.mplot3d import axes3d
            import matplotlib.pyplot as plt
            from matplotlib import cm

            config = 19
            if self.multi_atom:
                species =  list(X[config, 0].numbers)
                species = list(set(species))
                species.sort()
                n_species = len(species)
            else:
                species = [X[config, 0].numbers[0]]
                n_species = 1
            print('Atomic numbers: ', X[config, 0].numbers)

            for i in range(n_species):

                satoms = copy.deepcopy(X[config, 0])
                del satoms[[n for (n, atom) in enumerate(satoms) if satoms.numbers[n] != species[i]]]

                nl = ase.neighborlist.NeighborList(satoms.positions.shape[0] * [self.r_cut / 1.99],
                                                   skin=0., self_interaction=False, bothways=True)
                nl.update(satoms)
                indices, offsets = nl.get_neighbors(0)
                positions = np.empty((len(indices) + 1, 3))

                positions[0, :] = np.zeros(3)
                for n, (idx, offset) in enumerate(zip(indices, offsets)):
                    positions[n + 1, :] = satoms.positions[idx] + np.dot(offset, X[config, 0].get_cell()) - \
                                             satoms.positions[0]

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                nx = 200
                ny = 200
                rX = np.linspace(-1.2*self.r_cut, 1.2*self.r_cut, nx)
                rY = np.linspace(-1.2*self.r_cut, 1.2*self.r_cut, ny)
                rX, rY = np.meshgrid(rX, rY)
                rr = np.hstack((rX.reshape(nx*ny, 1), rY.reshape(nx*ny, 1), np.zeros((nx*ny, 1))))
                o = np.random.normal(size=(3,))
                print(np.linalg.norm(satoms.positions[0] - satoms.positions[1]))
                d = np.cross(satoms.positions[0] - o, satoms.positions[1] - o)
                dhat = d / np.linalg.norm(d)
                from utils.geometry import rotation2
                R = rotation2(np.array([0, 0, 1]), dhat)
                rr = np.dot(R, rr.T).T
                rZ = self.get_approx_density(atoms=positions, r=rr).reshape((nx, ny))

                ax.plot_surface(rX, rY, rZ, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1, antialiased=True, shade=True)#, alpha=1.)
                cset = ax.contour(rX, rY, rZ, zdir='z', offset=1.1, linewidth=0.1, cmap=cm.coolwarm)
                #cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
                #cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

                ax.set_xlabel('\nX [$\AA$]', linespacing=1)
                ax.set_xlim(-5.5, 5.5)
                ax.set_zlabel('\n$\\rho_0$', linespacing=1)
                ax.set_zlim(0.0, 1.1)
                ax.set_ylabel('\nY [$\AA$]', linespacing=1)
                #ax.set_zlim(-100, 100)
                plt.gcf().set_tight_layout(True)
                plt.savefig('pseudo_density_{}_{}.pdf'.format(config, species[i]))
                plt.show(block=True)
            sys.exit(0)

        if self.parallel_data:
            if self.optimize_sigma and self.derivative:
                Klocal = self._K_dK(X, X2, load_X, load_X2, material_id, material_id2)
                self.derivative = False
            else:
                Klocal = self._K(X, X2, load_X, load_X2, material_id, material_id2)
            comm.Allgatherv(Klocal, [K, chunksizes * X2_shape, offsets * X2_shape, MPI.DOUBLE])
        else:
            if self.optimize_sigma and self.derivative:
                K = self._K_dK(X, X2, load_X, load_X2, material_id, material_id2)
                self.derivative = False
            else:
                K = self._K(X, X2, load_X, load_X2, material_id, material_id2)

        self.kernel_times.append(timer() - start)
        return K

    def _K(self, X, X2, load_X, load_X2, material_id, material_id2):

        self.n_eval += 1
        if (X2 is not None) and (X.shape == X2.shape) and (X == X2).all():
            X2 = None
        if self.multi_atom:
            species = []
            for atoms in X[:, 0]:
                species += list(atoms.numbers)
            if X2 is not None:
                for atoms in X2[:, 0]:
                    species += list(atoms.numbers)
            species = list(set(species))
            species.sort()
            n_species = len(species)
            kappa_full = np.empty((n_species, n_species))
            for s1 in range(n_species):
                for s2 in range(n_species):
                    kappa_full[s1, s2] = self.similarity(species[s1], species[s2])
            kappa_all = {}
            if self.materials is not None:
                material_elements = {}
                for material in self.materials:
                    material_elements[material] = self.elements[material]
                    n_species = len(material_elements[material])
                for material1 in self.materials:
                    for material2 in self.materials:
                        kappa_all[(material1, material2)] = np.empty((n_species, n_species))
                        for s1 in range(n_species):
                            for s2 in range(n_species):
                                kappa_all[(material1, material2)][s1, s2] = self.similarity(self.elements[material1][s1],
                                                                                            self.elements[material2][s2])
            else:
                kappa = kappa_full
        else:   
            species = None
            n_species = 1 
            kappa = np.array([[1.]])

        if X2 is None:
            K = np.eye(X.shape[0])
            nl = []
            for d in range(X.shape[0]):
                nl.append(ase.neighborlist.NeighborList(X[d, 0].positions.shape[0] * [self.r_cut/1.99],
                                                        skin=0., self_interaction=False, bothways=True))
            
            pss = []
            for i in range(X.shape[0]):
                if self.materials is not None:
                    species = material_elements[material_id[i]]
                load = -1
                # Check if pss available and choose which one to load
                for j, k in load_X:
                    if i == j:
                        load = k
                        break
                if self.verbosity > 0 and load == -1:
                    self.print('\rPow. spec. {:02}/{:02}'.format(i + 1, X.shape[0]), end=''); sys.stdout.flush()

                # Load pss if available. Otherwise, calculate and save
                if load >= 0:
                    pss.append(self.pss_buffer[load][1])
                else:
                    nl[i].update(X[i, 0])
                    pss.append(self.get_all_power_spectrums(X[i, 0], nl[i], species))
                    for j, ll in enumerate(self.pss_buffer):
                        try:
                            if i == ll[1]:
                                self.pss_buffer[j][1] = pss[-1]
                                break
                        except:
                            pass

            start = timer()
            if self.parallel_cnlm:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                size = comm.Get_size()
                rank = comm.Get_rank()

                chunk, chunksizes, offsets = partition_ltri_rows(X.shape[0], rank, size)
                self.print(chunk, chunksizes, offsets)
                Klocal = np.zeros((chunksizes[rank], X.shape[0]))
                for i in range(chunk[0], chunk[1]):
                    Klocal[i - offsets[rank], i] = 1.
                comm.barrier()  # make sure all pss are available
            else:
                Klocal = K
                chunk = [0, X.shape[0]]
                chunksizes = [X.shape[0]]
                offsets = [0]
                rank = 0

            for d1 in range(chunk[0], chunk[1]):
                for d2 in range(d1):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id[d2])]
                        kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                        kappa22 = kappa_all[(material_id[d2], material_id[d2])]
                    else:
                        kappa = kappa_full
                        kappa11 = kappa_full
                        kappa22 = kappa_full
                    if self.verbosity > 1:
                        self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x '*(d2 + 1) + '. '*(d1 - d2 - 1) + '1 ',
                              end=''); sys.stdout.flush()
                    if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                        K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                        self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = self.Kcross_buffer[self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                    else:
                        K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]
                    if self.idx2folder[d1] not in self.Kdiag_buffer:
                        K00 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d1]), kappa11, kappa11).real
                        self.Kdiag_buffer[self.idx2folder[d1]] = K00
                    else:
                        K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                    if self.idx2folder[d2] not in self.Kdiag_buffer:
                        K11 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d2], pss[d2]), kappa22, kappa22).real
                        self.Kdiag_buffer[self.idx2folder[d2]] = K11
                    else:
                        K11 = self.Kdiag_buffer[self.idx2folder[d2]]

                    Kij = self.K_reduction(K01) / np.sqrt(self.K_reduction(K00) * self.K_reduction(K11))
                    Klocal[d1 - offsets[rank], d2] = Kij

            if self.parallel_cnlm:
                comm.Allgatherv(Klocal, [K, chunksizes * X.shape[0], offsets * X.shape[0], MPI.DOUBLE])

            if self.verbosity > 1:
                self.print('', end='\r')

            K += K.T - np.diag(K.diagonal())
            self.Km = np.power(K, self.exponent)
            self.reduction_times_X_X.append(timer() - start)
            return self.Km

        # else (X2 is not None)
        K = np.zeros((X.shape[0], X2.shape[0]))
        nl1 = []
        for d1 in range(X.shape[0]):
            nl1.append(ase.neighborlist.NeighborList(X[d1, 0].positions.shape[0] * [self.r_cut/1.99],
                                                     skin=0., self_interaction=False, bothways=True))
        nl2 =[]
        for d2 in range(X2.shape[0]):
            nl2.append(ase.neighborlist.NeighborList(X2[d2, 0].positions.shape[0] * [self.r_cut/1.99],
                                                     skin=0., self_interaction=False, bothways=True))

        pss = []
        for i in range(X.shape[0]):
            if self.materials is not None:
                species = material_elements[material_id[i]]
            load = -1
            # Check if pss available and choose which one to load
            for j, k in load_X:
                if i == j:
                    load = k
                    break
            if self.verbosity > 0 and load == -1:
                self.print('\rPow. spec. 1 {:02}/{:02}'.format(i + 1, X.shape[0]), end=''); sys.stdout.flush()
            # Load pss if available. Otherwise, calculate and save
            if load >= 0:
                pss.append(self.pss_buffer[load][1])
            else:
                nl1[i].update(X[i, 0])
                pss.append(self.get_all_power_spectrums(X[i, 0], nl1[i], species))
                for j, ll in enumerate(self.pss_buffer):
                    try:
                        if i == ll[1]:
                            self.pss_buffer[j][1] = pss[-1]
                            break
                    except:
                        pass

        # TODO: TEST pss buffering for X2
        pss2 = []
        for i in range(X2.shape[0]):
            if self.materials is not None:
                species = material_elements[material_id2[i]]

            load = -1
            # Check if pss available and choose which one to load
            for j, k in load_X2:
                if i == j:
                    load = k
                    break
            if self.verbosity > 0 and load == -1:
                print('\rPow. spec. {:02}/{:02}'.format(i + 1, X2.shape[0]), end=''); sys.stdout.flush()
            # Load pss if available. Otherwise, calculate and save
            if load >= 0:
                pss2.append(self.pss_buffer[load][1])
            else:
                nl2[i].update(X2[i, 0])
                pss2.append(self.get_all_power_spectrums(X2[i, 0], nl2[i], species))
                for j, ll in enumerate(self.pss_buffer):
                    try:
                        if i + X.shape[0] == ll[1]:
                            self.pss_buffer[j][1] = pss2[-1]
                            break
                    except:
                        pass

        if self.verbosity > 0:
            self.print('', end='\r')

        start = timer()
        if self.parallel_cnlm:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()

            chunk, chunksizes, offsets = partition1d(X.shape[0], rank, size)
            Klocal = np.zeros((chunksizes[rank], X2.shape[0]))
            comm.barrier()  # make sure all pss are available
        else:
            Klocal = K
            chunk = [0, X.shape[0]]
            chunksizes = [X.shape[0]]
            offsets = [0]
            rank = 0

        for d1 in range(chunk[0], chunk[1]):
            for d2 in range(X2.shape[0]):
                if self.materials is not None:
                    kappa = kappa_all[(material_id[d1], material_id2[d2])]
                    kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                    kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                else:
                    kappa = kappa_full
                    kappa11 = kappa_full
                    kappa22 = kappa_full
                if self.verbosity > 1:
                    self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[rank]) + 'x '*(d2 + 1) + '. '*(X2.shape[0] - d2 -1), end=''); sys.stdout.flush()
                """
                if self.idx2folder[d1] + self.idx2folderX2[d2] not in self.Kcross_buffer:
                    K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                    self.Kcross_buffer[self.idx2folder[d1] + self.idx2folderX2[d2]] = self.Kcross_buffer[
                        self.idx2folderX2[d2] + self.idx2folder[d1]] = K01
                else:
                    K01 = self.Kcross_buffer[self.idx2folder[d1] + self.idx2folderX2[d2]]
                if self.idx2folder[d1] not in self.Kdiag_buffer:
                    K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa, kappa).real
                    self.Kdiag_buffer[self.idx2folder[d1]] = K00
                else:
                    K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                if self.idx2folderX2[d2] not in self.Kdiag_buffer:
                    K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa, kappa).real
                    self.Kdiag_buffer[self.idx2folderX2[d2]] = K11
                else:
                    K11 = self.Kdiag_buffer[self.idx2folderX2[d2]]
                """
                K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real
                Kij = pow(self.K_reduction(K01) / \
                          np.sqrt(self.K_reduction(K00) * self.K_reduction(K11)), self.exponent)
                Kij = self.K_reduction(K01) / np.sqrt(self.K_reduction(K00) * self.K_reduction(K11))
                Klocal[d1 - offsets[rank], d2] = Kij
        if self.verbosity > 1:
            self.print('')

        if self.parallel_cnlm:
            comm.Allgatherv(Klocal, [K, chunksizes * X2.shape[0], offsets * X2.shape[0], MPI.DOUBLE])

        self.Km = np.power(K, self.exponent)
        self.reduction_times_X_X2.append(timer() - start)
        return self.Km

    def Kdiag(self, X):
        return np.ones(X.shape[0])

    def _K_dK(self, X, X2, load_X, load_X2, material_id, material_id2):

        self.n_eval += 2
        if (X2 is not None) and (X.shape == X2.shape) and (X == X2).all():
            X2 = None
        if self.multi_atom:
            species = []
            for atoms in X[:, 0]:
                species += list(atoms.numbers)
            if X2 is not None:
                for atoms in X2[:, 0]:
                    species += list(atoms.numbers)
            species = list(set(species))
            species.sort()
            n_species = len(species)
            kappa_full = np.empty((n_species, n_species))
            for s1 in range(n_species):
                for s2 in range(n_species):
                    kappa_full[s1, s2] = self.similarity(species[s1], species[s2])
            kappa_all = {}
            if self.materials is not None:
                material_elements = {}
                for material in self.materials:
                    material_elements[material] = self.elements[material]
                    n_species = len(material_elements[material])
                for material1 in self.materials:
                    for material2 in self.materials:
                        kappa_all[(material1, material2)] = np.empty((n_species, n_species))
                        for s1 in range(n_species):
                            for s2 in range(n_species):
                                kappa_all[(material1, material2)][s1, s2] = self.similarity(
                                    self.elements[material1][s1],
                                    self.elements[material2][s2])

            else:
                kappa = kappa_full
        else:
            species = None
            n_species = 1
            kappa = np.array([[1.]])

        if X2 is None:
            K = np.eye(X.shape[0])
            # dK_dalpha = np.zeros((X.shape[0], X.shape[0]))
            dK_dalpha = np.zeros((n_species, X.shape[0], X.shape[0])) # FIXME: multi-alpha
            nl = []
            for d in range(X.shape[0]):
                nl.append(ase.neighborlist.NeighborList(X[d, 0].positions.shape[0] * [self.r_cut / 1.99],
                                                        skin=0., self_interaction=False, bothways=True))

            dpss_dalpha = []
            pss = []
            for i in range(X.shape[0]):
                if self.materials is not None:
                    species = material_elements[material_id[i]]

                nl[i].update(X[i, 0])
                p, dp = self.get_all_power_spectrums(X[i, 0], nl[i], species, True)
                pss.append(p)
                dpss_dalpha.append(dp)
                for j, ll in enumerate(self.pss_buffer):
                    try:
                        if i == ll[1]:
                            self.pss_buffer[j][1] = pss[-1]
                            break
                    except:
                        pass

            if False:
                for d1 in range(X.shape[0]):
                    for d2 in range(d1):
                        if self.materials is not None:
                            kappa = kappa_all[(material_id[d1], material_id[d2])]
                            kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                            kappa22 = kappa_all[(material_id[d2], material_id[d2])]
                        else:
                            kappa = kappa_full
                            kappa11 = kappa_full
                            kappa22 = kappa_full
                        if self.verbosity > 1:
                            self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (
                            d1 - d2 - 1) + '1 ',
                                       end='');
                            sys.stdout.flush()
                        if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                            K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                            self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = self.Kcross_buffer[
                                self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                        else:
                            K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]
                        if self.idx2folder[d1] not in self.Kdiag_buffer:
                            K00 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d1]), kappa11,
                                            kappa11).real
                            self.Kdiag_buffer[self.idx2folder[d1]] = K00
                        else:
                            K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                        if self.idx2folder[d2] not in self.Kdiag_buffer:
                            K11 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d2], pss[d2]), kappa22,
                                            kappa22).real
                            self.Kdiag_buffer[self.idx2folder[d2]] = K11
                        else:
                            K11 = self.Kdiag_buffer[self.idx2folder[d2]]

                        dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1], pss[d2], kappa, kappa).real + \
                               np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d2], kappa, kappa).real
                        dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d1], kappa11, kappa11).real + \
                               np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1], pss[d1], kappa11, kappa11).real
                        dK11 = np.einsum('ijkl, mnol, jn, ko', pss[d2], dpss_dalpha[d2], kappa22, kappa22).real + \
                               np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d2], pss[d2], kappa22, kappa22).real

                        rdK00 = self.K_reduction(dK00)
                        rdK11 = self.K_reduction(dK11)
                        rdK01 = self.K_reduction(dK01)
                        rK00 = self.K_reduction(K00)
                        rK11 = self.K_reduction(K11)
                        rK01 = self.K_reduction(K01)

                        dKij = rdK01 / np.sqrt(rK00 * rK11)
                        dKij -= 0.5 * rK01 / (rK00 * rK11) ** (1.5) * (rdK00 * rK11 + rK00 * rdK11)
                        dKij *= self.exponent * pow(rK01 / np.sqrt(rK00 * rK11), self.exponent - 1.)

                        Kij = rK01 / np.sqrt(rK00 * rK11)  # pow(rK01 / np.sqrt(rK00 * rK11), self.exponent)

                        dK_dalpha[d1, d2] = dKij
                        K[d1, d2] = Kij
            else: # FIXME: multi-alpha
                for d1 in range(X.shape[0]):
                    for d2 in range(d1):
                        if self.materials is not None:
                            kappa = kappa_all[(material_id[d1], material_id[d2])]
                            kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                            kappa22 = kappa_all[(material_id[d2], material_id[d2])]
                        else:
                            kappa = kappa_full
                            kappa11 = kappa_full
                            kappa22 = kappa_full
                        if self.verbosity > 1:
                            self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (
                            d1 - d2 - 1) + '1 ',
                                       end='');
                            sys.stdout.flush()
                        if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                            K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                            self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = self.Kcross_buffer[
                                self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                        else:
                            K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]
                        if self.idx2folder[d1] not in self.Kdiag_buffer:
                            K00 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d1]), kappa11,
                                            kappa11).real
                            self.Kdiag_buffer[self.idx2folder[d1]] = K00
                        else:
                            K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                        if self.idx2folder[d2] not in self.Kdiag_buffer:
                            K11 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d2], pss[d2]), kappa22,
                                            kappa22).real
                            self.Kdiag_buffer[self.idx2folder[d2]] = K11
                        else:
                            K11 = self.Kdiag_buffer[self.idx2folder[d2]]

                        rK00 = self.K_reduction(K00)
                        rK11 = self.K_reduction(K11)
                        rK01 = self.K_reduction(K01)

                        Kij = rK01 / np.sqrt(rK00 * rK11)

                        K[d1, d2] = Kij
                # Derivatives
                for s1 in range(n_species):
                    for d1 in range(X.shape[0]):
                        for d2 in range(d1):
                            if self.materials is not None:
                                kappa = kappa_all[(material_id[d1], material_id[d2])]
                                kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                                kappa22 = kappa_all[(material_id[d2], material_id[d2])]
                            else:
                                kappa = kappa_full
                                kappa11 = kappa_full
                                kappa22 = kappa_full
                            if self.verbosity > 1:
                                self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (
                                d1 - d2 - 1) + '1 ',
                                           end='');
                                sys.stdout.flush()
                            if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                                K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                                self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = self.Kcross_buffer[
                                    self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                            else:
                                K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]
                            if self.idx2folder[d1] not in self.Kdiag_buffer:
                                K00 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d1]), kappa11,
                                                kappa11).real
                                self.Kdiag_buffer[self.idx2folder[d1]] = K00
                            else:
                                K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                            if self.idx2folder[d2] not in self.Kdiag_buffer:
                                K11 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d2], pss[d2]), kappa22,
                                                kappa22).real
                                self.Kdiag_buffer[self.idx2folder[d2]] = K11
                            else:
                                K11 = self.Kdiag_buffer[self.idx2folder[d2]]

                            dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1][:, s1], pss[d2], kappa, kappa).real + \
                                   np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d2][:, s1], kappa, kappa).real
                            dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d1][:, s1], kappa11, kappa11).real + \
                                   np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1][:, s1], pss[d1], kappa11, kappa11).real
                            dK11 = np.einsum('ijkl, mnol, jn, ko', pss[d2], dpss_dalpha[d2][:, s1], kappa22, kappa22).real + \
                                   np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d2][:, s1], pss[d2], kappa22, kappa22).real

                            rdK00 = self.K_reduction(dK00)
                            rdK11 = self.K_reduction(dK11)
                            rdK01 = self.K_reduction(dK01)
                            rK00 = self.K_reduction(K00)
                            rK11 = self.K_reduction(K11)
                            rK01 = self.K_reduction(K01)

                            dKij = rdK01 / np.sqrt(rK00 * rK11)
                            dKij -= 0.5 * rK01 / (rK00 * rK11) ** (1.5) * (rdK00 * rK11 + rK00 * rdK11)
                            dKij *= self.exponent * pow(rK01 / np.sqrt(rK00 * rK11), self.exponent - 1.)

                            dK_dalpha[s1, d1, d2] = dKij

            if self.verbosity > 1:
                self.print('', end='\r')

            K += K.T - np.diag(K.diagonal())
            self.Km = np.power(K, self.exponent)
            # self.dK_dalpha = dK_dalpha + dK_dalpha.T
            # FIXME: multi-alpha
            for i in range(dK_dalpha.shape[0]):
                dK_dalpha[i] += dK_dalpha[i].T
            self.dK_dalpha = dK_dalpha
            return self.Km

        # else (X2 is not None)
        K = np.zeros((X.shape[0], X2.shape[0]))
        dK_dalpha = np.zeros((X.shape[0], X2.shape[0]))
        # dK_dalpha = np.zeros((n_species, X.shape[0], X2.shape[0]))
        nl1 = []
        for d1 in range(X.shape[0]):
            nl1.append(ase.neighborlist.NeighborList(X[d1, 0].positions.shape[0] * [self.r_cut / 1.99],
                                                     skin=0., self_interaction=False, bothways=True))
        nl2 = []
        for d2 in range(X2.shape[0]):
            nl2.append(ase.neighborlist.NeighborList(X2[d2, 0].positions.shape[0] * [self.r_cut / 1.99],
                                                     skin=0., self_interaction=False, bothways=True))

        pss = []
        dpss_dalpha = []
        for i in range(X.shape[0]):
            if self.materials is not None:
                species = material_elements[material_id[i]]
            load = -1
            # Check if pss available and choose which one to load
            for j, k in load_X:
                if i == j:
                    load = k
                    break
            if self.verbosity > 0 and load == -1:
                self.print('\rPow. spec. 1 {:02}/{:02}'.format(i + 1, X.shape[0]), end='');
                sys.stdout.flush()
            nl1[i].update(X[i, 0])
            p, dp = self.get_all_power_spectrums(X[i, 0], nl[i], species, True)
            pss.append(p)
            dpss_dalpha.append(dp)
            for j, ll in enumerate(self.pss_buffer):
                try:
                    if i == ll[1]:
                        self.pss_buffer[j][1] = pss[-1]
                        break
                except:
                    pass

        # TODO: TEST pss buffering for X2
        pss2 = []
        dpss2_dalpha = []
        for i in range(X2.shape[0]):
            if self.materials is not None:
                species = material_elements[material_id2[i]]

            load = -1
            # Check if pss available and choose which one to load
            for j, k in load_X2:
                if i == j:
                    load = k
                    break
            if self.verbosity > 0 and load == -1:
                print('\rPow. spec. {:02}/{:02}'.format(i + 1, X2.shape[0]), end='');
                sys.stdout.flush()
            nl2[i].update(X2[i, 0])
            p, dp = self.get_all_power_spectrums(X2[i, 0], nl[i], species, True)
            pss2.append(p)
            dpss2_dalpha.append(dp)
            for j, ll in enumerate(self.pss_buffer):
                try:
                    if i + X.shape[0] == ll[1]:
                        self.pss_buffer[j][1] = pss2[-1]
                        break
                except:
                    pass

        if self.verbosity > 0:
            self.print('', end='\r')

        if False:
            for d1 in range(X.shape[0]):
                for d2 in range(X2.shape[0]):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id2[d2])]
                        kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                        kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                    else:
                        kappa = kappa_full
                        kappa11 = kappa_full
                        kappa22 = kappa_full
                    if self.verbosity > 1:
                        self.print(
                            '\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (X2.shape[0] - d2 - 1),
                            end='');
                        sys.stdout.flush()
                    """
                    if self.idx2folder[d1] + self.idx2folderX2[d2] not in self.Kcross_buffer:
                        K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                        self.Kcross_buffer[self.idx2folder[d1] + self.idx2folderX2[d2]] = self.Kcross_buffer[
                            self.idx2folderX2[d2] + self.idx2folder[d1]] = K01
                    else:
                        K01 = self.Kcross_buffer[self.idx2folder[d1] + self.idx2folderX2[d2]]
                    if self.idx2folder[d1] not in self.Kdiag_buffer:
                        K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa, kappa).real
                        self.Kdiag_buffer[self.idx2folder[d1]] = K00
                    else:
                        K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                    if self.idx2folderX2[d2] not in self.Kdiag_buffer:
                        K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa, kappa).real
                        self.Kdiag_buffer[self.idx2folderX2[d2]] = K11
                    else:
                        K11 = self.Kdiag_buffer[self.idx2folderX2[d2]]
                    """
                    K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                    K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                    K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real
                    dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1], pss2[d2], kappa, kappa).real + \
                           np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss2_dalpha[d2], kappa, kappa).real
                    dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d1], kappa11, kappa11).real + \
                           np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1], pss[d1], kappa11, kappa11).real
                    dK11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], dpss2_dalpha[d2], kappa22, kappa22).real + \
                           np.einsum('ijkl, mnol, jn, ko', dpss2_dalpha[d2], pss2[d2], kappa22, kappa22).real

                    rdK00 = self.K_reduction(dK00)
                    rdK11 = self.K_reduction(dK11)
                    rdK01 = self.K_reduction(dK01)
                    rK00 = self.K_reduction(K00)
                    rK11 = self.K_reduction(K11)
                    rK01 = self.K_reduction(K01)

                    dKij = rdK01 / np.sqrt(rK00 * rK11)
                    dKij -= 0.5 * rK01 / (rK00 * rK11) ** (1.5) * (rdK00 * rK11 + rK00 * rdK11)
                    dKij *= self.exponent * pow(rK01 / np.sqrt(rK00 * rK11), self.exponent - 1.)

                    Kij = rK01 / np.sqrt(rK00 * rK11)

                    dK_dalpha[d1, d2] = dKij
                    K[d1, d2] = Kij
        else:  # FIXME: multi-alpha
            for d1 in range(X.shape[0]):
                for d2 in range(X2.shape[0]):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id2[d2])]
                        kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                        kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                    else:
                        kappa = kappa_full
                        kappa11 = kappa_full
                        kappa22 = kappa_full
                    if self.verbosity > 1:
                        self.print(
                            '\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (
                            X2.shape[0] - d2 - 1),
                            end='');
                        sys.stdout.flush()

                    K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                    K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                    K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real

                    rK00 = self.K_reduction(K00)
                    rK11 = self.K_reduction(K11)
                    rK01 = self.K_reduction(K01)

                    Kij = rK01 / np.sqrt(rK00 * rK11)

                    K[d1, d2] = Kij
            # Derivatives
            for s1 in range(n_species):
                for d1 in range(X.shape[0]):
                    for d2 in range(X2.shape[0]):
                        if self.materials is not None:
                            kappa = kappa_all[(material_id[d1], material_id2[d2])]
                            kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                            kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                        else:
                            kappa = kappa_full
                            kappa11 = kappa_full
                            kappa22 = kappa_full
                        if self.verbosity > 1:
                            self.print(
                                '\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (
                                X2.shape[0] - d2 - 1),
                                end='');
                            sys.stdout.flush()

                        K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                        K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                        K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real
                        dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dsigma[d1][:, s1], pss2[d2], kappa, kappa).real + \
                               np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dsigma2[d2][:, s1], kappa, kappa).real
                        dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dsigma[d1][:, s1], kappa11, kappa11).real + \
                               np.einsum('ijkl, mnol, jn, ko', dpss_dsigma[d1][:, s1], pss[d1], kappa11, kappa11).real
                        dK11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], dpss_dsigma2[d2][:, s1], kappa22, kappa22).real + \
                               np.einsum('ijkl, mnol, jn, ko', dpss_dsigma2[d2][:, s1], pss2[d2], kappa22, kappa22).real

                        rdK00 = self.K_reduction(dK00)
                        rdK11 = self.K_reduction(dK11)
                        rdK01 = self.K_reduction(dK01)
                        rK00 = self.K_reduction(K00)
                        rK11 = self.K_reduction(K11)
                        rK01 = self.K_reduction(K01)

                        dKij = rdK01 / np.sqrt(rK00 * rK11)
                        dKij -= 0.5 * rK01 / (rK00 * rK11) ** (1.5) * (rdK00 * rK11 + rK00 * rdK11)
                        dKij *= self.exponent * pow(rK01 / np.sqrt(rK00 * rK11), self.exponent - 1.)

                        dK_dalpha[s1, d1, d2] = dKij

        if self.verbosity > 1:
            self.print('')

        self.Km = np.power(K, self.exponent)
        #self.dK_dalpha = dK_dalpha + dK_dalpha.T
        # FIXME: multi-alpha
        for i in range(dK_dalpha.shape[0]):
            dK_dalpha[i] += dK_dalpha[i].T
        self.dK_dalpha = dK_dalpha
        return self.Km

    def update_gradients_full(self, dL_dK, X, X2):
        if self.optimize_sigma:
            """
            # Numerical gradient
            self.n_eval += 2
            dsigma = 0.005
            soap = SOAP(self.soap_input_dim, self.sigma, self.r_cut, self.l_max, self.n_max, self.exponent,
                        self.r_grid_points, self.similarity, self.multi_atom, self.verbosity, self.structure_file)
            soap.sigma = soap.sigma + dsigma
            K1 = soap.K(X, X2)
            soap = SOAP(self.soap_input_dim, self.sigma, self.r_cut, self.l_max, self.n_max, self.exponent,
                        self.r_grid_points, self.similarity, self.multi_atom, self.verbosity, self.structure_file)
            soap.sigma = soap.sigma - dsigma
            K0 = soap.K(X, X2)
            self.sigma.gradient = np.sum(dL_dK * (K1 - K0) / (2 * dsigma))
            print('\ndK_dsigma (n): {}'.format(self.sigma.gradient))
            """
            # Analytical gradient
            #self.sigma.gradient = np.sum(dL_dK * self.dK_dalpha / (-self.sigma**3))
            # FIXME: multi-alpha
            for i, a in enumerate(self.alpha):
                self.sigma.gradient[i] = np.sum(dL_dK * self.dK_dalpha[i] / (-self.sigma[i] ** 3))
        if not (self.optimize_exponent or self.optimize_sigma):
            pass

    def update_gradients_diag(self, dL_dKdiag, X):
        if self.optimize_sigma:
            self.sigma.gradient = 0.
        if self.optimize_exponent:
            self.exponent.gradient = 0.
        if not (self.optimize_exponent or self.optimize_sigma):
            pass

    def parameters_changed(self):
        self.alpha = 1. / (2 * self.sigma ** 2)
        self.pss_buffer = []  # Forget values of the power spectrum
        self.Kcross_buffer = {}
        self.Kdiag_buffer = {}

        if self.optimize_sigma:
            self.alpha = 1. / (2 * self.sigma**2)
            self.pss_buffer = []    # Forget values of the power spectrum
            self.Kcross_buffer = {}
            self.Kdiag_buffer = {}
            self.derivative = True  # Calculate derivative next iteration
            self.dK_dalpha = None   # Invalidate derivative
        else:
            pass

    def gradients_X(self, dL_dK, X, X2=None):
        pass
