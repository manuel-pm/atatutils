# coding=utf-8
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
from exp_spherical_in import exp_spherical_in


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
    """Smooth radial cut-off function with parameters rcut, rdelta."""
    if r <= rcut-rdelta:
        return 1.
    elif r <= rcut:
        return 0.5*(1+np.cos(np.pi*(r-rcut+rdelta)/rdelta))
    return 0.


def nearest_neighbour_distance(atoms, which=0, largest=False):
    """Nearest neighbor distance between the atoms.

    Parameters
    ----------
    atoms : ase.Atoms object
        Atoms object where we look for the nearest neighbor distance.
    which : int >= 0
        Atom with respect to which we look for the NND.
    largest : bool, optional
        If True, return the maximum NND in the system. This
        ensures every atom in the system has at least one neighbor
        within a distance nnd.

    Returns
    -------
    nnd : float > 0
        NND of atom which or maximum nearest neighbor distance of all atoms.

    """
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
    return exp_spherical_in(y, n, x)


def c_ilm(l, m, alpha, ri, thetai, phii, x, derivative=False):
    I_01 = 4 * np.pi * exp_iv(-alpha * (x*x + ri*ri), l, 2 * alpha * x * ri) * np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        dI_01 = I_01 * (l / alpha - (x*x + ri*ri))
        dI_01 += 8 * np.pi * x * ri * exp_iv(-alpha * (x*x + ri*ri), l + 1, 2 * alpha * x * ri) * \
                 np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        return I_01, dI_01
    return I_01


def c_ilm2(l, m, alpha, ar2g2, thetai, phii, arg, derivative=False):
    I_01 = 4 * np.pi * exp_iv(-ar2g2, l, arg) * np.conj(sp.sph_harm(m, l, thetai, phii))
    if derivative:
        dI_01 = I_01 * ((l - ar2g2) / alpha)
        dI_01 += 4 * np.pi * arg / alpha * exp_iv(-ar2g2, l + 1, arg) * \
                 np.conj(sp.sph_harm(m, l, thetai, phii))
        return I_01, dI_01
    return I_01


def get_cnlm(atoms, n_max, l_max, alpha, gr2dr, r_grid, derivative=False, mpi_comm=None):
    r"""Calculate the coefficients of the expansion of the pseudo-density
    for the given atomic environment.

    Parameters
    ----------
    atoms : miniAtoms or ase.Atoms object
        Object representing an atomic environment.
    n_max : int > 0
        Maximum order of the radial expansion.
    l_max : int > 0
        Maximum order of the angular expansion.
    alpha : float > 0
        Precision of the Gaussian representing the atoms.
    gr2dr : 2-D np.ndarray of float
        Array containing the radial basis functions, one per row,
        evaluated at r_grid_points, such that
        np.dot(gr2dr[n], v) \approx \int r^2 g_n(r) v(r) dr.
    r_grid : 1-D np.ndarray
        Radial mesh to evaluate the integrals.
    derivative: bool
        Whether to return the derivatives with respect to alpha
    Returns
    -------
    c_nlm : 1-D ndarray of complex float
        Contains the flattened array :math:`c_{nlm}`.
    c_nlm : 1-D ndarray of complex float
        Contains the flattened array :math:`\partial c_{nlm}/\partial \alpha`.
    """
    parallel = True
    mpi_rank = 0
    mpi_size = 1
    if mpi_comm is None:
        parallel = False
    else:
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        from mpi4py import MPI

    c_nlm = np.zeros(n_max * l_max * l_max, dtype=complex)
    if derivative:
        dc_nlm = np.zeros(n_max * l_max * l_max, dtype=complex)
    r_cartesian = np.copy(atoms.positions)
    r, theta, phi = cart2sph(r_cartesian)

    r_grid2 = r_grid * r_grid
    ar2g2 = np.empty((r.shape[0], r_grid2.shape[0]))
    arg = np.empty((r.shape[0], r_grid2.shape[0]))
    for a in range(r.shape[0]):
        ar2g2[a] = alpha * (r[a] * r[a] + r_grid2)
        arg[a] = 2. * alpha * r[a] * r_grid

    I_all = np.zeros((sum_odd_integers(l_max - 1) * r.shape[0], r_grid.shape[0]), dtype=complex)
    dI_all = np.zeros((sum_odd_integers(l_max - 1) * r.shape[0], r_grid.shape[0]), dtype=complex)
    lchunk, chunksizes, offsets = partition1d(I_all.shape[0], mpi_rank, mpi_size)
    if parallel:
        I_local = np.zeros((chunksizes[mpi_rank], r_grid.shape[0]), dtype=complex)
        dI_local = np.zeros((chunksizes[mpi_rank], r_grid.shape[0]), dtype=complex)
    else:
        I_local = I_all
        dI_local = dI_all
    for idx in range(lchunk[0], lchunk[1]):
        l = int(np.sqrt(idx /r.shape[0]))
        m = idx /r.shape[0] - l**2
        a = idx % r.shape[0]
        if derivative:
            I_local[idx - offsets[mpi_rank]], dI_local[idx - offsets[mpi_rank]] = \
                c_ilm2(l, m - l, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
        else:
            I_local[idx - offsets[mpi_rank]] = c_ilm2(l, m - l, alpha, ar2g2[a], theta[a], phi[a], arg[a])
    if parallel:
        mpi_comm.Allgatherv(I_local.ravel(),
                            [I_all.ravel(), chunksizes * r_grid.shape[0],
                             offsets * r_grid.shape[0], MPI.DOUBLE_COMPLEX])
        if derivative:
            mpi_comm.Allgatherv(dI_local.ravel(),
                                [dI_all.ravel(), chunksizes * r_grid.shape[0],
                                 offsets * r_grid.shape[0], MPI.DOUBLE_COMPLEX])

    if parallel:
        lchunk, chunksizes, offsets = partition1d(n_max * l_max * l_max, mpi_rank, mpi_size)

        for idx in range(lchunk[0], lchunk[1]):
            n = idx / (l_max * l_max)
            nidx = idx % (l_max * l_max)
            l = int(np.sqrt(nidx))
            m = nidx - (l * (l + 1))
            for a in range(r.shape[0]):
                c_nlm[idx] += np.dot(gr2dr[n], I_all[(l**2 + m + l) * r.shape[0] + a])
                if derivative:
                    dc_nlm[idx] += np.dot(gr2dr[n], dI_all[(l**2 + m + l) * r.shape[0] + a])

        mpi_comm.Allgatherv(c_nlm[lchunk[0]: lchunk[1]],
                            [c_nlm, chunksizes, offsets, MPI.DOUBLE_COMPLEX])
        if derivative:
            mpi_comm.Allgatherv(dc_nlm[lchunk[0]: lchunk[1]],
                                [dc_nlm, chunksizes, offsets, MPI.DOUBLE_COMPLEX])

    else:
        for a, n, l in itertools.product(range(r.shape[0]), range(n_max), range(l_max)):
            for m in range(-l, l + 1):
                idx = n * l_max * l_max + l * l + (m + l)
                c_nlm[idx] += np.dot(gr2dr[n], I_all[(l ** 2 + m + l) * r.shape[0] + a])
                if derivative:
                    dc_nlm[idx] += np.dot(gr2dr[n], dI_all[(l ** 2 + m + l) * r.shape[0] + a])

    if derivative:
        return c_nlm, dc_nlm
    return c_nlm


def get_dcnlm_dalpha(atoms, n_max, l_max, alpha, gr2dr, r_grid, derivative=False, mpi_comm=None):
    r"""Calculate the derivative of the coefficients of the expansion of the pseudo-density
    for the given atomic environment.

    Parameters
    ----------
    atoms : miniAtoms or ase.Atoms object
        Object representing an atomic environment.
    n_max : int > 0
        Maximum order of the radial expansion.
    l_max : int > 0
        Maximum order of the angular expansion.
    alpha : float > 0
        Precision of the Gaussian representing the atoms.
    gr2dr : 2-D np.ndarray of float
        Array containing the radial basis functions, one per row,
        evaluated at r_grid_points, such that
        np.dot(gr2dr[n], v) \approx \int r^2 g_n(r) v(r) dr.
    r_grid : 1-D np.ndarray
        Radial mesh to evaluate the integrals.
    derivative: bool
        Whether to return the derivatives with respect to alpha
    Returns
    -------
    c_nlm : 1-D ndarray of complex float
        Contains the flattened array :math:`c_{nlm}`.
    c_nlm : 1-D ndarray of complex float
        Contains the flattened array :math:`\partial c_{nlm}/\partial \alpha`.
    """
    parallel = True
    mpi_rank = 0
    mpi_size = 1
    if mpi_comm is None:
        parallel = False
    else:
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        from mpi4py import MPI

    dc_nlm = np.zeros(n_max * l_max * l_max, dtype=complex)
    r_cartesian = np.copy(atoms.positions)
    r, theta, phi = cart2sph(r_cartesian)

    r_grid2 = r_grid * r_grid
    ar2g2 = np.empty((r.shape[0], r_grid2.shape[0]))
    arg = np.empty((r.shape[0], r_grid2.shape[0]))
    for a in range(r.shape[0]):
        ar2g2[a] = alpha * (r[a] * r[a] + r_grid2)
        arg[a] = 2. * alpha * r[a] * r_grid

    dI_all = np.zeros((sum_odd_integers(l_max - 1) * r.shape[0], r_grid.shape[0]), dtype=complex)
    lchunk, chunksizes, offsets = partition1d(I_all.shape[0], mpi_rank, mpi_size)
    if parallel:
        dI_local = np.zeros((chunksizes[mpi_rank], r_grid.shape[0]), dtype=complex)
    else:
        dI_local = dI_all
    for idx in range(lchunk[0], lchunk[1]):
        l = int(np.sqrt(idx / r.shape[0]))
        m = idx / r.shape[0] - l ** 2
        a = idx % r.shape[0]
        _, dI_local[idx - offsets[mpi_rank]] = \
            c_ilm2(l, m - l, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)

    if parallel:
        mpi_comm.Allgatherv(dI_local.ravel(),
                                [dI_all.ravel(), chunksizes * r_grid.shape[0],
                                 offsets * r_grid.shape[0], MPI.DOUBLE_COMPLEX])

    if parallel:
        lchunk, chunksizes, offsets = partition1d(n_max * l_max * l_max, mpi_rank, mpi_size)

        for idx in range(lchunk[0], lchunk[1]):
            n = idx / (l_max * l_max)
            nidx = idx % (l_max * l_max)
            l = int(np.sqrt(nidx))
            m = nidx - (l * (l + 1))
            for a in range(r.shape[0]):
                dc_nlm[idx] += np.dot(gr2dr[n], dI_all[(l ** 2 + m + l) * r.shape[0] + a])

        mpi_comm.Allgatherv(dc_nlm[lchunk[0]: lchunk[1]],
                                [dc_nlm, chunksizes, offsets, MPI.DOUBLE_COMPLEX])

    else:
        for a, n, l in itertools.product(range(r.shape[0]), range(n_max), range(l_max)):
            for m in range(-l, l + 1):
                idx = n * l_max * l_max + l * l + (m + l)
                dc_nlm[idx] += np.dot(gr2dr[n], dI_all[(l ** 2 + m + l) * r.shape[0] + a])

    return dc_nlm


def sum_squares_odd_integers(n):
    """Sum of the squares of the first n odd integers.

    """
    return n * (2 * n + 1) * (2 * n - 1) / 3


def sum_odd_integers(n):
    """Sum of the first n odd integers.

    """
    return (n + 1)**2


def cart2sph(coords):
    """Change coordinates from cartesian to spherical.

    Parameters
    ----------
    coords : 2-D np.ndarray of float > 0
        2-D array of shape (N, 3) with the Cartesian coordinates
        of the points.

    Returns
    -------
    r, theta, phi : 1-D np.ndarray of float
        1-D arrays of shape (N,) with the spherical
        coordinates of the points.

    """
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
    """Change coordinates from spherical to cartesian.

    Parameters
    ----------
    r : 1-D np.ndarray of float > 0
        1-D array of shape (N,) with radial coordinates.
    theta : 1-D np.ndarray of float

    phi : 1-D np.ndarray of float

    Returns
    -------
    x, y, z : 1-D np.ndarray of float
        Cartesian coordinates of the points.

    """
    rsin_theta = r * np.sin(theta)
    x = rsin_theta * np.cos(phi)
    y = rsin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def dirac_delta(i, j):
    """Dirac delta symbol between i and j.

    Parameters
    ----------
    i
        Object to be compared with j.
    j
        Object to be compared with i.

    Returns
    -------
    int {0, 1}
        1 if i==j and 0 otherwise.

    Notes
    -----
    The only condition to use the function is that
    the operator `==` must be defined between i and j.

    """
    if i == j:
        return 1
    return 0


def partition1d(ndata, rank, size):
    """Partitions a 1D array of size ndata between size processes.

    Parameters
    ----------
     ndata : int > 0
        Number of elements of the array.
    rank : int >=0
        Rank (id) of the process.
    size : int >=0
        Number of processes.

    Returns
    -------
    lchunk : list of 2 int
        Indices (first and one past last) of the chunk for process rank.
    chunksizes : 1-D ndarray of int
        Size (in rows) of all the chunks.
    offsets : 1-D ndarray of int
        Offsets (in rows) of all the chunks.

    """
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
    """Partitions a matrix assuming the cost of the partitioned
    computation depends only on the elements of the lower triangular part.

    Parameters
    ----------
    nrows : int > 0
        Number of rows of the matrix.
    rank : int >=0
        Rank (id) of the process.
    size : int >=0
        Number of processes.

    Returns
    -------
    lchunk : list of 2 int
        Indices (first and one past last) of the chunk for process rank.
    chunksizes : 1-D ndarray of int
        Size (in rows) of all the chunks.
    offsets : 1-D ndarray of int
        Offsets (in rows) of all the chunks.

    Notes
    -----
    The algorithm only partitions in rows, so a perfect balancing is not possible.
    This is motivated by the case where each row represents a data point so that
    we just distribute data points.

    """
    # TODO: Consider the case where we partition the memory of the 2-D array directly.
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
    """Partitions a matrix assuming the cost of the partitioned
    computation depends only on the elements of the upper triangular part.

    Parameters
    ----------
    nrows : int > 0
        Number of rows of the matrix.
    rank : int >=0
        Rank (id) of the process.
    size : int >=0
        Number of processes.

    Returns
    -------
    lchunk : list of 2 int
        Indices (first and one past last) of the chunk for process rank.
    chunksizes : 1-D ndarray of int
        Size (in rows) of all the chunks.
    offsets : 1-D ndarray of int
        Offsets (in rows) of all the chunks.

    Notes
    -----
    The algorithm only partitions in rows, so a perfect balancing is not possible.
    This is motivated by the case where each row represents a data point so that
    we just distribute data points.

    """
    # TODO: Consider the case where we partition the memory of the 2-D array directly.
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
    """Minimal atoms class including only positions
    and atomic numbers using the same ase.Atoms interface.

    Attributes
    ----------
    numbers : 1-D np.ndarray of int
        Atomic numbers of the represented atoms.
    positions : 2-D np.ndarray of float
        Positions of the represented atoms.

    Parameters
    ----------
    atoms : ase.Atoms object, optional
        ase.Atoms object to initialize from its positions and atomic numbers.
    positions : 2-D np.ndarray of float, optional
        Positions of the represented atoms.
    numbers : 1-D np.ndarray of int, optional
        Atomic numbers of the represented atoms.

    """
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

    def get_atomic_numbers(self):
        """Get an array with atomic numbers.

        Returns
        -------
        atomic_numbers : 1-D np.ndarray of int
            Array of shape (N,) with atomic numbers.

        """
        return self.numbers.copy()

    def set_atomic_numbers(self, atomic_numbers):
        """Set atomic numbers to new values. The user is
        responsible for keeping positions up to date as well.

        Parameters
        ----------
        atomic_numbers : 1-D np.ndarray of int
            Array of shape (N,) with atomic numbers.

        """
        assert len(atomic_numbers.shape) == 1
        self.atomic_numbers = atomic_numbers

    def get_positions(self):
        """Get an array with positions.

        Returns
        -------
        positions : 2-D np.ndarray of float
            Array of shape (N, 3) with positions.

        """
        return self.positions.copy()

    def set_positions(self, positions):
        """Set positions to new values. The user is responsible
        for keeping atomic numbers up to date as well.

        Parameters
        ----------
        positions : 2-D np.ndarray of floats
            Array of shape (N, 3) with positions.

        """
        assert positions.shape[1] == 3
        self.positions = positions

    def __eq__(self, other):
        """Check for identity of two miniAtoms objects.

        Parameters
        ----------
        other : ase.Atoms or miniAtoms object
            Atoms to compare to.

        Returns
        -------
        bool
            True if both objects have the same positions
            and atomic numbers.

        Notes
        -----
        The comparision od the positions is done directly in float,
        no tolerance is allowed.

        """
        return (self.positions == other.positions).all() and (self.numbers == other.numbers).all()

    def __ne__(self, other):
        """

        Parameters
        ----------
        other : ase.Atoms or miniAtoms object
            Atoms to compare to.

        Returns
        -------
        bool
            False if both objects have the same positions
            and atomic numbers.

        Notes
        -----
        The comparision od the positions is done directly in float,
        no tolerance is allowed.

        """
        return not self.__eq__(other)

    def __len__(self):
        """Length of the miniAtoms object, i.e., its number of atoms.

        Returns
        -------
        int > 0
            Number of atoms represented by the object.

        """
        return self.positions.shape[0]

    def __getitem__(self, i):
        """Return a subset of the atoms.

        Parameters
        ----------
        i : int >= 0
            Index or range of indices to select the subset of atoms.

        """
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
        """Delete a subset of the atoms.

        Parameters
        ----------
        i : int >= 0
            Index or range of indices to select the subset of atoms.

        """
        # li = range(len(self))
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

    def __repr__(self):
        """Representation of the object as a string.

        """
        tokens = []
        tokens.append('positions={}'.format(self.positions))
        tokens.append('numbers={}'.format(self.numbers))
        return '{}({})'.format(self.__class__.__name__, ', '.join(tokens))

    def __str__(self):
        """String with the positions and atomic numbers.

        """
        return '{}\n{}'.format(self.positions, self.numbers)


class SOAP(Kern):
    r"""Implementation of the SOAP kernel as a GPy kernel as described in [1]_, [2]_.

    Attributes
    ----------
    alpha : list of floats > 0
        Precision of the Gaussians representing the pseudo-density of
         each type of atom.
    derivative : bool
        Determines if the derivative will be calculated in the next call to
         the kernel function K(X, X').
    elements : dict materials: list of int
        Atomic number of each element for each material.
    exponent : int > 0
        Exponent for the definition of the kernel.
    kernel_times : list of float
        Times of each call to the kernel function K(X, X').
    l_max : int > 0
        Maximum order in the angular expansion of the pseudo-density.
    materials : list of str
        Path to the folder containing the data of the materials for the model.
    mpi_comm : MPI communicator
        Communicator to do the parallel processing.
    mpi_rank : int >= 0
        Id of the process.
    mpi_size : int > 0
        Number of processes.
    n_max : int > 0
        Maximum order in the radial expansion of the pseudo-density.
    num_diff : bool
        Determines if the derivatives are approximated numerically.
    optimize_sigma : bool
        Determines if the gradient of sigma is calculated for its optimization.
    parallel_cnlm : bool
        Determines if certain calculations are parallelized.
    power_spectrum_times : list of float
        Times of each call to get_all_power_spectrum
    r_cut : float > 0
        Cut-off radius for constructing the atomic neighbourhoods.
    r_grid_points : int > 0
        Number of grid points for numerical integration in the radial coordinate.
    reduction_times_X_X : list of float
        Times of the calculation of the similiarity among a single group of
         atomic structures given the power spectra.
    reduction_times_X_X2 : list of float
        Times of the calculation of the similiarity between the atomic structures
         of two groups given their power spectra.
    sigma : paramz Param
        Standard deviation of the Gaussians representing the pseudo-density of
         each type of atom.

    Parameters
    ----------
    input_dim : int > 0
        Dimensionality of the input space.
    sigma : iterable of float > 0.
        Standard deviation of the Gaussians representing the pseudo-density of
         each type of atom.
    r_cut : float > 0
        Cut-off radius for constructing the atomic neighbourhoods.
    l_max : int > 0
        Maximum order in the angular expansion of the pseudo-density.
    n_max : int > 0
        Maximum order in the radial expansion of the pseudo-density.
    exponent : int > 0
        Exponent for the definition of the kernel.
    r_grid_points : int > 0
        Number of grid points for numerical integration in the radial coordinate.
    similarity : function, optional
        Function which returns the similarity between elements of given atomic numbers.
    multi_atom : bool, optional
        Determines if the different types of atoms are treated differently.
    verbosity : int >= 0, optional
        Level of verbosity in the standard output.
    structure_file : str, optional
        Name of the file containing the description of the atomic configuration (ATAT format).
    parallel : str or bool, optional
        Parallel mode. If bool determines if running in parallel (`cnlm` mode).
    optimize_sigma : bool
        Determines if the gradient of sigma is calculated for its optimization.
    optimize_exponent : bool, optional
        Determines if the exponent is optimized. DEPRECATED.
    use_pss_buffer : bool, optional
        Determines if the power spectrums are buffered in memory for reuse.
    quadrature_order : int > 0, optional
        Order of the numerical quadrature for radial integration if using Gregory quadrature.
    quadrature_type : str, optional
        Type of numerical scheme for radial quadrature.
    materials : list of str
        Path to the folder containing the data of the materials for the model.
    elements : dict materials: list of int
        Atomic number of each element for each material.
    num_diff : bool
        Determines if the derivatives are approximated numerically.
    mpi_comm : MPI communicator, optional
        Communicator to do the parallel processing. If given parallel mode is activated.

    References
    ----------
    .. [1] A. P. Bart\'ok, R. Kondor, G. Cs\'anyi, *On representing chemical environments*.
        Physical Review B, 87(18), 184115 (2013). http://doi.org/10.1103/PhysRevB.87.184115
    .. [2] W. J. Szlachta, *First principles interatomic potential for tungsten based on Gaussian process regression.*
        University of Cambridge (September 2013). http://arxiv.org/abs/1403.3291

    """
    def __init__(self, input_dim, sigma=1., r_cut=5., l_max=10, n_max=10, exponent=1, r_grid_points=None,
                 similarity=None, multi_atom=False, verbosity=0, structure_file='str_relax.out',
                 parallel=None, optimize_sigma=True, optimize_exponent=False, use_pss_buffer=True,
                 quadrature_order=2, quadrature_type='gauss-legendre', materials=None, elements=None,
                 num_diff=False, mpi_comm=None):
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

        self.parallel_cnlm = False
        if mpi_comm is not None:
            parallel = True
        if parallel:
            from mpi4py import MPI
            self.MPI = MPI
            from utils.parprint import parprint
            self.print = parprint
            self.parallel_cnlm = True
            if mpi_comm is None:
                mpi_comm = self.MPI.COMM_WORLD
        if mpi_comm is not None:
            self.mpi_comm = mpi_comm
            self.mpi_size = self.mpi_comm.Get_size()
            self.mpi_rank = self.mpi_comm.Get_rank()
        else:
            self.mpi_comm = None
            self.mpi_size = 1
            self.mpi_rank = 0
            self.print = print

        self.optimize_sigma = optimize_sigma
        self.derivative = False
        self.num_diff = num_diff
        self.last_X_grad = None
        self.last_X2_grad = None

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
        """Creates a radial grid for numerical integration and calls self.update_G.

        Parameters
        ----------
        n_points : int >= 0, optional
            Number of points in the radial grid (defaults to self.n_max).

        """
        if n_points is None:
            self.r_grid = np.array([self.delta_r*i for i in range(self.n_max)])
        else:
            self.r_grid = np.linspace(0, self.r_cut, n_points)
        self.update_G()

    def update_G(self):
        r"""Creates an set of orthonormal radial basis functions G as well as an array Gr2dr
        to facilitate radial integration.

        Notes
        -----
        Starting with a set of equispaced Gaussians :math:`\phi_k(x)`, their overlap matrix
        and its Cholesky decomposition :math:`LL^T` are calculated. The orthonormal
        basis functions are then :math:`g_n(x) = \sum_k L^{-T}_{kn} \phi_k(x)`
        Finally, we define self.Gr2dr as :math:`g_n(r)*r^2` (at :math:`r=`self.r_grid) such that
        np.dot(self.Gr2dr[n, :], f[:]) approximates :math:`\int_0^{r_c}r^2 \g_n(r)f(r)\,dr`.

        """
        order = self.quad_order
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

    def compute_weights(self, r_nodes, delta_r, order, type='gregory'):
        """Compute quadrature weights.

        Parameters
        ----------
        r_nodes : int > 0
            number of nodes.
        delta_r : float > 0
            mesh spacing.
        order : int > 0
            quadrature order.
        type : str
            quadrature type.

        Returns
        -------
        array of float
            weight vector.

        """
        if type=='gregory':
            w = gregory_weights(r_nodes, delta_r, order)
        else:
            w = np.ones(r_nodes.shape[0])
        return w

    def inner_product(self, u, v, romberg=False):
        r"""Calculate inner product of u and v.

        Returns the **radial** inner product,

        .. math ::
            <u, v> = \int r^2 u(r) v(r) \,dr \approx \Delta r \sum_i w_i r_i^2 ui v_i

        Parameters
        ----------
        u : array of float
        v : array of float
        romberg : bool
            Determines if Romberg integration (from scipy) is used.

        Returns
        -------
        float
            Inner product of u and v.

        """
        if romberg:
            return scipy.integrate.romb(self.r2dr * u * v)
        return np.dot(self.r2dr * u, v)

    def inner_product_Gn(self, n, v, romberg=False):
        r"""Calculate inner product of G[n] (:math:`g_n`) and v.

        Returns the **radial** inner product,

        .. math ::
            <g_n, v> = \int r^2 g_n(r) v(r) \,dr \approx \Delta r \sum_i w_i r_i^2 g_{n, i} v_i

        Parameters
        ----------
        n : int > 0
            order of basis function $g_n$.
        v : array of float
        romberg : bool
            Determines if Romberg integration (from scipy) is used.

        Returns
        -------
        float
            Inner product of g_n and v.

        """
        if romberg:
            return scipy.integrate.romb(self.Gr2dr[n] * v)
        return np.dot(self.Gr2dr[n], v)

    def get_cnlm(self, atoms, derivative=False, alpha=None):
        r"""Calculate the coefficients of the expansion of the pseudo-density
        for the given atomic environment.

        Parameters
        ----------
        atoms : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        derivative : bool
            Determines if the derivative :math:`\partial c_{nlm}/\partial \alpha` is calculated.
        alpha : float > 0
            Precision of the Gaussian representing the atoms.
        Returns
        -------
        c_nlm : 1-D ndarray of complex float
            Contains the flattened array :math:`c_{nlm}`.
        c_nlm : 1-D ndarray of complex float
            Contains the flattened array :math:`\partial c_{nlm}/\partial \alpha`.
        """
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

        I_all = np.zeros((sum_odd_integers(self.l_max - 1) * r.shape[0], self.r_grid_points), dtype=complex)
        dI_all = np.zeros((sum_odd_integers(self.l_max - 1) * r.shape[0], self.r_grid_points), dtype=complex)
        lchunk, chunksizes, offsets = partition1d(I_all.shape[0], self.mpi_rank, self.mpi_size)
        if self.parallel_cnlm:
            I_local = np.zeros((chunksizes[self.mpi_rank], self.r_grid_points), dtype=complex)
            dI_local = np.zeros((chunksizes[self.mpi_rank], self.r_grid_points), dtype=complex)
        else:
            I_local = I_all
            dI_local = dI_all
        for idx in range(lchunk[0], lchunk[1]):
            l = int(np.sqrt(idx /r.shape[0]))
            m = idx /r.shape[0] - l**2
            a = idx % r.shape[0]
            if derivative:
                I_local[idx - offsets[self.mpi_rank]], dI_local[idx - offsets[self.mpi_rank]] = \
                    c_ilm2(l, m - l, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
            else:
                I_local[idx - offsets[self.mpi_rank]] = c_ilm2(l, m - l, alpha, ar2g2[a], theta[a], phi[a], arg[a])
        if self.parallel_cnlm:
            self.mpi_comm.Allgatherv(I_local.ravel(),
                                     [I_all.ravel(), chunksizes * self.r_grid_points,
                                      offsets * self.r_grid_points, self.MPI.DOUBLE_COMPLEX])
            if derivative:
                self.mpi_comm.Allgatherv(dI_local.ravel(),
                                         [dI_all.ravel(), chunksizes * self.r_grid_points,
                                          offsets * self.r_grid_points, self.MPI.DOUBLE_COMPLEX])

        if self.parallel_cnlm:
            lchunk, chunksizes, offsets = partition1d(self.n_max*self.l_max*self.l_max, self.mpi_rank, self.mpi_size)

            for idx in range(lchunk[0], lchunk[1]):
                n = idx / (self.l_max * self.l_max)
                nidx = idx % (self.l_max * self.l_max)
                l = int(np.sqrt(nidx))
                m = nidx - (l * (l + 1))
                for a in range(r.shape[0]):
                    #if derivative:
                    #    I_01, dI_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
                    #else:
                    #    I_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a])
                    #c_nlm[idx] += np.dot(self.Gr2dr[n], I_01)  # \sum_i\int r^2 g_n(r) c^i_{lm}(r)\,dr
                    #if derivative:
                    #    dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_01)  # \sum_i\int r^2 g_n(r) c^i_{lm}(r)'\,dr
                    #print(np.dot(self.Gr2dr[n], I_all[(l**2 + m + l) * r.shape[0] + a]), np.dot(self.Gr2dr[n], c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a])))
                    c_nlm[idx] += np.dot(self.Gr2dr[n], I_all[(l**2 + m + l) * r.shape[0] + a])
                    if derivative:
                        dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_all[(l**2 + m + l) * r.shape[0] + a])

            self.mpi_comm.Allgatherv(c_nlm[lchunk[0]: lchunk[1]],
                                 [c_nlm, chunksizes, offsets, self.MPI.DOUBLE_COMPLEX])
            if derivative:
                self.mpi_comm.Allgatherv(dc_nlm[lchunk[0]: lchunk[1]],
                                     [dc_nlm, chunksizes, offsets, self.MPI.DOUBLE_COMPLEX])

        else:
            for a, n, l in itertools.product(range(r.shape[0]), range(self.n_max), range(self.l_max)):
                for m in range(-l, l + 1):
                    # if derivative:
                    #     I_01, dI_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a], derivative)
                    # else:
                    #     I_01 = c_ilm2(l, m, alpha, ar2g2[a], theta[a], phi[a], arg[a])
                    idx = n * self.l_max * self.l_max + l * l + (m + l)
                    # c_nlm[idx] += np.dot(self.Gr2dr[n], I_01)
                    c_nlm[idx] += np.dot(self.Gr2dr[n], I_all[(l ** 2 + m + l) * r.shape[0] + a])
                    if derivative:
                        # dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_01)
                        dc_nlm[idx] += np.dot(self.Gr2dr[n], dI_all[(l ** 2 + m + l) * r.shape[0] + a])

        if derivative:
            return c_nlm, dc_nlm
        return c_nlm

    def get_power_spectrum(self, atoms, species=None, derivative=False):
        r"""Calculates the power spectrum :math:`p_{nll'}^{ss'}` of the pseudo-density for the
        given atomic environment and optionally its derivatives with respect
        to :math:`\alpha_i`.

        Parameters
        ----------
        atoms : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        species : list of int > 0
            Atomic numbers of the species in the atomic neighbourhood.
        derivative : bool
            Determines if the derivative :math:`\partial p_{nll'}^{ss'}/\partial \alpha_{i}` is calculated.

        Returns
        -------
        pspectrum : 3-D ndarray of complex float
            Power spectrum :math:`p_{nll'}^{ss'}`.
        dpspectrum : 4-D ndarray of complex float
            Derivatives of the power spectrum :math:`\partial p_{nll'}^{ss'}/\partial \alpha_{i}`.

        """
        if species is not None:
            n_species = len(species)
        else:
            n_species = 1
        pspectrum = np.empty((n_species, n_species, self.n_max, self.n_max, self.l_max), dtype=complex)
        c_nlm = np.empty((n_species, self.n_max, self.l_max*self.l_max), dtype=complex)
        if derivative:
            dc_nlm = np.ones((n_species, self.n_max, self.l_max * self.l_max), dtype=complex)
            dpspectrum = np.empty((n_species, n_species, n_species, self.n_max, self.n_max, self.l_max), dtype=complex)
        for i in range(n_species):
            satoms = copy.deepcopy(atoms)
            del satoms[[n for (n, atom) in enumerate(satoms) if atom.numbers[0]!=species[i]]]
            alpha = self.alpha[i]
            if derivative:
                c, dc = self.get_cnlm(satoms, derivative, alpha=alpha)
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
            return (pspectrum.reshape((n_species, n_species, self.n_max * self.n_max * self.l_max)) *
                    np.sqrt(8 * np.pi ** 2),
                    dpspectrum.reshape((n_species, n_species, n_species, self.n_max * self.n_max * self.l_max)) *
                    np.sqrt(8 * np.pi ** 2))
        return pspectrum.reshape((n_species, n_species, self.n_max*self.n_max*self.l_max)) * np.sqrt(8*np.pi**2)

    def get_approx_density(self, atoms, alpha, r):
        """Calculates  the pseudo-density with centres in atoms, precision alpha at the
        coordinates r.

        Parameters
        ----------
        atoms : 2-D ndarray (natoms, 3)
            Positions of the centres of the Gaussians.
        alpha : float > 0
            Precision of the Gaussians.
        r : 2-D ndarray (npoints, 3)
            Positions where the pseud-density is calculated.

        Returns
        -------
        2-D ndarray (npoints, 3)
            Pseudo-density evaluated at r.
        """

        if len(r.shape) == 1:
            r = r[np.newaxis, :]
        d = np.linalg.norm(atoms[:, :, np.newaxis] - r.T, axis=1)
        return np.exp(-alpha * d * d).sum(axis=0)
 
    def I_lmm(self, atoms0, atoms1, alpha):
        """Calculates the overlap integral between the
        pseudo-densities of two atomic environments.

        .. note:: Deprecated
                  `I_lmm` will be removed because it cannot
                  be used if the atomic environments have different alpha.

        Parameters
        ----------
        atoms0 : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        atoms1 : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        alpha : float > 0
            Precision of the Gaussians representing the pseudo-density.

        Returns
        -------
        1-D ndarray
            Flattened overlap integral for each lmm'.
        """
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
                            I_01 = np.exp(-alpha * (r0[i]**2 + r1[j]**2) / 2) * \
                                   iv(l, alpha * r0[i] * r1[j]) * \
                                   sp.sph_harm(m0 - l, l, theta0[i], phi0[i]) * \
                                   np.conj(sp.sph_harm(m1 - l, l, theta1[j], phi1[j]))
                            
                            I[idx] += I_01 / np.sqrt(2*l + 1)
                    idx += 1

        return I * np.sqrt(8 * np.pi**2) * 4 * np.pi * (np.pi / (2 * alpha))**(3. / 2)

    def I2(self, atoms0, atoms1, alpha):
        """Claculates the inner product between two overlap integrals.

        .. note:: Deprecated
                  `I` will be removed because it cannot
                  be used if the atomic environments have different alpha.

        Parameters
        ----------
        atoms0 : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        atoms1 : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        alpha : float > 0
            Precision of the Gaussians representing the pseudo-density.

        Returns
        -------
        float
            Inner product of the two integrals

        See Also
        --------
        I_lmm

        """
        I = self.I_lmm(atoms0, atoms1, alpha)
        return np.dot(np.conj(I),  I).real

    def K_reduction(self, K):
        """Reduction of the matrix containing the product
        of power spectrums between neighborhoods.

        Parameters
        ----------
        K : 2-D ndarray
            Array with the inner product of power spectrums
            between for each pair of neighborhoods.

        Returns
        -------
        float
            Reduction of the input matrix.

        """
        Kred = K.sum()
        return Kred

    def get_all_power_spectrums(self, atoms, nl, species, derivative=False):
        """Calculates the power spectrum of all neighborhoods in atoms.

        Parameters
        ----------
        atoms : miniAtoms or ase.Atoms object
            Object representing an atomic environment.
        nl : ase.NeighborList
            Object representing the neighborhoods of atoms.
        species : list of int > 0
            Atomic numbers of the species in the atomic neighbourhood.
        derivative : bool
            Determines if the derivative of the power spectrums is calculated.

        Returns
        -------
        list of 3-D ndarray of complex float
            Power spectrums :math:`p_{nll'}^{ss'}` for each neighborhood in atoms.
        list of 4-D ndarray of complex float
            Derivatives of the power spectrums :math:`\partial p_{nll'}^{ss'}/\partial \alpha_{i}`
            for each neighborhood in atoms.

        See Also
        --------
        get_power_spectrum

        """
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
            dp = np.empty((n, n_species, n_species, n_species, self.n_max * self.n_max * self.l_max), dtype=complex)
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
        """Calculates the SOAP kernel between structures in X and X2.

        Also, if the derivative is necessary, it will update the array
        self.dK_dalpha with the derivative of the kernel with respect
        to each alpha.

        Parameters
        ----------
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.
        X2 : 2-D ndarray (nstructures2, 1)
            Id's for another set of structures.

        Returns
        -------
        2-D ndarray of float
            Kernel matrix between the inputs in X and X2.

        Notes
        -----
        This functions takes structure names as inputs, i.e., the path to where
        its structure is defined (in a file using ATAT format). It then loads them
        into ase.Atoms objects that are used in the actual implementation
        of the SOAP kernel in self._K and self._K_dK.

        """
        start = timer()

        # self.print('K(X, X2)', X.shape, X2)
        if self.optimize_sigma and self.derivative:
            self.last_X_grad = X
            self.last_X2_grad = X2

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
        # self.print(X, X.shape)
        if X.dtype.kind == 'f' and X.shape[1] > 1:
            self.print('Here')
            baboom
        if X.dtype.kind == 'f':
            X = np.asarray(abs(np.asarray(X, dtype=int)), dtype=str)
            for i in range(X.shape[0]):
                # self.print(X[i][0])
                if int(X[i][0]) != 0 and int(X[i][0]) != 1:
                    X[i][0] = '{:05}'.format(int(X[i][0]))
        if X.dtype.kind == 'S':
            tmp = np.empty(X.shape, dtype=ase.Atoms)
            for i, folder in enumerate(X[:, 0]):
                tmp[i, 0] = ATAT2GPAW(os.path.join(folder, self.structure_file)).get_atoms()
                self.folder2idx[folder] = i
                self.idx2folder[i] = folder
                folder_material_id = '/'.join(folder.split('/')[0:-1])
                if folder_material_id == '':
                    folder_material_id = '.'
                material_id.append(folder_material_id)
            X = tmp
        # self.print('materials: {}'.format(material_id))
        # self.print('materials: {}'.format(self.elements))

        if X2 is not None:
            if X2.dtype.kind == 'f':
                X2 = np.asarray(abs(np.asarray(X2, dtype=int)), dtype=str)
                for i in range(X2.shape[0]):
                    # self.print(X2[i][0])
                    if int(X2[i][0]) != 0 and int(X2[i][0]) != 1:
                        X2[i][0] = '{:05}'.format(int(X2[i][0]))
            if X2.dtype.kind == 'S':
                tmp = np.empty(X2.shape, dtype=ase.Atoms)
                for i, folder in enumerate(X2[:, 0]):
                    tmp[i, 0] = ATAT2GPAW(os.path.join(folder, self.structure_file)).get_atoms()
                    self.idx2folderX2[i] = folder
                    folder_material_id = '/'.join(folder.split('/')[0:-1])
                    if folder_material_id == '':
                        folder_material_id = '.'
                    material_id2.append(folder_material_id)
                X2 = tmp

        """
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
                # Keep species i only and get all positions for the neighborhood of the atom 0
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

                # Calculate the density in a plane going through atoms 0 and 1 of the neighborhood
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
                rZ = self.get_approx_density(atoms=positions, alpha=self.alpha[i], r=rr).reshape((nx, ny))

                ax.plot_surface(rX, rY, rZ, rstride=1, cstride=1, cmap=cm.coolwarm,
                                linewidth=0.1, antialiased=True, shade=True)#, alpha=1.)
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
        """

        if self.optimize_sigma and self.derivative:
            #self.print('call _K_dK')
            K = self._K_dK(X, X2, load_X, load_X2, material_id, material_id2)
            self.derivative = False
        else:
            #self.print('call _K')
            K = self._K(X, X2, load_X, load_X2, material_id, material_id2)

        self.kernel_times.append(timer() - start)
        return K

    def _K(self, X, X2, load_X, load_X2, material_id, material_id2):
        """Computes the kernel matrix between X and X2.

        Parameters
        ----------
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.
        X2 : 2-D ndarray (nstructures2, 1)
            Id's for a set of structures.
        load_X : list of pairs
            Each pair contains the input in X to be loaded and the location in the pss buffer to load from.
        load_X2 : list of pairs
            Same as load_X but with inputs from X2.
        material_id : list of str
            For each input in X, the id of the material to which it belongs as specified in self.materials.
        material_id2
            For each input in X2, the id of the material to which it belongs as specified in self.materials.

        Returns
        -------
        2-D ndarray of float
            Kernel matrix between the inputs in X and X2.

        """
        # self.print(X, X2)
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
                                kappa_all[(material1, material2)][s1, s2] = \
                                    self.similarity(self.elements[material1][s1],
                                                    self.elements[material2][s2])
        else:   
            species = None

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
                    self.print('\rPSK {:02}/{:02}'.format(i + 1, X.shape[0]), end=''); sys.stdout.flush()

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
            chunk, chunksizes, offsets = partition_ltri_rows(X.shape[0], self.mpi_rank, self.mpi_size)
            if self.parallel_cnlm:
                Klocal = np.zeros((chunksizes[self.mpi_rank], X.shape[0]))
                for i in range(chunk[0], chunk[1]):
                    Klocal[i - offsets[self.mpi_rank], i] = 1.
                self.mpi_comm.barrier()  # make sure all pss are available
            else:
                Klocal = K

            # Pre-compute terms dependent on only one structure
            rK00 = np.empty(X.shape[0])
            for d1 in range(X.shape[0]):
                if self.materials is not None:
                    kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                else:
                    kappa11 = kappa_full
                if self.idx2folder[d1] not in self.Kdiag_buffer:
                    K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                    self.Kdiag_buffer[self.idx2folder[d1]] = K00
                else:
                    K00 = self.Kdiag_buffer[self.idx2folder[d1]]
                rK00[d1] = self.K_reduction(K00)

            # Compute kernel matrix K(X, X)
            for d1 in range(chunk[0], chunk[1]):
                for d2 in range(d1):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id[d2])]
                    else:
                        kappa = kappa_full
                    if self.verbosity > 1:
                        self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                        sys.stdout.flush()
                    if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                        # K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                        K01 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d2]).real, kappa, kappa)
                        self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = \
                        self.Kcross_buffer[self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                    else:
                        K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]

                    Kij = self.K_reduction(K01) / np.sqrt(rK00[d1] * rK00[d2])
                    Klocal[d1 - offsets[self.mpi_rank], d2] = Kij

            if self.parallel_cnlm:
                self.mpi_comm.Allgatherv(Klocal, [K, chunksizes * X.shape[0], offsets * X.shape[0], self.MPI.DOUBLE])

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
                self.print('\rPSK 1 {:02}/{:02}'.format(i + 1, X.shape[0]), end=''); sys.stdout.flush()
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
                self.print('\rPSK 2 {:02}/{:02}'.format(i + 1, X2.shape[0]), end=''); sys.stdout.flush()
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
        chunk, chunksizes, offsets = partition1d(X.shape[0], self.mpi_rank, self.mpi_size)
        if self.parallel_cnlm:
            Klocal = np.zeros((chunksizes[self.mpi_rank], X2.shape[0]))
            self.mpi_comm.barrier()  # make sure all pss are available
        else:
            Klocal = K

        # Pre-compute terms dependent on only one structure
        rK00 = np.empty(X.shape[0])
        rK11 = np.empty(X2.shape[0])
        for d1 in range(X.shape[0]):
            if self.materials is not None:
                kappa11 = kappa_all[(material_id[d1], material_id[d1])]
            else:
                kappa11 = kappa_full
            if self.idx2folder[d1] not in self.Kdiag_buffer:
                K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
                self.Kdiag_buffer[self.idx2folder[d1]] = K00
            else:
                K00 = self.Kdiag_buffer[self.idx2folder[d1]]
            rK00[d1] = self.K_reduction(K00)

        # Compute kernel matrix K(X, X2)
        for d2 in range(X2.shape[0]):
            if self.materials is not None:
                kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
            else:
                kappa22 = kappa_full
            if self.idx2folderX2[d2] not in self.Kdiag_buffer:
                K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real
                self.Kdiag_buffer[self.idx2folderX2[d2]] = K11
            else:
                K11 = self.Kdiag_buffer[self.idx2folderX2[d2]]
            rK11[d2] = self.K_reduction(K11)

        for d1 in range(chunk[0], chunk[1]):
            for d2 in range(X2.shape[0]):
                if self.materials is not None:
                    kappa = kappa_all[(material_id[d1], material_id2[d2])]
                else:
                    kappa = kappa_full
                if self.verbosity > 1:
                    self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                    sys.stdout.flush()

                # K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                K01 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss2[d2]).real, kappa, kappa)
                Kij = self.K_reduction(K01) / np.sqrt(rK00[d1] * rK11[d2])
                Klocal[d1 - offsets[self.mpi_rank], d2] = Kij
        if self.verbosity > 1:
            self.print('')

        if self.parallel_cnlm:
            self.mpi_comm.Allgatherv(Klocal, [K, chunksizes * X2.shape[0], offsets * X2.shape[0], self.MPI.DOUBLE])

        self.Km = np.power(K, self.exponent)
        self.reduction_times_X_X2.append(timer() - start)
        return self.Km

    def Kdiag(self, X):
        """Computes the diagonal of the kernel matrix between X and itself

        Parameters
        ----------
        X : 2-D ndarray (nstructures, 1)
            Id's for a set of structures.

        Returns
        -------
        1-D ndarray
            Diagonal of the kernel matrix between X and itself.

        """
        return np.ones(X.shape[0])

    def _K_dK(self, X, X2, load_X, load_X2, material_id, material_id2):
        """Computes the kernel matrix between X and X2 and its derivative with
        respect to the kernel parameters.

        Parameters
        ----------
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.
        X2 : 2-D ndarray (nstructures2, 1)
            Id's for a set of structures.
        load_X : list of pairs
            Each pair contains the input in X to be loaded and the location in the pss buffer to load from.
        load_X2 : list of pairs
            Same as load_X but with inputs from X2.
        material_id : list of str
            For each input in X, the id of the material to which it belongs as specified in self.materials.
        material_id2
            For each input in X2, the id of the material to which it belongs as specified in self.materials.

        Returns
        -------
        2-D ndarray of float
            Kernel matrix between the inputs in X and X2.

        Notes
        -----
        The derivative is not returned, but updated in self.dK_d*.

        """
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
            species = None
            n_species = 1

        if X2 is None:
            K = np.eye(X.shape[0])
            dK_dalpha = np.zeros((n_species, X.shape[0], X.shape[0]))
            # dK_dkappa = np.zeros((n_spececies, n_species, X.shape[0], X.shape[0])) # FIXME: kappa derivative
            nl = []
            for d in range(X.shape[0]):
                nl.append(ase.neighborlist.NeighborList(X[d, 0].positions.shape[0] * [self.r_cut / 1.99],
                                                        skin=0., self_interaction=False, bothways=True))
            dpss_dalpha = []
            pss = []
            for i in range(X.shape[0]):
                if self.verbosity > 1:
                    # self.print('\rPSdK {:02}/{:02}: '.format(i + 1, X.shape[0]), end='')
                    self.print('\rPSdK {:04.1f}%: '.format(100 * (i + 0.) / X.shape[0]), end='')
                    sys.stdout.flush()
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

            start = timer()
            dK_s = np.zeros((X.shape[0], X.shape[0]))
            chunk, chunksizes, offsets = partition_ltri_rows(X.shape[0], self.mpi_rank, self.mpi_size)
            if self.parallel_cnlm:
                Klocal = np.zeros((chunksizes[self.mpi_rank], X.shape[0]))
                for i in range(chunk[0], chunk[1]):
                    Klocal[i - offsets[self.mpi_rank], i] = 1.
                dK_slocal = np.zeros((chunksizes[self.mpi_rank], X.shape[0]))
                self.mpi_comm.barrier()  # make sure all pss are available
            else:
                Klocal = K
                dK_slocal = dK_s

            # Pre-compute terms dependent on only one structure
            rK00 = np.empty(X.shape[0])
            for d in range(X.shape[0]):
                if self.materials is not None:
                    kappa = kappa_all[(material_id[d], material_id[d])]
                else:
                    kappa = kappa_full
                if self.idx2folder[d] not in self.Kdiag_buffer:
                    K00 = np.einsum('ijkl, mnol, jn, ko', pss[d], pss[d], kappa, kappa).real
                    self.Kdiag_buffer[self.idx2folder[d]] = K00
                else:
                    K00 = self.Kdiag_buffer[self.idx2folder[d]]
                rK00[d] = self.K_reduction(K00)

            # Compute kernel matrix K(X, X)
            for d1 in range(chunk[0], chunk[1]):
                for d2 in range(d1):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id[d2])]
                    else:
                        kappa = kappa_full
                    if self.verbosity > 1:
                        # self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                        self.print('\rRed. {:04.1f}%: '.format(100 * (d1 + 0.) / chunksizes[self.mpi_rank]), end='')
                        sys.stdout.flush()
                    if (self.idx2folder[d1] + '-' + self.idx2folder[d2]) not in self.Kcross_buffer:
                        # K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d2], kappa, kappa).real
                        K01 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d2]).real, kappa, kappa)
                        self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]] = self.Kcross_buffer[
                            self.idx2folder[d2] + '-' + self.idx2folder[d1]] = K01
                    else:
                        K01 = self.Kcross_buffer[self.idx2folder[d1] + '-' + self.idx2folder[d2]]

                    rK01 = self.K_reduction(K01)
                    Kij = rK01 / np.sqrt(rK00[d1] * rK00[d2])

                    Klocal[d1 - offsets[self.mpi_rank], d2] = Kij

            if self.parallel_cnlm:
                self.mpi_comm.Allgatherv(Klocal, [K, chunksizes * X.shape[0], offsets * X.shape[0], self.MPI.DOUBLE])

            # Compute kernel matrix derivatives
            # FIXME: kappa derivative
            """
            # dK/d\kappa(X, X). s1 == s2 => dK/d\kappa = 0. dK/d\kappa_{s1s2} = dK/d\kappa_{s2s1}
            # Pre-compute terms dependent on only one structure
            rdK00 = np.empty((n_species, n_species, X.shape[0]))
            for s1 in range(n_species):
                for s2 in range(s1):
                    for d1 in range(X.shape[0]):
                        if self.materials is not None:
                            kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                        else:
                            kappa11 = kappa_full

                        dK00 = np.einsum('ijk, mnk, jn', pss[d1][:, s1, :, :], pss[d1][:, s2, :, :], kappa11).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, s2, :, :], pss[d1][:, s1, :, :], kappa11).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, :, s1, :], pss[d1][:, :, s2, :], kappa11).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, :, s2, :], pss[d1][:, :, s1, :], kappa11).real

                        rdK00[s1, s2, d1] = rdK00[s2, s1, d1] = self.K_reduction(dK00)

            for s1 in range(n_species):
                for s2 in range(s1):
                    for d1 in range(X.shape[0]):
                        for d2 in range(d1):
                            if self.materials is not None:
                                kappa = kappa_all[(material_id[d1], material_id[d2])]
                            else:
                                kappa = kappa_full
                            if self.verbosity > 1:
                                self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]), end='')
                                sys.stdout.flush()

                            dK01 = np.einsum('ijk, mnk, jn', pss[d1][:, s1, :, :], pss[d2][:, s2, :, :], kappa).real + \
                                   np.einsum('ijk, mnk, jn', pss[d1][:, s2, :, :], pss[d2][:, s1, :, :], kappa).real + \
                                   np.einsum('ijk, mnk, jn', pss[d1][:, :, s1, :], pss[d2][:, :, s2, :], kappa).real + \
                                   np.einsum('ijk, mnk, jn', pss[d1][:, :, s2, :], pss[d2][:, :, s1, :], kappa).real

                            rdK01 = self.K_reduction(dK01)
                            rK01 = K[d1, d2] * np.sqrt(rK00[d1] * rK00[d2])

                            dKij = rdK01 / np.sqrt(rK00[d1] * rK00[d2])
                            dKij -= 0.5 * rK01 / (rK00[d1] * rK00[d2]) ** (1.5) \
                                   * (rdK00[s1, s2, d1] * rK11[d2] + rK00[d1] * rdK11[s1, s2, d2])
                            dKij *= self.exponent * pow(K[d1, d2], self.exponent - 1.)

                            dK_dkappa[s1, s2, d1, d2] = dK_dkappa[s1, s2, d2, d1] = dKij
            """
            # dK/d\alpha(X, X)
            rdK00_s = np.zeros(X.shape[0])
            if self.parallel_cnlm:
                chunk1d, chunksizes1d, offsets1d = partition1d(X.shape[0], self.mpi_rank, self.mpi_size)
                rdK00local = np.zeros((chunksizes1d[self.mpi_rank],))
            else:
                chunk1d = [0, X.shape[0]]
                chunksizes1d = [X.shape[0]]
                offsets1d = [0]
                rdK00local = rdK00_s

            # Pre-compute terms dependent on only one structure
            rdK00 = np.empty((n_species, X.shape[0]))
            for s in range(n_species):
                for d in range(chunk1d[0], chunk1d[1]):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d], material_id[d])]
                    else:
                        kappa = kappa_full
                    dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d], dpss_dalpha[d][:, s], kappa, kappa).real + \
                           np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d][:, s], pss[d], kappa, kappa).real
                    rdK00local[d - offsets1d[self.mpi_rank]] = self.K_reduction(dK00)
                if self.parallel_cnlm:
                    self.mpi_comm.Allgatherv(rdK00local, [rdK00_s, chunksizes1d, offsets1d, self.MPI.DOUBLE])
                rdK00[s, :] = rdK00_s

            # Compute dk_d\alpha(X, X)
            for s1 in range(n_species):
                for d1 in range(chunk[0], chunk[1]):
                    for d2 in range(d1):
                        if self.materials is not None:
                            kappa = kappa_all[(material_id[d1], material_id[d2])]
                        else:
                            kappa = kappa_full
                        if self.verbosity > 1:
                            # self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                            self.print('\rRed. {:04.1f}%: '.format(100 * (d1 + 0.) / chunksizes[self.mpi_rank]), end='')
                            sys.stdout.flush()

                        #dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1][:, s1], pss[d2], kappa, kappa).real + \
                        #       np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d2][:, s1], kappa, kappa).real
                        dK01 = np.einsum('ijkmno, jn, ko',
                                         np.einsum('ijkl, mnol', dpss_dalpha[d1][:, s1], pss[d2]).real,
                                         kappa, kappa) + \
                               np.einsum('ijkmno, jn, ko',
                                         np.einsum('ijkl, mnol', pss[d1], dpss_dalpha[d2][:, s1]).real,
                                         kappa, kappa)

                        rdK01 = self.K_reduction(dK01)
                        rK01 = K[d1, d2] * np.sqrt(rK00[d1] * rK00[d2])

                        dKij = rdK01 / np.sqrt(rK00[d1] * rK00[d2])
                        dKij -= 0.5 * rK01 / (rK00[d1] * rK00[d2]) ** (1.5) * \
                                (rdK00[s1, d1] * rK00[d2] + rK00[d1] * rdK00[s1, d2])
                        dKij *= self.exponent * pow(rK01 / np.sqrt(rK00[d1] * rK00[d2]), self.exponent - 1.)

                        dK_slocal[d1 - offsets[self.mpi_rank], d2] = dKij

                if self.parallel_cnlm:
                    self.mpi_comm.Allgatherv(dK_slocal,
                                             [dK_s, chunksizes * X.shape[0], offsets * X.shape[0], self.MPI.DOUBLE])
                dK_dalpha[s1, :, :] = dK_s

            if self.verbosity > 1:
                self.print('', end='\r')

            K += K.T - np.diag(K.diagonal())
            self.Km = np.power(K, self.exponent)
            # FIXME: kappa derivative
            # for s1 in range(n_species):
            #     for s2 in range(s1 + 1, n_species):
            #         dK_dkappa[s1, s2, :, :] = dK_dkappa[s2, s1, :, :]
            # self.dK_dkappa = dK_dkappa
            for i in range(dK_dalpha.shape[0]):
                dK_dalpha[i] = dK_dalpha[i] + dK_dalpha[i].T
            self.dK_dalpha = dK_dalpha
            self.reduction_times_X_X.append(timer() - start)
            return self.Km

        # else (X2 is not None)
        K = np.zeros((X.shape[0], X2.shape[0]))
        dK_dalpha = np.zeros((n_species, X.shape[0], X2.shape[0]))
        # dK_dkappa = np.zeros((n_spececies, n_species, X.shape[0], X.shape[0])) # FIXME: kappa derivative
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
                self.print('\rPSdK 1 {:02}/{:02}'.format(i + 1, X.shape[0]), end='')
                sys.stdout.flush()
            nl1[i].update(X[i, 0])
            p, dp = self.get_all_power_spectrums(X[i, 0], nl1[i], species, True)
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
                print('\rPSdK 2 {:02}/{:02}'.format(i + 1, X2.shape[0]), end='')
                sys.stdout.flush()
            nl2[i].update(X2[i, 0])
            p, dp = self.get_all_power_spectrums(X2[i, 0], nl2[i], species, True)
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

        start = timer()
        dK_s = np.zeros((X.shape[0], X2.shape[0]))
        chunk, chunksizes, offsets = partition1d(X.shape[0], self.mpi_rank, self.mpi_size)
        if self.parallel_cnlm:
            Klocal = np.zeros((chunksizes[self.mpi_rank], X2.shape[0]))
            dK_slocal = np.zeros((chunksizes[self.mpi_rank], X2.shape[0]))
            self.mpi_comm.barrier()  # make sure all pss are available
        else:
            Klocal = K
            dK_slocal = dK_s

        # Pre-compute terms dependent on only one structure
        rK00 = np.empty(X.shape[0])
        rK11 = np.empty(X2.shape[0])
        for d1 in range(X.shape[0]):
            if self.materials is not None:
                kappa11 = kappa_all[(material_id[d1], material_id[d1])]
            else:
                kappa11 = kappa_full
            K00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss[d1], kappa11, kappa11).real
            rK00[d1] = self.K_reduction(K00)

        for d2 in range(X2.shape[0]):
            if self.materials is not None:
                kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
            else:
                kappa22 = kappa_full
            K11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], pss2[d2], kappa22, kappa22).real
            rK11[d2] = self.K_reduction(K11)

        # Compute kernel matrix K(X, X2)
        for d1 in range(chunk[0], chunk[1]):
            for d2 in range(X2.shape[0]):
                if self.materials is not None:
                    kappa = kappa_all[(material_id[d1], material_id2[d2])]
                else:
                    kappa = kappa_full
                if self.verbosity > 1:
                    self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                    sys.stdout.flush()

                # K01 = np.einsum('ijkl, mnol, jn, ko', pss[d1], pss2[d2], kappa, kappa).real
                K01 = np.einsum('ijkmno, jn, ko', np.einsum('ijkl, mnol', pss[d1], pss[d2]).real, kappa, kappa)
                rK01 = self.K_reduction(K01)

                Kij = rK01 / np.sqrt(rK00[d1] * rK11[d2])
                Klocal[d1 - offsets[self.mpi_rank], d2] = Kij

        if self.parallel_cnlm:
            self.mpi_comm.Allgatherv(Klocal, [K, chunksizes * X2.shape[0], offsets * X2.shape[0], self.MPI.DOUBLE])

        # Compute kernel matrix derivatives
        # FIXME: kappa derivative
        """
        # dK/d\kappa(X, X2). s1 == s2 => dK/d\kappa = 0. dK/d\kappa_{s1s2} = dK/d\kappa_{s2s1}
        # Pre-compute terms dependent on only one structure
        rdK00 = np.empty((n_species, n_species, X.shape[0]))
        for s1 in range(n_species):
            for s2 in range(s1):
                for d1 in range(X.shape[0]):
                    if self.materials is not None:
                        kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                    else:
                        kappa11 = kappa_full

                    dK00 = np.einsum('ijk, mnk, jn', pss[d1][:, s1, :, :], pss[d1][:, s2, :, :], kappa11).real + \
                           np.einsum('ijk, mnk, jn', pss[d1][:, s2, :, :], pss[d1][:, s1, :, :], kappa11).real + \
                           np.einsum('ijk, mnk, jn', pss[d1][:, :, s1, :], pss[d1][:, :, s2, :], kappa11).real + \
                           np.einsum('ijk, mnk, jn', pss[d1][:, :, s2, :], pss[d1][:, :, s1, :], kappa11).real

                    rdK00[s1, s2, d1] = rdK00[s2, s1, d1] = self.K_reduction(dK00)

        rdK11 = np.empty((n_species, n_species, X2.shape[0]))
        for s1 in range(n_species):
            for s2 in range(s1):
                for d2 in range(X2.shape[0]):
                    if self.materials is not None:
                        kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                    else:
                        kappa22 = kappa_full

                    dK11 = np.einsum('ijk, mnk, jn', pss2[d2][:, s1, :, :], pss2[d2][:, s2, :, :], kappa22).real + \
                           np.einsum('ijk, mnk, jn', pss2[d2][:, s2, :, :], pss2[d2][:, s1, :, :], kappa22).real + \
                           np.einsum('ijk, mnk, jn', pss2[d2][:, :, s1, :], pss2[d2][:, :, s2, :], kappa22).real + \
                           np.einsum('ijk, mnk, jn', pss2[d2][:, :, s2, :], pss2[d2][:, :, s1, :], kappa22).real

                    rdK11[s1, s2, d1] = rdK11[s2, s1, d1] = self.K_reduction(dK11)

        for s1 in range(n_species):
            for s2 in range(s1):
                for d1 in range(X.shape[0]):
                    for d2 in range(X2.shape[0]):
                        if self.materials is not None:
                            kappa = kappa_all[(material_id[d1], material_id2[d2])]
                        else:
                            kappa = kappa_full
                        if self.verbosity > 1:
                            # self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]) + 'x ' * (d2 + 1) + '. ' * (X2.shape[0] - d2 - 1), end='')
                            self.print('\r{:02}/{:02}: '.format(d1 + 1, X.shape[0]), end='')
                            sys.stdout.flush()

                        dK01 = np.einsum('ijk, mnk, jn', pss[d1][:, s1, :, :], pss2[d2][:, s2, :, :], kappa).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, s2, :, :], pss2[d2][:, s1, :, :], kappa).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, :, s1, :], pss2[d2][:, :, s2, :], kappa).real + \
                               np.einsum('ijk, mnk, jn', pss[d1][:, :, s2, :], pss2[d2][:, :, s1, :], kappa).real

                        rdK01 = self.K_reduction(dK01)
                        rK01 = K[d1, d2] * np.sqrt(rK00[d1] * rK00[d2])

                        dKij = rdK01 / np.sqrt(rK00[d1] * rK11[d2])
                        dKij -= 0.5 * rK01 / (rK00[d1] * rK11[d2]) ** (1.5) * (rdK00[s1, s2, d1] * rK11[d2] + rK00[d1] * rdK11[s1, s2, d2])
                        dKij *= self.exponent * pow(K[d1, d2], self.exponent - 1.)

                        dK_dkappa[s1, s2, d1, d2] = dKij
        """

        # dK/d\alpha
        # Pre-compute terms dependent on only one structure
        rdK00 = np.empty((n_species, X.shape[0]))
        rdK11 = np.empty((n_species, X2.shape[0]))
        for s1 in range(n_species):
            for d1 in range(X.shape[0]):
                if self.materials is not None:
                    kappa11 = kappa_all[(material_id[d1], material_id[d1])]
                else:
                    kappa11 = kappa_full
                dK00 = np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss_dalpha[d1][:, s1], kappa11, kappa11).real + \
                       np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1][:, s1], pss[d1], kappa11, kappa11).real
                rdK00[s1, d1] = self.K_reduction(dK00)

        for s1 in range(n_species):
            for d2 in range(X2.shape[0]):
                if self.materials is not None:
                    kappa22 = kappa_all[(material_id2[d2], material_id2[d2])]
                else:
                    kappa22 = kappa_full
                dK11 = np.einsum('ijkl, mnol, jn, ko', pss2[d2], dpss2_dalpha[d2][:, s1], kappa22, kappa22).real + \
                       np.einsum('ijkl, mnol, jn, ko', dpss2_dalpha[d2][:, s1], pss2[d2], kappa22, kappa22).real
                rdK11[s1, d2] = self.K_reduction(dK11)

        # Compute dK/d\alpha(X, X2)
        for s1 in range(n_species):
            for d1 in range(chunk[0], chunk[1]):
                for d2 in range(X2.shape[0]):
                    if self.materials is not None:
                        kappa = kappa_all[(material_id[d1], material_id2[d2])]
                    else:
                        kappa = kappa_full
                    if self.verbosity > 1:
                        self.print('\r{:02}/{:02}: '.format(d1 + 1, chunksizes[self.mpi_rank]), end='')
                        sys.stdout.flush()

                    # dK01 = np.einsum('ijkl, mnol, jn, ko', dpss_dalpha[d1][:, s1], pss2[d2], kappa, kappa).real + \
                    #        np.einsum('ijkl, mnol, jn, ko', pss[d1], dpss2_dalpha[d2][:, s1], kappa, kappa).real
                    dK01 = np.einsum('ijkmno, jn, ko',
                                     np.einsum('ijkl, mnol', dpss_dalpha[d1][:, s1], pss2[d2]).real,
                                     kappa, kappa) + \
                           np.einsum('ijkmno, jn, ko',
                                     np.einsum('ijkl, mnol', pss[d1], dpss2_dalpha[d2][:, s1]).real,
                                     kappa, kappa)

                    rdK01 = self.K_reduction(dK01)
                    rK01 = K[d1, d2] * np.sqrt(rK00[d1] * rK11[d2])

                    dKij = rdK01 / np.sqrt(rK00[d1] * rK11[d2])
                    dKij -= 0.5 * rK01 / (rK00[d1] * rK11[d2]) ** (1.5) * \
                            (rdK00[s1, d1] * rK11[d2] + rK00[d1] * rdK11[s1, d2])
                    dKij *= self.exponent * pow(K[d1, d2], self.exponent - 1.)

                    dK_slocal[d1 - offsets[self.mpi_rank], d2] = dKij

            if self.parallel_cnlm:
                self.mpi_comm.Allgatherv(dK_slocal,
                                         [dK_s, chunksizes * X2.shape[0], offsets * X2.shape[0], self.MPI.DOUBLE])
            dK_dalpha[s1, :, :] = dK_s

        if self.verbosity > 1:
            self.print('')

        self.Km = np.power(K, self.exponent)
        # FIXME: kappa derivative
        # for s1 in range(n_species):
        #     for s2 in range(s1 + 1, n_species):
        #         dK_dkappa[s1, s2, :, :] = dK_dkappa[s2, s1, :, :]
        self.dK_dalpha = dK_dalpha
        self.reduction_times_X_X2.append(timer() - start)
        return self.Km

    def update_gradients_full(self, dL_dK, X, X2):
        """Updates the derivatives of the parameters of the model.

        Parameters
        ----------
        dL_dK : float
            Derivative of the Likelihood/posterior with respect to the kernel.
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.
        X2 : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.

        Notes
        -----
        The analytical gradient assumes that K(X, X2) has been called
        before requesting the derivative.

        """
        if self.optimize_sigma:
            if self.num_diff:
                # Numerical gradient
                # self.n_eval += 2
                dsigma = 0.0005
                for i, a in enumerate(self.alpha):
                    self.print('+dK[{}]'.format(i))
                    soap = SOAP(self.soap_input_dim, self.sigma, self.r_cut, self.l_max, self.n_max, self.exponent,
                                self.r_grid_points, self.similarity, self.multi_atom, self.verbosity,
                                self.structure_file, parallel='cnlm', optimize_sigma=False, materials=self.materials,
                                elements=self.elements)
                    soap.sigma[i] = soap.sigma[i] + dsigma
                    K1 = soap.K(X, X2)
                    self.print('-dK[{}]'.format(i))
                    soap = SOAP(self.soap_input_dim, self.sigma, self.r_cut, self.l_max, self.n_max, self.exponent,
                                self.r_grid_points, self.similarity, self.multi_atom, self.verbosity,
                                self.structure_file, parallel='cnlm', optimize_sigma=False, materials=self.materials,
                                elements=self.elements)
                    soap.sigma[i] = soap.sigma[i] - dsigma
                    K0 = soap.K(X, X2)
                    self.sigma.gradient[i] = np.sum(dL_dK * (K1 - K0) / (2 * dsigma))
                    np.savez('dK_dsigma_numerical_{}'.format(i), dK=(K1 - K0) / (2 * dsigma))
            else:
                # Analytical gradient
                do_gradient = False
                if (self.last_X_grad is not None):
                    if ((X.shape == self.last_X_grad.shape)
                        and (X == self.last_X_grad).all()):
                        if X2 is None:
                            if self.last_X2_grad is None:
                                pass
                            else:
                                do_gradient = True
                        else:
                            if self.last_X2_grad is None:
                                do_gradient = True
                            else:
                                if ((X2.shape == self.last_X2_grad.shape)
                                    and (X2 == self.last_X2_grad).all()):
                                    pass
                                else:
                                    do_gradient = True
                    else:
                        do_gradient = True
                else:
                    do_gradient = True
                if do_gradient:
                    self.derivative = True
                    _ = self.K(X, X2)

                for i, a in enumerate(self.alpha):
                    self.sigma.gradient[i] = np.sum(dL_dK * self.dK_dalpha[i] / (-self.sigma[i] ** 3))
                    np.savez('dK_dsigma_analytical_{}'.format(i), dK=self.dK_dalpha[i] / (-self.sigma[i] ** 3))
            #self.print(self.sigma.values, self.sigma.gradient)
            #baboom
        if not (self.optimize_exponent or self.optimize_sigma):
            pass

    def update_gradients_diag(self, dL_dKdiag, X):
        """Updates the diagonal of the gradients of the kernel parameters.

        Parameters
        ----------
        dL_dKdiag : float
            Derivative of the Likelihood/posterior with respect to the diagonal of the kernel.
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.

        """
        if self.optimize_sigma:
            for i, a in enumerate(self.alpha):
                self.sigma.gradient[i] = 0.
        if self.optimize_exponent:
            self.exponent.gradient = 0.
        if not (self.optimize_exponent or self.optimize_sigma):
            pass

    def parameters_changed(self):
        """Updates the class when a model parameter is changed.

        """
        self.alpha = 1. / (2 * self.sigma ** 2)
        self.pss_buffer = []  # Forget values of the power spectrum
        self.Kcross_buffer = {}
        self.Kdiag_buffer = {}

        if self.optimize_sigma:
            self.alpha = 1. / (2 * self.sigma**2)
            self.pss_buffer = []    # Forget values of the power spectrum
            self.Kcross_buffer = {}
            self.Kdiag_buffer = {}
            self.derivative = True and (not self.num_diff) # Calculate analytical derivative next iteration
            self.dK_dalpha = None   # Invalidate derivative
        else:
            pass

    def gradients_X(self, dL_dK, X, X2=None):
        """Compute the derivatives with respect to the inputs of the model.

        Parameters
        ----------
        dL_dK : float
            Derivative of the Likelihood/posterior with respect to the kernel.
        X : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.
        X2 : 2-D ndarray (nstructures1, 1)
            Id's for a set of structures.

        Notes
        -----
        Not implemented, it cannot be defined as the inputs are not continuous.

        """
        pass
