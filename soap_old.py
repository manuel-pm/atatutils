import numpy as np
import scipy.special as sp

import ase


def iv(v, x):
    return sp.iv(v + 0.5, x) * np.sqrt(np.pi / (2 * x))


def sum_squares_odd_integers(n):
    return n * (2 * n + 1) * (2 * n - 1) / 3


def cart2sph(coords):
    r = np.linalg.norm(coords, axis=1)
    coords_hat = (coords.T / r).T
    theta = np.arccos(coords_hat[:, 2])
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    return r, theta, phi


def sph2cart(r, theta, phi):
    rsin_theta = r * np.sin(theta)
    x = rsin_theta * np.cos(phi)
    y = rsin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


class SOAP(object):
    def __init__(self, sigma=1., l_max=12):
        self.alpha = 1. / (2 * sigma)
        self.l_max = l_max

    def k(self, atoms0, atoms1, l_max=None):
        r0_cartesian = atoms0.positions
        r1_cartesian = atoms1.positions

        r0, theta0, phi0 = cart2sph(r0_cartesian)
        r1, theta1, phi1 = cart2sph(r1_cartesian)

        # print 'r, t, p 0:', r0, theta0, phi0
        # print 'r, t, p 1:', r1, theta1, phi1

        if l_max is None:
            l_max = self.l_max
        I = np.zeros(sum_squares_odd_integers(l_max + 1), dtype=complex)

        idx = 0
        for l in range(0, l_max):
            for m0 in range(2 * l + 1):
                for m1 in range(2 * l + 1):
                    for i in range(r0.shape[0]):
                        for j in range(r1.shape[0]):
                            I_01 = 4 * np.pi *\
                                   np.exp(-self.alpha *
                                          (r0[i] * r0[i] + r1[j] * r1[j]) /
                                          2) * \
                                   iv(l, self.alpha * r0[i] * r1[j]) * \
                                   sp.sph_harm(m0 - l, l, theta0[i], phi0[i])\
                                   * \
                                   np.conj(sp.sph_harm(m1 - l, l, theta1[j],
                                                   phi1[j]))
                            I[idx] += I_01
                    idx += 1

        return np.sum(np.conj(I) * I).real

    def K(self, atoms0, atoms1, exponent=1, l_max=None):
        k01 = self.k(atoms0, atoms1, l_max)
        k00 = self.k(atoms0, atoms0, l_max)
        k11 = self.k(atoms1, atoms1, l_max)
        # print 'ks: ', k01, k00, k11
        return pow(k01 / np.sqrt(k00 * k11), exponent)

if __name__ == '__main__':
    import numpy as np

    from ase import Atoms
    from ase.calculators.nwchem import NWChem
    from ase.optimize import BFGS

    from numpy import cross, eye
    from scipy.linalg import expm3, norm


    def M(axis, theta):
        return expm3(cross(eye(3), axis / norm(axis) * theta))

    h2 = Atoms('H2',
               positions=[[0, 0, 0],
                          [0, 0, 0.7]])
    h2.calc = NWChem(xc='PBE')
    opt = BFGS(h2)
    opt.run(fmax=0.02)

    h2.positions = np.dot(M(np.array([1., 1., 0.]), np.pi/4.), h2.positions.T).T
    h2prime = ase.Atoms('H2')
    h2prime.positions = np.copy(h2.positions)
    # print h2.positions
    angle = np.pi / 5.
    axis = np.array([1., 0., 0.])
    rotation = M(axis, angle)
    soap = SOAP(1.)
    for n in range(10):
        h2prime.positions = np.dot(rotation, h2prime.positions.T).T
        print (n + 1)*angle, soap.K(h2, h2prime)
    # print soap.k(h2prime, h2prime)
    # print soap.k(h2prime, h2)
    # print soap.k(h2, h2prime)
    # print soap.k(h2, h2)
    h2prime.positions = h2prime.positions + np.random.normal(
        size=h2prime.positions.shape)
    soap.K(h2, h2prime)

