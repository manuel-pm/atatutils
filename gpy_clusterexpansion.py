from __future__ import print_function

import os
import six

import numpy as np

import GPy  #_dev as GPy
from GPy.kern.src.basis_funcs import BasisFuncKernel as BasisFuncKernel
from paramz.caching import Cache_this

from atatutils.clusterexpansion import ClusterExpansion as CE

class ClusterExpansionBasisFuncKernel(BasisFuncKernel):
    def __init__(self, input_dim, ce_diameters, ce_fit_formation, ce_data_filename,
                 variance=1., active_dims=None, ARD=True, name='cluster_expansion_basis'):
        """GPy kernel with Cluster Expansion basis functions.
        Notes
        -----
        As a degenerate kernel, the number of basis functions
        is finite. The maximum number of points and maximum
        diameter for each type (no. of points) are input
        parameters to the kernel. There are no optimizable
        parameters in the kernel.
        """
        self.ce_fit_formation = ce_fit_formation
        self.ce_diameters = ce_diameters
        self.ce_fit_formation = ce_fit_formation
        self.ce = CE(diameters=ce_diameters, data_filename=ce_data_filename)
        super(ClusterExpansionBasisFuncKernel, self).__init__(input_dim, variance, active_dims, ARD, name)

    @Cache_this(limit=3, ignore_args=())
    def _phi(self, X):
        """
        
        Parameters
        ----------
        X : 
            Vector of inputs. 

        Returns
        -------
        design_matrix : 2-D np.ndarray
            Design matrix.

        """
        X = X.ravel()
        if not isinstance(X[0], six.string_types):
            X = np.asarray(abs(np.asarray(X, dtype=int)), dtype=str)
        design_matrix = np.empty((len(X), self.ce.n_basis))
        for i, folder in enumerate(X):
            base_folder = os.path.join(self.ce.base_dir, folder)
            x, x_1, x_t = self.ce.get_concentration(base_folder)
            design_matrix[i, :] = self.ce.get_design_vector(base_folder, x)
        return design_matrix

    def parameters_changed(self):
        BasisFuncKernel.parameters_changed(self)

