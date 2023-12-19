# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:14:43 2023

@author: halvorak
"""
from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as ut
from state_estimator import unscented_transform_sparse as ut_sp
import numpy as np
import scipy.sparse
import time

"""
Implementation of HD-UT, the sparse version

"""


def ut_high_dim_w_function_eval_sparse(x_dists_ind, func, dim_y, sigma_points_method = "julier", kwargs_sigma_points = {}, kwargs_ut = {}, ut_func = None):
    
    
    assert isinstance(x_dists_ind, list)
    dim_x_ind = len(x_dists_ind) #number of independent xi
    
    #total dimension of state vector and the mean of the distribution
    x_mean = []
    for i in range(dim_x_ind):
        x_mean.append(x_dists_ind[i]["mean"])
    x_mean = np.hstack(tuple(x_mean))
    dim_x = x_mean.shape[0]
    
    y_mean = scipy.sparse.csr_array((dim_y, 1))
    Py = scipy.sparse.csr_array((dim_y, dim_y))
    
    idx = 0
    ts = time.time()
    for i in range(dim_x_ind):
        
        #unpack the distribution
        xi_mean = x_dists_ind[i]["mean"]
        Pxi = x_dists_ind[i]["cov"]
        dim_xi = xi_mean.shape[0]
        xi_mean = np.zeros(dim_xi)
        assert (dim_xi, dim_xi) == Pxi.shape, f"Wrong dimensions, {Pxi.shape=} and expected to be square of {dim_xi=}"
        
        #select sigma-point method
        if sigma_points_method == "julier":
            points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "scaled":
            points = spc.ScaledSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        else:
            raise IndexError(f"{sigma_points_method=} is not implemented yet.")
        
        sigmas_aug = scipy.sparse.csr_array((dim_x, points.num_sigma_points()))
        
        #insert sigmas into the augmented sigmas
        sigmas_aug[idx:idx+dim_xi, :] = sigmas
        
        x_mean = x_mean.reshape(-1,1)
        
        if i == 0:
            yi_mean, Pyi = ut_sp.unscented_transform_w_function_eval(x_mean, sigmas_aug, Wm, Wc, func, **kwargs_ut, ut_func = ut_func)
        else:
            yi_mean, Pyi = ut_sp.unscented_transform_w_function_eval(x_mean, sigmas_aug, Wm, Wc, func, first_yi = scipy.sparse.csr_array((dim_y, 1)), **kwargs_ut, ut_func = ut_func)

        y_mean += yi_mean
        Py += Pyi
        
        idx += dim_xi
        
        if (i % 100)==0:
            tn = time.time()
            print(f"{i=}/{dim_x_ind}={i/dim_x_ind*100 :.2f}%, time: {tn-ts :.0f}s = {(tn-ts)/60 :.1f}min")
    
    return y_mean, Py

def is_sparse_diag(P):
    r, c = P.nonzero() #rows, columns of indices in P with non-zero entries
    is_diag = (r == c).all()
    return is_diag

 