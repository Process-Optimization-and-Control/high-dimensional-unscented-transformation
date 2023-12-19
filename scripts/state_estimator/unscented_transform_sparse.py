# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

import numpy as np
import scipy.linalg
import scipy.sparse



def unscented_transformation_gut(sigmas, wm, wc, symmetrization = True):
    """
    Calculates mean and covariance of sigma points by the unscented transform.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    symmetrization : TYPE bool, optional
        DESCRIPTION. Default is true. Symmetrize covariance/correlation matrix afterwards with Py = .5*(Py+Py.T)
    

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    Py : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

    """
    try:
        (n, dim_sigma) = sigmas.shape
    except ValueError: #sigmas is 1D
        sigmas = np.atleast_2d(sigmas)
        (n, dim_sigma) = sigmas.shape 
        assert dim_sigma == wm.shape[0], "Dimensions are wrong"
    print(f"{sigmas=}")
    print(f"{wm=}")
    mean = sigmas @ wm
    print(f"{mean=}")
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - mean.reshape(-1, 1)
    
    # Py = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas.T)])
    Py = (wc*sigmas) @ sigmas.T

    if symmetrization:
        Py = .5*(Py + Py.T)
    
    return mean, Py

def unscented_transformation_sparse(sigmas, wm, wc, symmetrization = True):
    """
    Calculates mean and covariance of sigma points by the unscented transform.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(n, dim_sigma)
        DESCRIPTION. Array of sigma points. Each column contains a sigma point
    wm : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the mean calculation of each sigma point.
    wc : TYPE np.array(dim_sigma,)
        DESCRIPTION. Weights for the covariance calculation of each sigma point.
    symmetrization : TYPE bool, optional
        DESCRIPTION. Default is true. Symmetrize covariance/correlation matrix afterwards with Py = .5*(Py+Py.T)
    

    Returns
    -------
    mean : TYPE np.array(dim_y,)
        DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
    Py : TYPE np.array(dim_y,dim_y)
        DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

    """
    try:
        (n, dim_sigma) = sigmas.shape
    except ValueError: #sigmas is 1D
        sigmas = np.atleast_2d(sigmas)
        (n, dim_sigma) = sigmas.shape 
        assert dim_sigma == wm.shape[0], "Dimensions are wrong"
    

    wm = scipy.sparse.csr_array(wm.reshape(-1,1))    
    wc = scipy.sparse.csr_array(np.ones((n,1)) @ wc[np.newaxis,:]) #stack vertically

    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - scipy.sparse.hstack([mean]*dim_sigma)
    
    Py = (wc*sigmas) @ sigmas.T

    if symmetrization:
        Py = .5*(Py + Py.T)
    return mean, Py


def unscented_transform_w_function_eval(x_mean, sigmas, wm, wc, func, first_yi = None, symmetrization = True, ut_func = None):
    
    dim_x, dim_sigma = sigmas.shape
    if ut_func is None:
        ut_func = unscented_transformation_gut
    
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = scipy.sparse.csr_array(func(x_mean + sigmas[:, [0]]))
        
    second_yi = scipy.sparse.csr_array(func(x_mean + sigmas[:, [1]]))
    dim_y = second_yi.shape[0]
    sig_y = scipy.sparse.csr_array((dim_y, dim_sigma))
    
    sig_y[:, [0]] = first_yi
    sig_y[:, [1]] = second_yi
    for i in range(2, dim_sigma):
        sig_y[:, [i]] = scipy.sparse.csr_array(func(x_mean + sigmas[:, [i]]))
    mean_y, P_y = ut_func(sig_y, wm, wc, symmetrization = symmetrization)
    return mean_y, P_y

