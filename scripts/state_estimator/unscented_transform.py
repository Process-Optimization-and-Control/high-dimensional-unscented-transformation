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
from . import sigma_points_classes as spc
import copy



def unscented_transformation(sigmas, wm, wc, symmetrization = True):
    """
    Calculates mean and covariance of sigma points by the unscented transformation.

    Parameters
    ----------
    sigmas : TYPE np.ndarray(dim_y, dim_sigma)
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
    
    mean = sigmas @ wm
    
    #will only work with residuals between sigmas and mean, so "recalculate" sigmas (it is a new variable - but save memory cost by calling it the same as sigmas)
    sigmas = sigmas - mean.reshape(-1, 1)
    
    # Py = sum([wc_i*(np.outer(sig_i, sig_i)) for wc_i, sig_i in zip(wc, sigmas.T)])
    Py = (wc*sigmas) @ sigmas.T

    if symmetrization:
        Py = .5*(Py + Py.T)
    
    return mean, Py


def unscented_transform_w_function_eval(sigmas, wm, wc, func, first_yi = None, symmetrization = True, ut_func = None):
    dim_x, dim_sigma = sigmas.shape
    
    if ut_func is None:
        ut_func = unscented_transformation
    
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = func(sigmas[:, 0])
        
    second_yi = func(sigmas[:, 1])
    dim_y = second_yi.shape[0]
    sig_y = np.zeros((dim_y, dim_sigma))
    sig_y[:, 0] = first_yi
    sig_y[:, 1] = second_yi
    for i in range(2, dim_sigma):
        sig_y[:, i] = func(sigmas[:, i]) 
    mean_y, P_y = ut_func(sig_y, wm, wc, symmetrization = symmetrization)
    return mean_y, P_y

def unscented_transform_w_function_eval_wslr(sigmas, wm, wc, func, first_yi = None, symmetrization = True, Px = None):
    dim_x, dim_sigma = sigmas.shape
    

    
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = func(sigmas[:, 0])
        
    second_yi = func(sigmas[:, 1])
    dim_y = second_yi.shape[0]
    sig_y = np.zeros((dim_y, dim_sigma))
    sig_y[:, 0] = first_yi
    sig_y[:, 1] = second_yi
    for i in range(2, dim_sigma):
        sig_y[:, i] = func(sigmas[:, i]) 
  
    mean_y = sig_y @ wm
    
    #normalized sigmas
    sig_yn = sig_y - mean_y.reshape(-1, 1)
    sig_yn_wc = wc*sig_yn
    Py = sig_yn_wc @ sig_yn.T
    Py = .5*(Py + Py.T) #symmetrize
    
    #cross co-variance
    sig_xn = sigmas - sigmas[:,0].reshape(-1,1)
    sig_xn_wc = wc*sig_xn
    Pxy = sig_xn_wc @ sig_yn.T
    
    #covariance of Px
    if Px is None:
        Px = sig_xn_wc @ sig_xn.T
        if symmetrization:
            Px = .5*(Px + Px.T) #symmetrize
        assert (dim_x, dim_x) == Px.shape, f"Px dimension wrong, dims are {Px.shape}"
        assert (dim_x, dim_y) == Pxy.shape, f"Pxy dimension wrong, dims are {Pxy.shape}"
    
    #Linear regression parameter
    # A = scipy.linalg.solve(Px, Pxy.T)
    A = Pxy.T @ np.linalg.inv(Px)
    
    return mean_y, Py, A
    
def hdut_map_func(xm, Px_sqrt, map_func, points_a, func_args = [], calc_Pxy = True, calc_A = True, constraint_func_sig_gen = None, return_sigmas_prop = False, sigma_func_arg_0 = False, Px_sqrt_inv = None):
    """
    Some steps takes to increase computational speed. High-dimensional UT using Julier's sigma-points with kappa = 3 - dim_x (minimizing 4th order moments for a normal distribution).

    Parameters
    ----------
    xm : TYPE np.array(dim_x,)
        DESCRIPTION. Mean of x
    Px : TYPE np.array((dim_x, dim_x))
        DESCRIPTION. Covariance matrix of x
    map_func : TYPE map function
        DESCRIPTION. (dim_y x (2*dim_x + 1)) <-- (dim_x, (2*dim_x + 1)). Maps y = f(x) for (2*dim_x + 1) samples of x. Output must be  a numpy array. Typically, this is a CasADi map function, or map_func = lambda sig: np.array(list(map(f, sig.T))).T if the function f is a numpy-based function
    calc_Pxy_and_A : TYPE, optional, bool
        DESCRIPTION. The default is True. Whether to calculate Pxy and A (WSLR matrix) or not. Uses some extra matrix operations to do this.

    Returns
    -------
    ym : TYPE np.array(dim_y)
        DESCRIPTION. Mean estimate of y
    Py : TYPE np.array((dim_y, dim_y))
        DESCRIPTION. Covariance estimate of y
    Pxy : TYPE np.array((dim_x, dim_y)) if calc_Pxy_and_A == True
        DESCRIPTION. Cross-covariance estimate
    A : TYPE np.array((dim_y, dim_x)) if calc_Pxy_and_A == True
        DESCRIPTION. UT implicitly does weighted statistical linear regression, i.e., y = A @ x + b + epsilon, where A and b are linearization parameters and epsilon is the zero-mean regression error (a RV). Returns A.

    """
    
    #check input
    dim_x = xm.shape[0]
    assert ((xm.ndim == 1) and (Px_sqrt.ndim == 2) and ((dim_x, dim_x) == Px_sqrt.shape)), "Wrong input."
    
    if calc_Pxy and (Px_sqrt_inv is None):
        Px_sqrt_inv = np.linalg.inv(Px_sqrt)
     
    
    #generate sigma-points
    sigmas_a = points_a.compute_sigma_points(xm, None, P_sqrt = Px_sqrt)[0]
    
    # constrain sigma-points (if constraint_func_sig_gen is provided)
    if not (constraint_func_sig_gen is None):
        sigmas_a = np.array(constraint_func_sig_gen(sigmas_a))
    
    # make arguments to sigmas_a. First argument in the list func_args is always sigmas_a.
    if sigma_func_arg_0:
        func_args[0] = sigmas_a
    
    #evaluate the sigma-points through the function y=f(x) by the map-construct
    sig_y = map_func(sigmas_a, *func_args) #evaluate sigma-points. 
    sig_y = np.array(sig_y)
    assert (isinstance(sig_y, np.ndarray) and (sig_y.shape[1] == (2*dim_x + 1)) and (sig_y.ndim == 2)), f"sig_y is wrong. {type(sig_y)=}. It should be a np.ndarray with 2 dimensions"
    dim_y = sig_y.shape[0]
    
    if return_sigmas_prop: #may be convenient for debugging
        return_sigmas_prop = copy.deepcopy(sig_y)
        
    #points belonging to the "first" sigma-point matrix (y1, y1.shape = (dim_y,3)) should not be changed. The rest should be subtracted y0
    ix = np.array([i for i in range(2, 2*dim_x + 1) if i != dim_x + 1])
    sig_y[:, ix] -= sig_y[:, 0, np.newaxis]
    
    #prepare for doing HD-UT. Allocate arrays and set weights for the 1D UT
    ym = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    if calc_Pxy:
        Pxy = np.zeros((dim_x, dim_y))
    
    #weights are the same for each 1D distribution. Create them here, before the for-loop
    if points_a.type == "ScaledSigmaPoints":
        points_new = spc.ScaledSigmaPoints(1, alpha = points_a.alpha, beta = points_a.beta, kappa = 3. - 1, kappa_func = points_a.kappa_func, suppress_init_warning = True)
        Wm_ei, Wc_ei = points_new.compute_weights()
    elif points_a.type == "JulierSigmaPoints":
        points_new = spc.JulierSigmaPoints(1, kappa = 3. - 1)
        Wm_ei = points_new.compute_weights()
        Wc_ei = Wm_ei.copy()
        
    #do nx one-dimensional UTs
    for i in range(dim_x):
        sig_yi = sig_y[:, [0, i + 1, dim_x + i +1]] #subset of evaluated sigmas to consider
        if i > 0: #if i == 0, we do not change
            sig_yi[:, 0] = 0.
        ymi, Pyi = unscented_transformation(sig_yi, Wm_ei, Wc_ei) #standard UT in 1D
        ym += ymi
        Py += Pyi
        
        if calc_Pxy:
            sig_yi_norm = sig_yi - ymi[:, np.newaxis]
            
            #select relevant sigma-points
            sig_ci= sigmas_a[i, [0, i + 1, dim_x + i + 1]] 
            
            #Calculate P_xiyi
            sig_ei = (sig_ci - sigmas_a[i, 0, np.newaxis])
            sig_ei_norm_w = Wc_ei*sig_ei #weighted and normalized (it has zero-mean)
            Pxy[i, :] = sig_ei_norm_w @ sig_yi_norm.T
        
    if calc_Pxy:  
        if calc_A:
            del sigmas_a, sig_y, sig_ei, sig_ci #free up space, A may be large
            An = Pxy.T # = Pxy.T @ I_inv 
            A = An @ Px_sqrt_inv

            return ym, Py, Pxy, A, return_sigmas_prop
        else: 
            return ym, Py, Pxy, return_sigmas_prop
    else:
        return ym, Py, return_sigmas_prop
        


def hdut_w_function_eval_wslr_1D(sigmas_e_all, sigmas_ei, wm, wc, func, first_yi = None, symmetrization = True, Pxi_inv = None):
    dim_x, dim_sigma = sigmas_e_all.shape
    dim_xi = sigmas_ei.shape[0]
    assert dim_sigma == sigmas_ei.shape[1]
    
    
    if first_yi is None: #the first (or zeroth) sigma-point has not been calculated outside this function. compute it here
        first_yi = func(sigmas_e_all[:, 0])
        
    second_yi = func(sigmas_e_all[:, 1])
    dim_y = second_yi.shape[0]
    sig_y = np.zeros((dim_y, dim_sigma))
    sig_y[:, 0] = first_yi
    sig_y[:, 1] = second_yi
    for i in range(2, dim_sigma):
        sig_y[:, i] = func(sigmas_e_all[:, i]) 
    
    mean_y = sig_y @ wm
    
    #normalized sigmas
    sig_yn = sig_y - mean_y.reshape(-1, 1)
    sig_yn_wc = wc*sig_yn
    Py = sig_yn_wc @ sig_yn.T
    Py = .5*(Py + Py.T) #symmetrize
    
    #cross co-variance
    sig_ei_n = sigmas_ei - sigmas_ei[:,0].reshape(-1,1)
    sig_ei_n_wc = wc*sig_ei_n
    Pxiy = sig_ei_n_wc @ sig_yn.T
    
    #covariance of Px
    if Pxi_inv is None:
        Pxi = sig_ei_n_wc @ sig_ei_n.T
        if symmetrization:
            Pxi = .5*(Pxi + Pxi.T) #symmetrize
        assert (dim_xi, dim_xi) == Pxi.shape, f"Px dimension wrong, dims are {Pxi.shape}"
        assert (dim_x, dim_y) == Pxiy.shape, f"Pxy dimension wrong, dims are {Pxiy.shape}"
    
        #Linear regression parameter
        # A = scipy.linalg.solve(Px, Pxy.T)
        A = Pxiy.T @ np.linalg.inv(Pxi)
    else:
        A = Pxiy.T @ Pxi_inv

        
    return mean_y, Py, A
