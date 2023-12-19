# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:14:43 2023

@author: halvorak
"""
from . import sigma_points_classes as spc
from . import unscented_transform as ut
from state_estimator import unscented_transform_sparse as ut_sp
import numpy as np
import scipy.sparse
import time

def ut_high_dim_w_function_eval(x_dists_ind, func, dim_y, sigma_points_method = "julier", kwargs_sigma_points = {}, kwargs_ut = {}, ut_func = None):
    
    
    assert isinstance(x_dists_ind, list)
    
    
    dim_x_ind = len(x_dists_ind) #number of independent xi
    
    #total dimension of state vector and the mean of the distribution
    x_mean = []
    for i in range(dim_x_ind):
        x_mean.append(x_dists_ind[i]["mean"])
    x_mean = np.hstack(tuple(x_mean))
    dim_x = x_mean.shape[0]
    
    y_mean = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    # print(f"{x_mean=} and {dim_x=}")
    idx = 0
    for i in range(dim_x_ind):
        
        #unpack the distribution
        xi_mean = x_dists_ind[i]["mean"]
        Pxi = x_dists_ind[i]["cov"]
        dim_xi = xi_mean.shape[0]
        assert (dim_xi, dim_xi) == Pxi.shape, f"Wrong dimensions, {Pxi.shape=} and expected to be square of {dim_xi=}"
        
        #select sigma-point method
        if sigma_points_method == "genut":
            points = spc.GenUTSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "julier":
            points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "scaled":
            points = spc.ScaledSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        else:
            raise IndexError(f"{sigma_points_method=} is not implemented yet.")
        
        sigmas_aug = np.tile(x_mean.reshape(-1,1), points.num_sigma_points())
        
        # print(f"{sigmas_aug=}")
        # print(f"{sigmas=}")
        
        #insert sigmas into the augmented sigmas
        sigmas_aug[idx:idx+dim_xi, :] = sigmas
        
        if i == 0:
            yi_mean, Pyi = ut.unscented_transform_w_function_eval(sigmas_aug, Wm, Wc, func, **kwargs_ut, ut_func = ut_func)
        else:
            yi_mean, Pyi = ut.unscented_transform_w_function_eval(sigmas_aug, Wm, Wc, func, first_yi = 0., **kwargs_ut, ut_func = ut_func)
        # print(f"{yi_mean=}, {Pyi=}")    
        
        y_mean += yi_mean
        Py += Pyi
        
        idx += dim_xi
        
        if (i % 10)==0:
            print(f"{i=}/{dim_x_ind}")
    
    return y_mean, Py

def ut_high_dim_w_function_eval_wslr(x_dists_ind, func, dim_y, sigma_points_method = "julier", kwargs_sigma_points = {}, kwargs_ut = {}, ut_func = None):
    
    
    assert isinstance(x_dists_ind, list)
    
    
    dim_x_ind = len(x_dists_ind) #number of independent xi
    
    #total dimension of state vector and the mean of the distribution
    x_mean = []
    for i in range(dim_x_ind):
        x_mean.append(x_dists_ind[i]["mean"])
    x_mean = np.hstack(tuple(x_mean))
    dim_x = x_mean.shape[0]
    
    y_mean = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    # print(f"{x_mean=} and {dim_x=}")
    idx = 0
    A_list = []
    for i in range(dim_x_ind):
        
        #unpack the distribution
        xi_mean = x_dists_ind[i]["mean"]
        Pxi = x_dists_ind[i]["cov"]
        dim_xi = xi_mean.shape[0]
        assert (dim_xi, dim_xi) == Pxi.shape, f"Wrong dimensions, {Pxi.shape=} and expected to be square of {dim_xi=}"
        
        #select sigma-point method
        if sigma_points_method == "genut":
            points = spc.GenUTSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "julier":
            points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "scaled":
            points = spc.ScaledSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        else:
            raise IndexError(f"{sigma_points_method=} is not implemented yet.")
        
        sigmas_aug = np.tile(x_mean.reshape(-1,1), points.num_sigma_points())
        
        # print(f"{sigmas_aug=}")
        # print(f"{sigmas=}")
        
        #insert sigmas into the augmented sigmas
        sigmas_aug[idx:idx+dim_xi, :] = sigmas
        
        if i == 0:
            yi_mean, Pyi, Ai = ut.unscented_transform_w_function_eval_wslr(sigmas_aug, Wm, Wc, func, **kwargs_ut)
        else:
            yi_mean, Pyi, Ai = ut.unscented_transform_w_function_eval_wslr(sigmas_aug, Wm, Wc, func, first_yi = 0., **kwargs_ut)
        # print(f"{yi_mean=}, {Pyi=}")    
        
        y_mean += yi_mean
        Py += Pyi
        
        A_list.append(Ai)
        
        idx += dim_xi
        
        if (i % 10)==0:
            print(f"{i=}/{dim_x_ind}")
    
    return y_mean, Py, A_list

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
    
    # print(f"{x_mean=} and {dim_x=}")
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
        if sigma_points_method == "genut":
            points = spc.GenUTSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "julier":
            points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        elif sigma_points_method == "scaled":
            points = spc.ScaledSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(xi_mean, Pxi)
        else:
            raise IndexError(f"{sigma_points_method=} is not implemented yet.")
        
        # sigmas_aug = np.tile(x_mean.reshape(-1,1), points.num_sigma_points())
        sigmas_aug = scipy.sparse.csr_array((dim_x, points.num_sigma_points()))
        
        # print(f"{sigmas=}")
        
        #insert sigmas into the augmented sigmas
        sigmas_aug[idx:idx+dim_xi, :] = sigmas
        
        x_mean = x_mean.reshape(-1,1)
        
        if i == 0:
            yi_mean, Pyi = ut_sp.unscented_transform_w_function_eval(x_mean, sigmas_aug, Wm, Wc, func, **kwargs_ut, ut_func = ut_func)
        else:
            yi_mean, Pyi = ut_sp.unscented_transform_w_function_eval(x_mean, sigmas_aug, Wm, Wc, func, first_yi = scipy.sparse.csr_array((dim_y, 1)), **kwargs_ut, ut_func = ut_func)
        # print(f"{yi_mean=}, {Pyi=}")    
        

        y_mean += yi_mean
        Py += Pyi
        
        idx += dim_xi
        
        if (i % 10)==0:
            tn = time.time()
            print(f"{i=}/{dim_x_ind}={i/dim_x_ind*100 :.2f}%, time: {tn-ts :.0f}s = {(tn-ts)/60 :.1f}min")
    
    return y_mean, Py

def is_sparse_diag(P):
    r, c = P.nonzero() #rows, columns of indices in P with non-zero entries
    is_diag = (r == c).all()
    return is_diag


def correlated_ut_high_dim_w_function_eval(x_dists_ind, eig_vec, func, dim_y, sigma_points_method = "julier", kwargs_sigma_points = {}, kwargs_ut = {}, ut_func = None):
    
    
    assert isinstance(x_dists_ind, list)
    
    
    dim_x_ind = len(x_dists_ind) #number of independent xi
    
    #total dimension of state vector and the mean of the distribution
    x_mean = []
    for i in range(dim_x_ind):
        x_mean.append(x_dists_ind[i]["mean"])
    x_mean = np.hstack(tuple(x_mean))
    dim_x = x_mean.shape[0]
    
    y_mean = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    # print(f"{x_mean=} and {dim_x=}")
    idx = 0
    for i in range(dim_x_ind):
        
        #unpack the distribution
        xi_mean = x_dists_ind[i]["mean"]
        Pxi = x_dists_ind[i]["cov"]
        dim_xi = xi_mean.shape[0]
        assert (dim_xi, dim_xi) == Pxi.shape, f"Wrong dimensions, {Pxi.shape=} and expected to be square of {dim_xi=}"
        
        #select sigma-point method

        if sigma_points_method == "julier":
            points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
            sigmas, Wm, Wc, P_sqrt = points.compute_sigma_points(0., Pxi)
        else:
            raise IndexError(f"{sigma_points_method=} is not implemented yet.")
        
        sigmas_aug = np.tile(x_mean.reshape(-1,1), points.num_sigma_points())
        
        # print(f"{sigmas_aug=}")
        # print(f"{sigmas=}")
        
        #insert sigmas into the augmented sigmas
        sigmas_aug[idx:idx+dim_xi, :] = sigmas
        
        if i == 0:
            yi_mean, Pyi = ut.unscented_transform_w_function_eval(sigmas_aug, Wm, Wc, func, **kwargs_ut, ut_func = ut_func)
        else:
            yi_mean, Pyi = ut.unscented_transform_w_function_eval(sigmas_aug, Wm, Wc, func, first_yi = 0., **kwargs_ut, ut_func = ut_func)
        # print(f"{yi_mean=}, {Pyi=}")    
        
        y_mean += yi_mean
        Py += Pyi
        
        idx += dim_xi
        
        if (i % 10)==0:
            print(f"{i=}/{dim_x_ind}")
    
    return y_mean, Py
    
def hdut_map_func(xm, Px, map_func, calc_Pxy_and_A = True):
    """
    High-dimensional UT using Julier's sigma-points with kappa = 3 - dim_x (minimizing 4th order moments for a normal distribution).

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
    assert ((xm.ndim == 1) and (Px.ndim == 2) and ((dim_x, dim_x) == Px.shape)), "Wrong input."
    
    #SVD and compute sqrt(Px)    
    eigval, eigvec = np.linalg.eig(Px)
    Px_sqrt = eigvec*np.sqrt(eigval) #equivalent to eigvec @ np.diag(np.sqrt(eigval)) @ I, but faster and less memory consumption
    assert np.allclose(Px_sqrt @ Px_sqrt.T, Px), "Px_sqrt is wrong." 
    
    #generate augmented sigma-points first
    points_a = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x, suppress_kappa_warning = True)
    sigmas_a, _, _, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt)
    
    #evaluate the sigma-points through the function y=f(x) by the map-construct
    sig_y = map_func(sigmas_a) #evaluate sigma-points. 
    assert (isinstance(sig_y, np.ndarray) and (sig_y.shape[1] == (2*dim_x + 1)) and (sig_y.ndim == 2)), f"sig_y is wrong. {type(sig_y)=}. It should be a np.ndarray with 2 dimensions"
    dim_y = sig_y.shape[0]
    
    #points belonging to the "first" sigma-point in the eigenspace should not be changed. The rest should be subtracted y0
    ix = np.array([i for i in range(2, 2*dim_x + 1) if i != dim_x + 1])
    sig_y[:, ix] -= sig_y[:, 0, np.newaxis]
    
    #prepare for doing HD-UT. Allocate arrays and set weights for the 1D UT
    ym = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    A = [] #WSLR matrix
    Pxy = []
    
    # A2 = np.zeros((dim_y, dim_x)) #speed-up
    Pxy2 = np.zeros((dim_x, dim_y))
    
    Wm_ei = np.array([4/6, 1/6, 1/6]) #always true for 1D distribution with kappa = 3-1
    Wc_ei = Wm_ei.copy()
    for i in range(dim_x):
        sig_yi = sig_y[:, [0, i + 1, dim_x + i +1]] #subset of evaluated sigmas to consider
        if i > 0: #if i == 0, we do not change
            sig_yi[:, 0] = 0.
        ymi, Pyi = ut.unscented_transformation_gut(sig_yi, Wm_ei, Wc_ei) #standard UT in 1D
        ym += ymi
        Py += Pyi
        
        if calc_Pxy_and_A:
            sig_yi_norm = sig_yi - ymi[:, np.newaxis]
            
            
            #x in cartesian space. Transform to eigenbasis and then obtain Pxiy
            sig_ci= sigmas_a[:, [0, i + 1, dim_x + i +1]] #cartesian
            
            sig_ei = eigvec.T @ (sig_ci - sigmas_a[:, 0, np.newaxis]) #eigenbasis
            sig_ei = sig_ei[np.newaxis, i, :]
            
            sig_ei_norm_w = Wc_ei*sig_ei #weighted and normalized (it has zero-mean)
            Pxiy = sig_ei_norm_w @ sig_yi_norm.T 
            print(f"\n\nfunc: {Pxiy.shape=}")
            Ai = Pxiy.T @ np.array([[1/eigval[i]]])
            A.append(Ai)
            Pxy.append(Pxiy)
            
            #possible speed-up by selecting appropriate eigenvector (avoids transpose and full matrix-vector multiplication) and setting elements directly into Pxy2 and A2 (avoid stacking later on)
            sig_ei2 = eigvec[:, i].reshape(1, -1) @ (sig_ci - sigmas_a[:, 0, np.newaxis])
            sig_ei_norm_w2 = Wc_ei*sig_ei2 #weighted and normalized (it has zero-mean)
            Pxy2[i, :] = sig_ei_norm_w2 @ sig_yi_norm.T
            print(f"{Pxy2[i, :, np.newaxis].shape=} and {np.array([[1/eigval[i]]]).shape=}")
            # A2[:, i] = (Pxy2[np.newaxis, i, :].T @ np.array([[1/eigval[i]]])).flatten()
            
            #this works
            # A2[:, i] = Pxy2[i, :, np.newaxis] @ np.array([1/eigval[i]])
            
            
            
            
            
            # #x in cartesian space. Transform to eigenbasis and then obtain Pxiy
            # sig_ci= sigmas_a[i, [0, i + 1, dim_x + i +1]] #cartesian
            # sig_ei = eigvec.T @ (sig_ci - sigmas_a[i, 0]) #eigenbasis
            # sig_ei_norm_w = Wc_ei*sig_ei #weighted and normalized (it has zero-mean)
            # Pxiy = sig_ei_norm_w @ sig_yi_norm.T 
            # print(f"{Pxiy=}")
            # Ai = Pxiy.T @ np.array([[1/eigval[i]]])
            # print(f"{Ai=}")
            # A.append(Ai)
            # Pxy.append(Pxiy)
        
    if calc_Pxy_and_A:
        A = np.hstack((A))
        Pxy = np.vstack((Pxy))
        
        assert np.allclose(Pxy, Pxy2)
        # A2 = Pxy2.T @ np.diag(1/eigval)
        # A2 = Pxy2.T * (1/eigval) #equivalent
        A2 = Pxy2.T / eigval # = Pxy.T * (1/eigval) = Pxy.T @ np.diag(1/eigval)
        assert np.allclose(A, A2)
        return ym, Py, Pxy, A
    else:
        return ym, Py
        
        
    
def hdut_map_func_fast(xm, Px, map_func, calc_Pxy = True, calc_A = True):
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
    assert ((xm.ndim == 1) and (Px.ndim == 2) and ((dim_x, dim_x) == Px.shape)), "Wrong input."
    
    #SVD and compute sqrt(Px)    
    eigval, eigvec = np.linalg.eig(Px)
    del Px #this matrix can be large
    
    Px_sqrt = eigvec*np.sqrt(eigval) #equivalent to eigvec @ np.diag(np.sqrt(eigval)) @ I, but faster and less memory consumption
    # assert np.allclose(Px_sqrt @ Px_sqrt.T, Px), "Px_sqrt is wrong." 
    
    #generate augmented sigma-points first
    points_a = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x, suppress_kappa_warning = True)
    sigmas_a, _, _, _ = points_a.compute_sigma_points(xm, None, P_sqrt = Px_sqrt)
    del Px_sqrt
    
    #evaluate the sigma-points through the function y=f(x) by the map-construct
    sig_y = map_func(sigmas_a) #evaluate sigma-points. 
    assert (isinstance(sig_y, np.ndarray) and (sig_y.shape[1] == (2*dim_x + 1)) and (sig_y.ndim == 2)), f"sig_y is wrong. {type(sig_y)=}. It should be a np.ndarray with 2 dimensions"
    dim_y = sig_y.shape[0]
    
    #points belonging to the "first" sigma-point in the eigenspace should not be changed. The rest should be subtracted y0
    ix = np.array([i for i in range(2, 2*dim_x + 1) if i != dim_x + 1])
    sig_y[:, ix] -= sig_y[:, 0, np.newaxis]
    
    #prepare for doing HD-UT. Allocate arrays and set weights for the 1D UT
    ym = np.zeros(dim_y)
    Py = np.zeros((dim_y, dim_y))
    if calc_Pxy:
        Pxy = np.zeros((dim_x, dim_y))
    
    Wm_ei = np.array([4/6, 1/6, 1/6]) #always true for 1D distribution with kappa = 3-1
    Wc_ei = Wm_ei.copy()
    for i in range(dim_x):
        sig_yi = sig_y[:, [0, i + 1, dim_x + i +1]] #subset of evaluated sigmas to consider
        if i > 0: #if i == 0, we do not change
            sig_yi[:, 0] = 0.
        ymi, Pyi = ut.unscented_transformation_gut(sig_yi, Wm_ei, Wc_ei) #standard UT in 1D
        ym += ymi
        Py += Pyi
        
        if calc_Pxy:
            sig_yi_norm = sig_yi - ymi[:, np.newaxis]
            
            #x in cartesian space. Transform to eigenbasis and then obtain Pxiy
            sig_ci= sigmas_a[:, [0, i + 1, dim_x + i +1]] #cartesian
            
            sig_ei = eigvec[np.newaxis, :, i] @ (sig_ci - sigmas_a[:, 0, np.newaxis])
            sig_ei_norm_w = Wc_ei*sig_ei #weighted and normalized (it has zero-mean)
            Pxy[i, :] = sig_ei_norm_w @ sig_yi_norm.T
           
        
    if calc_Pxy:
        if calc_A:
            del sigmas_a, sig_y, sig_ei, sig_ci #free up space, A may be large
            A = Pxy.T / eigval # = Pxy.T * (1/eigval) = Pxy.T @ np.diag(1/eigval)
            return ym, Py, Pxy, A
        else: 
            return ym, Py, Pxy
    else:
        return ym, Py
        
        
    
# def hdnut_map_func_fast(xm, Px, map_func, calc_Pxy = True, calc_A = True):
#     """
#     Some steps takes to increase computational speed. High-dimensional UT using Julier's sigma-points with kappa = 3 - dim_x (minimizing 4th order moments for a normal distribution).

#     Parameters
#     ----------
#     xm : TYPE np.array(dim_x,)
#         DESCRIPTION. Mean of x
#     Px : TYPE np.array((dim_x, dim_x))
#         DESCRIPTION. Covariance matrix of x
#     map_func : TYPE map function
#         DESCRIPTION. (dim_y x (2*dim_x + 1)) <-- (dim_x, (2*dim_x + 1)). Maps y = f(x) for (2*dim_x + 1) samples of x. Output must be  a numpy array. Typically, this is a CasADi map function, or map_func = lambda sig: np.array(list(map(f, sig.T))).T if the function f is a numpy-based function
#     calc_Pxy_and_A : TYPE, optional, bool
#         DESCRIPTION. The default is True. Whether to calculate Pxy and A (WSLR matrix) or not. Uses some extra matrix operations to do this.

#     Returns
#     -------
#     ym : TYPE np.array(dim_y)
#         DESCRIPTION. Mean estimate of y
#     Py : TYPE np.array((dim_y, dim_y))
#         DESCRIPTION. Covariance estimate of y
#     Pxy : TYPE np.array((dim_x, dim_y)) if calc_Pxy_and_A == True
#         DESCRIPTION. Cross-covariance estimate
#     A : TYPE np.array((dim_y, dim_x)) if calc_Pxy_and_A == True
#         DESCRIPTION. UT implicitly does weighted statistical linear regression, i.e., y = A @ x + b + epsilon, where A and b are linearization parameters and epsilon is the zero-mean regression error (a RV). Returns A.

#     """

# ### from the main script
# import sklearn.datasets
# import utils
# dim_x = 3
# # xm, Px, map_func, calc_Pxy = True, calc_A = True
# xm = np.zeros(dim_x)
# # xm = np.random.uniform(low = 1., high = 10., size = dim_x)
# Px = sklearn.datasets.make_spd_matrix(dim_x)

# std_dev, corr = utils.get_corr_std_dev(Px)
# del Px
# np.fill_diagonal(corr, 0.)
# assert ((corr > -1).all() and (corr < 1).all())
# np.fill_diagonal(corr, 1.)
# assert (np.diag(corr) == 1.).all()
# if True: #should be equivalent formulation, but avoid storing all the zeros
#     Px = std_dev.reshape(-1,1) * corr * std_dev
# else:
#     Px = np.diag(std_dev) @ corr @ np.diag(std_dev)
# del corr

# func = lambda x_in: x_in**2
# map_func = lambda s: np.array(list(map(func, s.T))).T
# calc_Pxy = True
# calc_A = True
# ###function

# #check input
# dim_x = xm.shape[0]
# assert ((xm.ndim == 1) and (Px.ndim == 2) and ((dim_x, dim_x) == Px.shape)), "Wrong input."

# std_dev, corr = utils.get_corr_std_dev(Px)

# #SVD and compute sqrt(Px)    
# eigval, eigvec = np.linalg.eig(Px)
# eigval2, eigvec2 = np.linalg.eig(corr)
# eigvec2 = eigvec2 @ np.diag(1/std_dev)
# norm = np.array([np.linalg.norm(eigvec2[:, i]) for i in range(dim_x)])
# norm = np.vstack([[norm] for i in range(dim_x)])
# eigvec3 = eigvec2

# assert np.allclose(eigvec3, eigvec)
# print(f"{eigvec3}")
# raise ValueError
# del Px #this matrix can be large

# Px_sqrt = eigvec*np.sqrt(eigval) #equivalent to eigvec @ np.diag(np.sqrt(eigval)) @ I, but faster and less memory consumption
# # assert np.allclose(Px_sqrt @ Px_sqrt.T, Px), "Px_sqrt is wrong." 

# #generate augmented sigma-points first
# points_a = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x, suppress_kappa_warning = True)
# sigmas_a, _, _, _ = points_a.compute_sigma_points(xm, None, P_sqrt = Px_sqrt)
# del Px_sqrt

# #evaluate the sigma-points through the function y=f(x) by the map-construct
# sig_y = map_func(sigmas_a) #evaluate sigma-points. 
# assert (isinstance(sig_y, np.ndarray) and (sig_y.shape[1] == (2*dim_x + 1)) and (sig_y.ndim == 2)), f"sig_y is wrong. {type(sig_y)=}. It should be a np.ndarray with 2 dimensions"
# dim_y = sig_y.shape[0]

# #points belonging to the "first" sigma-point in the eigenspace should not be changed. The rest should be subtracted y0
# ix = np.array([i for i in range(2, 2*dim_x + 1) if i != dim_x + 1])
# sig_y[:, ix] -= sig_y[:, 0, np.newaxis]

# #prepare for doing HD-UT. Allocate arrays and set weights for the 1D UT
# ym = np.zeros(dim_y)
# Py = np.zeros((dim_y, dim_y))
# if calc_Pxy:
#     Pxy = np.zeros((dim_x, dim_y))

# Wm_ei = np.array([4/6, 1/6, 1/6]) #always true for 1D distribution with kappa = 3-1
# Wc_ei = Wm_ei.copy()
# for i in range(dim_x):
#     sig_yi = sig_y[:, [0, i + 1, dim_x + i +1]] #subset of evaluated sigmas to consider
#     if i > 0: #if i == 0, we do not change
#         sig_yi[:, 0] = 0.
#     ymi, Pyi = ut.unscented_transformation_gut(sig_yi, Wm_ei, Wc_ei) #standard UT in 1D
#     ym += ymi
#     Py += Pyi
    
#     if calc_Pxy:
#         sig_yi_norm = sig_yi - ymi[:, np.newaxis]
        
#         #x in cartesian space. Transform to eigenbasis and then obtain Pxiy
#         sig_ci= sigmas_a[:, [0, i + 1, dim_x + i +1]] #cartesian
        
#         sig_ei = eigvec[np.newaxis, :, i] @ (sig_ci - sigmas_a[:, 0, np.newaxis])
#         sig_ei_norm_w = Wc_ei*sig_ei #weighted and normalized (it has zero-mean)
#         Pxy[i, :] = sig_ei_norm_w @ sig_yi_norm.T
       
    
# if calc_Pxy:
#     if calc_A:
#         del sigmas_a, sig_y, sig_ei, sig_ci #free up space, A may be large
#         A = Pxy.T / eigval # = Pxy.T * (1/eigval) = Pxy.T @ np.diag(1/eigval)
#         # return ym, Py, Pxy, A
#     else: 
#         pass
#         # return ym, Py, Pxy
# else:
#     pass
#     # return ym, Py
        
        

