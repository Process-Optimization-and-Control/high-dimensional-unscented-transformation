# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:16:16 2023

@author: halvorak
"""
import numpy as np
import scipy.stats
import scipy.linalg
import sklearn.decomposition
import sklearn.datasets

#Self-written modules
from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as ut
import utils

"""
Shows 
    - How to calculate linearization matrices
    - The non-recommended (but illustrative) implementation of HD-UT using normalized RV.

"""

#%% select sigma-point method
sp_method = "Julier"
# sp_method = "scaled"

sqrt_method = "chol"
# sqrt_method = "sqrtm"
# sqrt_method = "SVD"


#%% Def x-dist
dim_x = int(20) #can be varied
kwargs_sp_scaled = dict(alpha = 3e-1, beta = 2., kappa = 3.-dim_x)
xm = np.array([i for i in range(2, 2+dim_x)], dtype = float)

Px = sklearn.datasets.make_spd_matrix(dim_x)
std_dev_x, corr = utils.get_corr_std_dev(Px)
np.fill_diagonal(corr, 0.)
assert ((corr > -1).all() and (corr < 1).all())
np.fill_diagonal(corr, 1.)
assert (np.diag(corr) == 1.).all()

Px = np.diag(std_dev_x) @ corr @ np.diag(std_dev_x)
#%%Px_sqrt and SVD

#calculation of the three square-roots as in O. Straka, J. Duník, M. Šimandl, J. Havlík, Aspects and comparison of matrix decompositions in unscented Kalman filter, in:  2013 American Control Conference, 2013. Note: This step is of course not necessary, could instead calculate Px_sqrt directly using e.g. scipy.linalg.cholesky and scipy.linalg.sqrtm

D, U = np.linalg.eig(Px) #D: eigenvalues, U: eigenvectors
D_sqrt = np.diag(np.sqrt(D))

I = np.eye(dim_x)
I_sqrt = I 
I_sqrt_inv = I

if sqrt_method == "SVD":
    O = np.eye(dim_x)
elif sqrt_method == "chol":
    Q, R = scipy.linalg.qr((U @ D_sqrt).T)
    O = Q.copy()
    
elif sqrt_method == "sqrtm":
    O = U.T.copy()

Px_sqrt = U @ D_sqrt @ O
Px_sqrt_svd = U @ D_sqrt

assert np.allclose(Px_sqrt @ Px_sqrt.T, Px)

#%% Def x_n-dist and conversion x_n->x
#x_n ~ (0, I)

#conversion from x_n to x:
V = Px_sqrt
x_n2x = lambda x_n: xm + V @ x_n


#%% Def nonlinear func
func = lambda x_in: x_in**2 #"standard" function
func_n = lambda x_n_in: func(x_n2x(x_n_in)) #input using normalized RV

y0 = func(xm) 
func_hdut = lambda x_in: func(x_in) - y0
y0_n = func_n(np.zeros(dim_x)) #this is the same as y0
func_hdut_n = lambda x_n_int: func_n(x_n_int) - y0_n

dim_y = y0.shape[0]
#%% Test conversion on a point

# Define a point which only lies in 1D 
x_n_test = I_sqrt[:,0] #1 standard deviation along the first "component" in the x_e-distribution
x_test = x_n2x(x_n_test) 

y_test = func(x_test)
y_test_x_n = func_n(x_n_test)
assert np.allclose(y_test, y_test_x_n)
#%% UT-Normalized RV
if sp_method == "Julier":
    points_n = spc.JulierSigmaPoints(dim_x, kappa = 3. - dim_x)
elif sp_method == "scaled":
    points_n = spc.ScaledSigmaPoints(dim_x, **kwargs_sp_scaled)
sigmas_n, Wm_n, Wc_n, I_sqrt_spc = points_n.compute_sigma_points(np.zeros(dim_x), I)
assert np.allclose(I_sqrt, I_sqrt_spc)

kwargs_ut = {"symmetrization": True}
ym_n, Py_n, A_n = ut.unscented_transform_w_function_eval_wslr(sigmas_n, Wm_n, Wc_n, func_n, **kwargs_ut)
A_n2c = A_n @ np.linalg.inv(V) #this should be the same as the A-matrix by doing the UT directly on the x-dist

#%% UT - standard

if sp_method == "Julier":
    points_c = spc.JulierSigmaPoints(dim_x, kappa = 3. - dim_x)
elif sp_method == "scaled":
    points_c = spc.ScaledSigmaPoints(dim_x, **kwargs_sp_scaled)

if sqrt_method == "chol":
    sigmas_c, Wm_c, Wc_c, P_sqrt_c_spc = points_c.compute_sigma_points(xm, Px, P_sqrt = scipy.linalg.cholesky(Px, lower = True))
elif sqrt_method == "sqrtm":
    sigmas_c, Wm_c, Wc_c, P_sqrt_c_spc = points_c.compute_sigma_points(xm, Px, P_sqrt = scipy.linalg.sqrtm(Px))
elif sqrt_method == "SVD":
    sigmas_c, Wm_c, Wc_c, P_sqrt_c_spc = points_c.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_svd)
else:
    raise KeyError("Not implemented")

ym_c, Py_c, A_c = ut.unscented_transform_w_function_eval_wslr(sigmas_c, Wm_c, Wc_c, func, **kwargs_ut)


#%% HDUT (using normalized RV)

#one possible implementation of the HD-UT (not the recommended one)
x_dists_ind = [{"mean": np.array([xi]), "cov": np.array([[ei]])} for xi, ei in zip(xm, np.diag(I))]
dim_x_ind = len(x_dists_ind) #number of independent xi

ym_hdut= np.zeros(dim_y)
Py_hdut = np.zeros((dim_y, dim_y))
# print(f"{x_mean=} and {dim_x=}")
idx = 0

A_list = []
for i in range(dim_x_ind):
    
    #unpack the distribution
    # xi_mean = x_dists_ind[i]["mean"]
    Di = x_dists_ind[i]["cov"]
    dim_xi = Di.shape[0]
    assert ((Di.ndim == 2) and (dim_xi == 1))
    assert (dim_xi, dim_xi) == Di.shape, f"Wrong dimensions, {Di.shape=} and expected to be square of {dim_xi=}"
    
    if sp_method == "Julier":
        kwargs_sigma_points = {"kappa": 3. - dim_xi}
        points = spc.JulierSigmaPoints(dim_xi, **kwargs_sigma_points)
    elif sp_method == "scaled":
        kwargs_sp_scaled["kappa"] = 3. - dim_xi
        points = spc.ScaledSigmaPoints(dim_xi, **kwargs_sp_scaled)
    
    
    sigmas_ei, Wm_ei, Wc_ei, D_sqrt_i = points.compute_sigma_points(np.zeros(dim_xi), Di) #sigmas_ei.shape = ((1,3))
    
    #insert sigmas_ei into the correct row in the "full" sigma-point matrix
    sigmas_ei_all = np.zeros((dim_x, sigmas_ei.shape[1]))
    sigmas_ei_all[i] = sigmas_ei.flatten()
    
    #Do 1D UTs
    if i == 0:
        yi_mean, Pyi, Ai = ut.hdut_w_function_eval_wslr_1D(sigmas_ei_all, sigmas_ei, Wm_ei, Wc_ei, func_n, **kwargs_ut, Pxi_inv = 1/Di)
    else:
        yi_mean, Pyi, Ai = ut.hdut_w_function_eval_wslr_1D(sigmas_ei_all, sigmas_ei, Wm_ei, Wc_ei, func_hdut_n, first_yi = 0., **kwargs_ut, Pxi_inv = 1/Di)

    ym_hdut += yi_mean
    Py_hdut += Pyi
    A_list.append(Ai)
    
A_hdut_n = np.hstack(A_list) #linearization matrix in the "normalized" space
A_hdut_n2c = A_hdut_n @ np.linalg.inv(V) #linearization matrix in the standard cartesian space


#%% MC test to verify
N_mc = int(1e6)
x_samples = np.random.multivariate_normal(xm, Px, size = N_mc)

y_mc =np.array(list(map(func, x_samples)))
ym_mc = np.mean(y_mc, axis = 0)
Py_mc = np.cov(y_mc, rowvar = False)

print(f"{Wc_ei=}")

#%% Compare solutions
norm_Py_c = scipy.linalg.norm(Py_c - Py_mc, "fro")
norm_Py_n = scipy.linalg.norm(Py_n - Py_mc, "fro")
norm_Py_hdut = scipy.linalg.norm(Py_hdut - Py_mc, "fro")

print(f"{sqrt_method=} on standard UT")

#Py_c and Py_n should be the same, while Py_hdut should NOT be the same
print(f"{norm_Py_c=}")
print(f"{norm_Py_n=}")
print(f"{norm_Py_hdut=}")

#Means and linearization matrices should be equal
assert (np.allclose(ym_c, ym_n) and np.allclose(ym_c, ym_hdut))
Ac_equal = (np.allclose(A_c, A_n2c) and np.allclose(A_c, A_hdut_n2c))
if not Ac_equal:
    
    raise ValueError(f"{Ac_equal=}!")
else:
    print(f"Success! {Ac_equal=} and means are correct")