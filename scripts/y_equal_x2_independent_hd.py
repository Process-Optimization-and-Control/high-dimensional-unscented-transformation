
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:10:12 2021
@author: halvorak
"""

import numpy as np
import scipy.stats
import scipy.io
import matplotlib
import pathlib
import os
import copy
import casadi as ca
import scipy.sparse

#Self-written modules
from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as ut
from state_estimator import unscented_transform_sparse as ut_sp
import high_dim_ut as hd_ut
font = {'size': 14}
matplotlib.rc('font', **font)

#%%Define input variables, sigma-points and function
dim_x = int(1e5) #can be changed to e.g. 1e5
use_sparse = True
implemented_sigma_points_methods = ["julier", "scaled"]
sigma_points_method = implemented_sigma_points_methods[0]

x_mean = np.zeros(dim_x)
Px = np.square(np.random.uniform(low=0.8, high = 1.2, size = dim_x)) #NB: not diagonal, exploit sparsity

kappa_julier = np.max([0., 3. - dim_x])
alpha_scaled = 1e-3
beta_scaled = 2 
kappa_scaled = copy.copy(kappa_julier)

sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True) #this does not matter since Px is diagonal


x = ca.MX.sym("x", dim_x)
y = x**2
func = ca.Function("func", [x], [y])
dim_y = func.size_out(0)[0]

func_np = lambda x: np.array(func(x)).flatten()
func_np = lambda x: x*x

#%% Scaled UT
if dim_x <= 1e3:
    print("\nSUT start")
    points_sut = spc.ScaledSigmaPoints(dim_x, alpha = 1e-3, kappa = 1e-7, beta = 2., sqrt_method = sqrt_method)
    # points_sut = spc.ScaledSigmaPoints(dim_x, alpha = 0.3, kappa = 1e-7, beta = 2., sqrt_method = sqrt_method)
    sigmas_sut, wm_sut, wc_sut, _ = points_sut.compute_sigma_points(x_mean, np.diag(Px))
    ym_sut, Py_sut = ut.unscented_transform_w_function_eval(sigmas_sut, wm_sut, wc_sut, func_np)
    
    correct_mean = np.allclose(ym_sut, Px) #analytical solution is Px
    correct_Py_diag = np.allclose(np.diag(Py_sut), 2*np.square(Px))
    is_diag = np.allclose(Py_sut - np.diag(np.diagonal(Py_sut)), 0.)
    
    assert correct_mean, f"{correct_mean=}"
    print(f"{dim_x=}")
    print(f"{correct_mean=}")
    # assert correct_Py_diag, f"{correct_Py_diag=}"
    print(f"{correct_Py_diag=}")
    assert not is_diag
    print(f"{is_diag=}")
    
    scipy.linalg.cholesky(Py_sut)
    py_sut_spd = True 
    
    print(f"{py_sut_spd=}\nSUT end\n")
    # assert (correct_mean and correct_Py_diag and is_diag)


#%% UT high dim
x_dist_ind = [{"mean": np.array([xi]), "cov": np.array([[sig]])} for xi, sig in zip(x_mean, Px)]

dim_x_ind = len(x_dist_ind) #number of independent xi
kwargs_sigma_points = {"sqrt_method": sqrt_method, "kappa": 3-1.}
kwargs_ut = {"symmetrization": False}

ut_func = ut.unscented_transformation
ut_sp_func = ut_sp.unscented_transformation_sparse

if use_sparse:
    ym2, Py2 = hd_ut.ut_high_dim_w_function_eval_sparse(x_dist_ind, func_np, dim_y, sigma_points_method = sigma_points_method, kwargs_sigma_points=kwargs_sigma_points, kwargs_ut = kwargs_ut, ut_func = ut_sp_func)
else:
    ym2, Py2 = hd_ut.ut_high_dim_w_function_eval(x_dist_ind, func_np, dim_y, sigma_points_method = sigma_points_method, kwargs_sigma_points=kwargs_sigma_points, kwargs_ut = kwargs_ut, ut_func = ut_func)

try:
    correct_mean = np.isclose(ym2, Px).all() #analytical solution is Px
    correct_Py_diag = np.isclose(np.diag(Py2), 2*np.square(Px)).all()
except:
    correct_mean = np.isclose(ym2.toarray().flatten(), Px).all() #analytical solution is Px
    correct_Py_diag = np.isclose(Py2.diagonal(), 2*np.square(Px)).all()
    

assert correct_mean, f"{correct_mean=}"
print("\nHD-UT")
print(f"{correct_mean=}")
assert correct_Py_diag, f"{correct_Py_diag=}"
print(f"{correct_Py_diag=}")

if use_sparse:
    is_diag = hd_ut.is_sparse_diag(Py2)
    
    assert is_diag, "Py2 is not diagonal"
    print(f"{is_diag=}")
    
    if is_diag:
        Py_sqrt = np.sqrt(Py2.diagonal())
    else:
        Py_sqrt = scipy.sparse.csr_array(scipy.linalg.cholesky(Py2.toarray(), lower = True)) #check it is SPD
else:
    is_diag = np.allclose(Py2 - np.diag(np.diagonal(Py2)), 0.)
    assert is_diag, "Py2 is not diagonal"
    Py_sqrt = scipy.linalg.cholesky(Py2, lower = True) #check that it is SPD
    

if False: #save result
    project_dir = pathlib.Path(__file__).parent.parent
    dir_data = os.path.join(project_dir, "data")
    fpath_Py = os.path.join(dir_data, f"Py_dim{dim_x}.npz")
    fpath_Px = os.path.join(dir_data, f"Px_dim{dim_x}.npy")
    fpath_Px = f"Px_dim{dim_x}.npy"
    scipy.sparse.save_npz(fpath_Py, Py2)
    np.save(fpath_Px, Px)


    


