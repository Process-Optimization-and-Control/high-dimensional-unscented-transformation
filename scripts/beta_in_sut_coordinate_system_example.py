# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:16:16 2023

@author: halvorak
"""
import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import pandas as pd
import seaborn as sns
import casadi as ca
import scipy.sparse
import sklearn.decomposition
import sklearn.datasets
import time

#Self-written modules
from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as ut
import utils


np.set_printoptions(linewidth=np.nan)
font = {'size': 14}
matplotlib.rc('font', **font)
"""
Varying beta for the SUT and HD-SUT for the transformation of coordinates example

"""

project_dir = pathlib.Path(__file__).parent.parent

dir_data_load = os.path.join(project_dir, "data_example_coordinate_transf_paper") #only loaded if load_old_sim=True
dir_data_plot = os.path.join(dir_data_load, "plots") 

dim_x_list = np.array([2, 100]).astype(int)
beta_lst = np.arange(start = .01, stop = 4, step = 0.01)
beta_string = r"$\beta$"
dim_x_str = r"$n_x$"


error_eval = "absolute"
methods_implemented = ["SUT", "HD-SUT", "HD-UT"] 
methods_to_run = methods_implemented

sqrt_func = lambda P: scipy.linalg.cholesky(P, lower = True)
#making names for plotting later, e.g. HD-UT (Eig), HD-UT (Prin) etc
name_methods = methods_implemented

#prepare df to save results
col_df = name_methods.copy()
col_df.extend([beta_string, dim_x_str])
index = range(len(dim_x_list)*len(beta_lst))

df_Py_norm = pd.DataFrame(index = index, data = 0., columns = col_df)
df_std_norm = pd.DataFrame(index = index, data = 0., columns = col_df)
df_ym_norm = pd.DataFrame(index = index, data = 0., columns = col_df)

diverged_sim = []
not_spd_val = np.nan

idx_df = 0
for (i, dim_x) in enumerate(dim_x_list):
    #%% Def x-dist

    xm = np.hstack((np.zeros(int(dim_x/2)), np.ones(int(dim_x/2))*100))
    Px = np.diag(np.abs(np.random.normal(size = dim_x)))
    a_unif = np.sqrt(np.diag(Px)*3) #from definition of variance for uniform distribution
    kurtosis_unif = 1.8 # see kurtosis of uniform dist., Wikipedia
    
    N_mc = int(5e5) #Number of MC samples
    
    #%% Def func
    func = lambda x_in: np.hstack(
        (np.cos(x_in[:int(dim_x/2)])*x_in[int(dim_x/2):],
         np.sin(x_in[:int(dim_x/2)])*x_in[int(dim_x/2):]))
    
    
    n_threads = 16 #parallel evaluation (only for UT based methods)
    x_ca = ca.SX.sym("x_ca", dim_x)
    y_ca = ca.vertcat(ca.cos((x_ca[:int(dim_x/2)]))*x_ca[int(dim_x/2):],
        ca.sin((x_ca[:int(dim_x/2)]))*x_ca[int(dim_x/2):])
    ca_func = ca.Function("ca_func", [x_ca], [y_ca])
    ca_func_map = ca_func.map(int(2*dim_x + 1), "thread", n_threads)
    
    #%% Analytical solution, approximated by MC sim 
    
    x_samples = np.array([scipy.stats.uniform.rvs(loc = -ai, scale = 2*ai, size = N_mc) for ai in a_unif]) + xm.reshape(-1,1)
    
    y_mc = np.array(list(map(func, x_samples.T)))
    ym_ana = np.mean(y_mc, axis = 0)
    dim_y = ym_ana.shape[0]
    Py_ana = np.cov(y_mc, rowvar = False)
    
    assert Py_ana.shape == ((dim_x, dim_x))
        
    scipy.linalg.cholesky(Py_ana, lower = True) #check it is SPD
    std_ana, corr_ana = utils.get_corr_std_dev(Py_ana)
    
    #%% error_eval_functions
    
    val_func_Py = lambda mat: scipy.linalg.norm(mat - Py_ana, "fro")
    val_func_std = lambda vec: scipy.linalg.norm(vec - std_ana)
    val_func_ym = lambda vec: scipy.linalg.norm(vec - ym_ana)

    for beta_sut in beta_lst:
    
    
        ts = time.time()  
        
        
       
        #%% SUT
        if "SUT" in methods_to_run:
    
            Px_sqrt_i = sqrt_func(Px)
            
            points_a = spc.ScaledSigmaPoints(dim_x, alpha = 1e-2, beta = beta_sut, kappa = 1e-7, suppress_init_warning = True)
            sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
            
            kwargs_ut = {"symmetrization": False}
            yma, Pya, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
            dim_y = yma.shape[0]
            del Aa
            
            df_ym_norm.loc[beta_sut, "SUT"] = val_func_ym(yma)
            
            try: #check if covariance matrix is positive definite
                scipy.linalg.cholesky(Pya, lower = True)
                std_a = utils.get_corr_std_dev(Pya)[0]
                df_Py_norm.loc[idx_df, "SUT"] = val_func_Py(Pya)
                df_std_norm.loc[idx_df, "SUT"] = val_func_std(std_a)
                del Pya, std_a #depending on dim_x, these may take up significant space
            except scipy.linalg.LinAlgError:
                df_Py_norm.loc[idx_df, "SUT"] = not_spd_val
                df_std_norm.loc[idx_df, "SUT"] = not_spd_val
                diverged_sim.append({"Method": "SUT", "Px": Px, "Py": Pya, "beta": beta_sut, "dim_x": dim_x})
                del Pya
            # assert np.allclose(ym_ana, yma)
       
        #%% HD-SUT
        if "HD-SUT" in methods_to_run:
            
            points_x = spc.ScaledSigmaPoints(dim_x, alpha = 1e-2, kappa = kurtosis_unif - dim_x, beta = beta_sut, suppress_init_warning = True, kurtosis = kurtosis_unif)
                
            Px_sqrt_i = sqrt_func(Px)
            ym_hdsut, Py_hdsut, _ = ut.hdut_map_func(xm, Px_sqrt_i, ca_func_map, points_x, calc_Pxy = False, calc_A = False)
            
            df_ym_norm.loc[idx_df, "HD-SUT"] = val_func_ym(ym_hdsut)
            try: #check if covariance matrix is positive definite
                scipy.linalg.cholesky(Py_hdsut, lower = True)
                std_hdsut = utils.get_corr_std_dev(Py_hdsut)[0]
                df_Py_norm.loc[idx_df, "HD-SUT"] = val_func_Py(Py_hdsut)
                df_std_norm.loc[idx_df, "HD-SUT"] = val_func_std(std_hdsut)
            except scipy.linalg.LinAlgError:
                df_Py_norm.loc[idx_df, "HD-SUT"] = not_spd_val
                df_std_norm.loc[idx_df, "HD-SUT"] = not_spd_val
                diverged_sim.append({"Method": "HD-SUT", "Px": Px, "Py": Py_hdsut, "beta": beta_sut, "dim_x": dim_x})
                # del Py_hdsut
        
        
        
        
        #%% HD-UT
        if "HD-SUT" in methods_to_run:
            if beta_sut > 1.:
                kurtosis_unif = np.max([beta_sut, 1.1])
                points_x = spc.JulierSigmaPoints(dim_x, kappa = kurtosis_unif - dim_x, suppress_kappa_warning = True, kurtosis = kurtosis_unif)
                
                Px_sqrt_i = sqrt_func(Px)
                
                ym_hdut, Py_hdut, _,  = ut.hdut_map_func(xm, Px_sqrt_i, ca_func_map, points_x, calc_Pxy = False, calc_A = False)
                                    
                df_ym_norm.loc[idx_df, "HD-UT"] = val_func_ym(ym_hdut)
                try: #check if covariance matrix is positive definite
                    scipy.linalg.cholesky(Py_hdut, lower = True)
                    std_hdut = utils.get_corr_std_dev(Py_hdut)[0]
                    df_Py_norm.loc[idx_df, "HD-UT"] = val_func_Py(Py_hdut)
                    df_std_norm.loc[idx_df, "HD-UT"] = val_func_std(std_hdut)
                except scipy.linalg.LinAlgError:
                    df_Py_norm.loc[idx_df, "HD-UT"] = not_spd_val
                    df_std_norm.loc[idx_df, "HD-UT"] = not_spd_val
                    diverged_sim.append({"Method": "HD-UT", "Px": Px, "Py": Py_hdut, "idx_df": idx_df, "dim_x": dim_x})
                    del Py_hdut
            else:
                df_ym_norm.loc[idx_df, "HD-UT"] = np.nan
                df_Py_norm.loc[idx_df, "HD-UT"] = np.nan
                df_std_norm.loc[idx_df, "HD-UT"] = np.nan
                
        
        #%% Save data
        df_ym_norm.loc[idx_df, beta_string] = beta_sut
        df_Py_norm.loc[idx_df, beta_string] = beta_sut
        df_std_norm.loc[idx_df, beta_string] = beta_sut
        
        df_ym_norm.loc[idx_df, dim_x_str] = dim_x
        df_Py_norm.loc[idx_df, dim_x_str] = dim_x
        df_std_norm.loc[idx_df, dim_x_str] = dim_x
        #%% print progression
        t_iter = time.time() - ts
    
        
        #save dfs
        # df_ym_norm.to_pickle(os.path.join(dir_data, "df_ym_norm.pkl"))
        # df_Py_norm.to_pickle(os.path.join(dir_data, "df_Py_norm.pkl"))
        # df_std_norm.to_pickle(os.path.join(dir_data, "df_std_norm.pkl"))
    
        
        print(f"Iter {beta_sut=} and {t_iter= :.2f} for {dim_x=} s")
        idx_df += 1

#%% melt dataframes
df_ym_norm = df_ym_norm.melt(id_vars = [dim_x_str, beta_string], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
df_Py_norm = df_Py_norm.melt(id_vars = [dim_x_str, beta_string], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
df_std_norm = df_std_norm.melt(id_vars = [dim_x_str, beta_string], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")


# df_ym_norm[dim_x_str].astype(int)
df_Py_norm[dim_x_str] = df_Py_norm[dim_x_str].astype(int)
df_std_norm[dim_x_str] = df_std_norm[dim_x_str].astype(int)


df_ym_norm_hdut = df_ym_norm[df_ym_norm["Method"] == "HD-UT"]
df_ym_norm = df_ym_norm[df_ym_norm["Method"] != "HD-UT"]

df_Py_norm_hdut = df_Py_norm[df_Py_norm["Method"] == "HD-UT"]
df_Py_norm = df_Py_norm[df_Py_norm["Method"] != "HD-UT"]


df_std_norm_hdut = df_std_norm[df_std_norm["Method"] == "HD-UT"]
df_std_norm = df_std_norm[df_std_norm["Method"] != "HD-UT"]

#%% Plot

fig_ym, ax_ym = plt.subplots(1,1,layout = "constrained")

sns.lineplot(data = df_ym_norm[df_ym_norm["Method"] == "SUT"], x = beta_string, y = "norm_diff", hue = dim_x_str, ax = ax_ym)
# ax_ym.plot(df_ym_norm.index, df_ym_norm["HD-SUT"], label = "HD-SUT")
# ax_ym.plot(df_ym_norm.index, df_ym_norm["SUT"], label = "SUT")
ax_ym.set_ylabel(r"$||\hat{y} - \hat{y}^{MC}||_2$")
ax_ym.set_xlabel(beta_string)
ax_ym.legend()
ax_ym.set_yscale("log")

fig_Py, axes = plt.subplots(2,len(dim_x_list),layout = "constrained")
# (axes_Py1, axes_Py2, axes_Py3, axes_std1, axes_std2, axes_std3) = axes

for (i, dim_x) in enumerate(dim_x_list):
    axi = axes[0, i]
    sns.lineplot(data = df_Py_norm[df_Py_norm[dim_x_str] == int(dim_x)], x = beta_string, y = "norm_diff", ax = axi, hue = "Method")
    axi.set_xlabel(beta_string)
    axi.set_ylabel(r"$||P_y - P_y^{MC}||_F$")
    axi.set_yscale("log")
    
    axi = axes[1, i]
    sns.lineplot(data = df_std_norm[df_std_norm[dim_x_str] == int(dim_x)], x = beta_string, y = "norm_diff", ax = axi, hue = "Method")
    axi.set_xlabel(beta_string)
    axi.set_ylabel(r"$||\sigma_y - \sigma_y^{MC}||_2$")
    # axi.set_yscale("log")
    

fig_kappa, ax_kappa = plt.subplots(1,1,layout = "constrained")
sns.lineplot(data = df_ym_norm_hdut[df_ym_norm_hdut["Method"] == "HD-UT"], x = beta_string, y = "norm_diff", hue = dim_x_str, ax = ax_kappa)
ax_kappa.set_xlabel(r"$\gamma$ or used kurtosis")
ax_kappa.set_ylabel(r"$||\hat{y}^{HD-UT} - \hat{y}^{MC}||_2$")
