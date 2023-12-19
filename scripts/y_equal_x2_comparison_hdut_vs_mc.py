# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:16:16 2023

@author: halvorak
"""
import numpy as np
import scipy.stats
import scipy.linalg
# import scipy.integrate
import matplotlib.pyplot as plt
# import matplotlib
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

"""
Toy example: y=x^2 with correlated x.

We can re-use existing results by setting load_old_sim=True. If that one is False, simulation is re-run. Existing results are saved in dir_data_load and new results saved in dir_data.

"""

project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data_y_x2_corr_x") #save
# dir_data_load = os.path.join(dir_data, "100_iter_kappafunc_hdsut")
# dir_data_load = os.path.join(dir_data, "50_iter_231130_hdsut")
dir_data_load = os.path.join(dir_data, "100_iter_kappafixed_20231204")

#%% Def x-dist

N = 100 #number of times to repeat the simulation

#state dimensions to run

dim_x_list = np.array([2, 10, 25, 50, 100, 250, 500, 750, 1000])
# dim_x_list = np.array([250, 500, 750, 1000])
dim_x_list = dim_x_list.astype(int)
print(f"{dim_x_list=}")
load_old_sim = False #if True, it loads old simulations. False, it reruns simulations.

N_mc = int(3e4) #Number of MC samples

n_threads = 10 #parallel evaluation (only for UT based methods)
not_spd_val = np.nan #marks norm estimates of covariance with this value of Py is not SPD
# error_eval = "rel"
error_eval = "absolute"

diverged_sim = []
methods_implemented = ["MC", "HD-UT", "SUT", "HD-SUT", "UT"] 
# methods_to_run = [methods_implemented[m] for m in [1, 2, 3]] #select which method to run
methods_to_run = methods_implemented

sqrt_method = ["Eig", "Chol", "Prin"]
sqrt_func = [lambda P: spc.sqrt_eig(P), lambda P: scipy.linalg.cholesky(P, lower = True), lambda P: scipy.linalg.sqrtm(P)]

#making names for plotting later, e.g. HD-UT (Eig), HD-UT (Prin) etc
name_methods = []
for ni in methods_to_run:
    if "UT" in ni:
        for sqrt_m in sqrt_method:
            nim = ni + f" ({sqrt_m})"
            name_methods.append(nim)
    else:
        name_methods.append(ni)

#prepare df to save results
col_df = name_methods.copy()
col_df.extend(["dim_x", "iter"])
df_Py_norm = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = col_df)
df_std_norm = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = df_Py_norm.columns)

df_Py_cond = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = col_df)
df_cond_Px_sqrt = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = sqrt_method)

kwargs_ut = {"symmetrization": False}
idx_df = 0


if not load_old_sim:
    for ni in range(N):
        for j in range(len(dim_x_list)):
            ts = time.time()
        
            dim_x = dim_x_list[j]
            xm = np.zeros(dim_x)
            
            #%% Generate Px
            
            Px = sklearn.datasets.make_spd_matrix(dim_x)
            
            #verify everything is ok
            std_dev, corr = utils.get_corr_std_dev(Px)
            del Px
            np.fill_diagonal(corr, 0.)
            assert ((corr > -1).all() and (corr < 1).all())
            np.fill_diagonal(corr, 1.)
            assert (np.diag(corr) == 1.).all()
            if True: #equivalent formulation, but avoid storing all the zeros
                Px = std_dev.reshape(-1,1) * corr * std_dev
            else:
                Px = np.diag(std_dev) @ corr @ np.diag(std_dev)
            del corr


            #%% Def func
            func = lambda x_in: x_in**2

            x_ca = ca.SX.sym(f"x_ca_{j}", dim_x)
            y_ca = x_ca **2
            ca_func = ca.Function("ca_func", [x_ca], [y_ca])
            ca_func_map = ca_func.map(int(2*dim_x + 1), "thread", n_threads)
            
            #%% Analytical solution
            #analytical solution of y=x**2 when x~N(xm,Px)
            ym_ana = np.diag(Px) + xm**2
            Py_ana = np.zeros((dim_x, dim_x))
            
            dim_y = ym_ana.shape[0]
            
            for i in range(dim_x):
                Py_ana[:, i] = 2*Px[:, i]**2
            Py_ana = np.triu(Py_ana)
            Py_ana = Py_ana + Py_ana.T - np.diag(np.diag(Py_ana))
            np.fill_diagonal(Py_ana, 2*np.power(std_dev, 4))
            std_ana, corr_ana = utils.get_corr_std_dev(Py_ana)
            
            #%% error_eval_functions
            if error_eval == "rel": #relative error
                val_func_Py = lambda mat: scipy.linalg.norm(mat/Py_ana - 1, "fro")
                val_func_std = lambda vec: scipy.linalg.norm(vec/std_ana - 1)
    
            elif error_eval == "absolute":
                val_func_Py = lambda mat: scipy.linalg.norm(mat - Py_ana, "fro")
                val_func_std = lambda vec: scipy.linalg.norm(vec - std_ana)
            else:
                raise KeyError(f"{error_eval=} which is not supported")
            
            #%% MC solution
            if "MC" in methods_to_run: 
  
                x_samples = np.random.multivariate_normal(xm, Px, size = N_mc)
                y_mc = np.array(list(map(func, x_samples)))
                del x_samples
                ym_mc = np.mean(y_mc, axis = 0)
                Py_mc = np.cov(y_mc, rowvar = False)
                    
                del y_mc #depending on dim_x and N_MC, these may take up significant space
                try: #check if covariance matrix is positive definite
                    scipy.linalg.cholesky(Py_mc, lower = True)
                    df_Py_norm.loc[idx_df, "MC"] = val_func_Py(Py_mc)
                    df_Py_cond.loc[idx_df, "MC"] = np.linalg.cond(Py_mc)
                    std_mc = utils.get_corr_std_dev(Py_mc)[0]
                    del Py_mc
                    df_std_norm.loc[idx_df, "MC"] = val_func_std(std_mc)
                except scipy.linalg.LinAlgError:
                    df_Py_norm.loc[idx_df, "MC"] = not_spd_val
                    df_std_norm.loc[idx_df, "MC"] = not_spd_val
                    df_Py_cond.loc[idx_df, "MC"] = not_spd_val
                    diverged_sim.append({"Method": "MC", "Px": Px, "Py": Py_mc, "idx_df": idx_df, "dim_x": dim_x})
                    del Py_mc
            
            
            
            #%% UT - kappa=1e-7, standard UT
            if "UT" in methods_to_run:

                for label, sqrt_f in zip(sqrt_method, sqrt_func):
                    Px_sqrt_i = sqrt_f(Px)
                    
                    points_a = spc.JulierSigmaPoints(dim_x, kappa = 1e-7, suppress_kappa_warning = True)
                    sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
                    
                    kwargs_ut = {"symmetrization": False}
                    yma, Pya, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
                    dim_y = yma.shape[0]
                    assert np.allclose(Aa, 0.)
                    del Aa
                    
                    label = f" ({label})"
                    
                    try: #check if covariance matrix is positive definite
                        scipy.linalg.cholesky(Pya, lower = True)
                        std_a = utils.get_corr_std_dev(Pya)[0]
                        df_Py_norm.loc[idx_df, "UT" + label] = val_func_Py(Pya)
                        df_std_norm.loc[idx_df, "UT" + label] = val_func_std(std_a)
                        df_Py_cond.loc[idx_df, "UT" + label] = np.linalg.cond(Pya)
                        del Pya, std_a #depending on dim_x, these may take up significant space
                    except scipy.linalg.LinAlgError:
                        df_Py_norm.loc[idx_df, "UT" + label] = not_spd_val
                        df_std_norm.loc[idx_df, "UT" + label] = not_spd_val
                        df_Py_cond.loc[idx_df, "UT" + label] = not_spd_val
                        diverged_sim.append({"Method": "UT" + label, "Px": Px, "Py": Pya, "idx_df": idx_df, "dim_x": dim_x})
                        del Pya
                    assert np.allclose(ym_ana, yma)
            #%% SUT
            if "SUT" in methods_to_run:

                for label, sqrt_f in zip(sqrt_method, sqrt_func):
                    Px_sqrt_i = sqrt_f(Px)
                    
                    points_a = spc.ScaledSigmaPoints(dim_x, alpha = 1e-2, beta = 2., kappa = 1e-7, suppress_init_warning = True)
                    sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
                    
                    kwargs_ut = {"symmetrization": False}
                    yma, Pya, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
                    dim_y = yma.shape[0]
                    del Aa
                    
                    label = f" ({label})"
                    
                    try: #check if covariance matrix is positive definite
                        scipy.linalg.cholesky(Pya, lower = True)
                        std_a = utils.get_corr_std_dev(Pya)[0]
                        df_Py_norm.loc[idx_df, "SUT" + label] = val_func_Py(Pya)
                        df_std_norm.loc[idx_df, "SUT" + label] = val_func_std(std_a)
                        df_Py_cond.loc[idx_df, "SUT" + label] = np.linalg.cond(Pya)
                        del Pya, std_a #depending on dim_x, these may take up significant space
                    except scipy.linalg.LinAlgError:
                        df_Py_norm.loc[idx_df, "SUT" + label] = not_spd_val
                        df_std_norm.loc[idx_df, "SUT" + label] = not_spd_val
                        df_Py_cond.loc[idx_df, "SUT" + label] = not_spd_val
                        diverged_sim.append({"Method": "SUT" + label, "Px": Px, "Py": Pya, "idx_df": idx_df, "dim_x": dim_x})
                        del Pya
                    assert np.allclose(ym_ana, yma)
            
           
            #%% HD-UT
            if "HD-UT" in methods_to_run:
                points_x = spc.JulierSigmaPoints(dim_x, kappa = 3 - dim_x, suppress_kappa_warning = True)
                
                for label, sqrt_f in zip(sqrt_method, sqrt_func):
                    Px_sqrt_i = sqrt_f(Px)
                    
                    #check condition number of matrix square-root
                    df_cond_Px_sqrt.loc[idx_df, label] = np.linalg.cond(Px_sqrt_i)
                    
                    ym_hdut, Py_hdut, _, A_hdut, _ = ut.hdut_map_func_fast2(xm, Px_sqrt_i, ca_func_map, points_x)
                    
                    assert np.allclose(A_hdut, 0.)
                    
                    label = f" ({label})"
                    try: #check if covariance matrix is positive definite
                        scipy.linalg.cholesky(Py_hdut, lower = True)
                        std_hdut = utils.get_corr_std_dev(Py_hdut)[0]
                        df_Py_norm.loc[idx_df, "HD-UT" + label] = val_func_Py(Py_hdut)
                        df_std_norm.loc[idx_df, "HD-UT" + label] = val_func_std(std_hdut)
                        df_Py_cond.loc[idx_df, "HD-UT" + label] = np.linalg.cond(Py_hdut)
                    except scipy.linalg.LinAlgError:
                        df_Py_norm.loc[idx_df, "HD-UT" + label] = not_spd_val
                        df_std_norm.loc[idx_df, "HD-UT" + label] = not_spd_val
                        df_Py_cond.loc[idx_df, "HD-UT" + label] = not_spd_val
                        diverged_sim.append({"Method": "HD-UT" + label, "Px": Px, "Py": Py_hdut, "idx_df": idx_df, "dim_x": dim_x})
                        del Py_hdut
                        
            
            #%% HD-SUT
            if "HD-SUT" in methods_to_run:
                
                points_x = spc.ScaledSigmaPoints(dim_x, alpha = 1e-2, kappa = 3 - dim_x, beta = 2., suppress_init_warning = True)
                for label, sqrt_f in zip(sqrt_method, sqrt_func):
                    label = f" ({label})"
                    Px_sqrt_i = sqrt_f(Px)
                    ym_hdsut, Py_hdsut = ut.hdut_map_func_fast2(xm, Px_sqrt_i, ca_func_map, points_x)[:2]
                    
                    try: #check if covariance matrix is positive definite
                        scipy.linalg.cholesky(Py_hdsut, lower = True)
                        std_hdsut = utils.get_corr_std_dev(Py_hdsut)[0]
                        df_Py_norm.loc[idx_df, "HD-SUT" + label] = val_func_Py(Py_hdsut)
                        df_std_norm.loc[idx_df, "HD-SUT" + label] = val_func_std(std_hdsut)
                        df_Py_cond.loc[idx_df, "HD-SUT" + label] = np.linalg.cond(Py_hdsut)
                    except scipy.linalg.LinAlgError:
                        df_Py_norm.loc[idx_df, "HD-SUT" + label] = not_spd_val
                        df_std_norm.loc[idx_df, "HD-SUT" + label] = not_spd_val
                        df_Py_cond.loc[idx_df, "HD-SUT" + label] = not_spd_val
                        diverged_sim.append({"Method": "HD-SUT" + label, "Px": Px, "Py": Py_hdsut, "idx_df": idx_df, "dim_x": dim_x})
                        # del Py_hdsut
            
            
            
            #%% print progression
            t_iter = time.time() - ts
            
            df_Py_norm.loc[idx_df, "dim_x"] = dim_x
            df_std_norm.loc[idx_df, "dim_x"] = dim_x
            df_Py_norm.loc[idx_df, "Ni"] = ni
            df_std_norm.loc[idx_df, "Ni"] = ni
            df_Py_norm.loc[idx_df, "N_mc"] = N_mc
            df_std_norm.loc[idx_df, "N_mc"] = N_mc
            
            df_Py_cond.loc[idx_df, ["dim_x", "Ni", "N_mc"]] = [dim_x, ni, N_mc]
            idx_df +=1
            print(f"Iter {ni+1}/{N}, subiter {j+1}/{len(dim_x_list)}. {dim_x=} and {t_iter= :.2f} s")
    
    #%% Melt dataframe
    df_Py_norm = df_Py_norm.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
    df_Py_norm["diverged_old"] = np.where(df_Py_norm["norm_diff"] == not_spd_val, True, False)
    df_Py_norm["diverged"] = df_Py_norm["norm_diff"].isna()
    
    df_std_norm = df_std_norm.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
    
    df_Py_cond = df_Py_cond.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "cond")
    
    
else:
    df_Py_norm = pd.read_pickle(os.path.join(dir_data_load, "df_Py_norm.pkl"))
    df_std_norm = pd.read_pickle(os.path.join(dir_data_load, "df_std_norm.pkl"))
    
    df_Py_cond = pd.read_pickle(os.path.join(dir_data_load, "df_Py_cond.pkl"))
    
#%% Save dfs
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data_y_x2_corr_x")
df_Py_norm.to_pickle(os.path.join(dir_data, "df_Py_norm.pkl"))
df_std_norm.to_pickle(os.path.join(dir_data, "df_std_norm.pkl"))

df_Py_cond.to_pickle(os.path.join(dir_data, "df_Py_cond.pkl"))
#%%Plot norm cov

#warning: the plots are a mess if all methods (and all square-roots) are runned simulatenously

fig_py, ax_py = plt.subplots(1,1,layout = "constrained")

plt_type = sns.lineplot
plt_type = sns.scatterplot

order_method = ["HD-UT", "SUT", "MC", "HD-SUT"]
order_method = None

# df_Py_norm[df_Py_norm["norm_diff"] == -1.] = np.nan

plt_type(data = df_Py_norm, x = "dim_x", y = "norm_diff", hue = "Method", ax = ax_py, hue_order = order_method)
ax_py.set_ylabel(r"$||P_y - P_y^{ana}||_F$")
ax_py.set_xlabel(r"$n_x$")
xlim = ax_py.get_xlim()
ax_py.plot(list(xlim), [0,0], 'k')
ax_py.set_xlim(xlim)

#%%Plot cond cov
fig_py_cond, ax_py_cond = plt.subplots(1,1,layout = "constrained")
sns.scatterplot(data = df_Py_cond, x = "dim_x", y = "cond", hue = "Method", ax = ax_py_cond, hue_order = order_method)

ax_py_cond.set_ylabel(r"$\kappa (P_y) [-]$")
ax_py_cond.set_xlabel(r"$n_x$")
xlim = ax_py_cond.get_xlim()
ax_py_cond.plot(list(xlim), [0,0], 'k')
ax_py_cond.set_xlim(xlim)
ax_py_cond.set_yscale("log")


#%% Diverged sim vs nx
fig_dnx, ax_dnx = plt.subplots(1,1,layout = "constrained")
df_div2 = pd.DataFrame(index = dim_x_list, data = None, columns = name_methods)
for ci in name_methods:
    for dxi in dim_x_list:
        df_div2.loc[dxi, ci] = df_Py_norm[(df_Py_norm["Method"] == ci) & (df_Py_norm["dim_x"] == dxi)]["diverged"].sum()
df_div2.plot(ax = ax_dnx)
ax_dnx.set_xlabel(r"$n_x$")
ax_dnx.set_ylabel(f"#Diverged simulations ({N} total)")
# sns.lineplot(data = df_div2)


df_diverged = df_Py_norm.copy()
df_diverged.loc[df_diverged["norm_diff"].isna(), "diverged"] = True
df_diverged = df_diverged[["dim_x", "Method", "diverged"]]
df_div_group = df_diverged.groupby(["dim_x", "Method"])
df_diverged_sum = df_div_group.sum().unstack(level = 0)
df_diverged_sum.to_csv(os.path.join(dir_data, "simulations_diverged.csv"))


#%% Norm std_dev
if not load_old_sim:
    fig_std, ax_std = plt.subplots(1,1,layout = "constrained")
    plt_type(df_std_norm, x = "dim_x", y = "norm_diff", hue = "Method", ax = ax_std)
    ax_std.set_ylabel(r"$||\sigma_y - \sigma_y^{ana}||_2$")
    ax_std.set_xlabel(r"$n_x$")
    xlim = ax_std.get_xlim()
    ax_std.plot(list(xlim), [0,0], 'k')
    ax_std.set_xlim(xlim)
    
    fig_div, ax_div = plt.subplots(1,1,layout = "constrained")
    df_diverged = df_Py_norm.groupby("Method").sum()
    sns.barplot(df_diverged, x = df_diverged.index, y = "diverged", ax = ax_div)
    ax_div.set_ylabel(f"#Simulations (of {(df_Py_norm['Method'] == 'MC').shape[0]}) with " + r"$P_y<0$")


    #%% Check diverged sims
    div_hdut = []
    for d in diverged_sim:
        if ((d["Method"] == "HD-UT") or (d["Method"] == "HD-NUT") or (d["Method"] == "SUT")):
            # print(d)
            cond_Px = np.linalg.cond(d["Px"])
            std_x = utils.get_corr_std_dev(d["Px"])[0]
            cond_Py = np.linalg.cond(d["Py"])
            try:
                scipy.linalg.cholesky(d["Px"], lower = True)
            except scipy.linalg.LinAlgError:
                print(f"{d['Method']=} and {d['idx_df']=} not SPD for Px. Have {cond_Px=}")
            try:
                scipy.linalg.cholesky(d["Py"], lower = True)
            except scipy.linalg.LinAlgError:
                print(f"{d['Method']=} and {d['dim_x']=} not SPD for Px. Have {cond_Px=} and {cond_Py=}")
                


#%% Print statistics and save to file
df_Py_norm_c = df_Py_norm[df_Py_norm["diverged"] == False]
Py_norm_group = df_Py_norm_c.loc[:, ["dim_x", "Method", "norm_diff"]].groupby(by = ["dim_x", "Method"])
Py_norm_mean = Py_norm_group.mean().unstack(level=0)
Py_norm_std = Py_norm_group.std().unstack(level=0)
print(f"Py_norm_group.mean:\n{Py_norm_mean}")

df_Py_cond_c = df_Py_cond[df_Py_norm["diverged"] == False]
Py_cond_group = df_Py_cond_c.loc[:, ["dim_x", "Method", "cond"]].groupby(by = ["dim_x", "Method"])
Py_cond_mean = Py_cond_group.mean().unstack(level=0)
Py_cond_std = Py_cond_group.std().unstack(level=0)
print(f"Py_cond_group.mean:\n{Py_cond_mean}")

df_std_norm_c = df_std_norm[df_Py_norm["diverged"] == False]
std_norm_group = df_std_norm_c.loc[:, ["dim_x", "Method", "norm_diff"]].groupby(by = ["dim_x", "Method"])
std_norm_mean = std_norm_group.mean().unstack(level=0)
std_norm_std = std_norm_group.std().unstack(level=0)
print(f"std_norm_group.mean:\n{std_norm_mean}")

print(f"df_diverged_sum:\n{df_diverged_sum}")


#save to .csv files
Py_norm_mean.to_csv(os.path.join(dir_data, "Py_norm_mean.csv"))
Py_norm_std.to_csv(os.path.join(dir_data, "Py_norm_std.csv"))

std_norm_mean.to_csv(os.path.join(dir_data, "std_norm_mean.csv"))
std_norm_std.to_csv(os.path.join(dir_data, "std_norm_std.csv"))

Py_cond_mean.to_csv(os.path.join(dir_data, "Py_cond_mean.csv"))
Py_cond_std.to_csv(os.path.join(dir_data, "Py_cond_std.csv"))

#%% Compute p-values

dim_x_eval = [1000]
dim_x_eval = [500]
import scipy.stats
for dim_x in dim_x_eval:
    ut_chol = df_Py_norm.loc[(df_Py_norm["Method"] == "UT (Chol)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    ut_prin = df_Py_norm.loc[(df_Py_norm["Method"] == "UT (Prin)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    ut_eig = df_Py_norm.loc[(df_Py_norm["Method"] == "UT (Eig)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    
    sut_chol = df_Py_norm.loc[(df_Py_norm["Method"] == "SUT2 (Chol)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    sut_prin = df_Py_norm.loc[(df_Py_norm["Method"] == "SUT2 (Prin)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    sut_eig = df_Py_norm.loc[(df_Py_norm["Method"] == "SUT2 (Eig)") & (df_Py_norm["dim_x"] == dim_x), "norm_diff"]
    
    t_stat, p_val = scipy.stats.ttest_ind(sut_chol, sut_prin)
    print(f"{dim_x=}, ut_chol vs ut_prin: {p_val=}")
    
plt.figure()
fig, ax = plt.subplots(1,1, layout = "constrained")
ax.hist(sut_chol, alpha = .3, label = "SUT (Chol)")
ax.hist(sut_prin, alpha = .3, label = "SUT (Prin)")
ax.set_xlabel("RMSE")
ax.legend()
    

