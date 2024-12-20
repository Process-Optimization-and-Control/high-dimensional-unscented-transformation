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
Coordinate transformation example

We can re-use existing results by setting load_old_sim=True. If that one is False, simulation is re-run. Existing results are saved in dir_data_load and new results saved in dir_data.

"""

project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data_example_coordinate_transf") #directory where new data is saved
dir_data_load = os.path.join(project_dir, "data_example_coordinate_transf") #only loaded if load_old_sim=True
dir_plt = os.path.join(dir_data_load, "plots")

dirs = [dir_data, dir_data_load, dir_plt]
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)

#%% Set-up

load_old_sim = True #if True, it loads old simulations. False, it reruns simulations.
N = 20 #number of times to repeat the simulation

#state dimensions to run
dim_x_list = np.array([2, 10, 100, 1000]).astype(int)
print(f"{dim_x_list=}")

N_mc = int(5e5) #Number of MC samples

n_threads = 16 #parallel evaluation (only for UT based methods)
not_spd_val = np.nan #marks norm estimates of covariance with this value of Py is not SPD

diverged_sim = []
methods_implemented = ["HD-UT", "SUT", "HD-SUT", "UT", "Cubature"] 
methods_to_run = methods_implemented #select which method to run

sqrt_func = lambda P: scipy.linalg.cholesky(P, lower = True)

#making names for plotting later, e.g. HD-UT (Eig), HD-UT (Prin) etc
name_methods = methods_to_run

#prepare df to save results
col_df = name_methods.copy()
col_df.extend(["dim_x", "iter"])
df_Py_norm = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = col_df)
df_std_norm = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = df_Py_norm.columns)
df_ym_norm = pd.DataFrame(index = range(len(dim_x_list)*N), data = 0., columns = df_Py_norm.columns)

kwargs_ut = {"symmetrization": False}
idx_df = 0

if not load_old_sim:
    for ni in range(N):
        for j in range(len(dim_x_list)):
            ts = time.time()
        
            dim_x = dim_x_list[j]
            assert dim_x % 2 == 0, f"{dim_x=} and it is restricted to be even in this example."
            xm = np.zeros(dim_x)
            xm = np.hstack((np.zeros(int(dim_x/2)), np.ones(int(dim_x/2))*100))
            
            #%% Generate Px
            Px = np.diag(np.abs(np.random.normal(size = dim_x)))
            
            #uniform distribution with limits [-a, a].
            a_unif = np.sqrt(np.diag(Px)*3) #from definition of variance for uniform distribution
            kurtosis_unif = 1.8 # see kurtosis of uniform dist., Wikipedia
            beta_sut = 0.3

            dir_Px = os.path.join(dir_data, "Px")
            if not os.path.exists(dir_Px):
                os.mkdir(dir_Px)
            np.save(os.path.join(dir_Px, f"Px_Nmc{ni}_dimx{j}.npy"), Px)


            #%% Def func
            func = lambda x_in: np.hstack(
                (np.cos(x_in[:int(dim_x/2)])*x_in[int(dim_x/2):],
                 np.sin(x_in[:int(dim_x/2)])*x_in[int(dim_x/2):]))
            

            x_ca = ca.SX.sym(f"x_ca_{j}", dim_x)
            y_ca = ca.vertcat(ca.cos((x_ca[:int(dim_x/2)]))*x_ca[int(dim_x/2):],
                ca.sin((x_ca[:int(dim_x/2)]))*x_ca[int(dim_x/2):])
            ca_func = ca.Function("ca_func", [x_ca], [y_ca])
            ca_func_map = ca_func.map(int(2*dim_x + 1), "thread", n_threads)
            

            
            #%% Analytical solution, approximated by MC sim 
            
            if False: # random samples (standard MC simulations)
                x_samples = np.array([scipy.stats.uniform.rvs(loc = -ai, scale = 2*ai, size = N_mc) for ai in a_unif]) + xm.reshape(-1,1)  
            else: # LHS
                x_dists = np.array([scipy.stats.uniform(loc = -ai, scale = 2*ai) for ai in a_unif])
                x_samples = utils.get_lhs_points(x_dists, N_mc)
                
                #verify samples are in the region specified by the Uniform distribution
                check_lb = np.greater_equal(x_samples, -a_unif.reshape(-1,1))
                check_ub = np.less_equal(x_samples, a_unif.reshape(-1,1))
                assert (np.all(check_lb) and np.all(check_ub)), f"Somehting wrong with LHS. Have {np.all(check_lb)=} and {np.all(check_ub)=} for the generated samples"
                
                x_samples += xm.reshape(-1,1) #correct the mean
                
            
            y_mc = np.array(list(map(func, x_samples.T)))

            ym_ana = np.mean(y_mc, axis = 0)
            dim_y = ym_ana.shape[0]
            Py_ana = np.cov(y_mc, rowvar = False)
            
            
            assert Py_ana.shape == ((dim_x, dim_x))
                
            del y_mc #depending on dim_x and N_MC, these may take up significant space
            scipy.linalg.cholesky(Py_ana, lower = True)
            std_ana, corr_ana = utils.get_corr_std_dev(Py_ana)
            
            #%% error_eval_functions
            val_func_Py = lambda mat: scipy.linalg.norm(mat - Py_ana, "fro")
            val_func_std = lambda vec: scipy.linalg.norm(vec - std_ana)
            val_func_ym = lambda vec: scipy.linalg.norm(vec - ym_ana)
            
            #%% UT - kappa=1e-7, standard UT
            if "UT" in methods_to_run:

                Px_sqrt_i = sqrt_func(Px)
                
                points_a = spc.JulierSigmaPoints(dim_x, kappa = np.max([1e-7, kurtosis_unif - dim_x]), suppress_kappa_warning = True)
                sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
                
                kwargs_ut = {"symmetrization": False}
                yma, Pya, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
                dim_y = yma.shape[0]
                # assert np.allclose(Aa, 0.)
                del Aa
                
                df_ym_norm.loc[idx_df, "UT"] = val_func_ym(yma)
                try: #check if covariance matrix is positive definite
                    scipy.linalg.cholesky(Pya, lower = True)
                    std_a = utils.get_corr_std_dev(Pya)[0]
                    df_Py_norm.loc[idx_df, "UT"] = val_func_Py(Pya)
                    df_std_norm.loc[idx_df, "UT"] = val_func_std(std_a)
                    del Pya, std_a #depending on dim_x, these may take up significant space
                except scipy.linalg.LinAlgError:
                    df_Py_norm.loc[idx_df, "UT"] = not_spd_val
                    df_std_norm.loc[idx_df, "UT"] = not_spd_val
                    diverged_sim.append({"Method": "UT", "Px": Px, "Py": Pya, "idx_df": idx_df, "dim_x": dim_x})
                    del Pya
                # assert np.allclose(ym_ana, yma)
            #%% Cubature - kappa=0, standard UT
            if "Cubature" in methods_to_run:

                Px_sqrt_i = sqrt_func(Px)
                
                points_a = spc.JulierSigmaPoints(dim_x, kappa = 0., suppress_kappa_warning = True)
                sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
                
                kwargs_ut = {"symmetrization": False}
                yma, Py_cub, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
                dim_y = yma.shape[0]
                # assert np.allclose(Aa, 0.)
                del Aa
                
                
                df_ym_norm.loc[idx_df, "Cubature"] = val_func_ym(yma)
                try: #check if covariance matrix is positive definite
                    scipy.linalg.cholesky(Py_cub, lower = True)
                    std_a = utils.get_corr_std_dev(Py_cub)[0]
                    df_Py_norm.loc[idx_df, "Cubature"] = val_func_Py(Py_cub)
                    df_std_norm.loc[idx_df, "Cubature"] = val_func_std(std_a)
                    del Py_cub, std_a #depending on dim_x, these may take up significant space
                except scipy.linalg.LinAlgError:
                    df_Py_norm.loc[idx_df, "Cubature"] = not_spd_val
                    df_std_norm.loc[idx_df, "Cubature"] = not_spd_val
                    diverged_sim.append({"Method": "Cubature", "Px": Px, "Py": Py_cub, "idx_df": idx_df, "dim_x": dim_x})
                    del Py_cub
                # assert np.allclose(ym_ana, yma)
            #%% SUT
            if "SUT" in methods_to_run:

                Px_sqrt_i = sqrt_func(Px)
                
                points_a = spc.ScaledSigmaPoints(dim_x, alpha = 1e-2, beta = beta_sut, kappa = 1e-7, suppress_init_warning = True)
                sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
                
                kwargs_ut = {"symmetrization": False}
                yma, Pya, Aa = ut.unscented_transform_w_function_eval_wslr(sigmas_a, Wm_a, Wc_a, func, **kwargs_ut)
                dim_y = yma.shape[0]
                del Aa
                
                df_ym_norm.loc[idx_df, "SUT"] = val_func_ym(yma)
                
                try: #check if covariance matrix is positive definite
                    scipy.linalg.cholesky(Pya, lower = True)
                    std_a = utils.get_corr_std_dev(Pya)[0]
                    df_Py_norm.loc[idx_df, "SUT"] = val_func_Py(Pya)
                    df_std_norm.loc[idx_df, "SUT"] = val_func_std(std_a)
                    del Pya, std_a #depending on dim_x, these may take up significant space
                except scipy.linalg.LinAlgError:
                    df_Py_norm.loc[idx_df, "SUT"] = not_spd_val
                    df_std_norm.loc[idx_df, "SUT"] = not_spd_val
                    diverged_sim.append({"Method": "SUT", "Px": Px, "Py": Pya, "idx_df": idx_df, "dim_x": dim_x})
                    del Pya
                # assert np.allclose(ym_ana, yma)
        
           
            #%% HD-UT
            if "HD-UT" in methods_to_run:
                points_x = spc.JulierSigmaPoints(dim_x, kappa = kurtosis_unif - dim_x, suppress_kappa_warning = True, kurtosis = kurtosis_unif)
                
                Px_sqrt_i = sqrt_func(Px)
                
                #check condition number of matrix square-root
                
                ym_hdut, Py_hdut, _, A_wslr, _, = ut.hdut_map_func(xm, Px_sqrt_i, ca_func_map, points_x, calc_Pxy = True, calc_A = True)
                                    
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
                    diverged_sim.append({"Method": "HD-SUT", "Px": Px, "Py": Py_hdsut, "idx_df": idx_df, "dim_x": dim_x})
                    # del Py_hdsut
        
            
            
            #%% print progression
            t_iter = time.time() - ts
            
            df_ym_norm.loc[idx_df, "dim_x"] = dim_x
            df_Py_norm.loc[idx_df, "dim_x"] = dim_x
            df_std_norm.loc[idx_df, "dim_x"] = dim_x
            df_ym_norm.loc[idx_df, "Ni"] = ni
            df_Py_norm.loc[idx_df, "Ni"] = ni
            df_std_norm.loc[idx_df, "Ni"] = ni
            df_ym_norm.loc[idx_df, "Ni"] = N_mc
            df_Py_norm.loc[idx_df, "N_mc"] = N_mc
            df_std_norm.loc[idx_df, "N_mc"] = N_mc
            
            idx_df +=1
            
            #save dfs
            df_ym_norm.to_pickle(os.path.join(dir_data, "df_ym_norm.pkl"))
            df_Py_norm.to_pickle(os.path.join(dir_data, "df_Py_norm.pkl"))
            df_std_norm.to_pickle(os.path.join(dir_data, "df_std_norm.pkl"))

            print(f"Iter {ni+1}/{N}, subiter {j+1}/{len(dim_x_list)}. {dim_x=} and {t_iter= :.2f} s")
    
    #%% Melt dataframe
    df_ym_norm = df_ym_norm.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
    
    df_Py_norm = df_Py_norm.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
    df_Py_norm["diverged_old"] = np.where(df_Py_norm["norm_diff"] == not_spd_val, True, False)
    df_Py_norm["diverged"] = df_Py_norm["norm_diff"].isna()
    
    df_std_norm = df_std_norm.melt(id_vars = ["dim_x", "Ni"], value_vars = name_methods, var_name = "Method", value_name = "norm_diff")
    
else:
    df_Py_norm = pd.read_pickle(os.path.join(dir_data_load, "df_Py_norm.pkl"))
    df_ym_norm = pd.read_pickle(os.path.join(dir_data_load, "df_ym_norm.pkl"))
    df_std_norm = pd.read_pickle(os.path.join(dir_data_load, "df_std_norm.pkl"))   
    


#%% Save dfs

df_ym_norm.to_pickle(os.path.join(dir_data, "df_ym_norm.pkl"))
df_Py_norm.to_pickle(os.path.join(dir_data, "df_Py_norm.pkl"))
df_std_norm.to_pickle(os.path.join(dir_data, "df_std_norm.pkl"))


df_diverged = df_Py_norm.copy()
df_diverged.loc[df_diverged["norm_diff"].isna(), "diverged"] = True
df_diverged = df_diverged[["dim_x", "Method", "diverged"]]
df_div_group = df_diverged.groupby(["dim_x", "Method"])
df_diverged_sum = df_div_group.sum().unstack(level = 0)
df_diverged_sum.to_csv(os.path.join(dir_data, "simulations_diverged.csv"))



#%% Print statistics and save to file
ym_norm_group = df_ym_norm.loc[:, ["dim_x", "Method", "norm_diff"]].groupby(by = ["dim_x", "Method"])
ym_norm_mean = ym_norm_group.mean().unstack(level=0)
ym_norm_std = ym_norm_group.std().unstack(level=0)
print(f"ym_norm_group.mean:\n{ym_norm_mean}")

df_Py_norm_c = df_Py_norm[df_Py_norm["diverged"] == False]
Py_norm_group = df_Py_norm_c.loc[:, ["dim_x", "Method", "norm_diff"]].groupby(by = ["dim_x", "Method"])
Py_norm_mean = Py_norm_group.mean().unstack(level=0)
Py_norm_std = Py_norm_group.std().unstack(level=0)
print(f"Py_norm_group.mean:\n{Py_norm_mean}")


df_std_norm_c = df_std_norm[df_Py_norm["diverged"] == False]
std_norm_group = df_std_norm_c.loc[:, ["dim_x", "Method", "norm_diff"]].groupby(by = ["dim_x", "Method"])
std_norm_mean = std_norm_group.mean().unstack(level=0)
std_norm_std = std_norm_group.std().unstack(level=0)
print(f"std_norm_group.mean:\n{std_norm_mean}")

print(f"df_diverged_sum:\n{df_diverged_sum}")


#save to .csv files
Py_norm_mean.to_csv(os.path.join(dir_data, "Py_norm_mean.csv"))
Py_norm_std.to_csv(os.path.join(dir_data, "Py_norm_std.csv"))

ym_norm_mean.to_csv(os.path.join(dir_data, "ym_norm_mean.csv"))
ym_norm_std.to_csv(os.path.join(dir_data, "ym_norm_std.csv"))

std_norm_mean.to_csv(os.path.join(dir_data, "std_norm_mean.csv"))
std_norm_std.to_csv(os.path.join(dir_data, "std_norm_std.csv"))


#%% Plots
subset_method = [mi for mi in methods_implemented]
df_ym_norm_ss = df_ym_norm[df_ym_norm["Method"].isin(subset_method)]
df_ym_norm_ss["dim_x"] = df_ym_norm_ss.loc[:, "dim_x"].astype(int)
# df_ym_norm_ss = df_ym_norm_ss.loc[df_ym_norm_ss["diverged"], :]

df_Py_norm_ss = df_Py_norm[df_Py_norm["Method"].isin(subset_method)]
df_Py_norm_ss["dim_x"] = df_Py_norm_ss.loc[:, "dim_x"].astype(int)
df_Py_norm_ss = df_Py_norm_ss.loc[~df_Py_norm_ss["diverged"], :]

df_std_norm_ss = df_std_norm[df_std_norm["Method"].isin(subset_method)]
df_std_norm_ss["dim_x"] = df_std_norm_ss.loc[:, "dim_x"].astype(int)
df_std_norm_ss = df_std_norm_ss.dropna(how = "any")

hue_order = ["HD-UT", "HD-SUT", "SUT", "UT", "Cubature"]

fig_ym_pp, ax_ym_pp_ym = plt.subplots(1, 1, layout = "constrained", sharex = True)#, figsize = [6.4 , 5.05])
fg = sns.pointplot(df_ym_norm_ss, x = "dim_x", y = "norm_diff", hue = "Method", log_scale = (False, True),
                 estimator = "mean",
                 ax = ax_ym_pp_ym,
                 errorbar = ("ci", 95),
                  # kind = "point",
                 markers=["^", "o", "x", "v", "P"],
                 linestyles=["solid", "dotted", "dashed", "dashdot", (0, (3, 10, 1, 10))],
                 hue_order = hue_order,
                 # xlabel = r"$n_{x}$"
                 )



ax_ym_pp_ym.set_ylabel(r"$||\hat{y}-\hat{y}^{LHS}||_{2}$")
ax_ym_pp_ym.set_xlabel(r"$n_{x}$")
h, l = ax_ym_pp_ym.get_legend_handles_labels()
l = ["UT-C" if li == "Cubature" else li.split(" (Chol)")[0] for li in l]
ax_ym_pp_ym.get_legend().remove()
fig_ym_pp.legend(handles = h, labels = l, loc = "outside upper center", ncols = 3, frameon = False)
fname = "Figure 3 - ym_norm_x_unif"
[fig_ym_pp.savefig(os.path.join(dir_plt, f"{fname}.{ext}")) for ext in ["svg", "pdf", "eps"]]

fig_Py_pp, ax_Py_pp_ym = plt.subplots(1, 1, layout = "constrained", sharex = True)#, figsize = [6.4 , 5.05])
fg = sns.pointplot(df_Py_norm_ss, x = "dim_x", y = "norm_diff", hue = "Method", log_scale = (False, True),
                 estimator = "mean",
                 ax = ax_Py_pp_ym,
                 errorbar = ("ci", 95),
                  # kind = "point",
                 markers=["^", "o", "x", "v", "P"],
                 linestyles=["solid", "dotted", "dashed", "dashdot", (0, (3, 10, 1, 10))],
                 hue_order = hue_order
                 # xlabel = r"$n_{x}$"
                 )



ax_Py_pp_ym.set_ylabel(r"$||P_{y}-P_{y}^{LHS}||_{F}$")
ax_Py_pp_ym.set_xlabel(r"$n_{x}$")
h, l = ax_Py_pp_ym.get_legend_handles_labels()
l = ["UT-C" if li == "Cubature" else li.split(" (Chol)")[0] for li in l]
ax_Py_pp_ym.get_legend().remove()
fig_Py_pp.legend(handles = h, labels = l, loc = "outside upper center", ncols = 3, frameon = False)
fname = "Py_norm_x_unif"
[fig_Py_pp.savefig(os.path.join(dir_plt, f"{fname}.{ext}")) for ext in ["svg", "pdf", "eps"]]

fig_std_pp, ax_std_pp_ym = plt.subplots(1, 1, layout = "constrained", sharex = True)#, figsize = [6.4 , 5.05])
fg = sns.pointplot(df_std_norm_ss, x = "dim_x", y = "norm_diff", hue = "Method", log_scale = (False, True),
                 estimator = "mean",
                 ax = ax_std_pp_ym,
                 errorbar = ("ci", 95),
                  # kind = "point",
                 markers=["^", "o", "x", "v", "P"],
                 linestyles=["solid", "dotted", "dashed", "dashdot", (0, (3, 10, 1, 10))],
                 hue_order = hue_order
                 # xlabel = r"$n_{x}$"
                 )



ax_std_pp_ym.set_ylabel(r"$||\sigma_{y}-\sigma_{y}^{LHS}||_{2}$")
ax_std_pp_ym.set_xlabel(r"$n_{x}$")
h, l = ax_std_pp_ym.get_legend_handles_labels()
l = ["UT-C" if li == "Cubature" else li.split(" (Chol)")[0] for li in l]
ax_std_pp_ym.get_legend().remove()
fig_std_pp.legend(handles = h, labels = l, loc = "outside upper center", ncols = 3, frameon = False)
fname = "std_dev_norm_x_unif"
[fig_std_pp.savefig(os.path.join(dir_plt, f"{fname}.{ext}")) for ext in ["svg", "pdf", "eps"]]


fig_bp, (ax_bp_Py, ax_bp_std) = plt.subplots(1, 2, layout = "constrained")
fg = sns.pointplot(df_std_norm_ss, x = "dim_x", y = "norm_diff", hue = "Method", log_scale = (False, True),
                 estimator = "mean",
                 ax = ax_bp_std,
                 errorbar = ("ci", 95),
                  # kind = "point",
                 markers=["^", "o", "x", "v", "P"],
                 linestyles=["solid", "dotted", "dashed", "dashdot", (0, (3, 10, 1, 10))],
                 hue_order = hue_order
                 # xlabel = r"$n_{x}$"
                 )
fg = sns.pointplot(df_Py_norm_ss, x = "dim_x", y = "norm_diff", hue = "Method", log_scale = (False, True),
                 estimator = "mean",
                 ax = ax_bp_Py,
                 errorbar = ("ci", 95),
                  # kind = "point",
                 markers=["^", "o", "x", "v", "P"],
                 linestyles=["solid", "dotted", "dashed", "dashdot", (0, (3, 10, 1, 10))],
                 hue_order = hue_order
                 # xlabel = r"$n_{x}$"
                 )


ax_bp_std.set_ylabel(r"$||\sigma_{y}-\sigma_{y}^{LHS}||_{2}$")
ax_bp_Py.set_ylabel(r"$||P_{y}-P_{y}^{LHS}||_{F}$")
ax_bp_Py.set_xlabel(r"$n_{x}$")
ax_bp_std.set_xlabel(r"$n_{x}$")
h, l = ax_bp_Py.get_legend_handles_labels()
l = ["UT-C" if li == "Cubature" else li.split(" (Chol)")[0] for li in l]


ax_bp_std.get_legend().remove()
ax_bp_Py.get_legend().remove()
fig_bp.legend(handles = h, labels = l, loc = "outside upper center", ncols = 3, frameon = False)
fname = "Figure 4 - Py_and_std_dev_norm_together_x_unif"
[fig_bp.savefig(os.path.join(dir_plt, f"{fname}.{ext}")) for ext in ["svg", "pdf", "eps"]]


