# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:44:08 2023

@author: halvorak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pathlib
import os
import seaborn as sns
import utils_columnA as utils_colA

font = {'size': 16}
matplotlib.rc('font', **font)

"""
Plot results of the column A case study. Here, we only plot "statistical" results, i.e. RMSE and simulation times. Trajectories are plotted after running main_column_A.py

"""

#%% Def directories
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data_colA")

res_name = "res - 20231112 - 100iter cvodes dt2"
res_name = "res - 20231511 - 100iter rk4 dt5"
res_name = "res - 20231218 - 100iter"
# res_name = "res"
dir_res = os.path.join(dir_data, res_name)
info_sim = pd.read_pickle(os.path.join(dir_res, "info_sim.pickle")) #info (integrator type etc)

N_sim_to_plot = 100 #plot e.g. 50 of the first simulations

bleed_plt = False

t_arr = info_sim["t"]
t_end = 25 #[min]
try:
    ti_max = np.where(t_arr >= t_end)[0][0]
except IndexError:
    ti_max = t_arr.shape[0]-1
# ti_max = None

#read dataframes from the following subdirectories
sims = ["true", 
        "ekf", 
        "hd-ukf", 
        "s-ukf",
        ]
order_method = ["HD-UKF", "S-UKF", "EKF"]
# order_method = None


cost_func_type = "RMSE" #alternative ME (mean error)
# cost_func_type = "ME" #alternative ME
df = {si: utils_colA.read_df_in_dir(os.path.join(dir_res, si), concat = True, N_sim_end = N_sim_to_plot, ti_max = ti_max) for si in sims}

for si, dfi in df.items():
    dfi["State estimator"] = si.upper()

df = pd.concat(list(df.values()), ignore_index = True)


#%% Compute RMSE (absolute and relative)
col_states = ['x_1', 'x_2', 'x_3', 'x_4', 'M']
dim_x = len(col_states)
df_rmse = []
for si in sims[1:]:
    df_rmse.append(utils_colA.read_df_and_compute_rmse(os.path.join(dir_res, si), 
                                                  os.path.join(dir_res, "true"), 
                                                  col_states, cost_func_type = cost_func_type, N_sim_end = N_sim_to_plot, ti_max = ti_max))
    df_rmse[-1]["State estimator"] = si.upper()
    


df_rmse_rel = [(df_rmse[i].loc[:, col_states] / df_rmse[0].loc[:, col_states] -1.)*100 for i in range(1, len(df_rmse))]#relative error
for i in range(len(df_rmse_rel)):
    df_rmse_rel[i]["Tray"] = df_rmse[0]["Tray"]
    df_rmse_rel[i]["Ni"] = df_rmse[0]["Ni"]
    df_rmse_rel[i]["State estimator"] = (df_rmse[i+1]["State estimator"][0] + "/" +
                                         df_rmse[0]["State estimator"][0])

df_rmse_diff = [df_rmse[i].loc[:, col_states] - df_rmse[0].loc[:, col_states] for i in range(1, len(df_rmse))]#relative error
for i in range(len(df_rmse_diff)):
    df_rmse_diff[i]["Tray"] = df_rmse[0]["Tray"]
    df_rmse_diff[i]["Ni"] = df_rmse[0]["Ni"]
    df_rmse_diff[i]["State estimator"] = (df_rmse[i+1]["State estimator"][0] + "-" +
                                         df_rmse[0]["State estimator"][0])

df_rmse = pd.concat(df_rmse, ignore_index = True)
df_rmse_rel = pd.concat(df_rmse_rel, ignore_index = True)
df_rmse_diff = pd.concat(df_rmse_diff, ignore_index = True)

for se in sims:
    if se == "true":
        continue
    df_se = df_rmse[df_rmse["State estimator"] == se.upper()]
    
    

df_rmse_group = df_rmse.groupby(by = ["State estimator", "Ni", "Tray"]).mean() #multiindex
df_rmse_ave_tray = df_rmse_group.groupby(level = [0, 1]).mean()
df_rmse_ave_tray_ave_sim = df_rmse_ave_tray.groupby(level = 0).mean() #averaged over trays (inner) and simulations (outer)

try:
    print(f"df_rmse_ave_tray_ave_sim=\n{df_rmse_ave_tray_ave_sim}")
    print(f"df_rmse_ave_tray_ave_sim/df_rmse_ave_tray_ave_sim.loc['EKF',:]*100=\n{df_rmse_ave_tray_ave_sim/df_rmse_ave_tray_ave_sim.loc['EKF',:]*100}")
except KeyError:
    pass

df_rmse_ave_sim = df_rmse_group.groupby(level = [0,2]).mean() #averaged over trays (inner) and simulations (outer)
df_rmse_ave_sim = df_rmse_ave_sim.reset_index()

df_rmse_group = df_rmse.groupby(by = ["State estimator", "Ni"]).mean() #average over trays
df_rmse_group = df_rmse_group.drop(["Tray"], axis = 1)
df_rmse_group = df_rmse.groupby(by = ["State estimator"]).mean() #average over Ni
#%% Plot RMSE


fig_rmse, ax_rmse = plt.subplots(dim_x, 1, sharex = True, layout = "constrained", figsize = (8,6))
# fig_rmse.suptitle(f"{cost_func_type}")
state_label = [ r"$x_{LNK}$", r"$x_{LK}$", r"$x_{HK}$", r"$x_{HNK}$", r"M"]
uom = ["[-]", "[-]", "[-]", "[-]", "[kmol]"]
for i in range(dim_x):
    if bleed_plt:
        ax_rmse[i] = sns.lineplot(data = df_rmse, x = "Tray", y = col_states[i], hue = "State estimator", estimator = None, units = "Ni", ax = ax_rmse[i], hue_order = order_method)
    else:
        sns.lineplot(data = df_rmse, x = "Tray", y = col_states[i], hue = "State estimator", ax = ax_rmse[i], errorbar = ("sd", 1.), hue_order = order_method)
        # sns.lineplot(data = df_rmse, x = "Tray", y = col_states[i], hue = "State estimator", ax = ax_rmse[i])
    ax_rmse[i].set_ylabel(state_label[i] + " " + uom[i])
    # ax_rmse[i].set_yscale("log")
    # ax_rmse[i].ticklabel_format(useMathText=True)
    # ax_rmse[i].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base = 10))
    ax_rmse[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    if i > 0:
        ax_rmse[i].get_legend().remove()
ax_rmse[0].legend(ncol=2)        
#%% RMSE with manual control over whats plotted
fig_rmse2, ax_rmse2 = plt.subplots(dim_x, 1, sharex = True, layout = "constrained", figsize = (8,6))
fig_rmse2.suptitle(f"{cost_func_type}, manual eval")
for i in range(dim_x):
    sns.lineplot(data = df_rmse_ave_sim, x = "Tray", y = col_states[i], hue = "State estimator", ax = ax_rmse2[i], hue_order = order_method)
    ax_rmse2[i].set_ylabel(state_label[i] + " " + uom[i])
    if i > 0:
        ax_rmse2[i].get_legend().remove()    
ax_rmse2[0].legend(ncol=2)  


#%% Simulation time

df_t = []
df_si_iter = []
for i in range(1, len(sims)):
    si = sims[i]
    df_si = df[(df["State estimator"] == si.upper())]
    df_si_iter.append(pd.DataFrame(data = df_si["Ni"].unique()))
    df_si_iter[i-1] = pd.DataFrame(columns = ["State estimator", "Ni", "time_sim"])
    df_si_iter[i-1]["Ni"] = df_si["Ni"].unique()
    df_si_iter[i-1]["State estimator"] = si.upper()
    for ni in df_si["Ni"].unique():
        # print(df_si[df_si["Ni"] == ni].iloc[0]["time_sim"])
        df_si_iter[i-1].loc[ni, "time_sim"] = df_si[df_si["Ni"] == ni].iloc[0]["time_sim"]
        # print(ni)
df_t = pd.concat(df_si_iter, ignore_index= True)
del df_si_iter[:]
del df_si_iter    

fig_tsim, ax_tsim = plt.subplots(1,1, layout = "constrained")
sns.stripplot(df_t, x = "State estimator", y = "time_sim", ax = ax_tsim, order = [si.upper() for si in sims[1:]])


