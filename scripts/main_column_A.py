# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:34:34 2022

@author: halvorak
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import pathlib
import os
import scipy.linalg
import scipy.stats
import copy
import time
import timeit
import pandas as pd
import seaborn as sns
import casadi as ca
import pickle


#Self-written modules
from state_estimator import sigma_points_classes as spc
from state_estimator import UKF
from state_estimator import EKF

import utils_columnA as utils_colA
font = {'size': 14}
matplotlib.rc('font', **font)


#%% Set N simulation times
N_sim = 100 #this is how many times to repeat each iteration
n_threads_fx = int(16) #number of threads for fx and hx function in HD-UT
n_threads_hx = int(6)
n_threads_nlp_update = int(16)
integrator_casadi = "rk"
opts_integrator = {}

#select which filter(s) to run
filters_to_run = ["hd-ukf", "ekf", "s-ukf"]
# filters_to_run = ["hd-ukf", "s-ukf"]
# filters_to_run = [] #if only want to run the simulation (x_true)
load_sim_data = False # loads x_true, y and u_hist from a previous simulation. If False, it will simulate N_sim iterations
load_x0_kf = False
delete_old_folders = {"true": True, 
                      "ekf": True, 
                      "hd-ukf": True,
                      "s-ukf": True,
                      } #delete all previously saved data.
# Number of components
NC_used = 4
assert (NC_used == 2) or (NC_used == 4)

cost_func_type = "RMSE" #other valid option is "valappil"

#define time
dt = (1/60.)*5. # [min]
tf = 25 # [min]
t = np.arange(tf, step = dt)

print(f"{N_sim=}, {tf=} and {filters_to_run=}")
#%% Column data
# The following data need to be changed for a new column.
# These data are for "column A".
    
# Number of stages (including reboiler and total condenser: 
NT = 41 
# Location of feed stage (stages are counted from the bottom):
NF = 21
# Number of components
NC = 4
# Relative volatilities: alpha = [alpha1 alpha2...alphaNC]
alpha = np.array([4.35, 2.08, 1.44])

#data used in the measurement equations
idx_TIT = [0, 10, 20, NT-1] #which trays we have a temperature measurement
idx_TIT = [0, NT-1] #which trays we have a temperature measurement
idx_TIT = [5, NT-1-5] #which trays we have a temperature measurement
radius_col = .2  # [m] radius of column 
T_boiling_points= np.array([351.4, 370.4, 380.8, 391.9]) # [K], boiling points pure components
MW = np.array([46.1, 60.1, 74.1, 74.1]) # [g/mol], molar weight
dvapH = np.array([39.56, 41.17, 42.54, 43.06])*1000 # [J/mol], heat of vaporization
R = 8.314 #[J/K mol]


#%% Def directories
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data_colA")
dir_plots = os.path.join(project_dir, "plots_colA")

dir_res = os.path.join(dir_data, "res")
sims = copy.deepcopy(filters_to_run)
sims.extend(["true"])
for si in sims:
    dir_res_si = os.path.join(dir_res, si)
    if not os.path.exists(dir_res_si):
        os.mkdir(dir_res_si)
    if delete_old_folders[si]:
        utils_colA.delete_files_in_folder(dir_res_si) #clean directory from previous results

if not os.path.exists(dir_plots):
    os.mkdir(dir_plots)
    
info_sim = {"integrator": integrator_casadi, "n_threads_fx": n_threads_fx, "n_threads_hx": n_threads_hx, "NT": NT, "NF": NF, "t": t, "NC": NC_used, "idx_TIT": idx_TIT}

with open(os.path.join(dir_res, "info_sim.pickle"), "wb") as handle:
    pickle.dump(info_sim, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% Casadi integrator, jacobian df/dp
(F_dot, integrator, x_sym, M_sym, u_sym, d_sym, F_set_X, fx, fx_jac_func, 
 hx, hx_jac_func, func_plt, func_x2y, x_next_int, get_temp_profile, dim_TIT, dim_LIT) =  utils_colA.multicomA_ca(dt, NT, NF, NC, alpha, T_boiling_points, idx_TIT, MW, radius_col, dvapH, integrator_casadi = integrator_casadi, opts_integrator = opts_integrator)

#qp to constrain x_true (avoid e.g. negative concentrations)
opts_ipopt = {"print_time": 0, "ipopt": {"print_level": 0, "acceptable_tol": 1e-6, "linear_solver": "mumps"}}
S_qp, g_lb, g_ub = utils_colA.least_squares_clipping(ca.vertcat(x_sym, M_sym), func_plt, solver = "ipopt", opts_qp = opts_ipopt)

x_qp = ca.MX.sym("x_qp", x_sym.shape[0] + M_sym.shape[0])
sol_qp_sym = S_qp(x0 = x_qp, p = x_qp, lbg = g_lb, ubg = g_ub)
x_qp_sol = sol_qp_sym["x"]
func_constrain_by_qp = ca.Function("func_constrain_by_qp", [x_qp], [x_qp_sol])

#%%Def x0 and dimensions

_, x0, P0, u0, d0 = utils_colA.get_initial_conditions(NT, NC, NC_used)

x0_paper = np.array([0.25, .25, .25])
xc_0 = np.repeat(x0_paper[:, np.newaxis], NT)
x0 = np.hstack((xc_0, x0[-NT:]))
# x0 = F_set_X(np.ones(NT*(NC-1))*0.25, x0[-NT:]).toarray().flatten()
x0_dist = scipy.stats.multivariate_normal(mean = x0, cov = P0)

dim_t = t.shape[0]
dim_x = x0.shape[0]
dim_u = u0.shape[0]
dim_d = d0.shape[0]
y0 = hx(x0)
dim_y = y0.shape[0]

assert ((x_sym.shape[0] + M_sym.shape[0]) == dim_x)
assert (d_sym.shape[0] == dim_d)
assert (u_sym.shape[0] == dim_u)
assert (dim_y == dim_TIT + dim_LIT)


#%% Def noise
Q_nom = np.ones(dim_x)*1e-10
#repeatability of each sensor
accuracy_TT = 120*.25/100 # [degC] from DS
accuracy_dp = 320*.065/100 #[mbar] from DS
rep_TT = accuracy_TT/2. #assumed repeatability (noise) is 50% of accuracy
rep_LIT = accuracy_dp/2.
R_nom = np.hstack((np.ones(dim_TIT)*rep_TT**2,
                   np.ones(dim_LIT)*rep_LIT**2))

#%% Bounds on states (for x_true)
#set lower and upper bounds
eps = 1e-10
x_lb = np.ones(dim_x)*-eps
x_ub = np.hstack((np.ones(dim_x - NT)+eps, # mole fractions
                  np.ones(NT)*np.inf) #liquid hold-up
                 )

#%% Def HD-UKF
points_hdut = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x, suppress_kappa_warning = True)

#create map functions for fx and hx functions
if n_threads_fx == 1:
    fx_map_func = fx.map(points_hdut.dim_sigma)
else:
    fx_map_func = fx.map(points_hdut.dim_sigma, "thread", n_threads_fx)
if n_threads_hx == 1:
    hx_map_func = hx.map(points_hdut.dim_sigma)
else:
    hx_map_func = hx.map(points_hdut.dim_sigma, "thread", n_threads_hx)

hd_ukf = UKF.HD_UKF_additive_noise(x0.copy(), P0.copy(), fx_map_func, hx_map_func, points_hdut, Q_nom, R_nom, name="HD-UKF", P_sqrt_func = spc.sqrt_eig)
#%% Scaled UKF

points_s = spc.ScaledSigmaPoints(dim_x, kappa = 1e-7, alpha = 1e-3, beta = 2., sqrt_method = spc.sqrt_eig)

s_ukf = UKF.UKF_map_based(x0.copy(), P0.copy(), fx_map_func, hx_map_func, points_s, np.diag(Q_nom), np.diag(R_nom), name="S-UKF")

#%% Def EKF
ekf = EKF.EKF_additive_noise(x0.copy(), P0.copy(), fx, fx_jac_func, hx, hx_jac_func, Q_nom, R_nom) 

#%%Allocate array, N_sim
j_rmse_hd_ukf = np.zeros((dim_x, N_sim))
j_rmse_s_ukf = np.zeros((dim_x, N_sim))
j_rmse_ekf = np.zeros((dim_x, N_sim))

#good to check both mean and rmse
j_mean_hd_ukf = np.zeros((dim_x, N_sim))
j_mean_s_ukf = np.zeros((dim_x, N_sim))
j_mean_ekf = np.zeros((dim_x, N_sim))

#everything in dataframes
df_true = [[] for Ni in range(N_sim)]
df_hd_ukf = [[] for Ni in range(N_sim)]
df_s_ukf = [[] for Ni in range(N_sim)]
df_ekf = [[] for Ni in range(N_sim)]

time_sim_hd_ukf = np.zeros(N_sim)
time_sim_s_ukf = np.zeros(N_sim)
time_sim_ekf = np.zeros(N_sim)

Ni = 0
rand_seed = 6969
ts = time.time() #total start time
ti = time.time() #iteration time for a single run

print_subiter = True #print certain timesteps for a single case
num_exceptions = 0 #number of times we fail and start over
while Ni < N_sim:
    try:
        np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
        t_iter = time.time()
        t_iter_N_sim = time.time()
        
        
        #%% Allocate initial arrays and set x0
        if load_x0_kf:
            x0_kf = np.load(os.path.join(dir_res, "true", f"x0_kf_{Ni}.npy"))
        else: #draw random x0_kf
            x0_positive = False
            while not x0_positive:
                xc_0_kf = np.random.multivariate_normal(mean = x0[:NC-1], cov = P0[:(NC-1),:(NC-1)])
                x0_M = np.random.normal(loc = x0[-1], scale = np.sqrt(P0[-1,-1]))
                if ((xc_0_kf > 0.).all() and (xc_0_kf.sum() < 1.) and (x0_M > 0.)):
                    x0_positive = True
            xc_0_kf = np.repeat(xc_0_kf[:, np.newaxis], NT)
            x0_kf = np.hstack((xc_0_kf, np.ones(NT)*x0_M))
        
        
        #Arrays where values are stored
        x_true = np.zeros((dim_x, dim_t)) 
        x_dot = np.zeros((dim_x, dim_t)) 
        x_true_plt = np.zeros((dim_t, NC + 1, NT))
        u_hist = np.zeros((dim_u, dim_t)) 
        d_hist = np.zeros((dim_d, dim_t)) 
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        
        #Arrays where posterior prediction is stored
        x_post_hd_ukf = np.zeros((dim_x, dim_t))
        x_post_s_ukf = np.zeros((dim_x, dim_t))
        x_post_ekf = np.zeros((dim_x, dim_t))
        
        #diagnonal elements of covariance matrices
        P_diag_post_hd_ukf = np.zeros((dim_x, dim_t))
        P_diag_post_s_ukf = np.zeros((dim_x, dim_t))
        P_diag_post_ekf = np.zeros((dim_x, dim_t))
        
        #save the starting points for the true system and the filters
        x_true[:, 0] = x0
        u_hist[:, 0] = u0
        d_hist[:, 0] = d0
        
        x_post_hd_ukf[:, 0] = x0_kf.copy()
        x_post_s_ukf[:, 0] = x0_kf.copy()
        x_post_ekf[:, 0] = x0_kf.copy()
        x_ol[:, 0] = x0_kf.copy()
        
        x_comp_i, M_i = func_plt(x_true[:, 0])
        x_true_plt[0, :NC, :] = x_comp_i.toarray()
        x_true_plt[0, -1, :] = M_i.toarray().flatten()
        
        #save starting points for covariance matrices
        P_diag_post_hd_ukf[:, 0] = np.diag(P0.copy())
        P_diag_post_s_ukf[:, 0] = np.diag(P0.copy())
        P_diag_post_ekf[:, 0] = np.diag(P0.copy())
        
        y = np.zeros((dim_y, dim_t))
        vk = np.array([np.random.normal(0., sig_i) for sig_i in np.sqrt(R_nom)])
        y[:, 0] = hx(x_true[:, 0]).toarray().flatten() + vk
        
        #%% Initialize state estimators
        hd_ukf.x_post = x0_kf.copy()
        hd_ukf.P_post = P0.copy()
        
        s_ukf.x_post = x0_kf.copy()
        s_ukf.P_post = P0.copy()
        
        ekf.x_post = x0_kf.copy()
        ekf.P_post = P0.copy()

        
        #%% Simulate the plant
        if load_sim_data:
            x_true = np.load(os.path.join(dir_res, "true", f"x_true_{Ni}.npy"))
            y = np.load(os.path.join(dir_res, "true", f"y_{Ni}.npy"))
            u_hist = np.load(os.path.join(dir_res, "true", f"u_hist_{Ni}.npy"))
            print("'True' plant data loaded")
            # make things easier for plotting later
            for i in range(dim_t):
                x_comp_i, M_i = func_plt(x_true[:, i])
                x_true_plt[i, :NC, :] = x_comp_i.toarray()
                x_true_plt[i, -1, :] = M_i.toarray().flatten()
            
        else:
            for i in range(1, dim_t):
                
                #control action
                LT = 2.70629 
                u_hist[:, i-1] = utils_colA.P_controller(x_true[:, i-1], func_plt, LT)
                F = 1.
                d_hist[0,i-1] = F
                
                #Simulate the plant
                x_true[:, i] = fx(x_true[:, i-1], u_hist[:, i-1], d_hist[:, i-1]).toarray().flatten()
                
                #assert x_true is within bounds (and respects that the sum of mole fractions should be 1). Do this by the "QP" approach
                if not utils_colA.is_within_bounds(x_true[:, i], func_plt, x_lb, x_ub, eps = eps):
                    sol_x = S_qp(x0 = x_true[:, i], p = x_true[:, i], lbg = g_lb, ubg = g_ub)
                    assert S_qp.stats()["success"], f"{S_qp.stats()=}"
                    x_true[:, i] = sol_x["x"].toarray().flatten()
               
                # make things easier for plotting later
                x_comp_i, M_i = func_plt(x_true[:, i])
                x_true_plt[i, :NC, :] = x_comp_i.toarray()
                x_true_plt[i, -1, :] = M_i.toarray().flatten()
                
                
                #Simulate the open loop (kf distruabance and starting point)
                x_ol[:, i] = fx(x_ol[:, i-1], u_hist[:, i-1], d0).toarray().flatten()
    
                #Make a new measurement
                vk = np.array([np.random.normal(0., sig_i) for sig_i in np.sqrt(R_nom)])
                y[:, i] = hx(x_true[:, i]).toarray().flatten() + vk
                
                #disturbance value 
                d_hist[:, i] = d_hist[:, i-1]
            print("Plant simulated")
        #%% Run EKF on the entire measurement set
        if "ekf" in filters_to_run:
            
            ts_ekf = timeit.default_timer()
            P_post_ekf = P0.copy()
            for i in range(1, dim_t):

                ekf.predict(fx_args = [u_hist[:, i-1], d0])
                ekf.update(y[:, i])
                
                #Note that Jacobian evaluations are quite slow in ekf.predict(), seems there is something wrong with the sparsity of the Jacobian matrices. Numerically, the values are in the range 1e-50, but they are not marked as sparse and therefore it takes quite long time to calculate the full Jacobian. It is verified that the model has same values for dxdt as the Matlab model. In essence, the casadi model is correct (numerically, 1e-50=0 but it is not marked as sparse), but it has some efficiency issues.
                
                #Save estimates
                x_post_ekf[:, i] = ekf.x_post
                P_diag_post_ekf[:, i] = np.diag(ekf.P_post)
                    

                if (i%1 == 0):
                    t_iter = timeit.default_timer() - ts_ekf
                    print(f"EKF, iter {i}/{dim_t} completed on {t_iter :.2f} s")
            tf_ekf = timeit.default_timer()
            time_sim_ekf[Ni] = tf_ekf - ts_ekf
            del ekf.P_post, ekf.P_prior, ekf.x_prior, ekf.x_post
            print(f"EKF iter {Ni+1}/{N_sim} done, completed on {time_sim_ekf[Ni] :.0f}s = {time_sim_ekf[Ni]/60 :.1f} min")
       
      
        #%% Run HD-UKF on the entire measurement set
        if "hd-ukf" in filters_to_run:    
            ts_hd_ukf = timeit.default_timer()
            for i in range(1, dim_t):
                
                hd_ukf.predict(fx_args = [u_hist[:, i-1], d0])
                hd_ukf.update(y[:, i])

                #Save estimates
                x_post_hd_ukf[:, i] = hd_ukf.x_post
                P_diag_post_hd_ukf[:, i] = np.diag(hd_ukf.P_post)

                
                if (i%300 == 0):
                    t_iter = timeit.default_timer() - ts_hd_ukf
                    print(f"HD-UKF, iter {i}/{dim_t} completed on {t_iter :.2f} s. {n_threads_fx=} and {n_threads_hx=}")
            
                
            tf_hd_ukf = timeit.default_timer()
            time_sim_hd_ukf[Ni] = tf_hd_ukf - ts_hd_ukf
            print(f"HD-UKF iter {Ni+1}/{N_sim} done, completed on {time_sim_hd_ukf[Ni] :.0f}s = {time_sim_hd_ukf[Ni]/60 :.1f} min. {n_threads_fx=} and {n_threads_hx=}")
        
       
        #%% Run S-UKF on the entire measurement set
        if "s-ukf" in filters_to_run:    
            ts_s_ukf = timeit.default_timer()
            for i in range(1, dim_t):

                s_ukf.predict(fx_args = [u_hist[:, i-1], d0])
                s_ukf.update(y[:, i])

                #Save estimates
                x_post_s_ukf[:, i] = s_ukf.x_post
                P_diag_post_s_ukf[:, i] = np.diag(s_ukf.P_post)
                
                if (i%300 == 0):
                    t_iter = timeit.default_timer() - ts_s_ukf
                    print(f"s-ukf, iter {i}/{dim_t} completed on {t_iter :.2f} s. {n_threads_fx=} and {n_threads_hx=}")
            
                
            tf_s_ukf = timeit.default_timer()
            time_sim_s_ukf[Ni] = tf_s_ukf - ts_s_ukf
        
            del s_ukf.P_post, s_ukf.P_prior, s_ukf.x_post, s_ukf.x_prior
            print(f"s-ukf iter {Ni+1}/{N_sim} done, completed on {time_sim_s_ukf[Ni] :.0f}s = {time_sim_s_ukf[Ni]/60 :.1f} min. {n_threads_fx=} and {n_threads_hx=}")
        
        #%% Dataframes, performance index, save
        value_filter_not_run = 1 #same cost as OL response
        
        df_true[Ni] = utils_colA.states_one_sim_to_df(x_true, dim_t, func_plt)
        df_true[Ni]["Ni"] = int(Ni)
        
        df_true[Ni].to_pickle(os.path.join(dir_res, "true", f"df_{Ni}.pkl"))
        
        np.save(os.path.join(dir_res, "true", f"x_true_{Ni}.npy"), x_true)
        np.save(os.path.join(dir_res, "true", f"y_{Ni}.npy"), y)
        np.save(os.path.join(dir_res, "true", f"u_hist_{Ni}.npy"), u_hist)
        np.save(os.path.join(dir_res, "true", f"x0_kf_{Ni}.npy"), x0_kf)
        
       
        if "hd-ukf" in filters_to_run:
            j_rmse_hd_ukf[:, Ni] = utils_colA.compute_performance_index_rmse(x_post_hd_ukf, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_hd_ukf[:, Ni] = np.mean(x_post_hd_ukf - x_true, axis = 1)
            
            df_hd_ukf[Ni] = utils_colA.states_one_sim_to_df(x_post_hd_ukf, dim_t, func_plt)
            df_hd_ukf[Ni]["Ni"] = int(Ni)
            df_hd_ukf[Ni]["time_sim"] = time_sim_hd_ukf[Ni]
            df_hd_ukf[Ni].to_pickle(os.path.join(dir_res, "hd-ukf", f"df_{Ni}.pkl"))
        else:
            j_rmse_hd_ukf[:, Ni] = value_filter_not_run
        if "s-ukf" in filters_to_run:
            j_rmse_s_ukf[:, Ni] = utils_colA.compute_performance_index_rmse(x_post_s_ukf, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_s_ukf[:, Ni] = np.mean(x_post_s_ukf - x_true, axis = 1)
            
            df_s_ukf[Ni] = utils_colA.states_one_sim_to_df(x_post_s_ukf, dim_t, func_plt)
            df_s_ukf[Ni]["Ni"] = int(Ni)
            df_s_ukf[Ni]["time_sim"] = time_sim_s_ukf[Ni]
            df_s_ukf[Ni].to_pickle(os.path.join(dir_res, "s-ukf", f"df_{Ni}.pkl"))
        else:
            j_rmse_s_ukf[:, Ni] = value_filter_not_run
        if "ekf" in filters_to_run:
            j_rmse_ekf[:, Ni] = utils_colA.compute_performance_index_rmse(x_post_ekf, 
                                                                             x_ol, 
                                                                             x_true, cost_func = cost_func_type)
            j_mean_ekf[:, Ni] = np.mean(x_post_ekf - x_true, axis = 1)
            
            df_ekf[Ni] = utils_colA.states_one_sim_to_df(x_post_ekf, dim_t, func_plt)
            df_ekf[Ni]["Ni"] = int(Ni)
            df_ekf[Ni]["time_sim"] = time_sim_ekf[Ni]
            
            df_ekf[Ni].to_pickle(os.path.join(dir_res, "ekf", f"df_{Ni}.pkl"))
        else:
            j_rmse_ekf[:, Ni] = value_filter_not_run
       
        Ni += 1
        rand_seed += 1
        time_iteration = time.time() - t_iter
        if (Ni%1 == 0): #print every Xth iteration                                                               
            print(f"Iter {Ni}/{N_sim} done. t_iter = {time.time()-t_iter_N_sim: .1f} s and t_tot = {(time.time()-ts)/60: .1f} min")
    except KeyboardInterrupt as user_error:
        raise user_error
    except BaseException as e: #this is literally taking all possible exceptions
        print(e)
        raise e
        num_exceptions += 1
        rand_seed += 1
        print(f"Iter: {i}: Time spent, t_iter = {time.time()-ti: .2f} s ")

        continue
#%% Dataframes all sim       
df_true = pd.concat(df_true, ignore_index = True)
    

#%% Plot x, x_pred, y at time ti
ylabels = [ r"$x_{LNK}$ [-]", r"$x_{LK}$ [-]", r"$x_{HK}$ [-]", r"$x_{HNK}$ [-]"]
ylabels = [ r"$X^{LNK}$ [-]", r"$X^{LK}$ [-]", r"$X^{HK}$ [-]", r"$X^{HNK}$ [-]"]
ylabels_all = ylabels.copy()
ylabels_all.append(r"M [kmol]")
components = ["LNK", "LK", "HK", "HNK"]
print(f"Repeated {N_sim} time(s).")

plot_it = True
if plot_it:
    ti = 299
    alpha_fill = .2
    kwargs_pred = {}
    kwargs_hd_ukf = {"alpha": alpha_fill, "linestyle": "--"}
    kwargs_ekf = {"alpha": alpha_fill, "linestyle": "-."}
    
    # meas_idx = np.array([])
    # idx_y = 0
    filters_to_plot = [
        "hd_ukf",
        # "ekf"
        # "ol"
        ]
    x_idx = np.arange(NT)
    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    figsize = (12,9)
    figsize = (8,6)
    fig1, ax1 = plt.subplots(NC + 1, 1, sharex = True, layout = "constrained", figsize = figsize)
    fig2, ax2 = plt.subplots(NC, 1, sharex = True, layout = "constrained", figsize = figsize)
    vap_frac = func_x2y(x_true[:, ti]).toarray()
    vap_frac = np.vstack((vap_frac, (1. - vap_frac.sum(axis = 0))[np.newaxis, :]))
    
    plt_std_dev = True #plots 1 and 2 std dev around mean with shading
    for i in range(NC+1): #plot true states and ukf's estimates
        #plot true state
        ax1[i].plot(x_idx, x_true_plt[ti, i, :], label = r"$x_{true}$")
        
        if i < NC:
            ax2[i].plot(x_idx, vap_frac[i, :], label = r"$x_{true}$")
        # ax1[i].plot(t, x_true[i, :], label = r"$x_{true}$", color = 'b')
    
        #plot state predictions
        if "ekf" in filters_to_plot:
            # x_est_ekf = func_plt(x_post_ekf[:, ti])[0].toarray()
            x_est_ekf, x_M_ekf = func_plt(x_post_ekf[:, ti])
            x_est_ekf = np.vstack((x_est_ekf, x_M_ekf.toarray().flatten()))
            l_ekf = ax1[i].plot(x_idx, x_est_ekf[i, :], label = r"ekf", **kwargs_pred)
            
        if "hd_ukf" in filters_to_plot:
            # x_est_hd_ukf = func_plt(x_post_hd_ukf[:, ti])[0].toarray()
            x_est_hd_ukf, x_M_hd_ukf = func_plt(x_post_hd_ukf[:, ti])
            x_est_hd_ukf = np.vstack((x_est_hd_ukf, x_M_hd_ukf.toarray().flatten()))
            l_hd_ukf = ax1[i].plot(x_idx, x_est_hd_ukf[i, :], label = r"$\hat{x}^+_{HD-UKF}$", **kwargs_pred)
        
        if plt_std_dev: #plot shading around mean trajectory
            if i < (NC-1):
                if "ekf" in filters_to_plot:
                    kwargs_ekf.update({"color": l_ekf[0].get_color()})
                    P_x_est_ekf = func_plt(P_diag_post_ekf[:, ti])[0].toarray()
                    ax1[i].fill_between(x_idx, 
                                        x_est_ekf[i, :] + 2*np.sqrt(P_x_est_ekf[i,:]),
                                        x_est_ekf[i, :] - 2*np.sqrt(P_x_est_ekf[i,:]),
                                        **kwargs_ekf)
                    ax1[i].fill_between(x_idx, 
                                        x_est_ekf[i, :] + 1*np.sqrt(P_x_est_ekf[i,:]),
                                        x_est_ekf[i, :] - 1*np.sqrt(P_x_est_ekf[i,:]),
                                        **kwargs_ekf)
               
                
                #HD-ukf
                if "hd_ukf" in filters_to_plot:
                    kwargs_hd_ukf.update({"color": l_hd_ukf[0].get_color()})
                    P_x_est_hd_ukf = func_plt(P_diag_post_hd_ukf[:, ti])[0].toarray()
                    
                    ax1[i].fill_between(x_idx, 
                                        x_est_hd_ukf[i, :] + 2*np.sqrt(P_x_est_hd_ukf[i,:]),
                                        x_est_hd_ukf[i, :] - 2*np.sqrt(P_x_est_hd_ukf[i,:]),
                                        **kwargs_hd_ukf)
                    ax1[i].fill_between(x_idx, 
                                        x_est_hd_ukf[i, :] + 1*np.sqrt(P_x_est_hd_ukf[i,:]),
                                        x_est_hd_ukf[i, :] - 1*np.sqrt(P_x_est_hd_ukf[i,:]),
                                        **kwargs_hd_ukf)
        
        ylim_scaled = ax1[i].get_ylim()
        
        if "ol" in filters_to_plot:
            ax1[i].plot(x_idx, x_ol[i, :], label = "OL", **kwargs_pred)
        ax1[i].set_ylabel(ylabels_all[i])
        ax1[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    ax1[-1].set_ylim((.4,.6))
    ax1[-1].set_xlabel("Tray number")
    ax1[0].legend(ncol = 2, frameon = False) 
  
    ylim = ax1[1].get_ylim()
    ax1[1].set_ylim(ylim[0], 1.05)
    
#%% Plot x, x_pred, y with scatter at time ti

plot_it = True
if plot_it:
    ti = 299
    alpha_fill = .2
    kwargs_pred = {}
    kwargs_hd_ukf = {"alpha": alpha_fill, "linestyle": "--"}
    kwargs_ekf = {"alpha": alpha_fill, "linestyle": "-."}
    
    # meas_idx = np.array([])
    # idx_y = 0
    filters_to_plot = [
        "hd_ukf",
        # "ekf"
        # "ol"
        ]
    x_idx = np.arange(NT)

    font = {'size': 16}
    matplotlib.rc('font', **font)
    figsize = (12,9)
    figsize = (8,6)
    figs1, axs1 = plt.subplots(NC + 1, 1, sharex = True, layout = "constrained", figsize = figsize)
    figs2, axs2 = plt.subplots(NC, 1, sharex = True, layout = "constrained", figsize = figsize)
    vap_frac = func_x2y(x_true[:, ti]).toarray()
    vap_frac = np.vstack((vap_frac, (1. - vap_frac.sum(axis = 0))[np.newaxis, :]))
    
    plt_std_dev = False #plots 1 and 2 std dev around mean with shading
    for i in range(NC+1): #plot true states and ukf's estimates
        #plot true state
        axs1[i].scatter(x_idx, x_true_plt[ti, i, :], label = r"$x_{true}$")
        
        if i < NC:
            axs2[i].scatter(x_idx, vap_frac[i, :], label = r"$x_{true}$")
        # axs1[i].plot(t, x_true[i, :], label = r"$x_{true}$", color = 'b')
    
        #plot state predictions
        if "ekf" in filters_to_plot:
            # x_est_ekf = func_plt(x_post_ekf[:, ti])[0].toarray()
            x_est_ekf, x_M_ekf = func_plt(x_post_ekf[:, ti])
            x_est_ekf = np.vstack((x_est_ekf, x_M_ekf.toarray().flatten()))
            l_ekf = axs1[i].scatter(x_idx, x_est_ekf[i, :], label = r"ekf", **kwargs_pred)
            
        if "hd_ukf" in filters_to_plot:
            # x_est_hd_ukf = func_plt(x_post_hd_ukf[:, ti])[0].toarray()
            x_est_hd_ukf, x_M_hd_ukf = func_plt(x_post_hd_ukf[:, ti])
            x_est_hd_ukf = np.vstack((x_est_hd_ukf, x_M_hd_ukf.toarray().flatten()))
            if i < (NC-1):
                l_hd_ukf = axs1[i].errorbar(x_idx, x_est_hd_ukf[i, :], 
                                            yerr = 2*np.sqrt(P_x_est_hd_ukf[i,:]), label = r"$\hat{x}^+_{HD-UKF}$", color='#ff7f0e', fmt = "o")
            else:
                l_hd_ukf = axs1[i].scatter(x_idx, x_est_hd_ukf[i, :], 
                                            label = r"$\hat{x}^+_{HD-UKF}$", color='#ff7f0e')
                
        
        if plt_std_dev: #plot shading around mean trajectory
            if i < (NC-1):
                #HD-ukf
                if "hd_ukf" in filters_to_plot:
                    kwargs_hd_ukf.update({"color": l_hd_ukf.get_facecolor()})
                    P_x_est_hd_ukf = func_plt(P_diag_post_hd_ukf[:, ti])[0].toarray()
                    
                    axs1[i].errorbar(x_idx, 
                                        x_est_hd_ukf[i, :],
                                        yerr = 2*np.sqrt(P_x_est_hd_ukf[i,:]),
                                        **kwargs_hd_ukf)

        
        ylim_scaled = axs1[i].get_ylim()
        
        axs1[i].set_ylabel(ylabels_all[i])
        axs1[i].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    axs1[-1].set_ylim((.4,.6))
    axs1[-1].set_xlabel("Tray number")
    axs1[0].legend(ncol = 2, frameon = False) 
    

    ylim = axs1[1].get_ylim()
    axs1[1].set_ylim(ylim[0], 1.05)

    
#%% Plot traj at two times

plot_it = True
if plot_it:

    t1 = 0 
    t2 = 299
    
    
    t_idx = [t1, t2]
    alpha_fill = .2
    kwargs_pred = {}
    kwargs_hd_ukf = {"alpha": alpha_fill, "linestyle": "--"}
    kwargs_ekf = {"alpha": alpha_fill, "linestyle": "-."}
    
    # meas_idx = np.array([])
    # idx_y = 0
    filters_to_plot = [
        "hd_ukf",
        # "ekf"
        # "ol"
        ]
    x_idx = np.arange(NT)

    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    figsize = (12,9)
    fig_2traj, ax_2traj = plt.subplots(NC + 1, 2, sharex = True, sharey = True, layout = "constrained", figsize = figsize)

    plt_std_dev = True #plots 1 and 2 std dev around mean with shading
    for j in range(2):
        ti = t_idx[j]
        for i in range(NC+1): #plot true states and ukf's estimates
            #plot true state
            ax_2traj[i,j].plot(x_idx, x_true_plt[ti, i, :], label = r"$x_{true}$")
            
            if i < NC:
                ax2[i].plot(x_idx, vap_frac[i, :], label = r"$x_{true}$")
            # ax_2traj[i,j].plot(t, x_true[i, :], label = r"$x_{true}$", color = 'b')
        
            #plot state predictions
            if "ekf" in filters_to_plot:
                # x_est_ekf = func_plt(x_post_ekf[:, ti])[0].toarray()
                x_est_ekf, x_M_ekf = func_plt(x_post_ekf[:, ti])
                x_est_ekf = np.vstack((x_est_ekf, x_M_ekf.toarray().flatten()))
                l_ekf = ax_2traj[i,j].plot(x_idx, x_est_ekf[i, :], label = r"ekf", **kwargs_pred)
                
            if "hd_ukf" in filters_to_plot:
                # x_est_hd_ukf = func_plt(x_post_hd_ukf[:, ti])[0].toarray()
                x_est_hd_ukf, x_M_hd_ukf = func_plt(x_post_hd_ukf[:, ti])
                x_est_hd_ukf = np.vstack((x_est_hd_ukf, x_M_hd_ukf.toarray().flatten()))
                l_hd_ukf = ax_2traj[i,j].plot(x_idx, x_est_hd_ukf[i, :], label = r"$\hat{x}^+_{HD-UKF}$", **kwargs_pred)
            
            if plt_std_dev: #plot shading around mean trajectory
                if i < (NC-1):
                    if "ekf" in filters_to_plot:
                        kwargs_ekf.update({"color": l_ekf[0].get_color()})
                        P_x_est_ekf = func_plt(P_diag_post_ekf[:, ti])[0].toarray()
                        ax_2traj[i,j].fill_between(x_idx, 
                                            x_est_ekf[i, :] + 2*np.sqrt(P_x_est_ekf[i,:]),
                                            x_est_ekf[i, :] - 2*np.sqrt(P_x_est_ekf[i,:]),
                                            **kwargs_ekf)
                        ax_2traj[i,j].fill_between(x_idx, 
                                            x_est_ekf[i, :] + 1*np.sqrt(P_x_est_ekf[i,:]),
                                            x_est_ekf[i, :] - 1*np.sqrt(P_x_est_ekf[i,:]),
                                            **kwargs_ekf)
                   
                    
                    #HD-ukf
                    if "hd_ukf" in filters_to_plot:
                        kwargs_hd_ukf.update({"color": l_hd_ukf[0].get_color()})
                        P_x_est_hd_ukf = func_plt(P_diag_post_hd_ukf[:, ti])[0].toarray()
                        
                        ax_2traj[i,j].fill_between(x_idx, 
                                            x_est_hd_ukf[i, :] + 2*np.sqrt(P_x_est_hd_ukf[i,:]),
                                            x_est_hd_ukf[i, :] - 2*np.sqrt(P_x_est_hd_ukf[i,:]),
                                            **kwargs_hd_ukf)
                        ax_2traj[i,j].fill_between(x_idx, 
                                            x_est_hd_ukf[i, :] + 1*np.sqrt(P_x_est_hd_ukf[i,:]),
                                            x_est_hd_ukf[i, :] - 1*np.sqrt(P_x_est_hd_ukf[i,:]),
                                            **kwargs_hd_ukf)
            
            ylim_scaled = ax_2traj[i,j].get_ylim()
            
            if "ol" in filters_to_plot:
                ax_2traj[i,j].plot(x_idx, x_ol[i, :], label = "OL", **kwargs_pred)
            if j == 0:
                ax_2traj[i,j].set_ylabel(ylabels_all[i])
            # ax2[i].set_ylabel(f"y({components[i]})")
            # ax_2traj[i,j].legend(frameon = False, ncol = 3) 
        ax_2traj[-1, j].set_xlabel("Tray number")
        ax_2traj[0, 0].legend(ncol = 2, frameon = False) 

    
    #%% Plot measurements, y
    fig_y, ax_y = plt.subplots(dim_y, 1, sharex = True, layout = "constrained")
    ylabels_meas = ["TIT [K]" if i < dim_TIT else "LIT [mbar]" for i in range(dim_y)]
    
    for i in range(dim_y):
        ax_y[i].plot(t, y[i,:])
        ax_y[i].set_ylabel(ylabels_meas[i])
    ax_y[-1].set_xlabel("Time [min]")
    
    
    
    #%%Temperature profile
    ti = 50
    fig_ti, ax_ti = plt.subplots(1,1, layout = "constrained")
    # temp_traj = 
    ax_ti.plot(np.arange(NT), get_temp_profile(x_true[:, ti]).toarray(), label = "Temperature profile")
    ax_ti.scatter(idx_TIT, y[:dim_TIT, ti], label = "TIT")
    # ax_ti.plot(x_idx[::-1], y[:, ti])
    ax_ti.set_ylabel(r"T [K]")
    ax_ti.set_xlabel(r"Tray number")
    ax_ti.legend()
    
    temp_profile_true = np.hstack([get_temp_profile(x_true[:, j]).toarray() for j in range(dim_t)])

    #%% Animationof T
    from matplotlib.animation import FuncAnimation
    fig_anim2, ax_anim2 = plt.subplots(1, 1, layout = "constrained")

    plt_kwargs = [{"label": "Temp profile"}, {"linewidth": 0, "marker": "x", "markersize": 10, "label": "TIT"}]
    lines = []
    for idx in range(len(plt_kwargs)): #create line objects
        lobj = ax_anim2.plot([], [], **plt_kwargs[idx])[0]
        lines.append(lobj)
    
    
    time_template = 'time = %.1fs'
    # time_text = ax_anim2.text(300, 370, '')
    time_text = ax_anim2.text(0.05, 0.9, '', transform=ax_anim2.transAxes)
    def init2():
        ax_anim2.set_xlim((-.5, NT))
        # ax_anim2.set_ylim((y[:dim_TIT:].min()-2, y[:dim_TIT:].max()+2))
        ax_anim2.set_ylim((temp_profile_true.min()-2, 
                     temp_profile_true.max()+2))
        ax_anim2.set_xlabel("Tray number")
        ax_anim2.set_ylabel("T [K]")
        ax_anim2.legend()

        return *lines,
    
    def update2(i):
        time_text.set_text(f"{i*dt :.1f} min of {t[-1] :.1f} min")
        
        x_data = [np.arange(NT), #line 1
                 idx_TIT #line 2
                 ]
        temp_traj = get_temp_profile(x_true[:, i]).toarray().flatten()
        y_data = [temp_traj, #line 1
                  y[:dim_TIT, i] #line2
                  ]

        for idx in range(len(lines)):
            lines[idx].set_data(x_data[idx], y_data[idx])

        return *lines, time_text

    ani = FuncAnimation(fig_anim2, update2, frames= np.arange(start = 0, stop = dim_t, step = 1),
                        init_func=init2, blit=True, interval = 10)
    plt.show()
    # writergif = matplotlib.animation.PillowWriter(fps=30)
    # ani.save(os.path.join(dir_plots, "temp_profile.gif"),writer=writergif)
    
    #%% Animationof M
    from matplotlib.animation import FuncAnimation
    fig_anim3, ax_anim3 = plt.subplots(1, 1, layout = "constrained")

    
    line = ax_anim3.plot([], [], **plt_kwargs[idx])[0]
    
    M_profile_true = np.hstack([func_plt(x_true[:, j])[1].toarray() for j in range(dim_t)])
    
    time_template = 'time = %.1fs'
    # time_text = ax_anim3.text(300, 370, '')
    time_text = ax_anim3.text(0.05, 0.9, '', transform=ax_anim3.transAxes)
    def init3():
        ax_anim3.set_xlim((-.5, NT))
        # ax_anim3.set_ylim((y[:dim_TIT:].min()-2, y[:dim_TIT:].max()+2))
        ax_anim3.set_ylim((M_profile_true.min()-.2, 
                     M_profile_true.max()+.2))
        ax_anim3.set_xlabel("Tray number")
        ax_anim3.set_ylabel("M [kmol]")
        ax_anim3.legend()

        return line,
    
    def update3(i):
        time_text.set_text(f"{i*dt :.1f} min of {t[-1] :.1f} min")
        
        x_data = np.arange(NT)
        
        y_data = M_profile_true[:, i]

        line.set_data(x_data, y_data)

        return line, time_text

    ani = FuncAnimation(fig_anim3, update3, frames= np.arange(start = 0, stop = dim_t, step = 1),
                        init_func=init3, blit=True, interval = 10)
    plt.show()
    writergif = matplotlib.animation.PillowWriter(fps=30)
    # ani.save(os.path.join(dir_plots, "temp_profile.gif"),writer=writergif)
    
    #%% Composition contour plot
    labels_comp = [r"$X^{LNK}$", r"$X^{LK}$", r"$X^{HK}$", r"$X^{HNK}$"]
    fig_mi3, ax_mi3 = plt.subplots(2,2,layout = "constrained", figsize = (12,9))
    ax_mi3 = ax_mi3.flatten()
    color_lev = np.linspace(x_true_plt[:, :NC, :].min(),x_true_plt[:, :NC, :].max(),100)
    for i in range(NC):
        if ((NC_used == 2) and ((i == 0) or (i == 3))):
            continue
        # color_lev = np.arange(x_true_plt[:, i, :].min(),x_true_plt[:, i, :].max(),.01)
        cont = ax_mi3[i].contourf(t, x_idx, x_true_plt[:, i, :].T, color_lev)
        ax_mi3[i].set_ylabel("Tray number")
        ax_mi3[i].set_xlabel("Time [min]")
        ax_mi3[i].set_title(f"{labels_comp[i]}" + " [-]")
    
    cbar = fig_mi3.colorbar(cont, ax=ax_mi3.ravel().tolist())    
    cbar.ax.set_ylabel("Liquid mole fraction [-]")
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    # fig_mi3.suptitle("Composition trajectories")

    