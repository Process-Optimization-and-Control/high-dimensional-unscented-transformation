# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:27:29 2023

@author: halvorak
"""

import numpy as np
import casadi as ca
import scipy.stats
import pandas as pd
import os
import shutil

def get_initial_conditions(NT, NC, NC_used):
    t = [0, 20000]
    
    #initial conditions for the states
    X = 0.5*np.ones(4*NT)
    dim_x = X.shape[0]
    #initial covariance estimate
    P0 = np.diag(np.ones(dim_x)*5e-4)
    # X = np.arange(1, 4*41+1)
    
    # Inputs and disturbances
    LT = 2.70629                          # Reflux
    VB = 3.20629                          # Boilup
    F = 1.0 + 0.00                        # Feedrate
    qF = 1.0                              # Feed liquid fraction
    
    if NC_used == 2:
        zF =  np.array([0, 0.5, 0.5, 0])  # Feed compositions for component A to component NC 
    else:
        zF =  np.array([0.05, 0.35, 0.35, 0.25])  # Feed compositions for component A to component NC 
    assert zF.sum() == 1., f"{zF.sum()=}, it should be 1. {zF=}"
    dim_zF = zF.shape[0]
    #zF = [0.05 0.45 0.45 0.05]  # Feed compositions for component A to component NC 
    #zF = [0.25 0.25 0.25 0.25]  # Feed compositions for component A to component NC 
    
    
    # P-Controllers for control of reboiler and condenser hold up.
    KcB = 10
    KcD = 10                     # controller gains
    MDs = 0.5 
    MBs = 0.5                    # Nominal holdups - these are rather small  
    Ds = 0.5 
    Bs = 0.5                      # Nominal flows
    MB = X[(NC-1)*NT] 
    MD = X[NC*NT-1]    # Actual reboiler and condenser holdup
    D = Ds + (MD-MDs)*KcD                   # Distillate flow
    B = Bs + (MB-MBs)*KcB                   # Bottoms flow     
    
    # Store all inputs and disturbances
    
    U = np.zeros(6)
    U[0] = LT 
    U[1] = VB 
    U[2] = D 
    U[3] = B 
    U[4] = F 
    U[5] = qF 
    A = zF 
    U = np.hstack([U, A])
    
    u = np.array([LT, VB, D, B])
    d = np.array([F, qF])
    d = np.hstack((d, A))
    return t, X, P0, u, d
    # return t,X,U

def P_controller(x, func_plt, LT):
    x_frac, M = func_plt(x) #mole fraction and liquid hold-up
    M = M.toarray().flatten()
    # P-Controllers for control of reboiler and condenser hold up.
    
    # if ti > 10:
    #     LT = 3.                          # Reflux
    # else:
    #     LT = 2.70629                          # Reflux
    VB = LT + 0.4                         # Boilup
    
    KcB = 10 # controller gains
    KcD = 10                     # controller gains
    MDs = 0.5 
    MBs = 0.5                    # Nominal holdups - these are rather small  
    
    Ds = 0.4 
    Bs = 0.6                      # Nominal flows
    
    MB = M[0] #Reboiler hold-up
    MD = M[-1] # Distillate hold-up

    D = Ds + (MD-MDs)*KcD                   # Distillate flow
    B = Bs + (MB-MBs)*KcB          #bottoms flow
    
    u = np.array([LT, VB, D, B])
    return u    


def multicomA_ca(dt, NT, NF, NC, alpha, T_bp, idx_TIT, MW, radius_col, dvapH, integrator_casadi = "cvodes", opts_integrator = {}, dim_zF = 4, symvar1 = ca.MX.sym, symvar2 = ca.MX.sym):
    """
    CasADi model. Info about the model is given below. This is originally a Matlab-model, converted to Python. Original model is found at Sigurd Skogestad's homepage, https://folk.ntnu.no/skoge/. Documentation below is for the Matlab model.
    
    *************************************************************************************
    ***************** Program written by Stathis Skouras at March, 2001. *****************  
    *************************************************************************************
    
    multicomA- This is a nonlinear model of a continuous distillation column with
                NT-1 theoretical stages including a reboiler (stage 1) plus a
                total condenser ("stage" NT). 
                Model assumptions: 
                NC components (multicomponent mixture); Component NC is the heavy component
                constant relative volatilities; 
                no vapor holdup; 
                one feed and two products;
                constant molar flows (same vapor flow on all stages); 
                Liquid flow dynamics modelled by Franci's Weir Formula.
                total condenser.  
    
                The model is based on column A in Skogestad and Postlethwaite
                (1996). The model has NC*NT states.
    
    Inputs:    t    - time in [min].
                X    - State, the first NT states are compositions of light
                      component A, the next NT states are compositions of component B, etc,
                      compositions of the heavy component are not reported directly  
                      reboiler/bottom stage is stage (1) and condenser is stage (41). 
                      The last NT states are liquid holdups in each stage. 
                U(1) - reflux L,
                U(2) - boilup V,
                U(3) - top or distillate product flow D,
                U(4) - bottom product flow B,
                U(5) - feed rate F,
                U(6) - feed liquid fraction, qF.
                A   - feed compositions, zF.
            U=[U A] - Disturbances vector
    Outputs:   xprime - vector with time derivative of all the states 
    
    ########################################################
    
    ------------------------------------------------------------
    """
    
    
    # Splitting the states
    dim_x_sym = (NC-1)*NT
    x_sym = symvar1("x_sym", dim_x_sym)              # Liquid compositions from btm to top
    M = symvar1("M", NT)                # Liquid hold up from btm to top [kmol]
    X = ca.vertcat(x_sym, M) #state vector
    dim_x = X.shape[0]
    
    # Inputs and disturbances
    # Inputs
    LT = symvar1("LT", 1)                             # Reflux
    VB = symvar1("VB", 1)                             # Boilup
    D = symvar1("D", 1)                             # Distillate
    B = symvar1("B", 1)                             # Bottoms
    u = ca.vertcat(LT, VB, D, B)
    # Disturbances
    F = symvar1("F", 1)                              # Feedrate
    qF = symvar1("qF", 1)                              # Feed liquid fraction
    zF = symvar1("zF", dim_zF)                    # Feed compositions
    d = ca.vertcat(F, qF, zF) #disturbances

    
    # Rearrange elements of composition vector (x) for later use
    Iu  =  np.arange(NT).reshape(-1,1) @ np.ones((1, NC-1), dtype = int)
    Iu += NT * np.ones((NT, 1), dtype = int) @ np.arange(0, NC-2 + 1).reshape(1,-1) 
    x = x_sym[Iu].T

    x_comp = ca.horzcat(x.T, ca.DM.ones(NT, 1) - x.T @ ca.DM.ones(NC-1, 1))
    x_plt = x_comp.T
    
    # THE MODEL
    # Vapour-liquid equilibria (multicomponent ideal VLE, Stichlmair-Fair, 'Distillation', p. 36, 1998)
    y = (alpha[:NC-1, np.newaxis] @ np.ones((1,NT)) * x) / (np.ones((NC-1, 1)) @ (1 + (alpha[np.newaxis, :NC-1] - 1) @ x))
    
    #y_comp only used to calculate temperature profile in the column
    y_comp = ca.horzcat(y.T, ca.DM.ones(NT, 1) - y.T @ ca.DM.ones(NC-1, 1))
    
    # Vapor Flows assuming constant molar flows
    i = np.arange(NT-1)    
    V = VB*np.ones(NT-1)
    i = np.arange(NF-1, NT-1) 
    V[i] += (1-qF)*F
    
    
    # Liquid flows are given by Franci's Weir Formula L(i) = K*Mow(i)^1.5 
    # Liquid flow L(i) dependent only on the holdup over the weir Mow(i)
    # M(i) =  Mow(i) + Muw(i) (Total holdup  =  holdup over weir + holdup below weir) 
    
    Kuf = 21.65032                                # Constant above feed
    Kbf = 29.65032                                # Constant below feed
    Muw = 0.25                                    # Liquid holdup under weir (Kmol)  
    
    
    L = ca.MX.zeros(NT)
    i = np.arange(1, NF)
    L[i] =  Kbf * (M[i] - Muw)**1.5      # Liquid flows below feed (Kmol/min)    
    i = np.arange(NF, NT-1)
    L[i] = Kuf * ((M[i] - Muw)**1.5)      # Liquid flows above feed (Kmol/min) 
    L[-1] = LT                                   # Condenser's liquid flow (Kmol/min)  
    
    
    # Time derivatives from material balances for 
    # 1) total holdup and 2) component holdup
    
    # Column
    j = np.arange(2 - 1, NT-1)
    dMdt = ca.MX.zeros(NT)
    dMdt[j] = L[j+1] - L[j] + V[j-1] - V[j]
         
    dMxdt = ca.MX.zeros(NC-1, NT)
    for i in range(NC-1):
        for j in range(1, NT-1):
            dMxdt[i,j] = L[j+1]*x[i,j+1] - L[j]*x[i,j] + V[j-1]*y[i,j-1] - V[j]*y[i,j]
    
    
    # Correction for feed at the feed stage
    # The feed is assumed to be mixed into the feed stage
    
    dMdt[NF-1] = dMdt[NF-1] + F
    
    dMxdt[:,NF-1] = dMxdt[:,NF-1] + F * zF[:NC-1]
    
    # Reboiler (assumed to be an equilibrium stage)
    dMdt[0] = L[1] - V[0] - B
    
    i = np.arange(NC-1)
    dMxdt[i,0] =  L[1]*x[i,1] - V[0]*y[i,0] - B*x[i,0]
    
    # Total condenser [no equilibrium stage]
    dMdt[NT-1]  =  V[NT-1-1]           - LT         - D
    
    i = np.arange(NC-1)
    dMxdt[i,NT-1] =  V[NT-1-1]*y[i,NT-1-1] - LT*x[i,NT-1] - D*x[i,NT-1]
    
    # Compute the derivative for the mole fractions from d(Mx)  =  x dM + M dx
    dxdt = (dMxdt - x * (np.ones((NC-1,1)) @ dMdt.T)) / (np.ones((NC-1,1))  @ M.T)
    
    # Rearrange elements of composition vector (dxdt) for later use
    dxdt = dxdt.T.reshape((dim_x_sym, 1))

    # Output
    xprime = ca.vertcat(dxdt, dMdt)    
    
    #Create CasADi functions and integrators
    ode = {"x": X, "u": u, "p": d, "ode": xprime}
    integrator = ca.integrator("integrator", integrator_casadi, ode, 0., dt, opts_integrator)
    
    x_next = integrator(x0 = X, u = u, p = d)["xf"]
    fx = ca.Function("fx", [X, u, d], [x_next], ["x", "u", "d"], ["x_next"])
    
    #Jacobian df/dx for EKF
    jac = ca.jacobian(x_next, X)
    fx_jac_func = ca.Function("jac_func", [X, u, d], [jac])
    
    #Measurements:
    # Temperature profile along the column    
    T_col = ((y_comp + x_comp)/2.) @ T_bp[:, np.newaxis]
    #select which tempeatures should be used as measurements (TIT)
    dim_TIT = len(idx_TIT)
    selector_TIT = np.zeros((dim_TIT, NT))
    for i in range(dim_TIT):
        selector_TIT[i, idx_TIT[i]] = 1
    TIT = selector_TIT @ T_col
    
    #Level transmitter. Measures the dP [mbar] according to p=rho*g*h=rho*g*V/A=g/A*m=g/A*M*MW_ave, where A is cross-section of column, MW_ave average molecular weight and M [kmol]
    M_measured = M[[0, -1]] #[kmol] - liquid hold up in reboiler and condenser
    MW_ave = x_comp[[0, -1],:] @ MW #[g/mol]
    A = np.pi*radius_col**2 # [m^2]
    LIT = (9.81/A)*M_measured*MW_ave #[kg/s^2] = [Pa]
    LIT = LIT/100 #[mbar]
    dim_LIT = LIT.shape[0]
    
    #Measurement function h(x)
    y_meas = ca.vertcat(TIT, LIT)
    hx = ca.Function("hx", [X], [y_meas])
    hx_jac = ca.jacobian(hx(X), X)
    hx_jac_func = ca.Function("hx_jac_func", [X], [hx_jac])
    
    get_temp_profile = ca.Function("get_temp_profile", [X], [T_col])
    
    #plotting functions
    F_set_X = ca.Function("F_set_X", [x_sym, M], [X])
    func_plt = ca.Function("func_plt", [X], [x_plt, M], ["X"], ["x_plt", "M"])
    
    func_x2y = ca.Function("func_x2y", [X], [y])

    
    F_dot = ca.Function("F_dot", [X, u, d], [xprime], ["X", "u", "d"], ["xprime"])
    
    return (F_dot, integrator, x_sym, M, u, d, F_set_X, fx, fx_jac_func, hx, hx_jac_func, func_plt, func_x2y, x_next,
 get_temp_profile, dim_TIT, dim_LIT)



def least_squares_clipping(sig, func_plt, solver = "qpoases", opts_qp = {}):
    #Kolås 2009: constrained nonlinear state estimation based on the UKF approach. Implement methods in section 5.1.3 of the paper.
    
    dim_x = sig.shape[0]
    
    #parameters in the optimization problem
    sig_unc = ca.MX.sym("sig_unc", dim_x) #unconstrained siga-point

    W = ca.MX.eye(dim_x)
    
    
    J = (sig - sig_unc).T @ W @ (sig - sig_unc) #cost function
    
    #create constraints
    #split the states in composition part and hold-up part
    sig_comp, sig_M = func_plt(sig) 
    
    NT = sig_M.shape[0] #number of trays
    NC = sig_comp.shape[0] #number of components
    #Note: sig_comp[-1,i] = 1 - sig_comp[:-1, i].sum(). The decision variables are in sig_comp[:-1,i], and the last row gives the total balance (linear equation) 
    assert sig_comp.shape == (4, sig_M.shape[0])
    
    #define linear constraints
    g = []
    for i in range(NC):
        g = ca.vertcat(g, sig_comp[i,:].T)
    g = ca.vertcat(g, sig_M)
    
    #values for the constraint
    g_lb_comp = np.zeros((NC)*NT)
    g_ub_comp = np.ones((NC)*NT)
    
    g_lb_M = np.zeros(NT)
    g_ub_M = np.ones(NT)*np.inf
    
    g_lb = np.hstack((g_lb_comp, g_lb_M))
    g_ub = np.hstack((g_ub_comp, g_ub_M))

    #since cost function is quadratic and constraints are linear we have a QP
    qp = {"x": sig, "f": J, "g": g, "p": sig_unc}
    if solver == "qpoases":
        S_qp = ca.qpsol("S_qp", "qpoases", qp, opts_qp)
    elif solver == "ipopt":
        S_qp = ca.nlpsol("S_qp", "ipopt", qp, opts_qp)
    else:
        raise KeyError(f"{solver=} is not implemented")
    return S_qp, g_lb, g_ub
    

def get_positive_point(dist, eps = 1e-8, N = 1):
    """
    Sample a point from the distribution dist, and requires all points to be positive

    Returns
    -------
    point : TYPE np.array((dim_x,))
        DESCRIPTION. Points with all positive values

    """
    dim_x = dist.rvs(size = 1).shape[0]
    points = np.zeros((N, dim_x))
    
    sampled_points = dist.rvs(size = 10*N)
    k = 0
    for i in range(sampled_points.shape[0]):
        if (sampled_points[i,:] > eps).all(): #accept the point
            points[k, :] = sampled_points[i,:]
            k += 1
            if k >= N:
                break
    assert k >= N, "Did not find enough points above the constraint"
    assert (points > eps).all(), "Not all points are above the constraint"
    
    if N == 1:
        points = points.flatten()
    return points
    

def get_corr_std_dev(P):
    std_dev = np.sqrt(np.diag(P))
    std_dev_inv = np.diag(1/std_dev)
    corr = std_dev_inv @ P @ std_dev_inv
    return std_dev, corr
    
def compute_performance_index_rmse(x_kf, x_ol, x_true, cost_func = "RMSE"):
    if cost_func == "RMSE":
        # J = np.linalg.norm(x_kf - x_true, axis = 1, ord = 2)
        J = np.sqrt(np.square(x_kf - x_true).mean(axis = 1))
    elif cost_func == "ME":
        J = (x_kf - x_true).mean(axis = 1)
    elif cost_func == "valappil": #valappil's cost index
        J = np.divide(np.linalg.norm(x_kf - x_true, axis = 1, ord = 2),
                      np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    else:
        raise ValueError("cost function is wrongly specified. Must be RMSE or valappil.")
    return J

def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def states_one_sim_to_df(x, dim_t, func_plt):
    df = [[] for i in range(dim_t)]
    x_comp_i, M_i = func_plt(x[:, 0])
    NT = M_i.shape[0]
    col_names = [f"x_{i+1}" for i in range(x_comp_i.shape[0])]
    col_names.extend(["M", "Tray"])
    for ti in range(dim_t):
        x_comp_i, M_i = func_plt(x[:, ti])
        df[ti] = pd.DataFrame(data = np.hstack((x_comp_i.T, M_i, np.arange(NT, dtype = int)[:, np.newaxis])), columns = col_names)
        df[ti]["ti"] = ti
    df = pd.concat(df, ignore_index = True)
    df["Tray"] = pd.to_numeric(df["Tray"], downcast = "integer")
    return df

def read_df_in_dir(dir_res, concat = True, N_sim_end = None, ti_max = None):
    df = []
    i = 0
    while True:
        try:
            df.append(pd.read_pickle(os.path.join(dir_res, f"df_{i}.pkl")))
            if ti_max is not None:
                df[i] = df[i][df[i]["ti"] <= ti_max]
            i += 1
        except FileNotFoundError:
            break #we are out of files
    if N_sim_end is not None: #specify which simulations should be returned
        df = df[:N_sim_end]
    if concat:
        df = pd.concat(df, ignore_index = True)
    
    return df

def read_df_and_compute_rmse(dir_se, dir_true, col_states, cost_func_type = "RMSE", N_sim_end = None, ti_max = None):
    df_se_list = read_df_in_dir(dir_se, N_sim_end = N_sim_end, concat = False, ti_max = ti_max)
    df_true_list = read_df_in_dir(dir_true, N_sim_end = N_sim_end, concat = False, ti_max = ti_max)
    df_rmse = [[] for i in range(len(df_se_list))]
    
    for i in range(len(df_se_list)): #each simulation
        trays = df_se_list[i]["Tray"].unique()
        df_rmse[i] = pd.DataFrame(index = trays, columns= col_states, data = 0.)
        for ti in range(len(trays)): #each tray
            for si in col_states: #each component
                df_se_i = df_se_list[i].loc[df_se_list[i]["Tray"] == ti, si]
                df_true_i = df_true_list[i].loc[df_true_list[i]["Tray"] == ti, si] 
                df_rmse[i].loc[ti, si] = compute_performance_index_rmse(df_se_i.to_numpy()[np.newaxis,:], None, df_true_i.to_numpy()[np.newaxis,:], cost_func = cost_func_type)

        #add simulation number info
        assert df_se_list[i]["Ni"].unique().shape[0] == 1, "There should be data from one simulation in this df"
        df_rmse[i]["Ni"] = df_se_list[i]["Ni"][0]
        df_rmse[i]["Tray"] = trays
    
    df_rmse = pd.concat(df_rmse, ignore_index = True)
    return df_rmse

def is_within_bounds(x, func_plt, x_lb, x_ub, eps = 1e-10):
    
    if not ((x >= x_lb).all() and (x <= x_ub).all()):
        return False
    #be sure that mole fractions are correct
    x_comp = func_plt(x)[0]
    x_comp = x_comp.toarray()
    if not (np.allclose(x_comp.sum(axis = 0), 1.) and (x_comp >= -eps).all() and (x_comp <= 1.+eps).all()):
        return False
    return True

