# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:16:16 2023

@author: halvorak
"""
import numpy as np
import scipy.stats
import scipy.linalg
import matplotlib.pyplot as plt

#Self-written modules
from state_estimator import sigma_points_classes as spc
import plotter

"""
This script visualizes the following
    - The UT interpreted as WSLR in the case of y=x^2
    - Mapping from a normalized RV, x^N~(0,I) to x~(xm, Px) using x = xm + sqrt(Px)@x^n

"""
plotter.illustrate_ut_1d()

np.set_printoptions(linewidth=np.nan)

#%% Def x-dist
xm = np.array([4.,3.])
std_dev_x = np.array([4., 4.,])
corr_val = 0.4
corr = np.array([[1., corr_val],
                 [corr_val, 1.,]])
Px = np.diag(std_dev_x) @ corr @ np.diag(std_dev_x)

x_samples = np.random.multivariate_normal(xm, Px, size = int(1e5))
dim_x = Px.shape[0]

#%%Px_sqrt and SVD

eigval, eigvec = np.linalg.eig(Px)
Px_sqrt = eigvec @ np.diag(np.sqrt(eigval))


sqrt_method = [r"$U\sqrt{D}$", 
                "Chol",
                "Prin"
               ]
Px_sqrts = [eigvec @ np.diag(np.sqrt(eigval)), 
            scipy.linalg.cholesky(Px, lower = True) ,
            scipy.linalg.sqrtm(Px)
            ]
assert np.array([np.allclose(Px_sqrt @ Px_sqrt.T, Px) for Px_sqrt in Px_sqrts]).all()

#%%Plot sigmas with different sqrt(Px)

fig_sig_all, ax_sig_all = plt.subplots(1,2, layout = "constrained", figsize = (12,6))
ax_sig_norm, ax_sig_c = ax_sig_all

for label, Px_sqrt_i in zip(sqrt_method, Px_sqrts):
    points_a = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x)
    sigmas_a, Wm_a, Wc_a, _ = points_a.compute_sigma_points(xm, Px, P_sqrt = Px_sqrt_i)
    
    l = ax_sig_c.scatter(sigmas_a[0,1:], sigmas_a[1,1:], label = r"$\sqrt{P_x}=$" + label)
    
    if label == sqrt_method[0]:
        color_eigvec = l.get_facecolor()

#plot confidence ellipse
n_std = range(1,3)
for ni in n_std:
    if ni == 1:
        # plotter.confidence_ellipse(xm, Px, ax_sig, n_std = ni, edgecolor = "red", label = r"$i \sigma$")
        plotter.confidence_ellipse(xm, Px, ax_sig_c, n_std = ni, edgecolor = "red",
                                   # label = r"$i \sigma$"
                                   )
    else:
        # plotter.confidence_ellipse(xm, Px, ax_sig, n_std = ni, edgecolor = "red")
        plotter.confidence_ellipse(xm, Px, ax_sig_c, n_std = ni, edgecolor = "red")

#Plot eigenvectors
xlim = ax_sig_c.get_xlim()        
ylim = ax_sig_c.get_ylim()      

ax_sig_c.plot(xm[0] + np.array([eigvec[0,0], -eigvec[0,0]])*1e2, 
              xm[1] + np.array([eigvec[1,0], -eigvec[1,0]])*1e2, 
              color = color_eigvec,
              # label = "U"
              )
ax_sig_c.plot(xm[0] + np.array([eigvec[0,1], -eigvec[0,1]])*1e2, 
              xm[1] + np.array([eigvec[1,1], -eigvec[1,1]])*1e2, 
              color = color_eigvec)

ax_sig_c.set_xlim(xlim)  
ax_sig_c.set_ylim(ylim)  


ax_sig_c.scatter(*xm, label = "Mean")

xlim = ax_sig_c.get_xlim()
ylim = ax_sig_c.get_ylim()
ax_sig_c.plot(list(xlim), [0,0], 'k')
ax_sig_c.plot([0,0], list(ylim), 'k')
ax_sig_c.set_xlim(xlim)
ax_sig_c.set_ylim(ylim)
ax_sig_c.set_xlabel(r"$x_1$")
ax_sig_c.set_ylabel(r"$x_2$")
ax_sig_c.legend()
ax_sig_c.title.set_text(r"$x\sim (\hat{x}, P_x)$")

#%% Def x_n dist

x_nm = np.zeros(dim_x)
I = np.eye(dim_x)

points_n = spc.JulierSigmaPoints(dim_x, kappa = 3.-dim_x)
sigmas_norm, _, _, _ = points_n.compute_sigma_points(x_nm, I)

l = ax_sig_norm.scatter(*sigmas_norm, label = r"$\chi^{N}$")

#plot confidence ellipse
n_std = range(1,3)
for ni in n_std:
    if ni == 1:
        plotter.confidence_ellipse(x_nm, I, ax_sig_norm, n_std = ni, edgecolor = "red", 
                                   # label = r"$i \sigma$"
                                   )
    else:
        plotter.confidence_ellipse(x_nm, I, ax_sig_norm, n_std = ni, edgecolor = "red")
        
xlim = ax_sig_norm.get_xlim()
ylim = ax_sig_norm.get_ylim()

ax_sig_norm.plot(list(xlim), [0,0], 'k')
ax_sig_norm.plot([0,0], list(ylim), 'k')
ax_sig_norm.set_xlim(xlim)
ax_sig_norm.set_ylim(ylim)

ax_sig_norm.set_xlabel(r"$x_1^N$")
ax_sig_norm.set_ylabel(r"$x_2^N$")
ax_sig_norm.legend()
ax_sig_norm.title.set_text(r"$x^N\sim (0, I)$")

#%% Plot the linear transformation x = xm + sqrt(Px)@xn
if True:# Check that the linear transformation x=xm + P_sqrt @ x_n works
    for label, Px_sqrt_i in zip(sqrt_method, Px_sqrts):
        sig_conv = xm[:, np.newaxis] + Px_sqrt_i @ sigmas_norm
        ax_sig_c.scatter(*sig_conv, label = r"$x_{conv}: $" + label, s = 12)
    ax_sig_c.legend()


