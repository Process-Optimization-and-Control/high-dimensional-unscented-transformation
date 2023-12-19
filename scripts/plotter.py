# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:15:30 2023

@author: halvorak
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.patches
import matplotlib.transforms

from state_estimator import sigma_points_classes as spc
from state_estimator import unscented_transform as ut

font = {'size': 16}
matplotlib.rc('font', **font)

def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    assert (cov.ndim == 2) and (cov.shape[0] == cov.shape[1]) and (cov == cov.T).all() and (mean.ndim == 1) and (mean.shape[0] == 2) 
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def illustrate_ut_1d():
    """
    The UT interpreted as WSLR in 1D using y=x^2

    Returns
    -------
    None.

    """
    xm = np.array([0.])
    Px = np.array([[1.]])
    x = np.random.normal(loc=xm[0], scale = Px[0,0], size = int(5e2))
    func = lambda xi: xi**2
    y = func(x)
    
    df = pd.DataFrame(data = np.hstack((x[:,np.newaxis],y[:,np.newaxis])), columns = ["x", "y"])
    ax_sns = sns.jointplot(data = df, x = "x", y = "y", label = "MC samples")
    # ax_sns.ax_marg_y.remove()
    ax = ax_sns.ax_joint
    ax.set_xlabel(r"$x\sim \mathcal{N} (0,1)$")
    ax.set_ylabel(r"$y=x^2$")
    
    points = spc.JulierSigmaPoints(1, kappa = 3.-1)
    sigmas, Wm, Wc, _ = points.compute_sigma_points(xm, Px)
    sigmas_prop = np.array(list(map(func, sigmas.T))).T
    ym, Py, A = ut.unscented_transform_w_function_eval_wslr(sigmas, Wm, Wc, func)
    b = ym - A @ xm[:,np.newaxis]
    Py_nom = A @ Px @ A.T
    P_eps = Py - Py_nom
    
    color_ut = '#ff7f0e'
    # color_std = '#1f77b4'
    # color_norm = '#ff7f0e'
    ax.scatter(sigmas, sigmas_prop, color = color_ut, label = "Sigma-points")
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(list(xlim), [0., 0.,], color = 'k')
    ax.plot([0., 0.,], list(ylim), color = 'k')
    
    y_wslr = np.zeros(2)
    y_wslr[0] = A @ np.array(list(xlim))[0,np.newaxis] + b
    y_wslr[1] = A @ np.array(list(xlim))[1,np.newaxis] + b
    ax.plot(list(xlim), y_wslr, color = color_ut, label = r"$y^{UT}=Ax+b+\epsilon$")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    return None

# illustrate_ut_1d()

