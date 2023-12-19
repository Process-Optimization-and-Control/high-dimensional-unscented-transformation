# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:50:10 2023

@author: halvorak
"""

import numpy as np
import matplotlib.pyplot as plt
from state_estimator import sigma_points_classes as spc

#%%find alpha in SUT which makes W^((0))>=0
wc0_f = lambda a: 3*a**2*(4-a**2)-1
av = np.linspace(1e-4, 1, num = int(1e2))
wc0 = []
for ai in av:
    points_dummy = spc.ScaledSigmaPoints(1, beta = 2., kappa = 3.-1, alpha = ai)
    sig, wm, wc, _ = points_dummy.compute_sigma_points(np.array([1.]), np.eye(1))
    
    #uncomment to test other input variables for the SUT
    # points_dummy = spc.ScaledSigmaPoints(100, beta = 2., kappa = 1e-7, alpha = ai, suppress_init_warning = True)
    # sig, wm, wc, _ = points_dummy.compute_sigma_points(np.zeros(100), np.eye(100))
    wc0.append(wc[0])


plt.plot(av, wc0)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$W_c^{(0)}$")
plt.ylim((-1,1))
plt.title("SUT")