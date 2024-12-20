# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats.qmc

def get_corr_std_dev(P, fast_calc = True):
    std_dev = np.sqrt(np.diag(P))
    if not fast_calc:
        std_dev_inv = np.diag(1/std_dev) 
        corr = std_dev_inv @ P @ std_dev_inv
    else: #do not store all the unnecessary zeros. This is an equivalent formulation to the one above
        std_dev_inv = 1/std_dev
        corr = std_dev_inv.reshape(-1,1) * P * std_dev_inv
    return std_dev, corr
    

def get_lhs_points(dist_list, N_lhs):
    """
    Return N_lhs samples from the Latin Hypercube, based on the distributinos in dist_list

    Parameters
    ----------
    dist_list : TYPE list
        DESCRIPTION. List of scipy.stats distributions
    N_lhs : TYPE int
        DESCRIPTION. Number of samples from the Latin Hypercube

    Returns
    -------
    lhs_samples : TYPE np.array((len(dist_list)), N_lhs)
        DESCRIPTION. Samples from the Latin Hypercube

    """
    assert np.ndim(dist_list) == 1
    dim_x = len(dist_list)
    
    sampler = scipy.stats.qmc.LatinHypercube(d = dim_x)
    sample = sampler.random(N_lhs) #samples are done in the CDF (cumulative distribution function)
        
    #convert to values in the state domain
    lhs_samples = np.array([dist.ppf(sample[:, i]) for (i, dist) in enumerate(dist_list)])
    assert lhs_samples.shape == (dim_x, N_lhs)
    
    return lhs_samples
    