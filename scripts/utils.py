# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np

def get_corr_std_dev(P, fast_calc = True):
    std_dev = np.sqrt(np.diag(P))
    if not fast_calc:
        std_dev_inv = np.diag(1/std_dev) 
        corr = std_dev_inv @ P @ std_dev_inv
    else: #do not store all the unnecessary zeros. This is an equivalent formulation to the one above
        std_dev_inv = 1/std_dev
        corr = std_dev_inv.reshape(-1,1) * P * std_dev_inv
    return std_dev, corr
    