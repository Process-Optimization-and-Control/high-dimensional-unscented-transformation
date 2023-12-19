# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:54:47 2023

@author: halvorak
"""
import numpy as np 
import scipy.linalg

class EKF_additive_noise():
    def __init__(self, x0, P0, fx, jac_f, hx, jac_h, Q, R):
        self.dim_x = x0.shape[0]
        self.dim_y = hx.size1_out(0)
        self.x_post = x0
        self.P_post = P0
        self.fx = fx
        self.fx = fx
        self.jac_f = jac_f
        self.hx = hx
        self.jac_h = jac_h
        assert Q.shape[0] == self.dim_x
        assert R.shape[0] == self.dim_y
        self.Q = Q
        self.R = R 

    def predict(self, fx=None, jac_f = None, Q = None, fx_args = [], symmetrize = True):
        self.x_prior = self.fx(self.x_post, *fx_args).toarray().flatten()
        self.F = self.jac_f(self.x_post, *fx_args).toarray()
        
        self.P_prior = self.F @ self.P_post @ self.F.T
        if Q is None: #use self.Q
            if self.Q.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + self.Q)
            else:
                self.P_prior += self.Q
        else: #use assigned Q
            if Q.ndim == 1: #1D-array interpreted as a diagonal Q-matrix (save memory cost for large dim_x)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + Q)
            else:
                self.P_prior += Q
        
        if symmetrize:
            self.P_prior = self.symmetrize_matrix(self.P_prior)
            
    def update(self, y, symmetrize = True):
        y_pred = self.hx(self.x_prior).toarray().flatten()
        self.res = y - y_pred
        
        self.H = self.jac_h(self.x_prior).toarray()
        self.Py_pred = self.H @ self.P_prior @ self.H.T #+ self.R 
        if self.R.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
            np.fill_diagonal(self.Py_pred, self.Py_pred.diagonal() + self.R)
        else:
            self.Py_pred += self.R
        
        if symmetrize:
            self.Py_pred = self.symmetrize_matrix(self.Py_pred)
        
        self.K = scipy.linalg.solve(self.Py_pred, self.H @ self.P_prior, check_finite = False, assume_a = "pos").T

        
        
        assert (self.dim_x, self.dim_y) == self.K.shape

        self.x_post = self.x_prior + (self.K @ self.res[:, np.newaxis]).flatten()
        self.P_post = (np.eye(self.dim_x) - self.K @ self.H) @ self.P_prior
        
        if symmetrize:
            self.P_post = self.symmetrize_matrix(self.P_post)
        
    def symmetrize_matrix(self, P):
        # default method
        P = .5* (P + P.T)
        return P
    
# class
        