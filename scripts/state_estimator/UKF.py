# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:17:56 2022

@author: halvorak
"""

from . import unscented_transform as ut

# from copy import deepcopy
import numpy as np
import scipy.linalg
import casadi as ca

class HD_UKF_additive_noise():
    
    def __init__(self, x0, P0, fx_map, hx_map, points_x, Q, R, P_sqrt_func = None,
                 w_mean = None, v_mean = None, name=None, func_constrain_sigmas_gen = None):
        """
        UKF using the HD-UT. .

        """
        
        # check inputs
        dim_x = x0.shape[0]
        assert x0.ndim == 1, f"x0 should be 1d array, it is {x0.ndim}"
        assert P0.ndim == 2, f"P0 should be 2d array, it is {P0.ndim}"
        assert (dim_x, dim_x) == P0.shape 
        assert (P0 == P0.T).all() #symmtrical
        
        dim_y = hx_map.size1_out(0) #this fails if hx_map is not a casadi.casadi.Function
        

        assert (dim_x == Q.shape[0]), f"{Q.shape=} and {dim_x=}"
        assert (dim_y == R.shape[0])

        
        if w_mean is None:
            w_mean = np.zeros((dim_x,))
        
        if v_mean is None:
            v_mean = np.zeros((dim_y,))
            
        if P_sqrt_func is None:
            P_sqrt_func = points_x.sqrt

        #set init values
        self.x_post = x0
        self.P_post = P0
        self.x_prior = self.x_post.copy()
        self.P_prior = self.P_post.copy()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.w_mean = w_mean
        self.Q = Q
        self.v_mean = v_mean
        self.R = R
        
        #save functions etc
        self.points_x = points_x
        self.dim_sigmas = points_x.num_sigma_points()
        self.hx_map = hx_map
        self.fx_map = fx_map
        self.P_sqrt_func = P_sqrt_func
        self.name = name  # object name, handy when printing from within class
        

        self.func_constrain_sigmas_gen = func_constrain_sigmas_gen
 
    

    def predict(self, fx=None, w_mean = None, Q = None, fx_args = [], func_constrain_sigmas_gen = None):
        r"""
        Performs the predict step of the UKF. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.
        
        
        Solves the equation
        wk = fx(x, p) - fx(x_post, E[p])
        fx(x,p) = fx(x_post, E[p]) + wk

        Parameters
        ----------

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, kwargs_sigma_points), optional
            Optional function to compute the unscented transform for the sigma
            points passed. If the points are GenUT, you can pass 3rd and 4th moment through kwargs_sigma_points (see description of sigma points class for details)

    

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if func_constrain_sigmas_gen is None:
            func_constrain_sigmas_gen = self.func_constrain_sigmas_gen

        if fx_args: #there are additinoal arguments for the map function
            for i in range(len(fx_args)):
                fx_args[i] = ca.repmat(fx_args[i], 1, self.dim_sigmas)
        # print(f"{fx_args=}")
        
        # self.x_prior, self.P_prior = ut.hdut_map_func_fast(self.x_post, self.P_post, self.fx_map, self.points_x, func_args = fx_args, calc_Pxy = False, calc_A = False, constraint_func_sig_gen = func_constrain_sigmas_gen)[:2]
        self.x_prior, self.P_prior = ut.hdut_map_func(self.x_post, self.P_sqrt_func(self.P_post), self.fx_map, self.points_x, func_args = fx_args, calc_Pxy = False, calc_A = False, constraint_func_sig_gen = func_constrain_sigmas_gen)[:2]
        
        
        #add process noise
        self.x_prior += w_mean
        if Q is None:
            if self.Q.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + self.Q)
            else:
                self.P_prior += self.Q
        else: #Q is provided
            if Q.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + Q)
            else:
                self.P_prior += Q
            del Q

    def update(self, y, R=None, v_mean = None, func_constrain_sigmas_gen = None):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        y : numpy.array of shape (dim_y)
            measurement vector

        R : numpy.array((dim_y, dim_y)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.
        v_mean : numpy.array((dim_y,)), optional
            Mean of measurement noise. If provided, it is added to self.y_pred

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. 

        hx : callable h(x, **hx_args), optional
            Measurement function. If not provided, the default
            function passed in during construction will be used.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if y is None:
            self.x_post = self.x_prior.copy()
            self.P_post = self.P_prior.copy()
            return

        if v_mean is None:
            v_mean = self.v_mean
            
        if func_constrain_sigmas_gen is None:
            func_constrain_sigmas_gen = self.func_constrain_sigmas_gen

        self.y_pred, self.Py_pred, self.Pxy = ut.hdut_map_func(self.x_prior, self.P_sqrt_func(self.P_prior), self.hx_map, self.points_x, calc_Pxy = True, calc_A = True, constraint_func_sig_gen = func_constrain_sigmas_gen)[:3]
        # self.y_pred, self.Py_pred, self.Pxy, self.H = ut.hdut_map_func_fast(self.x_prior, self.P_prior, self.hx_map, self.points_x, calc_Pxy = True, calc_A = True, constraint_func_sig_gen = func_constrain_sigmas_gen)[:4]

        
        # add measurement noise
        self.y_pred += v_mean
        if R is None:
            if self.R.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.Py_pred, self.Py_pred.diagonal() + self.R)
            else:
                self.Py_pred += self.R
        else: #R is provided
            if R.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.Py_pred, self.Py_pred.diagonal() + R)
            else:
                self.Py_pred += R
            del R


        # Innovation term of the UKF
        self.y_res = y - self.y_pred

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        # self.K = scipy.linalg.solve(self.Py_pred.T, self.Pxy.T, assume_a = "pos").T
        self.K = self.Pxy @ np.linalg.inv(self.Py_pred)
        # self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self.dim_x, self.dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ self.Py_pred @ self.K.T
        
 
        
class UKF_map_based(HD_UKF_additive_noise):
    
    def __init__(self, x0, P0, fx_map, hx_map, points_x, Q, R,
                 w_mean = None, v_mean = None, name=None, func_constrain_sigmas_gen = None):
        """
        Create a Kalman filter. IMPORTANT: Additive white noise is assumed!

        """
        super().__init__(x0, P0, fx_map, hx_map, points_x, Q, R, 
                     w_mean = w_mean, v_mean = v_mean, name = name, func_constrain_sigmas_gen = func_constrain_sigmas_gen)
        
        

    def predict(self,fx=None, w_mean = None, Q = None, fx_args = [], func_constrain_sigmas_gen = None):
        r"""
        Performs the predict step of the UKF. On return, self.x_prior and
        self.P_prior contain the predicted state (x) and covariance (P). '

     
        """
        
        
        if w_mean is None:
            w_mean = self.w_mean
        
        if func_constrain_sigmas_gen is None:
            func_constrain_sigmas_gen = self.func_constrain_sigmas_gen
            
        # calculate sigma points for given mean and covariance for the states
        sigmas_raw_fx, self.Wm_x, self.Wc_x = self.points_x.compute_sigma_points(
            self.x_post, self.P_post)[:3]

        if fx_args: #there are additinoal arguments for the map function
            for i in range(len(fx_args)):
                fx_args[i] = ca.repmat(fx_args[i], 1, self.dim_sigmas)
        
        sigmas_prop = self.fx_map(sigmas_raw_fx, *fx_args)
        sigmas_prop = np.array(sigmas_prop)
        
        self.x_prior, self.P_prior = ut.unscented_transformation(sigmas_prop, self.Wm_x, self.Wc_x)
        
        
        #add process noise
        self.x_prior += w_mean
        if Q is None:
            if self.Q.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + self.Q)
            else:
                self.P_prior += self.Q
        else: #Q is provided
            if Q.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.P_prior, self.P_prior.diagonal() + Q)
            else:
                self.P_prior += Q
            del Q
        


    def update(self, y, R=None, v_mean = None, func_constrain_sigmas_gen = None):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        y : numpy.array of shape (dim_y)
            measurement vector

        R : numpy.array((dim_y, dim_y)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.
        v_mean : numpy.array((dim_y,)), optional
            Mean of measurement noise. If provided, it is added to self.y_pred

        """
        if y is None:
            self.x_post = self.x_prior.copy()
            self.P_post = self.P_prior.copy()
            return

        if v_mean is None:
            v_mean = self.v_mean
            
        if func_constrain_sigmas_gen is None:
            func_constrain_sigmas_gen = self.func_constrain_sigmas_gen
            
            
        # recreate sigma points
        (sigmas_raw_hx,
         self.Wm, self.Wc) = self.points_x.compute_sigma_points(self.x_prior,
                                                       self.P_prior
                                                       )[:3]
        sigmas_meas = self.hx_map(sigmas_raw_hx)
        sigmas_meas = np.array(sigmas_meas)


        # compute mean and covariance of the predicted measurement
        self.y_pred, self.Py_pred = ut.unscented_transformation(sigmas_meas, self.Wm, self.Wc)

        # add measurement noise
        self.y_pred += v_mean
        if R is None:
            if self.R.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.Py_pred, self.Py_pred.diagonal() + self.R)
            else:
                self.Py_pred += self.R
        else: #R is provided
            if R.ndim == 1: #1D-array interpreted as a diagonal matrix (save memory cost for large dim)
                np.fill_diagonal(self.Py_pred, self.Py_pred.diagonal() + R)
            else:
                self.Py_pred += R
            del R

        
        # #Kalman gain. Start with cross_covariance
        self.Pxy = self.cross_covariance(sigmas_raw_hx - self.x_prior.reshape(-1,1),
                                    sigmas_meas - self.y_pred.reshape(-1,1), self.Wc)
        
        # Innovation term of the UKF
        self.y_res = y - self.y_pred

        # Kalman gain
        # solve K@Py_pred = P_xy <=> PY_pred.T @ K.T = P_xy.T
        # self.K = scipy.linalg.solve(self.Py_pred.T, self.Pxy.T, assume_a = "pos").T
        self.K = self.Pxy @ np.linalg.inv(self.Py_pred)
        # self.K = np.linalg.solve(Py_pred.T, Pxy.T).T
        # self.K = np.linalg.lstsq(Py_pred.T, Pxy.T)[0].T #also an option
        assert self.K.shape == (self.dim_x, self.dim_y)

        # calculate posterior
        self.x_post = self.x_prior + self.K @ self.y_res
        self.P_post = self.P_prior - self.K @ self.Py_pred @ self.K.T
        
    def cross_covariance(self, sigmas_x, sigmas_y, W_c):
        """
        Cross-covariance between two probability distribution x,y which are already centered around their mean values x_mean, y_mean

        Parameters
        ----------
        sigmas_x : TYPE np.array((dim_x, dim_sigmas))
            DESCRIPTION. Sigma-points created from the x-distribution, centered around x_mean
        sigmas_y : TYPE np.array((dim_y, dim_sigmas))
            DESCRIPTION. Sigma-points created from the y-distribution, centered around y_mean
        W_c : TYPE np.array(dim_sigmas,)
            DESCRIPTION. Weights to compute the covariance

        Returns
        -------
        P_xy : TYPE np.array((dim_x, dim_y))
            DESCRIPTION. Cross-covariance between x and y

        """
        try:
            (dim_x, dim_sigmas_x) = sigmas_x.shape
        except ValueError:  # sigmas_x is 1D
            sigmas_x = np.atleast_2d(sigmas_x)
            (dim_x, dim_sigmas_x) = sigmas_x.shape
            assert dim_sigmas_x == W_c.shape[0], "Dimensions are wrong"
        try:
            (dim_y, dim_sigmas_y) = sigmas_y.shape
        except ValueError:  # sigmas_y is 1D
            sigmas_y = np.atleast_2d(sigmas_y)
            (dim_y, dim_sigmas_y) = sigmas_y.shape
            assert dim_sigmas_y == dim_sigmas_x, "Dimensions are wrong"
        
        #NB: could/should be changed to matrix product
        #Calculate cross-covariance -
        P_xy = sum([Wc_i*np.outer(sig_x,sig_y) for Wc_i, sig_x, sig_y in zip(W_c, sigmas_x.T, sigmas_y.T)])
        assert (dim_x, dim_y) == P_xy.shape
        return P_xy
        