# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:54:11 2021

@author: halvorak
"""

import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
# import colorcet as cc
# import pathlib
# import os
import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms


class SigmaPoints():
    """
    Parent class when sigma points algorithms are constructed. All points tru to estimate mean and covariance of Y, where Y=f(X)
    """
    def __init__(self, n, sqrt_method = scipy.linalg.sqrtm):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky

        Returns
        -------
        None.

        """
        self.n = n
        
        #principal matrix square root or Cholesky factorization (only lower triangular factorization) is supported. Default for np.linalg.cholesky is lower factorization, default for scipy.linalg.cholesky is upper factorization
        if sqrt_method is scipy.linalg.cholesky:
            sqrt_method = lambda P: scipy.linalg.cholesky(P, lower = True)
        self.sqrt = sqrt_method
        
    def num_sigma_points(self):
        """
        Returns the number of sigma points. Most algorithms return (2n+1) points, can be overwritten by child class

        Returns
        -------
        TYPE int
            DESCRIPTION. dim_sigma, number of sigma points

        """
        return 2*self.n + 1
    
    def is_matrix_pos_def(self, a_matrix):
        """
        Checks if a matrix is positive definite by checking if all eigenvalues are positive

        Parameters
        ----------
        a_matrix : TYPE np.array((n,n))
            DESCRIPTION. A matrix

        Returns
        -------
        TYPE bool
            DESCRIPTION. True if the matrix is pos def, else False

        """
        return np.all(np.linalg.eigvals(a_matrix) > 0)
    

    
class JulierSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Julier's original paper. It assumes that the distribtions are symmetrical.
    
    @TECHREPORT{Julier96ageneral,
    author = {Simon Julier and Jeffrey K. Uhlmann},
    title = {A General Method for Approximating Nonlinear Transformations of Probability Distributions},
    institution = {},
    year = {1996}
}
    
    """
    def __init__(self, n, kappa = 0., sqrt_method = scipy.linalg.sqrtm, suppress_kappa_warning = False, kurtosis = 3.):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        # print("WARNING: This class has NOT been verified yet")
        # raise ValueError("This class has NOT been verified yet!")
        if not (kappa == np.max([(3-n), 0])):
            if not suppress_kappa_warning:
                print(f"warning: kappa is not set to kappa = max([(3-n),0]) = max([{3-n},0]), which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.kappa = kappa
        self.dim_sigma = self.num_sigma_points()
        self.Wm = self.compute_weights()
        self.Wc = self.Wm.copy()
        self.kurtosis = kurtosis
        
        self.type = "JulierSigmaPoints" #useful when checking types
        
    # def compute_weights(self)
    def compute_sigma_points(self, mu, P, P_sqrt = None):
        """
        Computes the sigma points based on Julier's paper

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        P_sqrt : TYPE np.array(n,n), optional
            DESCRIPTION. default is None. If supplied, algorithm does not compute sqrt(P).

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

        """
        assert ((self.n == mu.shape[0]) and mu.ndim == 1), f"{self.n=} while {mu.shape=}."
                
        sqrt_factor = np.sqrt(self.n+self.kappa)
        self.sqrt_factor = sqrt_factor
        if P_sqrt is None:
            assert ((self.n, self.n) == P.shape), f"P.shape = {P.shape}, it must be ({self.n, self.n})"
            try:
                P_sqrt = self.sqrt(P)
            except np.linalg.LinAlgError as LinAlgError:
                print("P is not positive definite.")
                raise LinAlgError
                
        self.P_sqrt_weight = sqrt_factor*P_sqrt
        
        sigmas = mu[:, np.newaxis] + np.hstack((np.zeros((self.n, 1)), self.P_sqrt_weight, -self.P_sqrt_weight))
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
    def compute_weights(self, n = None):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        if n is None:
            n = self.n
            dim_sigma = self.dim_sigma
        else:
            assert isinstance(n, int)
            dim_sigma = 2*n + 1
        
        weights = np.array([1/(2*(n + self.kappa)) for i in range(dim_sigma)])
        weights[0] = self.kappa/(n + self.kappa)
        return weights

class ScaledSigmaPoints(SigmaPoints):
    """
    From
    JULIER, S. J. The Scaled Unscented Transformation.  Proceedings of the American Control Conference, 2002 2002 Anchorage. 4555-4559 vol.6.

    """
    
    def __init__(self, n, alpha = 1e-3, beta = 2., kappa = 0., sqrt_method = scipy.linalg.sqrtm, suppress_init_warning = False, force_Wm_sum_zero = False, kappa_func = None, kurtosis = 3.):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is scipy.linalg.sqrtm (principal matrix square root). Method to calculate the square root of a matrix. The other choice is typically np.linalg.cholesky
        alpha : TYPE, optional float
            DESCRIPTION. The default is 1e-3. Determines the scaling of the sigma-points (how far away they are from the mean.
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.alpha = alpha
        self.beta = beta
        self.kappa_func = kappa_func
        if self.kappa_func is not None:
            kappa = self.kappa_func(n)
        self.kappa = kappa
        self.kurtosis = kurtosis
        self.force_Wm_sum_zero = force_Wm_sum_zero
        self.lam = self.calculate_lam()
        if ((self.kappa != np.max([(3-n), 0])) and (suppress_init_warning == False)):
            print(f"warning: kappa is not set to kappa = max([(3-n),0]) = max([{3-n},0]), which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.dim_sigma = self.num_sigma_points()
        self.Wm, self.Wc = self.compute_weights(force_Wm_sum_zero = force_Wm_sum_zero)
        
        self.type = "ScaledSigmaPoints" #useful when checking types
    
    def calculate_lam(self):
        lam = (self.alpha**2)*(self.n + self.kappa) - self.n
        return lam
        
    def compute_weights(self, n = None, force_Wm_sum_zero = False):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        if n is None:
            n = self.n
            dim_sigma = self.dim_sigma
        else:
            assert isinstance(n, int)
            dim_sigma = 2*n + 1
        alpha = self.alpha
        beta = self.beta
        
        lam = self.calculate_lam()
        
        Wm = np.array([1/(2*(n + lam)) for i in range(dim_sigma)])
        Wc = Wm.copy()
        
        if force_Wm_sum_zero:
            Wm[0] = 1 - Wm[1:].sum()
        else:
            Wm[0] = lam/(lam + n)
        Wc[0] = lam/(lam + n) + (1 - alpha**2 + beta) #this does not add up to 1 (Wm.sum()=1, Wm.sum() != 1)
        return Wm, Wc
    
    def compute_sigma_points(self, mu, P, P_sqrt = None):
        """
        Computes the sigma points based on Julier's paper

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X
        P_sqrt : TYPE np.array(n,n), optional
            DESCRIPTION. default is None. If supplied, algorithm does not compute sqrt(P).

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        # if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
        #     raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            sqrt_factor = np.sqrt(n+self.lam)
            if P_sqrt is None:
                P_sqrt = self.sqrt(P)
            P_sqrt_weight = sqrt_factor*P_sqrt
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt_weight[:, i]
            sigmas[:, 1 + n + i] = mu - P_sqrt_weight[:, i]
        
        return sigmas, self.Wm, self.Wc, P_sqrt
        
       

def sqrt_eig(P):
    eigval, eigvec = np.linalg.eig(P)
    Px_sqrt = eigvec*np.sqrt(eigval) #equivalent to eigvec @ eigval since eigval is diagonal
    return Px_sqrt
