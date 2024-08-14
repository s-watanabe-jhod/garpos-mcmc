"""
Created:
	12/01/2021 by S. Watanabe
Contains:
	hparam_to_real
	E_matrix
	sampling_a
"""
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, identity
from sksparse.cholmod import cholesky

########################

def hparam_to_real(hp, hp_init, hp_prior_l, hp_prior_u):
	"""
	Translate the hyperparameters in real-scale.
	
	Parameters
	----------
	hp : ndarray
		Hyperparameter vector (used for MCMC sampling).
	hp_init : ndarray
		Priors of hyperparameter vector.
	hp_prior_l : ndarray
		Lower limits as prior for hyperparameter vector.
	hp_prior_u : ndarray
		Upper limits as prior for hyperparameter vector.
	
	Returns
	-------
	sigma, mu_t, mu_m, 	nu0, nu1, nu2, rho2, kappa12 : float
		Hyperparameters in real scale (as defined in obs. eq.).
	hppenalty : Bool
		Whether the hps are set in the upper/lower limits.
	"""
	
	sigma = 10.**hp[0]
	mu_t = hp[1] * 60.
	mu_m = hp[2]
	nu0 = 10.**hp[3]
	nu1 = 10.**hp[4]
	nu2 = 10.**hp[5]
	rho2 = 10.**hp[6]
	kappa12 = hp[7]
	
	# Penalty from the prior distribution of hyperparameters
	hp_l = hp - hp_prior_l
	hp_u = hp_prior_u - hp
	hppenalty = np.all( hp_l > 0. ) and np.all( hp_u > 0. )
	
	return sigma, mu_t, mu_m, nu0, nu1, nu2, rho2, kappa12, hppenalty


def E_matrix(mu_t, mu_m, ndata, mat_dt, diff_mt, same_mt, mat_tt0, fpower, rr):
	"""
	Set the covariance matrix for the data-error vector.
	
	Parameters
	----------
	mu_t : float
		Hyperparameter (mu_t).
		Controls the correlation length for acoustic data.
	mu_m : float
		Hyperparameter (mu_MT).
		Controls the inter-transponder data correlation.
	ndata : int
		Number of used acoustic data.
	mat_dt : csc_matrix
		Difference time matrix for every shots.
	diff_mt : csc_matrix
		Matrix for MT[i] != MT[j].
	same_mt : csc_matrix
		Matrix for MT[i] == MT[j].
	mat_tt0 :csc_matrix
		Weight matrix for covariance.
	fpower : float
		Number of significant digits for correlation coefficient.
	rr : float
		Scale for i != j shot (to avoid rank deficient).
	
	Returns
	-------
	E_factor : Factor object
		Cholesky decomposition of E0.
	E0 : csc_matrix
		Covariance matrix for the data-error vector.
	"""
	
	# cut-off for data covariance
	ff = 10. ** fpower 
	# Set observation error covariance
	E0 = csc_matrix((np.floor(np.exp(-mat_dt.data/mu_t)*ff)/ff, 
	                 mat_dt.indices, mat_dt.indptr), 
	                 shape=mat_dt.shape) * rr
	mteff = mu_m * diff_mt + same_mt
	E0 += csc_matrix(identity(ndata))
	E0 = E0.multiply(mteff)
	E0 = E0.multiply(mat_tt0)
	# cholesky decomposition
	E_factor = cholesky(E0, ordering_method="natural")
	
	return E_factor, E0


def sampling_a(nmp, sigma, a_star, Ci_factor):
	"""
	Stochastic sampling of vector a from a*, 
	by considering the estimated covariance (C).
	
	Parameters
	----------
	nmp : int
		length of vector a*.
	sigma : float
		Scale factor for the data-error covariance.
	a_star : ndarray
		Conditional maximum-likelihood solution for the vector a.
	Ci_factor : Factor object
		Cholesky decomposition of C-inverse.
	
	Returns
	-------
	a_sample : ndarray
		Sampled vector a.
	"""
	
	### sampling for a ###
	zz = np.random.standard_normal(nmp)
	var_a = Ci_factor.solve_Lt(zz, use_LDLt_decomposition=False)
	a_sample = a_star + sigma**0.5 * var_a
	
	return a_sample

