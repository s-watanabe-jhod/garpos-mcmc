"""
Created:
	12/01/2021 by S. Watanabe
Contains:
	hparam_to_real
	E_matrix
	log_likelihood
	sampling_a
"""
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, identity
from sksparse.cholmod import cholesky


########################

def hparam_to_real(hp, hp_init, hp_prior):
	"""
	Translate the hyperparameters in real-scale.
	
	Parameters
	----------
	hp : ndarray
		Hyperparameter vector (used for MCMC sampling).
	hp_init : ndarray
		Priors of hyperparameter vector.
	hp_prior : ndarray
		Prior standard deviations of hyperparameter vector.
	
	Returns
	-------
	sigma, mu_t, mu_m, 	nu0, nu1, nu2, rho2, kappa12 : float
		Hyperparameters in real scale (as defined in obs. eq.).
	hppenalty : float
		L2 penalty related to the hyperparameters' priors.
	"""
	
	sigma = 10.**hp[0]
	mu_t = 1. / (1. + math.e**-hp[1]) * 6. * 60. + 10.
	mu_m = 1. / (1. + math.e**-hp[2])
	nu0 = 10.**hp[3]
	nu1 = 10.**hp[4]
	nu2 = 10.**hp[5]
	rho2 = 10.**hp[6]
	kappa12 = 1. / (1. + math.e**-hp[7])
	
	# Penalty from the prior distribution of hyperparameters
	hppenalty = [((hp[k]-hp_init[k])/hp_prior[k])**2. for k in range(len(hp))]
	hppenalty = -0.5 * sum(hppenalty)
	
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


def log_likelihood(sigma, ndata, jcb, E_factor, H, rankH, loglikeH, y):
	"""
	Calculate the conditional maximum-likelihood solution for the vector a
	log-likelihood of given parameters.
	
	Parameters
	----------
	sigma : float
		Scale factor for the data-error covariance
	ndata : int
		Number of used acoustic data.
	jcb : csc_matrix
		Jacobian matrix for gamma. 
	E_factor : Factor object
		Cholesky decomposition of E0.
	H : csc_matrix
		Matrix for the smoothness constraint.
	rankH : int
		Rank of matrix H.
	loglikeH : float
		The value of log(||Lambda_H||).
	y : ndarray
		Vector of log(TravelTime_obs) - log(TravelTime_ref).
	
	Returns
	-------
	loglike : ndarray
		Log-likelihood for an MCMC sample.
	a : ndarray
		Conditional maximum-likelihood solution for the vector a.
	Ci_factor : Factor object
		Cholesky decomposition of C-inverse.
	"""
	
	logdetEi = -E_factor.logdet()
	
	# Calc for perturbation model vector a 
	LiG = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
	GtEiG = LiG.T @ LiG
	
	Eiy = E_factor(y)
	GtEiy = jcb @ Eiy
	
	Ci = GtEiG + H
	try:
		Ci_factor = cholesky(Ci.tocsc(), ordering_method="natural")
	except:
		loglike = np.nan
		a = 0.
		Ci_factor = 0.
		return loglike, a, Ci_factor
	
	logdetC = -Ci_factor.logdet()
	a = Ci_factor(GtEiy)
	
	gamma = -jcb.T @ a
	R = y + gamma
	EiR = E_factor(R)
	
	misfit = R @ EiR
	penalty = ( a @ H ) @ a
	s0 = misfit + penalty
	
	loglike  = -(ndata+rankH-len(a)) * math.log(sigma)
	loglike += loglikeH
	loglike += logdetEi + logdetC - s0/sigma
	loglike = 0.5 * loglike
	
	return loglike, a, Ci_factor


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

