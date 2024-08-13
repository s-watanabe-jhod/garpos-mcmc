"""
Created:
	12/01/2021 by S. Watanabe
Modified
	08/13/2024 by S. Watanabe: Change in creating matrix H
Contains:
	jacobian
	H_matrix
	a_to_mp
"""
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

# GARPOS module
from .forward import calc_gamma


def jacobian(nmppos, shots, spdeg, knots):
	"""
	Calculate Jacobian matrix for vector a.
	
	Parameters
	----------
	nmppos : int
		Number of position parameters.
	shots : DataFrame
		GNSS-A shot dataset.
	spdeg : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".
	
	Returns
	-------
	slvidx : list
		Indices of model parameters to be solved. (for gamma's coeff.)
	imp0 : ndarray (len=7)
		Indices where the type of model parameters change. The last two is for delta_alpha2.
	jcb0 : csc_matrix
		Jacobian matrix for vector a (only for alpha0 and alpha1).
	jcb2 : csc_matrix
		Jacobian matrix for vector a (only for alpha2), to tie alpha1 and alpha2.
	"""
	
	# set pointers for model parameter vector
	ncps = [ max([0, len(kn)-spdeg-1]) for kn in knots]
	imp0 = np.cumsum(np.array([nmppos] + ncps + [1, 1]))
	imp0 = imp0.astype(int)
	
	# set solve index for model parameter vector
	slvidx = np.arange(imp0[0],imp0[3],dtype=int)
	slvidx = np.append(slvidx, np.array(imp0[5:7]))
	slvidx = slvidx.astype(int)
	
	nmp = len(slvidx)
	ndata = len(shots.index)
	
	# Jacobian
	jcb0 = lil_matrix( (nmp, ndata) )
	mpj = np.zeros(imp0[5])
	imp = 0
	for impsv in range(imp0[0],imp0[3]):
		mpj[impsv] = 1.
		gamma, a = calc_gamma(mpj, shots, imp0, spdeg, knots)
		jcb0[imp,:] = -gamma
		imp += 1
		mpj[impsv] = 0.
	
	mpj[imp0[3]:imp0[4]] = 1.
	gamma, a = calc_gamma(mpj, shots, imp0, spdeg, knots)
	jcb0[-2,:] = -gamma
	mpj = np.zeros(imp0[5])
	mpj[imp0[4]:imp0[5]] = 1.
	gamma, a = calc_gamma(mpj, shots, imp0, spdeg, knots)
	jcb0[-1,:] = -gamma
	mpj = np.zeros(imp0[5])
	
	jcb0 = jcb0.tocsc() # jcb = Gt
	
	jcb2 = lil_matrix( (nmp, ndata) )
	imp = imp0[1]-imp0[0]
	for impsv in range(imp0[3],imp0[5]):
		mpj[impsv] = 1.
		gamma, a = calc_gamma(mpj, shots, imp0, spdeg, knots)
		jcb2[imp,:] = -gamma
		imp += 1
		mpj[impsv] = 0.
	jcb2 = jcb2.tocsc()
	
	return slvidx, imp0, jcb0, jcb2


def H_matrix(nu0, nu1, nu2, rho2, nknot, H0):
	"""
	Set the smoothness constraint matrix H.
	
	Parameters
	----------
	nu0, nu1, nu2, rho2 : float
		hyperparameters in real scale (as defined in obs. eq.).
	nknot : int
		Number of parameter for B-spline for each component.
	H0 : csc_matrix
		2nd derivative matrix of the B-spline basis.
	
	Returns
	-------
	H : csc_matrix
		Matrix for the smoothness constraint.
	"""
	
	H = lil_matrix( (nknot*3+2, nknot*3+2) )
	H[nknot*0:nknot*1, nknot*0:nknot*1] = H0 / nu0
	H[nknot*1:nknot*2, nknot*1:nknot*2] = H0 / (nu0*nu1)
	H[nknot*2:nknot*3, nknot*2:nknot*3] = H0 / (nu0*nu1)
	# covariance matrix for ridge constraint
	H[-2,-2] = 1./rho2
	H[-1,-1] = 1./rho2
	
	return H


def H_params(nu0, nu1, nu2, rho2, rankH0):
	"""
	Calc. the rank(H) and log(||Lambda_H||).
	
	Parameters
	----------
	nu0, nu1, nu2, rho2 : float
		hyperparameters in real scale (as defined in obs. eq.).
	rankH0 : int
		rank of H0
	
	Returns
	-------
	rankH : int
		Rank of matrix H.
	loglikeH : float
		The value of log(||Lambda_H||).
	"""
	rankHg2 = 2
	logdetHg2 = -2. * math.log(rho2)
	
	rankH = rankH0*3 + rankHg2
	logdetH  = -rankH0*3 * math.log(nu0)
	logdetH += -rankH0*2 * math.log(nu1)
	logdetH += logdetHg2

	return rankH, logdetH


def a_to_mp(imp0, kappa12, mp, a):
	"""
	Insert "a*" into the vector "mp".
	
	Parameters
	----------
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	kappa12 : float
		Hyperparameter related to the gradient-layer's depth.
	mp : ndarray
		Model parameter vector.
	a : ndarray
		Conditional maximum-likelihood solution for the vector a.
	
	Returns
	-------
	mp : ndarray
		Model parameter vector.
	"""
	
	mp[imp0[0]:imp0[3]] = a[:-2]
	mp[imp0[3]:imp0[4]] = kappa12 * mp[imp0[1]:imp0[2]] + a[-2]
	mp[imp0[4]:imp0[5]] = kappa12 * mp[imp0[2]:imp0[3]] + a[-1]
	
	return mp

