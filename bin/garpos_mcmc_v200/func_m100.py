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
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix, coo_matrix
from sksparse.cholmod import cholesky

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
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	jcb0 : csc_matrix
		Jacobian matrix for vector a.
	jcb2 : csc_matrix
		Zero matrix for this constrint.
	"""
	
	# set pointers for model parameter vector
	ncps = [ max([0, len(kn)-spdeg-1]) for kn in knots]
	imp0 = np.cumsum(np.array([nmppos] + ncps))
	imp0 = imp0.astype(int)
	
	# set solve index for model parameter vector
	slvidx = np.arange(imp0[0],imp0[5],dtype=int)
	slvidx = slvidx.astype(int)
	
	nmp = len(slvidx)
	ndata = len(shots.index)
	
	# Jacobian
	jcb0 = lil_matrix( (nmp, ndata) )
	mpj = np.zeros(imp0[5])
	imp = 0
	for impsv in range(imp0[0],imp0[5]):
		mpj[impsv] = 1.
		gamma, a = calc_gamma(mpj, shots, imp0, spdeg, knots)
		jcb0[imp,:] = -gamma
		imp += 1
		mpj[impsv] = 0.
	jcb0 = jcb0.tocsc() # jcb = Gt
	
	jcb2 = lil_matrix( (nmp, ndata) )
	jcb2 = jcb2.tocsc()
	
	return slvidx, imp0, jcb0, jcb2


def H_matrix(nu0, nu1, nu2, rho2, H0s):
	"""
	Set the smoothness constraint matrix H.
	
	Parameters
	----------
	nu0, nu1, nu2, rho2 : float
		hyperparameters in real scale (as defined in obs. eq.).
	H0s : list of csc_matrix
		Base-matrices for the smoothness constraint.
	
	Returns
	-------
	H : csc_matrix
		Matrix for the smoothness constraint.
	"""
	
	H = H0s[0]/nu0 + H0s[1]/(nu0*nu1) + H0s[2]/(nu0*nu2)
	
	return H


def H_bases(nknot, H0):
	"""
	Set the base-matrices for smoothness constraint matrix H.
	
	Parameters
	----------
	nknot : int
		Number of parameter for B-spline for each component.
	H0 : csc_matrix
		2nd derivative matrix of the B-spline basis.
	
	Returns
	-------
	H0s : list of csc_matrix
		Base-matrices for the smoothness constraint.
	"""
	
	H00 = lil_matrix( (nknot*5, nknot*5) )
	H01 = lil_matrix( (nknot*5, nknot*5) )
	H02 = lil_matrix( (nknot*5, nknot*5) )
	H00[nknot*0:nknot*1, nknot*0:nknot*1] = H0
	H01[nknot*1:nknot*2, nknot*1:nknot*2] = H0
	H01[nknot*2:nknot*3, nknot*2:nknot*3] = H0
	H02[nknot*3:nknot*4, nknot*3:nknot*4] = H0
	H02[nknot*4:nknot*5, nknot*4:nknot*5] = H0
	H00 = H00.tocsc()
	H01 = H01.tocsc()
	H02 = H02.tocsc()
	
	return [H00, H01, H02]


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
	rankH = rankH0*5
	logdetH  = -rankH0*5 * math.log(nu0)
	logdetH += -rankH0*2 * (math.log(nu1)+math.log(nu2))

	return rankH, logdetH


def a_to_mp(imp0, kappa12, mp, a):
	"""
	Insert "a*" into the vector "mp".
	
	Parameters
	----------
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	kappa12 : float
		Hyperparameter related to the gradient-layer's depth (not used in this version).
	mp : ndarray
		Model parameter vector.
	a : ndarray
		Conditional maximum-likelihood solution for the vector a.
	
	Returns
	-------
	mp : ndarray
		Model parameter vector.
	"""
	
	mp[imp0[0]:imp0[5]] = a
	
	return mp

