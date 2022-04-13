"""
Created:
	12/01/2021 by S. Watanabe
Contains:
	jacobian
	H_matrix
	a_to_mp
"""
import math
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

# GARPOS module
from .eachstep import derivative2
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
		Jacobian matrix for vector a (only for alpha0 and alpha1).
	jcb2 : csc_matrix
		Jacobian matrix for vector a (only for alpha2), to tie alpha1 and alpha2.
	"""
	
	# set pointers for model parameter vector
	ncps = [ max([0, len(kn)-spdeg-1]) for kn in knots]
	imp0 = np.cumsum(np.array([nmppos] + ncps))
	imp0 = imp0.astype(int)
	
	# set solve index for model parameter vector
	slvidx = np.arange(imp0[0],imp0[3],dtype=int)
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


def H_matrix(nu0, nu1, nu2, rho2, imp0, spdeg, knots):
	"""
	Set the smoothness constraint matrix H.
	
	Parameters
	----------
	nu0, nu1, nu2, rho2 : float
		hyperparameters in real scale (as defined in obs. eq.).
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	spdeg : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".
	
	Returns
	-------
	H : csc_matrix
		Matrix for the smoothness constraint.
	rankH : int
		Rank of matrix H.
	loglikeH : float
		The value of log(||Lambda_H||).
	"""
	
	lambdas = [1.] + [nu1]*2
	diff = lil_matrix( (imp0[len(lambdas)], imp0[len(lambdas)]) )
	
	for k, lamb in enumerate(lambdas):
		
		knot = knots[k]
		if len(knot) == 0:
			continue
		dk = derivative2(spdeg, knot)
		diff[imp0[k]:imp0[k+1], imp0[k]:imp0[k+1]] = dk / lamb
	
	H0 = diff[imp0[0]:imp0[1],imp0[0]:imp0[1]]
	
	rankThr = 1.e-9
	rankH0 = np.linalg.matrix_rank(H0.toarray(), tol=rankThr)
	
	H = diff[imp0[0]:,imp0[0]:]
	H = H.tocsc() / nu0
	
	rankH = rankH0*3
	loglikeH  = -rankH0*3 * math.log(nu0)
	loglikeH += -rankH0*2 * math.log(nu1)
	
	return H, rankH, loglikeH


def a_to_mp(slvidx, imp0, kappa12, mp, a):
	"""
	Insert "a*" into the vector "mp".
	
	Parameters
	----------
	slvidx : list
		Indices of model parameters to be solved. (for gamma's coeff.)
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
	
	for j in range(len(a)):
		mp[slvidx[j]] = a[j]
	mp[imp0[3]:imp0[5]] = kappa12 * mp[imp0[1]:imp0[3]]
	
	return mp

