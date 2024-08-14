"""
Created:
	12/05/2021 by S. Watanabe
Modified
	08/13/2024 by S. Watanabe: Change settings file to YAML
Contains:
	init_position
	make_knots
	derivative2
	data_var_base
	setup_hparam
"""
import sys
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix, linalg


def init_position(cfg, MTs):
	"""
	Calculate Jacobian matrix for positions.

	Parameters
	----------
	cfg : configparser
		Config file for site paramters.
	MTs : list
		List of transponders' name.

	Returns
	-------
	mp : ndarray
		complete model parameter vector. (only for position)
	names : ndarray
		names of variable
	slvidx_pos : list
		Indices of model parameters to be solved. (only for position)
	mtidx : dictionary
		Indices of mp for each MT.
	"""

	mtidx = {}
	mp = np.array([])
	ae = np.array([])
	names = []
	for imt, mt in enumerate(MTs):
		mtidx[mt] = imt * 3
		dpos = cfg.get("Model-parameter", mt + "_dPos").split()
		dpos = list(map(float, dpos))
		mp = np.append(mp, dpos[0:3])
		ae = np.append(ae, dpos[3:6])
		names += [mt+"_de", mt+"_dn", mt+"_du"]

	dcnt = cfg.get("Model-parameter", "dCentPos").split()
	dcnt = list(map(float, dcnt))
	mp = np.append(mp, dcnt[0:3])
	ae = np.append(ae, dcnt[3:6])
	names += ["cnt_de", "cnt_dn", "cnt_du"]

	atd = cfg.get("Model-parameter", "ATDoffset").split()
	atd = list(map(float, atd))
	mp = np.append(mp, atd[0:3])
	if atd[3] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)
	if atd[4] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)
	if atd[5] > 1.e-8:
		ae = np.append(ae, 3.0)
	else:
		ae = np.append(ae, 0.0)

	slvidx_pos = np.where( ae > 1.e-14 )[0]

	return mp, np.array(names), slvidx_pos, mtidx


def make_knots(shotdat, spdeg, knotintervals):
	"""
	Create the B-spline knots for correction value "gamma".

	Parameters
	----------
	shotdat : DataFrame
		GNSS-A shot dataset.
	spdeg : int
		spline degree (=3).
	knotintervals : list of int (len=5)
		approximate knot intervals.

	Returns
	-------
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".
	"""

	sets = shotdat['SET'].unique()
	st0s = np.array([shotdat.loc[shotdat.SET==s, "ST"].min() for s in sets])
	stfs = np.array([shotdat.loc[shotdat.SET==s, "RT"].max() for s in sets])

	st0 = shotdat.ST.values.min()
	stf = shotdat.RT.values.max()
	obsdur = stf - st0

	nknots = [ int(obsdur/knint) for knint in knotintervals ]
	knots = [ np.linspace(st0, stf, nall+1) for nall in nknots ]

	for k, cn in enumerate(knots):

		if nknots[k] == 0:
			knots[k] = np.array([])
			continue

		rmknot = np.array([])
		for i in range(len(sets)-1):
			isetkn = np.where( (knots[k]>stfs[i]) & (knots[k]<st0s[i+1]) )[0]
			if len(isetkn) > 2*(spdeg+2):
				rmknot = np.append(rmknot, isetkn[spdeg+1:-spdeg-1])
		rmknot = rmknot.astype(int)
		if len(rmknot) > 0:
			knots[k] = np.delete(knots[k], rmknot)

		dkn = (stf-st0)/float(nknots[k])
		addkn0 = np.array( [st0-dkn*(n+1) for n in reversed(range(spdeg))] )
		addknf = np.array( [stf+dkn*(n+1) for n in range(spdeg)] )
		knots[k] = np.append(addkn0, knots[k])
		knots[k] = np.append(knots[k], addknf)

	return knots


def derivative2(p, knot):
	"""
	Calculate the matrix for 2nd derivative of the B-spline basis
	for a certain component in "gamma" model.
	
	Parameters
	----------
	p : int
		spline degree (=3).
	knot : ndarray
		B-spline knot vector for a target component.
	
	Returns
	-------
	H0 : csc_matrix
		2nd derivative matrix of the B-spline basis.
	"""
	
	# smoothing constraints
	nn = len(knot)-p-1
	delta =  lil_matrix( (nn-2, nn) )
	w = lil_matrix( (nn-2, nn-2) )
	
	for j in range(nn-2):
		dkn0 = (knot[j+p+1] - knot[j+p  ])/3600.
		dkn1 = (knot[j+p+2] - knot[j+p+1])/3600.
		
		delta[j,j]   =  1./dkn0
		delta[j,j+1] = -1./dkn0 -1./dkn1
		delta[j,j+2] =  1./dkn1
		
		if j >= 1:
			w[j,j-1] = dkn0 / 6.
			w[j-1,j] = dkn0 / 6.
		w[j,j] = (dkn0 + dkn1) / 3.
	
	delta = delta.tocsr()
	w = w.tocsr()
	
	H0 = (delta.T @ w) @ delta
	
	return H0


def data_var_base(shotdat, T0, dt_thr):
	"""
	Set the variables for covariance matrix for data.

	Parameters
	----------
	shotdat : DataFrame
		GNSS-A shot dataset.
	T0 : float
		Typical travel time (in sec.).
	dt_thr : float
		threshold for correlating data (typically < 10 min.) (in sec.).

	Returns
	-------
	mat_dt : csc_matrix
		Difference time matrix for every shots.
	mat_tt0 :csc_matrix
		Weight matrix for covariance.
	same_mt : csc_matrix
		Matrix for MT[i] == MT[j].
	diff_mt : csc_matrix
		Matrix for MT[i] != MT[j].
	"""

	TT0 = shotdat.TT.values / T0

	ndata = shotdat.index.size
	sts = shotdat.ST.values
	mtids = shotdat.mtid.values
	lines = shotdat.LN.values
	
	negativedST = shotdat[ (shotdat.ST.diff(1) == 0.) & (shotdat.mtid.diff(1) == 0.) ]
	if len(negativedST) > 0:
		print(negativedST.index)
		print("error in data_var_base; see setup_model.py")
		sys.exit(1)
	
	mat_dt = lil_matrix( (ndata, ndata) )
	mat_tt0 = lil_matrix( (ndata, ndata) )
	same_mt = lil_matrix( (ndata, ndata) )
	diff_mt = lil_matrix( (ndata, ndata) )
	for i, (iMT, iST, iLN) in enumerate(zip( mtids, sts, lines )):
		idx = shotdat[ ( abs(sts - iST) < dt_thr)].index
		mat_dt[i,idx] = np.abs(iST - sts[idx]) * (iLN==lines[idx])
		mat_tt0[i,idx] = 1. / TT0[i] / TT0[idx]
		same_mt[i,idx] = 1. * (iMT==mtids[idx])
		diff_mt[i,idx] = 1. * (iMT!=mtids[idx])
		
	mat_dt = mat_dt.tocsc()
	mat_tt0 = mat_tt0.tocsc()
	same_mt = same_mt.tocsc()
	diff_mt = diff_mt.tocsc()

	return mat_dt, mat_tt0, same_mt, diff_mt


def setup_hparam(icfg):
	"""
	Setup hyperparameter vecotor and its cotrol parameter vectors. 

	Parameters
	----------
	icfg : configparser
		Config file for MCMC setting.

	Returns
	-------
	hp_init : ndarray
		Initial values of hyperparameter.
	hp_proposal : ndarray
		Standard deviations of proposal distribution for hyperparameter.
	hp_prior_l : ndarray
		Lower limits as prior for hyperparameter vector.
	hp_prior_u : ndarray
		Upper limits as prior for hyperparameter vector.
	"""

	# for position
	sgmx = float(icfg["MCMC-parameter"]["sigma_x"]) 
	# for hyperparameters
	logsgm  = icfg["MCMC-parameter"]["log10_sigma_sq"]
	mu_t = icfg["MCMC-parameter"]["mu_t_minute"]
	mu_m = icfg["MCMC-parameter"]["mu_m_range"]
	lognu0 = icfg["MCMC-parameter"]["log10_nu0_sq"]
	lognu1 = icfg["MCMC-parameter"]["log10_nu1_sq"]
	lognu2 = icfg["MCMC-parameter"]["log10_nu2_sq"]
	logrho2 = icfg["MCMC-parameter"]["log10_rho2_sq"]
	kappa12 = icfg["MCMC-parameter"]["kappa12"]

	# store in ndarray
	hp_init = [logsgm[0], mu_t[0], mu_m[0], lognu0[0], lognu1[0], lognu2[0], logrho2[0], kappa12[0]]
	hp_init = np.array(hp_init)
	hp_proposal = [logsgm[1], mu_t[1], mu_m[1], lognu0[1], lognu1[1], lognu2[1], logrho2[1], kappa12[1]]
	hp_proposal = np.array(hp_proposal)
	hp_prior_l = [logsgm[2], mu_t[2], mu_m[2], lognu0[2], lognu1[2], lognu2[2], logrho2[2], kappa12[2]]
	hp_prior_l = np.array(hp_prior_l)
	hp_prior_u = [logsgm[3], mu_t[3], mu_m[3], lognu0[3], lognu1[3], lognu2[3], logrho2[3], kappa12[3]]
	hp_prior_u = np.array(hp_prior_u)

	return hp_init, hp_proposal, hp_prior_l, hp_prior_u
