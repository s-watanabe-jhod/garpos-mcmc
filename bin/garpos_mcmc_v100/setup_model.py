"""
Created:
	12/05/2021 by S. Watanabe
Contains:
	init_position
	make_knots
	derivative2
	data_var_base
	setup_hparam
"""
import sys
import json
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
	dk : ndarray
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
	
	dk = (delta.T @ w) @ delta
	
	return dk


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
	hp_prior : ndarray
		Standard deviations of prior distribution for hyperparameter.
	"""

	# for position
	sgmx = float(icfg.get("MCMC-parameter","sigma_x")) 
	# for hyperparameters
	logsigma, sgmsgm, prisgm = json.loads(icfg.get("MCMC-parameter","log10_sigma_sq"))
	mu_t, sgmmut, primut = json.loads(icfg.get("MCMC-parameter","sigmoid_mu_t"))
	mu_m, sgmmum, primum = json.loads(icfg.get("MCMC-parameter","sigmoid_mu_m"))
	lognu0, sgmnu0, prinu0 = json.loads(icfg.get("MCMC-parameter","log10_nu0_sq"))
	lognu1, sgmnu1, prinu1 = json.loads(icfg.get("MCMC-parameter","log10_nu1_sq"))
	lognu2, sgmnu2, prinu2 = json.loads(icfg.get("MCMC-parameter","log10_nu2_sq"))
	logrho2, sgmrho2, prirho2 = json.loads(icfg.get("MCMC-parameter","log10_rho2_sq"))
	kappa12, sgmkappa12, prikappa12 = json.loads(icfg.get("MCMC-parameter","sigmoid_kappa12"))

	# store in ndarray
	hp_init = [logsigma, mu_t, mu_m, lognu0, lognu1, lognu2, logrho2, kappa12]
	hp_init = np.array(hp_init)
	hp_proposal = [sgmsgm, sgmmut, sgmmum, sgmnu0, sgmnu1, sgmnu2, sgmrho2, sgmkappa12]
	hp_proposal = np.array(hp_proposal)
	hp_prior = [prisgm, primut, primum, prinu0, prinu1, prinu2, prirho2, prikappa12]
	hp_prior = np.array(hp_prior)

	return hp_init, hp_proposal, hp_prior

