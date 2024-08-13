"""
Created:
	12/05/2021 by S. Watanabe
Contains:
	calc_forward
	atd2enu
	calc_gamma
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix

# garpos module
from .coordinate_trans import corr_attitude
from .traveltime_rt import calc_traveltime_raytrace


def calc_forward(shots, mp, nMT, svp, T0, icfg):
	"""
	Calculate the forward modeling of observation eqs.

	Parameters
	----------
	shots : DataFrame
		GNSS-A shot dataset.
	mp : ndarray
		complete model parameter vector.
	nMT : int
		number of transponders.
	svp : DataFrame
		Sound speed profile.
	T0 : float
		Typical travel time.

	Returns
	-------
	shots : DataFrame
		GNSS-A shot dataset in which calculated data is added.
	"""

	# calc ATD offset
	calATD = np.vectorize(corr_attitude)
	pl0 = mp[(nMT+1)*3+0]
	pl1 = mp[(nMT+1)*3+1]
	pl2 = mp[(nMT+1)*3+2]
	hd0 = shots.head0.values
	hd1 = shots.head1.values
	rl0 = shots.roll0.values
	rl1 = shots.roll1.values
	pc0 = shots.pitch0.values
	pc1 = shots.pitch1.values
	ple0, pln0, plu0 = calATD(pl0, pl1, pl2, hd0, rl0, pc0)
	ple1, pln1, plu1 = calATD(pl0, pl1, pl2, hd1, rl1, pc1)
	shots['ple0'] = ple0
	shots['pln0'] = pln0
	shots['plu0'] = plu0
	shots['ple1'] = ple1
	shots['pln1'] = pln1
	shots['plu1'] = plu1

	# calc Residuals
	cTT, cTO = calc_traveltime_raytrace(shots, mp, nMT, icfg, svp)
	logTTc = np.log( cTT/T0 ) - shots.gamma.values
	ResiTT = shots.logTT.values - logTTc

	shots['calcTT'] = cTT
	shots['TakeOff'] = cTO
	shots['logTTc'] = logTTc
	shots['ResiTT'] = ResiTT
	# approximation log(1 + x) ~ x
	shots['ResiTTreal'] = ResiTT * shots.TT.values
	
	return shots

def atd2enu(shots, atd):
	"""
	Calculate the ATD offset's effect into ENU coords.

	Parameters
	----------
	shots : DataFrame
		GNSS-A shot dataset.
	atd : ndarray
		ATD offset vector.

	Returns
	-------
	shots : DataFrame
		GNSS-A shot dataset with ENU offset is added.
	"""

	# calc ATD offset
	calATD = np.vectorize(corr_attitude)
	pl0 = atd[0] #mp[(nMT+1)*3+0]
	pl1 = atd[1] #mp[(nMT+1)*3+1]
	pl2 = atd[2] #mp[(nMT+1)*3+2]
	hd0 = shots.head0.values
	hd1 = shots.head1.values
	rl0 = shots.roll0.values
	rl1 = shots.roll1.values
	pc0 = shots.pitch0.values
	pc1 = shots.pitch1.values
	ple0, pln0, plu0 = calATD(pl0, pl1, pl2, hd0, rl0, pc0)
	ple1, pln1, plu1 = calATD(pl0, pl1, pl2, hd1, rl1, pc1)
	shots['ple0'] = ple0
	shots['pln0'] = pln0
	shots['plu0'] = plu0
	shots['ple1'] = ple1
	shots['pln1'] = pln1
	shots['plu1'] = plu1
	
	return shots


def calc_gamma(mp, shotdat, imp0, spdeg, knots):
	"""
	Calculate correction value "gamma" in the observation eqs.

	Parameters
	----------
	mp : ndarray
		complete model parameter vector.
	shotdat : DataFrame
		GNSS-A shot dataset.
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	spdeg : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".

	Returns
	-------
	gamma : ndarray
		Values of "gamma". Note that scale facter is not applied.
	a : 2-d list of ndarray
		[a0[<alpha>], a1[<alpha>]] :: a[<alpha>] at transmit/received time.
		<alpha> is corresponding to <0>, <1E>, <1N>, <2E>, <2N>.
	"""

	a0 = []
	a1 = []
	for k, kn in enumerate(knots):
		if len(kn) == 0:
			a0.append( 0. )
			a1.append( 0. )
			continue
		ct = mp[imp0[k]:imp0[k+1]]
		bs = BSpline(kn, ct, spdeg, extrapolate=False)
		a0.append( bs(shotdat.ST.values) )
		a1.append( bs(shotdat.RT.values) )

	ls = 1000.  # m/s/m to m/s/km order for gradient

	de0 = shotdat.de0.values
	de1 = shotdat.de1.values
	dn0 = shotdat.dn0.values
	dn1 = shotdat.dn1.values
	mte = shotdat.mtde.values
	mtn = shotdat.mtdn.values

	gamma0_0 =  a0[0]
	gamma0_1 = (a0[1] * de0 + a0[2] * dn0) / ls
	gamma0_2 = (a0[3] * mte + a0[4] * mtn) / ls

	gamma1_0 =  a1[0]
	gamma1_1 = (a1[1] * de1 + a1[2] * dn1) / ls
	gamma1_2 = (a1[3] * mte + a1[4] * mtn) / ls

	gamma0 = gamma0_0 + gamma0_1 + gamma0_2
	gamma1 = gamma1_0 + gamma1_1 + gamma1_2

	gamma = (gamma0 + gamma1)/2.
	a = [a0, a1]

	return gamma, a

