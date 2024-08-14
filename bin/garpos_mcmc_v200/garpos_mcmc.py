"""
Created:
	08/13/2024 by S. Watanabe
Contains:
	garpos_mcmc
"""
import os
import sys
import time
import datetime
import math
import configparser
import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity
from scipy.interpolate import BSpline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import yaml

# GARPOS modules
from .setup_model import init_position, make_knots, derivative2, data_var_base, setup_hparam
from .forward import calc_forward, calc_gamma, atd2enu
from .eachstep import hparam_to_real, E_matrix, sampling_a
from .traveltime_rt import calc_traveltime_raytrace
from .traveltime_d import calc_traveltime, calc_snell
from .output import outresults
from .resplot import plot_residuals

def garpos_mcmc(cfgf, icfgf, odir, suffix, mode, ext, iwbic):
	
	if not os.path.exists(odir):
		os.makedirs(odir)
	odir = odir+"/"
	
	# Read a configuration file
	with open(icfgf, 'r') as icfgyml:
		icfg = yaml.safe_load(icfgyml)
	
	if mode == "m100":
		from .func_m100 import jacobian, H_bases, H_matrix, H_params, a_to_mp
	elif mode == "m101":
		from .func_m101 import jacobian, H_bases, H_matrix, H_params, a_to_mp
	elif mode == "m102":
		from .func_m102 import jacobian, H_bases, H_matrix, H_params, a_to_mp
	else:
		print("Set mode appropriately :: %s \n" % options.mode + modetxt)
		sys.exit(1)
	
	# MCMC sampling parameters
	if iwbic:
		sample_scale = 1
	else:
		sample_scale = int(icfg["MCMC-parameter"]["final_sample_scale"])
	sample_rate = int(icfg["MCMC-parameter"]["sample_rate"])
	sample_size = int(icfg["MCMC-parameter"]["sample_size"]) * sample_scale
	skip_sample = int(icfg["MCMC-parameter"]["skip_sample"]) * sample_scale
	burn_in = int(icfg["MCMC-parameter"]["burn_in"]) * sample_scale

	# Std. dev. of proposal distribution for position
	sgmx = float(icfg["MCMC-parameter"]["sigma_x"]) 

	# Set priors for hyperparameters
	hp_init, hp_proposal, hp_prior_l, hp_prior_u = setup_hparam(icfg)

	spdeg = 3
	knotint0 = float(icfg["Config-parameter"]["knotint0"]) * 60.

	dt_thr = float(icfg["Config-parameter"]["dt_thr_minute"]) * 60.
	fpower = float(icfg["Config-parameter"]["fpower"])
	rr = float(icfg["Config-parameter"]["rr"])

	# Read GNSS-A data
	cfg = configparser.ConfigParser()
	cfg.read(cfgf, 'UTF-8')
	site = cfg.get("Obs-parameter", "Site_name")
	height0 = float( cfg.get("Site-parameter", "Height0") )

	# Read obs file
	obsfile = cfg.get("Data-file", "datacsv")
	shots = pd.read_csv(obsfile, comment='#', index_col=0)
	shots = shots[~shots.isnull().any(axis=1)].reset_index(drop=True) # check NaN in shotdata
	shots = shots[~(shots.TT <= 0.)].reset_index(drop=True) # check TT > 0 in shotdata
	shots = shots[~shots['flag']].reset_index(drop=True).copy()
	ndata = len(shots.index)

	# Set beta for WBIC calc.
	if iwbic:
		beta = 1./math.log(ndata)
		hp_proposal = hp_proposal * beta
	else:
		beta = 1.
	
	# Sound speed profile
	svpf = cfg.get("Obs-parameter", "SoundSpeed")
	svp = pd.read_csv(svpf, comment='#')

	# IDs of existing transponder
	MTs = cfg.get("Site-parameter", "Stations").split()
	MTs = [ str(mt) for mt in MTs ]
	nMT = len(MTs)

	# Set Model parameter for positions
	mppos0, names, slvidx_pos, mtidx = init_position(cfg, MTs)
	slvx = mppos0[slvidx_pos]
	atd0 = mppos0[-3:]

	# MT index for model parameter
	shots['mtid'] = [ mtidx[mt] for mt in shots['MT'] ]

	# Initial parameters for gradient gamma
	cnt0 = np.array([ mppos0[imt*3:imt*3+3] for imt in range(nMT)])
	cnt0 = np.mean(cnt0, axis=0)
	shots['sta0_e'] = mppos0[shots['mtid']+0] + mppos0[len(MTs)*3+0]
	shots['sta0_n'] = mppos0[shots['mtid']+1] + mppos0[len(MTs)*3+1]
	shots['sta0_u'] = mppos0[shots['mtid']+2] + mppos0[len(MTs)*3+2]
	shots['mtde'] = (shots['sta0_e'].values - cnt0[0])
	shots['mtdn'] = (shots['sta0_n'].values - cnt0[1])
	shots['de0'] = shots['ant_e0'].values - shots['ant_e0'].values.mean()
	shots['dn0'] = shots['ant_n0'].values - shots['ant_n0'].values.mean()
	shots['de1'] = shots['ant_e1'].values - shots['ant_e0'].values.mean()
	shots['dn1'] = shots['ant_n1'].values - shots['ant_n0'].values.mean()
	shots['iniflag'] = shots['flag'].copy()

	# set ATD offset in ENU coordinate
	shots = atd2enu(shots, atd0)

	# calc average depth*2 (characteristic length)
	L0 = np.array([(mppos0[i*3+2] + mppos0[nMT*3+2]) for i in range(nMT)]).mean()
	L0 = abs(L0 * 2.)

	# calc depth-averaged sound speed (characteristic length/time)
	vl = svp.speed.values
	dl = svp.depth.values
	avevlyr = [ (vl[i+1]+vl[i])*(dl[i+1]-dl[i])/2. for i in svp.index[:-1]]
	V0 = np.array(avevlyr).sum()/(dl[-1]-dl[0])

	S0, Snell_interpolate = calc_snell(shots, mppos0, nMT, icfg, svp, height0)

	# calc characteristic time
	T0 = L0 / V0
	shots["logTT"] = np.log(shots.TT.values/T0)

	# Set Model Parameters for gamma
	knotintervals = [knotint0] * 5
	knots = make_knots(shots, spdeg, knotintervals)
	H0 = derivative2(spdeg, knots[0])
	H0 = H0.tocsc()
	nknot = len(knots[0])-spdeg-1
	rankThr = 1.e-9
	rankH0 = np.linalg.matrix_rank(H0.toarray(), tol=rankThr)
	H0s = H_bases(nknot, H0)

	nmppos = len(mppos0)
	slvidx, imp0, jcb0, jcb2 = jacobian(nmppos, shots, spdeg, knots)
	nmp = len(slvidx)

	# Set E-inverse (or Sigma_d-inverse)
	mat_dt, mat_tt0, same_mt, diff_mt  = data_var_base(shots, T0, dt_thr)

	# for MCMC samples
	samples = []
	resi_samples = []
	sumflag = 0
	# initial value
	dx0 = slvx.copy()
	hp0 = hp_init.copy()
	dx1 = dx0.copy()
	hp1 = hp0.copy()

	start = time.time()
	pbar = tqdm(range(sample_size+1))
	for i in pbar:
		
		flag = 0
		if i > 0:
			x_proposal = np.array([sgmx]*len(slvx))
			q_x = np.round(np.random.normal(0, x_proposal), 8)
			dx1 = dx0 + q_x
			q_hp = np.round(np.random.normal(0, hp_proposal), 8)
			hp1 = hp0 + q_hp
		
		sigma, mu_t, mu_m, nu0, nu1, nu2, rho2, kappa12, hppenalty \
		= hparam_to_real(hp1, hp_init, hp_prior_l, hp_prior_u)
		
		if not hppenalty:
			accept_prob = -1.
		else:
			# update position by dx1
			mppos = mppos0.copy()
			for j, dx in enumerate(dx1):
				mppos[slvidx_pos[j]] = dx
			mp = np.zeros(imp0[5])
			mp[:imp0[0]] = mppos.copy()
			cnt1 = np.array([ mppos[imt*3:imt*3+3] for imt in range(nMT)])
			cnt1 = np.mean(cnt1, axis=0) + mppos[nMT*3:nMT*3+3]
			
			# Calc Log-likelihood
			cTT, cTO = calc_traveltime(shots, mp, nMT, S0, Snell_interpolate)
			if i == 0:
				cTTrt, cTOrt = calc_traveltime_raytrace(shots, mp, nMT, icfg, svp)
				diff_dtt = np.max(np.abs(cTTrt-cTT))*1000.
				#print("Max. difference of interpolated travel time: %8.3e" % diff_dtt + " msec.")
				if diff_dtt > 1.e-3:
					print("Error in interpolation of travel time (difference is larger than microsec)!!!")
					sys.exit(1)
			
			y = shots.logTT.values - np.log( cTT/T0 )
			E_factor, E0 = E_matrix(mu_t, mu_m, ndata, mat_dt, diff_mt, same_mt, mat_tt0, fpower, rr)
			logdetEi = -E_factor.logdet()
			rankH, logdetH = H_params(nu0, nu1, nu2, rho2, rankH0)
			H = H_matrix(nu0, nu1, nu2, rho2, H0s)
			
			# Calc for perturbation model vector a 
			jcb = jcb0 + kappa12 * jcb2
			LiG = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
			GtEiG = LiG.T @ LiG
			Ci = (beta * GtEiG + H).tocsc()
			try:
				Ci_factor = cholesky(Ci, ordering_method="natural")
			except:
				print(i, ": Ci may not be positive definite")
				#print(np.linalg.matrix_rank(Ci.toarray(), tol=rankThr))
				#print(Ci.toarray().shape)
				#print(np.linalg.det(Ci.toarray()))
				i -= 1
				continue
				return ["Error_in_Ci", np.nan, np.nan, np.zeros(3)]
			
			Eiy = E_factor(y)
			GtEiy = jcb @ Eiy
			a_star = beta * Ci_factor(GtEiy)
			gamma = -jcb.T @ a_star
			R = y + gamma
			LiR = E_factor.solve_L(R, use_LDLt_decomposition=False)
			misfit = LiR.T @ LiR
			penalty = ( a_star @ H ) @ a_star
			s0 = beta * misfit + penalty
			
			logdetC = -Ci_factor.logdet()
			loglike  = -(ndata*beta+rankH-len(a_star)) * math.log(sigma)
			loglike += logdetH + logdetEi + logdetC - s0/sigma
			loglike = 0.5 * loglike
			
			# sampling for a
			a_sample = sampling_a(nmp, sigma, a_star, Ci_factor)
			
			if iwbic:
				alpha = np.zeros(5)
				gamma_k = -jcb.T @ a_sample
				Rk = y + gamma_k
				LiRk = E_factor.solve_L(Rk, use_LDLt_decomposition=False)
				misfit_sample = LiRk.T @ LiRk
				wbic = -logdetEi + ndata * math.log(sigma) + misfit_sample/sigma
				wbic = wbic * 0.5
			else:
				a0 = []
				a1 = []
				mp = a_to_mp(imp0, kappa12, mp, a_sample)
				for k, kn in enumerate(knots):
					if len(kn) == 0:
						a0.append( 0. )
						a1.append( 0. )
						continue
					ct = mp[imp0[k]:imp0[k+1]]
					bs = BSpline(kn, ct, spdeg, extrapolate=False)
					a0.append( bs(shots.ST.values) )
					a1.append( bs(shots.RT.values) )
				dv0 = (np.mean(a0[0])*0.5 + np.mean(a1[0])*0.5) * V0
				g1e = (np.mean(a0[1])*0.5 + np.mean(a1[1])*0.5) * V0
				g1n = (np.mean(a0[2])*0.5 + np.mean(a1[2])*0.5) * V0
				g2e = (np.mean(a0[3])*0.5 + np.mean(a1[3])*0.5) * V0
				g2n = (np.mean(a0[4])*0.5 + np.mean(a1[4])*0.5) * V0
				alpha = np.array([dv0, g1e, g1n, g2e, g2n])
				wbic = 0.
			
			if i == 0:
				loglike0 = loglike - 10.
				loglike_map = loglike0
			
			##################################
			# Acception ratio of MCMC sample #
			##################################
			dloglike = min(1., loglike - loglike0)
			dloglike = max(-1.e2, dloglike)
			accept_prob = min(1, np.exp(dloglike))
		
		# Metropolis-Hastings Algorithm
		if accept_prob > np.random.uniform():
			dx0 = dx1.copy()
			cnt0 = cnt1.copy()
			hp0 = hp1.copy()
			alpha0 = alpha.copy()
			R0 = R.copy()
			loglike0 = loglike
			wbic0 = wbic
			flag = 1 # flag for acception
			sumflag += 1
			
			if loglike0 > loglike_map:
				mp_map = a_to_mp(imp0, kappa12, mp, a_star)
				Ci_factor_map = Ci_factor.copy()
				hp_map = hp0.copy()
				loglike_map = loglike0
				wbic_map = wbic0
		
		# sampling at "sample_rate"
		if i % sample_rate == 0:
			res = np.append(dx0, cnt0)
			res = np.append(res, hp0)
			res = np.append(res, alpha0)
			res = np.append(res, loglike0)
			res = np.append(res, wbic0)
			res = np.append(res, flag)
			samples.append(res)
		
		if i >= burn_in and i % skip_sample == 0:
			tt_resi = R0 * shots.TT.values
			resi_samples.append(tt_resi)
		
		aratio = round(sumflag * 100./(i+1), 1)
		postfix = f"AccRatio: {aratio:.1f}%, logL: {loglike:.2f}"
		pbar.set_postfix_str(postfix)
	
	##########
	# Output #
	##########
	# MCMC result
	clmn  = list(names[slvidx_pos])
	clmn += ['e', 'n', 'u']
	clmn += ['log10_sigma', 'mu_t', 'mu_m', 
			 'log10_nu0', 'log10_nu1', 'log10_nu2', 
			 'log10_rho2', 'kappa12']
	clmn += ['dV0', 'g1e', 'g1n', 'g2e', 'g2n', 'loglikelihood', 'wbic']
	clmn0 = clmn.copy()
	clmn += ['flag']
	
	clmn_chain  = list(names[slvidx_pos])
	if len(slvidx_pos) > 3:
		clmn_chain += ['e', 'n', 'u']
	clmn_chain += ['log10_sigma', 'mu_t', 'mu_m', 
				   'log10_nu0', 'log10_nu1', 'log10_nu2', 
				   'log10_rho2', 'kappa12', 'loglikelihood']
	
	clmn_hist  = list(names[slvidx_pos])
	if len(slvidx_pos) > 3:
		clmn_hist += ['e', 'n', 'u']
	clmn_hist += ['log10_sigma', 'mu_t', 'mu_m', 
				  'log10_nu0', 'log10_nu1', 'log10_nu2', 
				  'log10_rho2', 'kappa12']
	if not iwbic:
		clmn_hist += ['dV0', 'g1e', 'g1n', 'g2e', 'g2n']

	if mode == "m100":
		clmn_chain.remove('log10_rho2')
		clmn_chain.remove('kappa12')
		clmn_hist.remove('log10_rho2')
		clmn_hist.remove('kappa12')
	elif mode == "m101":
		clmn_chain.remove('log10_rho2')
		clmn_chain.remove('log10_nu2')
		clmn_hist.remove('log10_rho2')
		clmn_hist.remove('log10_nu2')
	elif mode == "m102":
		clmn_chain.remove('log10_nu2')
		clmn_hist.remove('log10_nu2')

	df_result = pd.DataFrame(samples, columns = clmn)
	df_resi = pd.DataFrame(resi_samples)

	# WBIC value
	wbic = df_result.loc[burn_in::skip_sample,"wbic"].values.mean()
	aveloglike = df_result.loc[burn_in::skip_sample,"loglikelihood"].values.mean()
	if iwbic:
		### Observation parameter ###
		site = cfg.get("Obs-parameter", "Site_name")
		camp = cfg.get("Obs-parameter", "Campaign")
		# filenames to output
		filebase = site + "." + camp + suffix
		resf  = odir + filebase + "-res.dat"
	
	# output the MAP solution
	gamma, a  = calc_gamma(mp_map, shots, imp0, spdeg, knots)
	shots["gamma"] = gamma
	shots = calc_forward(shots, mp_map, nMT, svp, T0, icfg)
	av = np.array(a) * V0
	shots['dV'] = shots.gamma * V0
	slvidx0 = np.array([])
	comment = ""
	C_map = Ci_factor_map(csc_matrix(identity(nmp)))
	Cov = C_map.toarray()
	resf, dcpos = outresults(odir, suffix, cfg, imp0, slvidx0,
							 Cov, mp_map, shots, comment, MTs, mtidx, av)
	date1 = cfg.get("Obs-parameter", "Date(UTC)")
	d0 = datetime.datetime.strptime(date1,"%Y-%m-%d")
	mpf  = resf.replace('res.dat','m.p.dat')
	obsf = resf.replace('res.dat','obs.csv')
	plot_residuals(resf, obsf, mpf, d0, MTs, V0, ext)
	
	# chain
	chainf = resf.replace("res.dat","chain.csv")
	df_result.to_csv(chainf, index = None)
	resif = resf.replace("res.dat","chain_resi.csv")
	df_resi.to_csv(resif, index = None)

	# Plot MCMC sample chain
	nax = len(clmn_chain)
	fig, ax = plt.subplots(nax, figsize = (15, nax*4))
	for i, clm in enumerate(clmn_chain):
		ax[i].plot(df_result[clm])
		ax[i].set_ylabel(clm)
	chainfig = resf.replace("res.dat","chain.pdf")
	fig.savefig(chainfig)
	plt.close()

	# percentile
	perf = resf.replace('res.dat','percentile.csv')
	dfp = df_result.loc[burn_in:, clmn_hist].quantile([0.03, 0.25, 0.5, 0.75, 0.97])
	dfp.to_csv(perf)

	# Plot a histogram
	df = df_result.loc[burn_in::skip_sample,clmn_hist]
	df = df.replace([np.inf, -np.inf], np.nan)
	sns.set_style("white")
	g = sns.PairGrid(df)
	g.map_upper(plt.scatter, s=10)
	g.map_diag(sns.histplot, kde=False)
	g.map_lower(sns.kdeplot, cmap="Blues_d")
	histfig = resf.replace("res.dat","hist.pdf")
	g.savefig(histfig)
	plt.close()
	
	print(resf, "WBIC = ", wbic)
	return [resf, wbic, aveloglike]
