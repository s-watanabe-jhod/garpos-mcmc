#!/usr/bin/env python
# coding: utf-8
"""
Created:
	12/01/2021 by S. Watanabe
"""
from optparse import OptionParser
import os
import sys
import time
import datetime
import math
import configparser
import json
import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity
from scipy.interpolate import BSpline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# GARPOS modules
from garpos_mcmc_v100.setup_model import init_position, make_knots, derivative2, data_var_base, setup_hparam
from garpos_mcmc_v100.forward import calc_forward, calc_gamma, atd2enu
from garpos_mcmc_v100.eachstep import hparam_to_real, E_matrix, log_likelihood, sampling_a
from garpos_mcmc_v100.traveltime import calc_traveltime
from garpos_mcmc_v100.output import outresults
from garpos_mcmc_v100.resplot import plot_residuals

modetxt = u"  m100: double-grad\n  m101: single-grad \n  m102: alpha2-offset"

######################################################################
usa = u"Usage: %prog [options] "
opt = OptionParser(usage=usa)
opt.add_option( "-i", action="store", type="string",
				default="", dest="invcfg",
				help=u"Path to the setup file" 
				)
opt.add_option( "-f", action="store", type="string",
				default="", dest="cfgfile",
				help=u'Path to the site-parameter file'
				)
opt.add_option( "-d", action="store", type="string",
				default="./result/", dest="directory",
				help=u"Set save directory"
				)
opt.add_option( "--mode", action="store", type="string", 
				default="na", dest="mode",
				help=u'Set mode (m100/m101/m102)' + modetxt
				)
opt.add_option( "--suffix", action="store", type="string",
				default="", dest="suffix",
				help=u"Set suffix for result files"
				)
opt.add_option( "--ext", action="store", type="string", 
				default="png", dest="ext",
				help=u'Set extention for residual plot'
				)
(options, args) = opt.parse_args()
#####################################################################

mode = options.mode
if mode == "m100":
	from garpos_mcmc_v100.func_m100 import jacobian, H_matrix, a_to_mp
elif mode == "m101":
	from garpos_mcmc_v100.func_m101 import jacobian, H_matrix, a_to_mp
elif mode == "m102":
	from garpos_mcmc_v100.func_m102 import jacobian, H_matrix, a_to_mp
else:
	print("Set mode appropriately :: %s \n" % options.mode + modetxt)
	sys.exit(1)

if not os.path.isfile(options.invcfg) or options.invcfg == "":
	print("NOT FOUND (setup file) :: %s" % options.invcfg)
	sys.exit(1)
if not os.path.isfile(options.cfgfile) or options.cfgfile == "":
	print("NOT FOUND (site paramter file) :: %s" % options.cfgfile)
	sys.exit(1)

cfgf = options.cfgfile
icfgf = options.invcfg
suffix = options.suffix
odir = options.directory+"/"
ext = options.ext

if not os.path.exists(odir):
	os.makedirs(options.directory)

# Read a configuration file
icfg = configparser.ConfigParser()
icfg.read(icfgf, 'UTF-8')

# MCMC sampling parameters
sample_rate = int(icfg.get("MCMC-parameter","sample_rate"))
sample_size = int(icfg.get("MCMC-parameter","sample_size"))
skip_sample = int(icfg.get("MCMC-parameter","skip_sample"))
burn_in = int(icfg.get("MCMC-parameter","burn_in"))

# Std. dev. of proposal distribution for position
sgmx = float(icfg.get("MCMC-parameter","sigma_x")) 

# Set priors for hyperparameters
hp_init, hp_proposal, hp_prior = setup_hparam(icfg)

spdeg = 3
knotint0 = float(icfg.get("Config-parameter","knotint0")) * 60.

dt_thr = float(icfg.get("Config-parameter","dt_thr_minute")) * 60.
fpower = float(icfg.get("Config-parameter","fpower"))
rr = float(icfg.get("Config-parameter","rr"))

# Read GNSS-A data
cfg = configparser.ConfigParser()
cfg.read(cfgf, 'UTF-8')
site = cfg.get("Obs-parameter", "Site_name")

# Read obs file
obsfile = cfg.get("Data-file", "datacsv")
shots = pd.read_csv(obsfile, comment='#', index_col=0)
shots = shots[~shots.isnull().any(axis=1)].reset_index(drop=True) # check NaN in shotdata
shots = shots[~(shots.TT <= 0.)].reset_index(drop=True) # check TT > 0 in shotdata
shots = shots[~shots['flag']].reset_index(drop=True).copy()

# Sound speed profile
svpf = cfg.get("Obs-parameter", "SoundSpeed")
svp = pd.read_csv(svpf, comment='#')

# IDs of existing transponder
MTs = cfg.get("Site-parameter", "Stations").split()
MTs = [ str(mt) for mt in MTs ]
nMT = len(MTs)


# Set Model parameter for positions
mppos0, names, slvidx_pos, mtidx = init_position(cfg, MTs)
print(names[slvidx_pos])
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

# calc characteristic time
T0 = L0 / V0
shots["logTT"] = np.log(shots.TT.values/T0)

mat_dt, mat_tt0, same_mt, diff_mt  = data_var_base(shots, T0, dt_thr)

# Set Model Parameters for gamma
knotintervals = [knotint0] * 5
knots = make_knots(shots, spdeg, knotintervals)
H0 = derivative2(spdeg, knots[0])
rankThr = 1.e-9
rankH0 = np.linalg.matrix_rank(H0.toarray(), tol=rankThr)

nmppos = len(mppos0)
slvidx, imp0, jcb0, jcb2 = jacobian(nmppos, shots, spdeg, knots)

nmp = len(slvidx)
ndata = len(shots.index)


# for MCMC samples
samples = []
sumflag = 0
# initial value
dx0 = slvx.copy()
hp0 = hp_init.copy()

pbar = tqdm(range(sample_size+1))
for i in pbar:
	
	flag = 0
	
	if i > 0:
		x_proposal = np.array([sgmx]*len(slvx))
		q_x = np.round(np.random.normal(0, x_proposal), 8)
		dx1 = dx0 + q_x
		q_hp = np.round(np.random.normal(0, hp_proposal), 8)
		hp1 = hp0 + q_hp
	else:
		dx1 = dx0.copy()
		hp1 = hp0.copy()
	
	sigma, mu_t, mu_m, nu0, nu1, nu2, rho2, kappa12, hppenalty = hparam_to_real(hp1, hp_init, hp_prior)
	
	# update position by dx1
	mppos = mppos0.copy()
	for j, dx in enumerate(dx1):
		mppos[slvidx_pos[j]] = dx
	mp = np.zeros(imp0[5])
	mp[:imp0[0]] = mppos.copy()
	
	cnt1 = np.array([ mppos[imt*3:imt*3+3] for imt in range(nMT)])
	cnt1 = np.mean(cnt1, axis=0) + mppos[nMT*3:nMT*3+3]
	
	jcb = jcb0 + kappa12 * jcb2
	
	# Calc Log-likelihood
	cTT, cTO = calc_traveltime(shots, mp, nMT, icfg, svp)
	y = shots.logTT.values - np.log( cTT/T0 )
	E_factor, E0 = E_matrix(mu_t, mu_m, ndata, mat_dt, diff_mt, same_mt, mat_tt0, fpower, rr)
	H, rankH, loglikeH = H_matrix(nu0, nu1, nu2, rho2, imp0, spdeg, H0, rankH0)
	loglike, a_star, Ci_factor = log_likelihood(sigma, ndata, jcb, E_factor, H, rankH, loglikeH, y)
	
	if np.isnan(loglike):
		print(i, ": Ci is not positive definite")
		exit()
		continue
	
	loglike += hppenalty
	
	if i == 0:
		loglike0 = loglike - 10.
	
	# Acception ration of MCMC sample
	dloglike = min(1., loglike - loglike0)
	dloglike = max(-1.e2, dloglike)
	accept_prob = min(1, np.exp(dloglike))
	
	# sampling for a
	a_sample = sampling_a(nmp, sigma, a_star, Ci_factor)
	mp = a_to_mp(slvidx, imp0, kappa12, mp, a_sample)
	
	a0 = []
	a1 = []
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
	
	# Metropolis-Hastings Algorithm
	if accept_prob > np.random.uniform():
		dx0 = dx1.copy()
		cnt0 = cnt1.copy()
		hp0 = hp1.copy()
		alpha0 = alpha.copy()
		loglike0 = loglike
		flag = 1 # flag for acception
		sumflag += 1
		
		if accept_prob == 1.:
			mp_map = a_to_mp(slvidx, imp0, kappa12, mp, a_star)
			Ci_factor_map = Ci_factor.copy()
			hp_map = hp0.copy()
			loglike_map = loglike0
	
	# sampling at "sample_rate"
	if i % sample_rate == 0:
		res = np.append(dx0, cnt0)
		res = np.append(res, hp0)
		res = np.append(res, alpha0)
		res = np.append(res, loglike0)
		res = np.append(res, flag)
		samples.append(res)
	
	aratio = round(sumflag * 100./(i+1), 1)
	postfix = f"AccRatio: {aratio:.1f}%, logL: {loglike:.2f}"
	pbar.set_postfix_str(postfix)


# MCMC result
clmn  = list(names[slvidx_pos])
clmn += ['e', 'n', 'u']
clmn += ['log10_sigma', 'mu_t', 'sigmo_mu_m', 
         'log10_nu0', 'log10_nu1', 'log10_nu2', 
         'log10_rho2', 'sigmo_kappa12']
clmn += ['dV0', 'g1e', 'g1n', 'g2e', 'g2n', 'loglikelihood']
clmn0 = clmn.copy()
clmn += ['flag']

clmn_chain  = list(names[slvidx_pos])
if len(slvidx_pos) > 3:
	clmn_chain += ['e', 'n', 'u']
clmn_chain += ['log10_sigma', 'mu_t', 'sigmo_mu_m', 
               'log10_nu0', 'log10_nu1', 'log10_nu2', 
               'log10_rho2', 'sigmo_kappa12', 'loglikelihood']

clmn_hist  = list(names[slvidx_pos])
if len(slvidx_pos) > 3:
	clmn_hist += ['e', 'n', 'u']
clmn_hist += ['log10_sigma', 'mu_t', 'sigmo_mu_m', 
              'log10_nu0', 'log10_nu1', 'log10_nu2', 
              'log10_rho2', 'sigmo_kappa12']
clmn_hist += ['dV0', 'g1e', 'g1n', 'g2e', 'g2n']

if mode == "m100":
	clmn_chain.remove('log10_rho2')
	clmn_chain.remove('sigmo_kappa12')
	clmn_hist.remove('log10_rho2')
	clmn_hist.remove('sigmo_kappa12')
elif mode == "m101":
	clmn_chain.remove('log10_rho2')
	clmn_chain.remove('log10_nu2')
	clmn_hist.remove('log10_rho2')
	clmn_hist.remove('log10_nu2')
elif mode == "m102":
	clmn_chain.remove('log10_nu2')
	clmn_hist.remove('log10_nu2')

df_result = pd.DataFrame(samples, columns = clmn)

gamma, a  = calc_gamma(mp_map, shots, imp0, spdeg, knots)
shots["gamma"] = gamma
shots = calc_forward(shots, mp_map, nMT, svp, T0, icfg)

av = np.array(a) * V0
shots['dV'] = shots.gamma * V0
slvidx0 = np.array([])
comment = ""

C = Ci_factor_map(csc_matrix(identity(nmp)))
Cov = C.toarray()

resf, dcpos = outresults(odir, suffix, cfg, imp0, slvidx0,
                         Cov, mp, shots, comment, MTs, mtidx, av)
chainf = resf.replace("res.dat","chain.csv")
df_result.to_csv(chainf, index = None)


# MCMC sample chain
nax = len(clmn_chain)
fig, ax = plt.subplots(nax, figsize = (15, nax*4))

for i, clm in enumerate(clmn_chain):
	ax[i].plot(df_result[clm])
	ax[i].set_ylabel(clm)

chainfig = resf.replace("res.dat","chain.pdf")
fig.savefig(chainfig)

date1 = cfg.get("Obs-parameter", "Date(UTC)")
d0 = datetime.datetime.strptime(date1,"%Y-%m-%d")
mpf  = resf.replace('res.dat','m.p.dat')
obsf = resf.replace('res.dat','obs.csv')
perf = resf.replace('res.dat','percentile.csv')

plot_residuals(resf, obsf, mpf, d0, MTs, V0, ext)
dfp = df_result.loc[burn_in:, clmn_hist].quantile([0.025, 0.25, 0.5, 0.75, 0.975])
dfp.to_csv(perf)

# Plot a histogram
df = df_result.loc[burn_in::skip_sample,clmn_hist]
sns.set(style="white")
g = sns.PairGrid(df)
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.histplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
histfig = resf.replace("res.dat","hist.pdf")
g.savefig(histfig)

