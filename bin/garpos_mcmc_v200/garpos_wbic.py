"""
Created:
	08/13/2024 by S. Watanabe
Contains:
	parallelrun
	model_search_wbic
"""
import os
import glob
import math
import shutil
import configparser
from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# garpos module
from .garpos_mcmc import garpos_mcmc

def custom_error_callback(error):
	print(f'Got an error: {error}')

def parallelrun(inplist, maxcore):
	"""
	Run the model parameter estimation in parallel.
	
	Parameters
	----------
	inplist : DataFrame
		List of arguments for the function.
	maxcore : int
		maximum number of parallelization.
	
	Returns
	-------
	inplist : DataFrame
		List of arguments for the function in which brief results are added.
	"""
	
	npara = len(inplist.index)
	mc = min(maxcore, npara)
	
	# Input files
	i0 = inplist.cfgfile
	i1 = inplist.invcfg
	
	# Output parameters
	o1 = inplist.outdir
	o2 = inplist.suffix
	o3 = inplist.ext
	
	# Models
	s0 = inplist.slvmode
	s1 = inplist.iwbic
	
	inp = list(zip(i0,i1,o1,o2,s0,o3,s1))
	
	with Pool(processes=mc) as p:
		reslist = p.starmap(garpos_mcmc, inp)
		p.close()
	
	inplist["resfile"] = [ r[0] for r in reslist ]
	inplist["wbic"] = [ r[1] for r in reslist ]
	inplist["ave_loglike"] = [ r[2] for r in reslist ]
	
	return inplist


def model_search_wbic(cfgf, icfgf, outdir, suf, ext):
	"""
	This module is used for model selection by WBIC.
	This is a main driver to run GARPOS-MCMC with WBIC.
	
	Parameters
	----------
	cfgf : string
		Path to the site-parameter file.
	icfgf : stirng
		Path to the analysis-setting file.
	outdir : string
		Directory to store the results.
	suf : string
		Suffix to be added for result files.
	mode : string
		
	ext : string
		
	Returns
	-------
	resf : string
		Result site-paramter file name (min-ABIC model).
	"""
	#######################################
	# Set Input parameter list for Search #
	#  * it might be better to be changed #
	#######################################
	maxcore = 1
	models = ["m100", "m101", "m102"]
	nmodels = len(models)
	#######################################
	
	if nmodels == 1:
		wkdir = outdir+"/"
	elif nmodels > 1:
		wkdir  = outdir+ "/tested_models/"
	else:
		print("error in setting for model selection")
		sys.exit(1)
	
	if not os.path.exists(wkdir+"/"):
		os.makedirs(wkdir)
	
	# Set File Name
	cfg = configparser.ConfigParser()
	cfg.read(cfgf, 'UTF-8')
	site = cfg.get("Obs-parameter", "Site_name")
	camp = cfg.get("Obs-parameter", "Campaign")
	filebase = site + "." + camp + suf
	
	sufs = [ suf ] * nmodels
	for i in range(nmodels):
		if nmodels > 1:
			sufs[i] += "_" + models[i]
	
	inputs = pd.DataFrame(sufs, columns = ['suffix'])
	inputs["slvmode"] = models
	
	print(models)
	
	inputs["cfgfile"] = cfgf
	inputs["invcfg"] = icfgf
	inputs["outdir"] = wkdir
	inputs["ext"] = ext
	inputs["iwbic"] = True
	
	outputs = parallelrun(inputs, maxcore)
	
	resf = outputs.resfile[0]
	score='wbic'
	
	df = outputs.sort_values(score, ascending=True).reset_index(drop=True)
	resf = df.resfile[0]
	
	if nmodels > 1:
		bestfile = os.path.basename(resf)
		dfl = os.path.abspath(resf)
		dfl = os.path.dirname(dfl)
		
		# to summarize the results
		df = df.loc[:,[score,"slvmode","ave_loglike","resfile"]]
		print(df)
		of = wkdir + "searchres-%s.csv" % filebase
		df.to_csv(of)
		
		dfplot = df.sort_values("slvmode", ascending=True).reset_index(drop=True)
		wbicf = wkdir + "fig_searchres-%s.png" % filebase
		plt.title(filebase)
		plt.grid()
		#plt.xticks(models)
		plt.xlabel("tested models")
		plt.ylabel("relative WBIC value")
		plt.plot(dfplot.slvmode, dfplot.wbic)
		plt.scatter(dfplot.slvmode, dfplot.wbic, marker="o")
		plt.savefig(wbicf, bbox_inches='tight', pad_inches=0.1)
		plt.close()
		
		# to calc the posterior for beta = 1
		mode = df.slvmode[0]
		suffix = suf + "_" + mode
		print("\nPreferred model is " + mode + "\n")
		
		odir = outdir+"/"
		[resf, wbic, aveloglike] = garpos_mcmc(cfgf, icfgf, odir, suffix, mode, ext, False)
	
	return resf
	
