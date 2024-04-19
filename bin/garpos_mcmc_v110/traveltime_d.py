"""
Created:
	07/08/2022 by S. Watanabe
"""
import sys
import math
import ctypes
import numpy as np
from scipy import interpolate

def calc_snell(shotdat, mp, nMT, icfg, svp, height0, 
               ndist, ddep, distmargin=2.0, depmargin=1.5):
	
	# fortran library
	libdir = icfg.get("Inv-parameter","lib_directory")
	lib_raytrace = icfg.get("Inv-parameter","lib_raytrace")
	
	dep0 = svp.depth.values[-1]
	t0 = np.linspace(0., distmargin, ndist)
	dst = t0 * dep0
	
	# calc depth-averaged sound slowness (characteristic length/time)
	vl = svp.speed.values
	dl = svp.depth.values
	aveslyr = [ (1./vl[i+1]+1./vl[i])*(dl[i+1]-dl[i])/2. for i in svp.index[:-1]]
	S0 = np.array(aveslyr).sum()/(dl[-1]-dl[0])
	
	# station depth (from height0)
	sta0_u = [mp[iMT*3+2] + mp[nMT*3+2] for iMT in range(nMT)]
	depmax = math.ceil(max(sta0_u)+depmargin)
	depmin = math.floor(min(sta0_u)-depmargin)
	depths = np.arange(depmin,depmax,ddep)
	
	# surface unit depth
	u0 = shotdat.ant_u0.values + shotdat.plu0.values
	u1 = shotdat.ant_u1.values + shotdat.plu1.values
	surf  = np.append(u0, u1)
	surfmax = math.ceil(max(surf)+1.)
	surfmin = math.floor(min(surf)-1.)
	surfs = np.arange(surfmin,surfmax,ddep)
	
	SnellTT3d = np.zeros((len(surfs),len(depths),ndist))
	DirectTT3d = np.zeros((len(surfs),len(depths),ndist))
	
	for i, surf in enumerate(surfs):
		for j, dep in enumerate(depths):
			
			yd  = np.zeros(len(dst)) + dep
			ys  = np.zeros(len(dst)) + surf
			dsv = np.zeros(len(dst))
			
			# sv layer
			l_depth = svp.depth.values
			l_speed = svp.speed.values
			
			#######################################
			# for call f90 library (calc 1way TT) #
			#######################################
			nl = len(l_depth)
			nn = ctypes.byref(ctypes.c_int32(ndist))
			nl = ctypes.byref(ctypes.c_int32(nl))
			
			# output
			ctm = np.zeros_like(dst)
			cag = np.zeros_like(dst)
			f90 = np.ctypeslib.load_library(lib_raytrace, libdir)
			f90.raytrace_.argtypes = [
				ctypes.POINTER(ctypes.c_int32), # n
				ctypes.POINTER(ctypes.c_int32), # nlyr
				np.ctypeslib.ndpointer(dtype=np.float64), # l_depth
				np.ctypeslib.ndpointer(dtype=np.float64), # l_speed
				np.ctypeslib.ndpointer(dtype=np.float64), # dist
				np.ctypeslib.ndpointer(dtype=np.float64), # yd
				np.ctypeslib.ndpointer(dtype=np.float64), # ys
				np.ctypeslib.ndpointer(dtype=np.float64), # dsv
				np.ctypeslib.ndpointer(dtype=np.float64), # ctm (output)
				np.ctypeslib.ndpointer(dtype=np.float64)  # cag (output)
				]
			f90.raytrace_.restype = ctypes.c_void_p
			f90.raytrace_(nn, nl, l_depth, l_speed, dst, yd, ys, dsv, ctm, cag)
			
			ScalTime  = np.array(ctm)
			SnellTT3d[i,j,:] = ScalTime
			
			#############################
			# Direct Travel-time
			#############################
			
			DcalTime  = (dst**2. + (surf-dep)**2.)**0.5 * S0
			DirectTT3d[i,j,:] = DcalTime
	
	DiffTT = SnellTT3d - DirectTT3d
	
	surfs = np.array(surfs)
	depths = np.array(depths)
	dists = np.array(dst)
	
	snell_interpolate = interpolate.RegularGridInterpolator((surfs, depths, dists), DiffTT, method="linear")
	
	return S0, snell_interpolate


def calc_traveltime(shotdat, mp, nMT, S0, Snell_interpolate):
	"""
	Calculate the round-trip travel time.
	
	Parameters
	----------
	shotdat : DataFrame
		GNSS-A shot dataset.
	mp : ndarray
		complete model parameter vector.
	nMT : int
		number of transponders.
	S0 : float
		Average sound slowness.
	
	Returns
	-------
	calTT : ndarray
		Calculated travel time (sec.).
	calA0 : ndarray
		Calculated take-off angle (degree).
	"""
	
	ndat = len(shotdat.index)
	
	# station pos
	sta0_e = mp[shotdat['mtid']+0] + mp[nMT*3+0]
	sta0_n = mp[shotdat['mtid']+1] + mp[nMT*3+1]
	sta0_u = mp[shotdat['mtid']+2] + mp[nMT*3+2]
	
	e0 = shotdat.ant_e0.values + shotdat.ple0.values
	n0 = shotdat.ant_n0.values + shotdat.pln0.values
	u0 = shotdat.ant_u0.values + shotdat.plu0.values
	e1 = shotdat.ant_e1.values + shotdat.ple1.values
	n1 = shotdat.ant_n1.values + shotdat.pln1.values
	u1 = shotdat.ant_u1.values + shotdat.plu1.values
	
	dist0 = ((e0 - sta0_e)**2. + (n0 - sta0_n)**2.)**0.5
	dist1 = ((e1 - sta0_e)**2. + (n1 - sta0_n)**2.)**0.5
	
	rng0 = (dist0**2. + (u0 - sta0_u)**2.)**0.5
	rng1 = (dist1**2. + (u1 - sta0_u)**2.)**0.5
	angle0 = np.arctan(dist0/np.abs(u0-sta0_u))
	angle1 = np.arctan(dist1/np.abs(u1-sta0_u))
	
	par0 = np.array([u0, sta0_u, dist0]).T
	par1 = np.array([u1, sta0_u, dist1]).T
	
	directTT0 = rng0 * S0 + Snell_interpolate(par0)
	directTT1 = rng1 * S0 + Snell_interpolate(par1)
	
	calTT = directTT0 + directTT1
	calA0 = (angle0 + angle1)/2. * 180./math.pi
	
	return calTT, calA0
	
