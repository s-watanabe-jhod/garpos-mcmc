#!/usr/bin/env python
"""
Created:
	08/13/2024 by S. Watanabe
"""
from optparse import OptionParser
import os
import sys

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from garpos_mcmc_v200.garpos_wbic import model_search_wbic

if __name__ == '__main__':

	######################################################################
	usa = u"Usage: %prog [options] "
	modetxt = u"  m100: double-grad\n  m101: single-grad \n  m102: alpha2-offset"
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
	opt.add_option( "--suffix", action="store", type="string",
					default="", dest="suf",
					help=u"Set suffix for result files"
					)
	opt.add_option( "--ext", action="store", type="string", 
					default="png", dest="ext",
					help=u'Set extention for residual plot'
					)
	(options, args) = opt.parse_args()
	#####################################################################
	
	if not os.path.isfile(options.invcfg) or options.invcfg == "":
		print("NOT FOUND (setup file) :: %s" % options.invcfg)
		sys.exit(1)
	if not os.path.isfile(options.cfgfile) or options.cfgfile == "":
		print("NOT FOUND (site parameter file) :: %s" % options.cfgfile)
		sys.exit(1)
	
	ext = options.ext
	odir = options.directory
	rf = model_search_wbic(options.cfgfile, options.invcfg, odir, options.suf, ext)

	exit()
