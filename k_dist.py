#!/usr/bin/env python

import os, sys, shutil, glob, pickle, copy

os.chdir('/global/u1/p/pernak18/RRTMGP/g-point-reduction')

# "standard" install
import numpy as np

from multiprocessing import Pool

# directory in which libraries installed with conda are saved
PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
PATHS = ['common', PIPPATH]
for path in PATHS: sys.path.append(path)

# needed at AER unless i update `pandas`
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# user must do `pip install xarray` on cori (or other NERSC machines)
import xarray as xa

# local module
import by_band_lib as BYBAND

PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
REFDIR = '{}/reference_netCDF/g-point-reduce'.format(PROJECT)

KFULLNC = '{}/rrtmgp-data-lw-g256-2018-12-04.nc'.format(REFDIR)
GARAND = '{}/multi_garand_template_single_band.nc'.format(REFDIR)

# test (RRTMGP) and reference (LBL) flux netCDF files
TESTNC = '{}/rrtmgp-lw-flux-inputs-outputs-garandANDpreind.nc'.format(REFDIR)
REFNC = '{}/lblrtm-lw-flux-inputs-outputs-garandANDpreind.nc'.format(REFDIR)
PATHS = [KFULLNC, EXE, TESTNC, REFNC, GARAND]

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

for PATH in PATHS: BYBAND.pathCheck(PATH)

CWD = os.getcwd()

# only do one domain or the other
DOLW = True
DOSW = not DOLW
DOMAIN = 'LW' if DOLW else 'SW'
NBANDS = 16 if DOLW else 14

# forcing scenario (0 is no forcing...need a more comprehensive list)
IFORCING = 0

# does band-splitting need to be done, or are there existing files 
# that have divided up the full k-distribution?
BANDSPLIT = False

# remove the netCDFs that are generated for all of the combinations 
# and iterations of combinations in bandOptimize()
CLEANUP = False

# number of iterations for the optimization
NITER = 1

# cost function variables
CFCOMPS = ['band_flux_dn', 'band_flux_up']
CFLEVS = [0, 10000, 102000] # pressure levels of interest in Pa
CFWGT = [0.5, 0.5]

kFiles = sorted(glob.glob('{}/coefficients_{}_band??.nc'.format(
        BANDSPLITDIR, DOMAIN)))

test = False

# this should be parallelized; also is part of preprocessing so we 
# shouldn't have to run it multiple times
kBandDict = {}
for iBand, kFile in enumerate(kFiles):
    if test and iBand != 0: continue
    band = iBand + 1
    kObj = BYBAND.gCombine_kDist(kFile, iBand, DOLW, 1, 
        fullBandKDir=BANDSPLITDIR, 
        fullBandFluxDir=FULLBANDFLUXDIR, cleanup=CLEANUP)
    kObj.gPointCombine()
    kBandDict['band{:02d}'.format(band)] = kObj

    print('Band {} complete'.format(band))
# end kFile loop

#with open('temp.pickle', 'wb') as fp: pickle.dump(kBandDict, fp)