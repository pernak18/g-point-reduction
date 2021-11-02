#!/usr/bin/env python

import os, sys, shutil, glob, pickle, copy

os.chdir('/global/u1/p/pernak18/RRTMGP/g-point-reduction')
os.chdir('/global/u2/k/kcadyper/g-point-reduction')

# "standard" install
import numpy as np

# directory in which libraries installed with conda are saved
PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'lib/python3.8/site-packages'
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

GARAND = '/global/project/projectdirs/e3sm/pernak18/reference_netCDF/' + \
    'g-point-reduce/multi_garand_template_single_band.nc'
BYBAND.pathCheck(GARAND)

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

CWD = os.getcwd()
print (CWD)

# only do one domain or the other
DOLW = True
DOSW = not DOLW
DOMAIN = 'LW' if DOLW else 'SW'
NBANDS = 16 if DOLW else 14

# remove the netCDFs that are generated for all of the combinations 
# and iterations of combinations in bandOptimize()
CLEANUP = True

kFiles = sorted(glob.glob('{}/coefficients_{}_band??.nc'.format(
        BANDSPLITDIR, DOMAIN)))
print (len(kFiles))

test = True

# this should be parallelized; also is part of preprocessing so we 
# shouldn't have to run it multiple times
kBandDict = {}
for iBand, kFile in enumerate(kFiles):
    print(iBand)
    print (kFile)
    #if test and iBand != 0: continue
    band = iBand + 1
    kObj = BYBAND.gCombine_kDist(kFile, iBand, DOLW, 1, 
        fullBandKDir=BANDSPLITDIR, 
        fullBandFluxDir=FULLBANDFLUXDIR, cleanup=CLEANUP)
    kObj.gPointCombine()
    kBandDict['band{:02d}'.format(band)] = kObj

    print('Band {} complete'.format(band))
# end kFile loop

with open('{}_k-dist.pickle'.format(DOMAIN), 'wb') as fp: pickle.dump(kBandDict, fp)
