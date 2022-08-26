#!/usr/bin/env python

import os, sys, glob, pickle

PATHS = ['common']
for path in PATHS: sys.path.append(path)

# needed at AER unless i update `pandas`
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# local modules
import g_point_reduction as REDUX
from rrtmgp_cost_compute import flux_cost_compute as FCC

GARAND = '/global/project/projectdirs/e3sm/pernak18/reference_netCDF/' + \
    'g-point-reduce/multi_garand_template_single_band.nc'
FCC.pathCheck(GARAND)

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
    #if test and iBand != 0: continue
    band = iBand + 1
    kObj = REDUX.gCombine_kDist(kFile, iBand, DOLW, 1, 
        fullBandKDir=BANDSPLITDIR, 
        fullBandFluxDir=FULLBANDFLUXDIR, cleanup=CLEANUP)
    kObj.gPointCombine()
    kBandDict['band{:02d}'.format(band)] = kObj

    print('Band {} complete'.format(band))
# end kFile loop

with open('{}_k-dist.pickle'.format(DOMAIN), 'wb') as fp:
  pickle.dump(kBandDict, fp)
