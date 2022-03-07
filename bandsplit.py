#!/usr/bin/env python

import os, sys, shutil, glob

# "standard" install
import numpy as np

from multiprocessing import Pool

# directory in which libraries installed with conda are saved
PIPPATH = '/global/homes/k/kcadyper/.local/lib/python3.8/site-packages/'
# PIPPATH = '/global/homes/e/emlawer/.local/cori/3.8-anaconda-2020.11/' + \
#     'lib/python3.8/site-packages'
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

# only do one domain or the other
DOLW = True
DOMAIN = 'LW' if DOLW else 'SW'
NBANDS = 16 if DOLW else 14

PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
#EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
#    PROJECT)
REFDIR = '{}/reference_netCDF/g-point-reduce'.format(PROJECT)

# test (RRTMGP) and reference (LBL) flux netCDF files, full k-distributions, 
# and by-band Garand input file
fluxSuffix = 'flux-inputs-outputs-garandANDpreind.nc'
if DOLW:
    GARAND = '{}/multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-lw-g256-2018-12-04.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-lw-g256-jen-xs.nc'.format(REFDIR)
else:
    GARAND = '{}/charts_multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-sw-g224-2018-12-04.nc'.format(REFDIR)
# endif LW

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

PATHS = [GARAND, KFULLNC]
for PATH in PATHS: BYBAND.pathCheck(PATH)

print('Band splitting commenced')
BYBAND.pathCheck(BANDSPLITDIR, mkdir=True)
BYBAND.pathCheck(FULLBANDFLUXDIR, mkdir=True)

for iBand in range(NBANDS):
    print('Splitting Band {:d} of {:d}'.format(iBand+1, NBANDS))

    # divide full k-distribution into subsets for each band
    kObj = BYBAND.gCombine_kDist(KFULLNC, iBand, DOLW, 1, 
        fullBandKDir=BANDSPLITDIR, fullBandFluxDir=FULLBANDFLUXDIR, 
        profilesNC=GARAND)
    kObj.kDistBand()
    print('k distribution split complete')

    # quick, non-parallelized flux calculations (because the 
    # executable is run in one directory)
    # TO DO: HAVEN'T TESTED THIS SINCE IT HAS BEEN MOVED OUT OF THE CLASS
    BYBAND.fluxCompute(kObj.kBandNC, kObj.profiles, kObj.exe, 
                       kObj.fullBandFluxDir, kObj.fluxBandNC)
    print('Flux split complete')
# end band loop

print('Band splitting completed')
