#!/usr/bin/env python

import os, sys, shutil, glob, pickle, copy, time

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
CFCOMPS = ['band_flux_up', 'band_flux_dn']
CFLEVS = [0, 26, 42] # pressure levels of interest in Pa
CFWGT = [0.5, 0.5]

fullBandFluxes = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
        FULLBANDFLUXDIR, DOMAIN)))

with open('temp.pickle', 'rb') as fp: kBandDict = pickle.load(fp)

CFDIR = 'sfc_tpause_TOA_band_flux_up_down'

coObj = BYBAND.gCombine_Cost(
    kBandDict, fullBandFluxes, REFNC, TESTNC, 
    IFORCING, 1, profilesNC=GARAND, exeRRTMGP=EXE, 
    cleanup=CLEANUP, 
    costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
    costWeights=CFWGT, test=False, optDir='./{}'.format(CFDIR))

NITER = 2
DIAGNOSTICS = True
for i in range(1, NITER+1):
    t1 = time.process_time()

    print('Iteration {}'.format(i))
    temp = time.process_time()
    coObj.kMap()
    print('kMap: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxComputePool()
    print('Flux Compute: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxCombine()
    print('Flux Combine: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.costFuncComp(init=True)
    coObj.costFuncComp()
    print('Cost function computation: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.findOptimal()
    print('findOptimal: {:.4f}'.format(time.process_time()-temp))

    if coObj.optimized: break
    if DIAGNOSTICS: coObj.costDiagnostics()

    coObj.setupNextIter()

    print('Full iteration: {:.4f}'.format(time.process_time()-t1))
    coObj.calcOptFlux(
        KFULLNC, fluxOutNC='optimized_fluxes_iter{:03d}.nc'.format(i))
# end iteration loop

t1 = time.process_time()
coObj.calcOptFlux(KFULLNC)
print('New k-file {:.4f}'.format(time.process_time()-t1))

print('Optimization complete')