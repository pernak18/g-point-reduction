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

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

for PATH in PATHS: BYBAND.pathCheck(PATH)

CWD = os.getcwd()

# only do one domain or the other
DOLW = False
DOSW = not DOLW
DOMAIN = 'LW' if DOLW else 'SW'
NBANDS = 16 if DOLW else 14

# test (RRTMGP) and reference (LBL) flux netCDF files, full k-distributions, 
# and by-band Garand input file
fluxSuffix = 'flux-inputs-outputs-garandANDpreind.nc'
if DOLW:
    GARAND = '{}/multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-lw-g256-2018-12-04.nc'.format(REFDIR)
    REFNC = '{}/lblrtm-lw-{}'.format(REFDIR, fluxSuffix)
    TESTNC = '{}/rrtmgp-lw-{}'.format(REFDIR, fluxSuffix)
    #TESTNC = '{}/profile-stats-plots/rrtmgp-lw-flux-inputs-outputs-garand-all.nc'.format(REFDIR)
    #TESTNC = '{}/g-point-reduce/rrtmgp-lw-flux-inputs-outputs-garandANDpreind.nc'.format(REFDIR)
else:
    GARAND = '{}/charts_multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-sw-g224-2018-12-04.nc'.format(REFDIR)
    REFNC = '{}/charts-sw-{}'.format(REFDIR, fluxSuffix)
    TESTNC = '{}/rrtmgp-sw-{}'.format(REFDIR, fluxSuffix)
# endif LW

PATHS = [KFULLNC, EXE, TESTNC, REFNC, GARAND]

# does band-splitting need to be done, or are there existing files 
# that have divided up the full k-distribution?
BANDSPLIT = True

# remove the netCDFs that are generated for all of the combinations 
# and iterations of combinations in bandOptimize()
CLEANUP = False

# number of iterations for the optimization
NITER = 1

CFCOMPS = ['band_flux_up', 'band_flux_dn']
CFCOMPS = ['heating_rate', 'flux_net', 'flux_net_forcing_2']
CFCOMPS = ['flux_dir_dn', 'flux_dif_dn']

# level indices for each component 
# (e.g., 0 for surface, 41 for Garand TOA)
# one dictionary key per component so each component
# can have its own set of level indices
CFLEVS = {}
# CFLEVS['heating_rate'] = range(41)
# CFLEVS['flux_net'] = [0, 26, 42]
# CFLEVS['flux_net_forcing'] = [0, 26, 42]
CFLEVS['flux_dir_dn'] = [0, 26]
CFLEVS['flux_dif_dn'] = [26, 42]

# weights for each cost function component
CFWGT = [0.5, 0.5]

fullBandFluxes = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
        FULLBANDFLUXDIR, DOMAIN)))

with open('k-dist.pickle', 'rb') as fp: kBandDict = pickle.load(fp)

CFDIR = 'sfc_tpause_TOA_band_flux_up_down'
CFDIR = 'salami'
CFDIR = 'direct-down'

RESTORE = True
pickleCost = 'cost-optimize.pickle'

if RESTORE:
    assert os.path.exists(pickleCost), 'Cannot find {}'.format(pickleCost)
    print('Restoring {}'.format(pickleCost))
    with open(pickleCost, 'rb') as fp: coObj = pickle.load(fp)
else:
    coObj = BYBAND.gCombine_Cost(
        kBandDict, fullBandFluxes, REFNC, TESTNC, 1, DOLW, 
        profilesNC=GARAND, exeRRTMGP=EXE, cleanup=CLEANUP, 
        costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
        costWeights=CFWGT, test=False, optDir='./{}'.format(CFDIR))
# endif RESTORE

NITER = 2
DIAGNOSTICS = True
for i in range(coObj.iCombine, NITER+1):
    t1 = time.process_time()

    wgtInfo = ['{:.2f} ({})'.format(
        wgt, comp) for wgt, comp in zip(CFWGT, CFCOMPS)]
    wgtInfo = ' '.join(wgtInfo)
    print('Weights: {}'.format(wgtInfo))
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

    temp = time.process_time()
    with open(pickleCost, 'wb') as fp: pickle.dump(coObj, fp)
    print('Pickle: {:.4f}'.format(time.process_time()-temp))

    print('Full iteration: {:.4f}'.format(time.process_time()-t1))
    coObj.calcOptFlux(
        fluxOutNC='optimized_fluxes_iter{:03d}.nc'.format(i))
# end iteration loop

t1 = time.process_time()
#coObj.kDistOpt(KFULLNC)
coObj.calcOptFlux()
print('New k-file {:.4f}'.format(time.process_time()-t1))

print('Optimization complete')