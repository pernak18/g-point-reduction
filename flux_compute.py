#!/usr/bin/env python

import os, sys, shutil, glob, pickle, copy, time

os.chdir('/global/u2/k/kcadyper/g-point-reduction/')

# "standard" install
import numpy as np

from multiprocessing import Pool

import pathlib as PL

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

PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
REFDIR = '{}/reference_netCDF/g-point-reduce'.format(PROJECT)

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

for PATH in PATHS: BYBAND.pathCheck(PATH)

CWD = os.getcwd()

# only do one domain or the other
DOLW = True
DOMAIN = 'LW' if DOLW else 'SW'
NBANDS = 16 if DOLW else 14

# test (RRTMGP) and reference (LBL) flux netCDF files, full k-distributions, 
# and by-band Garand input file
fluxSuffix = 'flux-inputs-outputs-garandANDpreind.nc'
if DOLW:
    GARAND = '{}/multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-lw-g256-2018-12-04.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-lw-g256-jen-xs.nc'.format(REFDIR)
    REFNC = '{}/lblrtm-lw-{}'.format(REFDIR, fluxSuffix)
    TESTNC = '{}/rrtmgp-lw-{}'.format(REFDIR, fluxSuffix)
    #TESTNC = './io.nc'
    #TESTNC = '{}/profile-stats-plots/rrtmgp-lw-flux-inputs-outputs-garand-all.nc'.format(REFDIR)
    #TESTNC = '{}/g-point-reduce/rrtmgp-lw-flux-inputs-outputs-garandANDpreind.nc'.format(REFDIR)
else:
    GARAND = '{}/charts_multi_garand_template_single_band.nc'.format(REFDIR)
    KFULLNC = '{}/rrtmgp-data-sw-g224-2018-12-04.nc'.format(REFDIR)
    REFNC = '{}/charts-sw-{}'.format(REFDIR, fluxSuffix)
    TESTNC = '{}/rrtmgp-sw-{}'.format(REFDIR, fluxSuffix)
# endif LW

PATHS = [KFULLNC, EXE, TESTNC, REFNC, GARAND]

# remove the netCDFs that are generated for all of the combinations 
# and iterations of combinations in bandOptimize()
CLEANUP = False

CFCOMPS = ['band_flux_up', 'band_flux_dn']
CFCOMPS = ['flux_net_forcing_19', 'flux_dir_dn', 'flux_dif_dn', 'flux_net', 'flux_net_forcing_5']
CFCOMPS = ['flux_net', 'band_flux_net', 'heating_rate',
  'heating_rate_7', 'flux_net_forcing_5', 'flux_net_forcing_6',
  'flux_net_forcing_7', 'flux_net_forcing_9', 'flux_net_forcing_10',
  'flux_net_forcing_11', 'flux_net_forcing_12', 'flux_net_forcing_13',
  'flux_net_forcing_14', 'flux_net_forcing_15', 'flux_net_forcing_16',
  'flux_net_forcing_17', 'flux_net_forcing_18']
#CFCOMPS = ['flux_dif_net', 'flux_dir_dn', 'heating_rate']
# CFCOMPS = ['flux_net','flux_net_forcing_8','flux_net_forcing_9','flux_net_forcing_10',
#            'flux_net_forcing_11','flux_net_forcing_12','flux_net_forcing_13','flux_net_forcing_14',
#            'flux_net_forcing_15','flux_net_forcing_16','flux_net_forcing_17']

# level indices for each component 
# (e.g., 0 for surface, 41 for Garand TOA)
# one dictionary key per component so each component
# can have its own set of level indices
CFLEVS = {}
#CFLEVS['heating_rate'] = range(41)
#CFLEVS['flux_up'] = [26, 42]
LEVELS = {}
LEVELS['flux_net'] = [0, 26, 42]
LEVELS['band_flux_net'] = [42]
LEVELS['heating_rate'] = range(42)
LEVELS['heating_rate_7'] = range(42)
LEVELS['flux_net_forcing_5'] = [0, 26, 42]
LEVELS['flux_net_forcing_6'] = [0, 26, 42]
LEVELS['flux_net_forcing_7'] = [0, 26, 42]
LEVELS['flux_net_forcing_9'] = [0, 26, 42]
LEVELS['flux_net_forcing_10'] = [0, 26, 42]
LEVELS['flux_net_forcing_11'] = [0, 26, 42]
LEVELS['flux_net_forcing_12'] = [0, 26, 42]
LEVELS['flux_net_forcing_13'] = [0, 26, 42]
LEVELS['flux_net_forcing_14'] = [0, 26, 42]
LEVELS['flux_net_forcing_15'] = [0, 26, 42]
LEVELS['flux_net_forcing_16'] = [0, 26, 42]
LEVELS['flux_net_forcing_17'] = [0, 26, 42]
LEVELS['flux_net_forcing_18'] = [0, 26, 42]
CFLEVS = dict(LEVELS)
#CFLEVS['flux_up_forcing_7'] = [26, 42]
# CFLEVS['flux_dif_net'] = [0, 26, 42]
# CFLEVS['flux_dir_dn'] = [0, 26]
# CFLEVS['heating_rate'] = range(41)
# for comp in CFCOMPS: CFLEVS[comp] = [0, 26, 42]

# weights for each cost function component
CFWGT = [0.6, 0.04, 0.12, 0.12, 0.01, 0.02, 0.04, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005]
#CFWGT = [0.99] + [0.001] * 10

"""
CFCOMPS = ['flux_net','band_flux_net','heating_rate','heating_rate_7',
           'flux_net_forcing_5','flux_net_forcing_6','flux_net_forcing_7',
           'flux_net_forcing_9','flux_net_forcing_10','flux_net_forcing_11',
           'flux_net_forcing_12','flux_net_forcing_13','flux_net_forcing_14',
           'flux_net_forcing_15','flux_net_forcing_16','flux_net_forcing_17',
           'flux_net_forcing_18']
# level indices for each component
# (e.g., 0 for surface, 41 for Garand TOA)
# one dictionary key per component so each component
# can have its own set of level indices
CFLEVS = {}
CFLEVS['flux_net'] = [0, 26, 42]
CFLEVS['band_flux_net'] = [42]
CFLEVS['heating_rate'] = list(range(41))
CFLEVS['heating_rate_7'] = list(range(41))
CFLEVS['flux_net_forcing_5'] = [0, 26, 42]
CFLEVS['flux_net_forcing_6'] = [0, 26, 42]
CFLEVS['flux_net_forcing_7'] = [0, 26, 42]
CFLEVS['flux_net_forcing_9'] = [0, 26, 42]
CFLEVS['flux_net_forcing_10'] = [0, 26, 42]
CFLEVS['flux_net_forcing_11'] = [0, 26, 42]
CFLEVS['flux_net_forcing_12'] = [0, 26, 42]
CFLEVS['flux_net_forcing_13'] = [0, 26, 42]
CFLEVS['flux_net_forcing_14'] = [0, 26, 42]
CFLEVS['flux_net_forcing_15'] = [0, 26, 42]
CFLEVS['flux_net_forcing_16'] = [0, 26, 42]
CFLEVS['flux_net_forcing_17'] = [0, 26, 42]
CFLEVS['flux_net_forcing_18'] = [0, 26, 42]
# weights for each cost function component
CFWGT = [0.6, 0.04, 0.12, 0.12,
         0.01, 0.02, 0.04,
        0.005, 0.005, 0.005,
        0.005, 0.005, 0.005,
        0.005, 0.005, 0.005,
        0.005]
"""

# Modified g-point weighting used when cost function starts to increase
xWeight = 0.1

fullBandFluxes = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
        FULLBANDFLUXDIR, DOMAIN)))

#with open('{}_k-dist.pickle'.format(DOMAIN), 'rb') as fp: kBandDict = pickle.load(fp)
with open('{}/{}_k-dist.pickle'.format(os.getcwd(), DOMAIN), 'rb') as fp: kBandDict = pickle.load(fp)

CFDIR = 'sfc_tpause_TOA_band_flux_up_down'
CFDIR = 'salami'
CFDIR = 'direct-down'
CFDIR = 'xsecs_test_eli'

RESTORE = False
pickleCost = '{}_cost-optimize.pickle'.format(DOMAIN)

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

# number of iterations for the optimization
coSave = ' '
NITER = 140
print (coObj.iCombine)
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
#     print('kMap: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxComputePool()
#     print('Flux Compute: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxCombine()
#     print('Flux Combine: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.costFuncComp(init=True)
    coObj.costFuncComp()
#     print('Cost function computation: {:.4f}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.findOptimal()
#     print('findOptimal: {:.4f}'.format(time.process_time()-temp))


    print(len(coObj.totalCost))
    print(coObj.dCost[coObj.iOpt])

    if coObj.optimized: break
    if DIAGNOSTICS: coObj.costDiagnostics()

    import copy
    print(coObj.dCost[coObj.iOpt]-coObj.deltaCost0)

# Start of special g-point combination branch
    #if coObj.dCost[coObj.iOpt]-coObj.deltaCost0 > 0.0:
    if coObj.dCost[coObj.iOpt]-coObj.deltaCost0 > -2.01:
       print ('will change here')
       delta0 = coObj.dCost[coObj.iOpt]-coObj.deltaCost0
       print (delta0)
       print (type(delta0))
       bandObj = coObj.distBands
       if (coObj.optBand+1) < 10:
           bandKey='band0{}'.format(coObj.optBand+1)
       else:
           bandKey='band{}'.format(coObj.optBand+1)
       #sys.exit() 
       newObj = BYBAND.gCombine_kDist(bandObj[bandKey].kInNC, coObj.optBand, DOLW,
            bandObj[bandKey].iCombine, fullBandKDir=BANDSPLITDIR,
            fullBandFluxDir=FULLBANDFLUXDIR)
       curkFile = os.path.basename(coObj.optNC)
       ind = curkFile.find('_g')
       g1 = int(curkFile[ind+2:ind+4])
       g2 = int(curkFile[ind+5:ind+7])
       gCombine =[[g1-1,g2-1]]

       print  ("   ")
       print (newObj.workDir)
       ind = curkFile.find('coeff')
       print (curkFile)
       fluxPath = PL.Path(curkFile[ind:]).with_suffix('')
       fluxDir = '{}/{}'.format(newObj.workDir,fluxPath)

       parr =['plus','minus']
       for pmFlag in parr:
           coCopy = copy.deepcopy(coObj)
           print ("  ")
           print (pmFlag)
           newObj.gPointCombineSglPair(pmFlag,gCombine,xWeight)
           newCoefFile = '{}/{}_{}.nc'.format(newObj.workDir,fluxPath,pmFlag)
           fluxFile = os.path.basename(newCoefFile).replace('coefficients', 'flux')
           print (fluxFile)
           print (newCoefFile)
           BYBAND.fluxCompute(newCoefFile,GARAND,EXE,fluxDir,fluxFile)

           trialNC = '{}/{}'.format(fluxDir,fluxFile)
           coCopy.combinedDS[coObj.iOpt] = BYBAND.combineBandsSgl( 
                   coObj.optBand, DOLW,trialNC,coObj.fullBandFluxes)
           coCopy.costFuncCompSgl(coCopy.combinedDS[coObj.iOpt])
           coCopy.findOptimal()
        
           print ("len total cost", "dcost")
           print(len(coCopy.totalCost))
           print(coCopy.dCost[coObj.iOpt])
           if DIAGNOSTICS: coCopy.costDiagnostics()
           print ("delta cost")
           print(coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0)
           if(pmFlag == 'plus'):
               deltaPlus = coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0
           if(pmFlag == 'minus'):
               deltaMinus = coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0

# Fit cost of three g-point options to a parabola and find minimum weight
       print (delta0,deltaPlus,deltaMinus)
       xArr = [-xWeight,0.,xWeight]
       yArr = [deltaMinus,delta0,deltaPlus]
       coeff = np.polyfit(xArr,yArr,2)
       xWeightNew = -coeff[1]/(2.*coeff[0])
       print (xWeightNew)

# Define newest g-point combination
       coCopy = copy.deepcopy(coObj)
       pmFlag = 'mod'
       print ("  ")
       print (pmFlag)
       newObj.gPointCombineSglPair(pmFlag,gCombine,xWeightNew)
       newCoefFile = '{}/{}_{}.nc'.format(newObj.workDir,fluxPath,pmFlag)
       fluxFile = os.path.basename(newCoefFile).replace('coefficients', 'flux')
       print (fluxFile)
       print (newCoefFile)
       BYBAND.fluxCompute(newCoefFile,GARAND,EXE,fluxDir,fluxFile)

       trialNC = '{}/{}'.format(fluxDir,fluxFile)
       coCopy.combinedDS[coObj.iOpt] = BYBAND.combineBandsSgl( 
               coObj.optBand, DOLW,trialNC,coObj.fullBandFluxes)
       coCopy.costFuncCompSgl(coCopy.combinedDS[coObj.iOpt])
       coCopy.findOptimal()
    
       print ("len total cost", "dcost")
       print(len(coCopy.totalCost))
       print(coCopy.dCost[coObj.iOpt])
       if DIAGNOSTICS: coCopy.costDiagnostics()
       print ("delta cost")
       print(coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0)
       coObj = copy.deepcopy(coCopy)
       del coCopy
# End of spoecial g-point combination branch
    
    coObj.setupNextIter()

    temp = time.process_time()
    with open(pickleCost, 'wb') as fp: pickle.dump(coObj, fp)
#     print('Pickle: {:.4f}'.format(time.process_time()-temp))

#     print('Full iteration: {:.4f}'.format(time.process_time()-t1))
    coObj.calcOptFlux(
        fluxOutNC='optimized_fluxes_iter{:03d}.nc'.format(i))
# end iteration loop

t1 = time.process_time()
KOUTNC = 'rrtmgp-data-{}-g-red.nc'.format(DOMAIN)

ncFiles = [coObj.distBands[key].kInNC for key in coObj.distBands.keys()]
# kDistOpt(KFULLNC, kOutNC=KOUTNC)
coObj.kDistOpt(KFULLNC, kOutNC=KOUTNC)
coObj.calcOptFlux()
# print('New k-file {:.4f}'.format(time.process_time()-t1))

print('Optimization complete')
