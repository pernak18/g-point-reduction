#!/usr/bin/env python

import os, sys, glob, pickle, copy, time

# "standard" install
import numpy as np

import pathlib as PL

# directory in which libraries installed with conda are saved
PATHS = ['common']
for path in PATHS: sys.path.append(path)

# needed at AER unless i update `pandas`
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# local modules
import g_point_reduction as REDUX
import flux_cost_compute as FCC

PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
REFDIR = '{}/reference_netCDF/g-point-reduce'.format(PROJECT)

BANDSPLITDIR = 'band_k_dist'
FULLBANDFLUXDIR = 'full_band_flux'

for PATH in PATHS: FCC.pathCheck(PATH)

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
CFLEVS = {}

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

# weights for each cost function component
CFWGT = [0.6, 0.04, 0.12, 0.12, 0.01, 0.02, 0.04, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005]

fullBandFluxes = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
        FULLBANDFLUXDIR, DOMAIN)))

with open('{}/{}_k-dist.pickle'.format(os.getcwd(), DOMAIN), 'rb') as fp: kBandDict = pickle.load(fp)

CFDIR = 'sfc_tpause_TOA_band_flux_up_down'
CFDIR = 'salami'
CFDIR = 'direct-down'
CFDIR = 'xsecs_test_eli'

RESTORE = True
pickleCost = '{}_cost-optimize.pickle'.format(DOMAIN)

if RESTORE:
    assert os.path.exists(pickleCost), 'Cannot find {}'.format(pickleCost)
    print('Restoring {}'.format(pickleCost))
    with open(pickleCost, 'rb') as fp: coObj = pickle.load(fp)
else:
    coObj = FCC.gCombine_Cost(
        kBandDict, fullBandFluxes, REFNC, TESTNC, 1, DOLW, 
        profilesNC=GARAND, exeRRTMGP=EXE, cleanup=CLEANUP, 
        costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
        costWeights=CFWGT, test=False, optDir='./{}'.format(CFDIR))
# endif RESTORE

# number of iterations for the optimization
NITER = 1
DIAGNOSTICS = True
for i in range(coObj.iCombine, NITER+1):
    t1 = time.process_time()


    wgtInfo = ['{:.2f} ({})'.format(
        wgt, comp) for wgt, comp in zip(CFWGT, CFCOMPS)]
    wgtInfo = ' '.join(wgtInfo)
    print  ("  ")
    print  ("  ")
    print('Weights: {}'.format(wgtInfo))
    print('Iteration {}'.format(i))
    temp = time.process_time()
    coObj.kMap()
    # print('kMap: {:.4f}, {:10.4e}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxComputePool()
    # print('Flux Compute: {:.4f}, {:10.4e}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.fluxCombine()
    # print('Flux Combine: {:.4f}, {:10.4e}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.costFuncComp(init=True)
    coObj.costFuncComp()
    # print('Cost function computation: {:.4f}, {:10.4e}'.format(time.process_time()-temp))

    temp = time.process_time()
    coObj.findOptimal()
    # print('findOptimal: {:.4f}, {:10.4e}'.format(time.process_time()-temp))


    if coObj.optimized: break
    if DIAGNOSTICS: coObj.costDiagnostics()

# Start of special g-point combination branch
    if coObj.dCost[coObj.iOpt]-coObj.deltaCost0 > 0.1:
    #if coObj.dCost[coObj.iOpt]-coObj.deltaCost0 > -2.01:
       print ('will change here')
       xWeight = 0.05
       delta0 = coObj.dCost[coObj.iOpt]-coObj.deltaCost0
       bandObj = coObj.distBands
       if (coObj.optBand+1) < 10:
           bandKey='band0{}'.format(coObj.optBand+1)
       else:
           bandKey='band{}'.format(coObj.optBand+1)
       #sys.exit() 
        
       newObj = REDUX.gCombine_kDist(bandObj[bandKey].kInNC, coObj.optBand, DOLW,
            i, fullBandKDir=BANDSPLITDIR,
            fullBandFluxDir=FULLBANDFLUXDIR)
       curkFile = os.path.basename(coObj.optNC)
       ind = curkFile.find('_g')
       g1 = int(curkFile[ind+2:ind+4])
       g2 = int(curkFile[ind+5:ind+7])
       gCombine =[[g1-1,g2-1]]

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
           FCC.fluxCompute(newCoefFile,GARAND,EXE,fluxDir,fluxFile)

           trialNC = '{}/{}'.format(fluxDir,fluxFile)
           coCopy.combinedNC[coObj.iOpt] = REDUX.combineBandsSgl( 
                   coObj.optBand, coObj.fullBandFluxes,trialNC,DOLW)
           coCopy.costFuncCompSgl(coCopy.combinedNC[coObj.iOpt])
           #coCopy.findOptimal()
        
           if DIAGNOSTICS: coCopy.costDiagnostics()
           if(pmFlag == 'plus'):
               deltaPlus = coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0
           if(pmFlag == 'minus'):
               deltaMinus = coCopy.dCost[coObj.iOpt]-coCopy.deltaCost0

# Fit change in cost of three g-point options to a parabola and find minimum weight if parabola is concave;
#  if that weight is outside the (-0.1,0.1) range or the parabola is convex use the weight that leads to 
# the smallest increase or largest decrease in the  cost function;

       print ("  ")
       print ("New weight calculation")
       xArr = [-xWeight,0.,xWeight]
       yArr = [deltaMinus,delta0,deltaPlus]
       print (yArr)
       ymin = min(yArr)
       imin = yArr.index(ymin)
       coeff = np.polyfit(xArr,yArr,2)
       print (coeff[0])
       xMin = -coeff[1]/(2.*coeff[0])
       if (coeff[0] >0.):
           print ("concave parabola ",xMin)
           if (xMin < -xWeight or xMin > xWeight):
               xWeightNew = xArr[imin]
           else:
               xWeightNew = xMin
       else:
           xWeightNew = xArr[imin]
       print (xWeightNew)

# Define newest g-point combination
       coCopy = copy.deepcopy(coObj)
       pmFlag = 'mod'
       print ("  ")
       print (pmFlag)
       newObj.gPointCombineSglPair(pmFlag,gCombine,xWeightNew)
       newCoefFile = '{}/{}_{}.nc'.format(newObj.workDir,fluxPath,pmFlag)
       fluxFile = os.path.basename(newCoefFile).replace('coefficients', 'flux')
       print (newCoefFile)
       print (newCoefFile,file=open('new_weight_diag.txt','a'))
       print (coeff[0],xMin,yArr,xWeightNew,file=open('new_weight_diag.txt','a'))
       print ("  ",file=open('new_weight_diag.txt','a'))
       FCC.fluxCompute(newCoefFile,GARAND,EXE,fluxDir,fluxFile)

       trialNC = '{}/{}'.format(fluxDir,fluxFile)
       coCopy.combinedNC[coObj.iOpt] = REDUX.combineBandsSgl( 
               coObj.optBand, coObj.fullBandFluxes,trialNC,DOLW,)
       coCopy.costFuncCompSgl(coCopy.combinedNC[coObj.iOpt])
       #coCopy.findOptimal()
    
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
# end iteration loop

t1 = time.process_time()
KOUTNC = 'rrtmgp-data-{}-g-red.nc'.format(DOMAIN)

ncFiles = [coObj.distBands[key].kInNC for key in coObj.distBands.keys()]
# kDistOpt(KFULLNC, kOutNC=KOUTNC)
coObj.kDistOpt(KFULLNC, kOutNC=KOUTNC)

# combine flux netCDFs after optimized solutions over all trials are found
REDUX.combineBands(0, coObj.fullBandFluxes, coObj.fullBandFluxes[0], 
                    coObj.doLW, finalDS=True, outNC='optimized_fluxes.nc')
# print('New k-file {:.4f}'.format(time.process_time()-t1))

print('Optimization complete')
