# standard libraries
import os, sys
import pathlib as PL
from copy import deepcopy as DCP

# pip installs
import numpy as np
import xarray as xa
from tqdm import tqdm

# local module
import g_point_reduction as REDUX

TRIALVARS = [
  'fluxInputsAll', 'trialNC', 'combinedNC', 'dCost', 'totalCost'
]

def kModInit(kBand, coIter, weight=0.05, doLW=False, 
             kDir='band_k_dist', fluxDir='fluxes'):
  """
  https://github.com/pernak18/g-point-reduction/issues/19#issuecomment-1370224270
  kBand -- list of gCombine_kDist objects (one for each band)
  coIter -- current iteration of cost optimization process
  """
  # gPointCombineSglPair scales initial weights with either 
  scaleWeight = ['plus', '2plus']

  kBandDict = {}
  kBandDict['init'] = dict(kBand)

  for sWgt in scaleWeight:
      # print(sWgt)
      # replace existing kBandDict with modified objects
      kBandDict[sWgt] = {}
      for iBand, key in enumerate(kBand.keys()):
        
        band = iBand+1
        kNC = kBand[key].kInNC
        kObjMod = REDUX.gCombine_kDist(kNC, iBand, doLW, coIter, 
          fullBandKDir=kDir, fullBandFluxDir=fluxDir)

        # how many combinations exist for band?
        with xa.open_dataset(kNC) as ds: nCombine = ds.dims['gpt']-1

        # g-point indices, and its combination "partner"
        gCombine = [[x, x+1] for x in range(nCombine)]

        # generate the k-dist netCDF for each combination in band
        for comb in gCombine:
          kObjMod.gPointCombineSglPair(sWgt, [comb], weight)

        kBandDict[sWgt]['band{:02d}'.format(band)] = kObjMod
      # end kBand loop
  # end scaleWeight loop

  return kBandDict
# end kModInit()

def costModInit(coObj0, bandKObjMod, scaleWeight=['plus', '2plus'], 
               diagnostics=False):
  """
  coObj0 -- gCombine_Cost object before modified g-point combining
  bandKObjMod -- dictionary containing gCombine_kDist objects for 
    each band (1 key per band)
  """

  dCostMod = {}
  dCostMod['init'] = np.array(coObj0.totalCost) - coObj0.winnerCost

  for sWgt in scaleWeight:
      coObjMod = REDUX.gCombine_Cost(
          bandKObjMod[sWgt], coObj0.fullBandFluxes, 
          coObj0.lblNC, coObj0.rrtmgpNC, coObj0.iCombine, 
          coObj0.doLW, profilesNC=coObj0.profiles, exeRRTMGP=coObj0.exe, 
          costFuncComp=coObj0.compNameCF, costFuncLevs=coObj0.pLevCF, 
          costWeights=coObj0.costWeights, optDir=coObj0.optDir)

      coObjMod.kMap()
      coObjMod.fluxComputePool()
      coObjMod.fluxCombine()

      # use coObj init costs so we don't have to do that calculation again
      coObjMod.optNC = coObj0.optNC
      coObjMod.costComp0 = coObj0.costComp0
      coObjMod.iOpt = coObj0.iOpt

      # grab components up until current, modified iteration
      for comp, cost0 in coObj0.cost0.items():
          coObjMod.cost0[comp] = cost0[:-1]
          coObjMod.dCost0[comp] = coObj0.dCost0[comp][:-1]
          coObjMod.costComps[comp] = coObj0.costComps[comp][:-1]
          coObjMod.dCostComps[comp] = coObj0.dCostComps[comp][:-1]
          coObjMod.costComp0[comp] = coObj0.costComp0[comp]
          coObjMod.dCostComps0[comp] = coObj0.dCostComps0[comp]
      # end comp loop

      # proceed with subsequent cost optimization methods
      coObjMod.costFuncComp()
      coObjMod.winnerCost = coObj0.winnerCost
      if diagnostics:
        print('{} Diagnostics:'.format(sWgt))
        coObjMod.costDiagnostics()
      # endif diag

      dCostMod[sWgt] = np.array(coObjMod.totalCost) - coObj0.winnerCost
  # end scaleWeight loop

  return dCostMod, coObjMod
# end costModInit()

def scaleWeightRegress(dCostMod, returnCoeffs=False):
  """
  dCostMod -- dictionary from costModInit
  """
  # fit line for zero-crossing estimation
  abscissa = np.array([0, 1, 2])
  ordinates = np.vstack([dCost for dCost in dCostMod.values()])
  iMinDelta = np.argmin(np.abs(ordinates), axis=0)

  coeffs = np.polyfit(abscissa, ordinates, 2)
  newScales = np.array(iMinDelta).astype(float)
  cross = []
  for iTrial, coeff in enumerate(coeffs.T):
    poly = np.poly1d(coeff)
    roots = np.roots(coeff)

    # TO DO: root limits should be a flexible parameter
    iReplace = np.where((
      roots >= 0) & (roots <= 2) & np.all(np.isreal(roots)))[0]
    nReplace = iReplace.size

    # usually nReplace is just 1, unless there are complex roots
    if nReplace == 1:
      newScales[iTrial] = roots[iReplace]
      cross.append(True)
    else:
      cross.append(False)
    # endif nReplace
  # end coeff loop

  cross = np.array(cross)
  # TO DO: need to replace some trials in newScales with parabola
  # minimum -- we've only replaced with dCost min or a zero-crossing

  return newScales, cross
# end scaleWeightRegress()

def whereRecompute(kBandAll, coObjMod, trialZero, scales, 
                   weight=0.05, doBand=None):
  """
  Determine where new flux and cost computations need to be made 
  (i.e., at zero crossings), recombine g-points, then replace 
  flux computation arguments with corresponding inputs

  trialZero -- boolean array, nTrial elements
    does a given trial have a zero crossing in the weight scales?
  kBandAll -- list of gCombine_kDist, one for each band
  coObjMod -- gCombine_Cost object, likely output from costModInit
  scales -- float array [nTrial], scale factor used with `weight` in 
    modified g-point combining
  doBand -- int, single unit-offset band number of band to process
    if not set, all are processed
  """

  gCombineAll = []
  for iBand, key in enumerate(kBandAll.keys()):
    if doBand is not None and iBand != doBand: continue
    band = iBand+1
    kNC = kBandAll[key].kInNC

    # how many combinations exist for band?
    with xa.open_dataset(kNC) as ds: nCombine = ds.dims['gpt']-1
    # print(kNC, nCombine)

    # g-point indices, and its combination "partner"
    gCombineAll += [[x, x+1] for x in range(nCombine)]
  # end band loop

  # full k and flux directories are the same for all bands
  kBand = kBandAll['band01']

  # loop over all trials with a zero crossing
  iReprocess = np.where(trialZero)[0]

  # label regression trials
  scaleStr = np.repeat('_regress', iReprocess.size)

  # the trials where the 'plus' modification 
  # are better than '2plus' should be pretty small, so we'll just 
  # recalculate these guys as well
  # TO DO: not optimal
  i1 = np.where(scales == 1)[0]
  iReprocess = np.append(iReprocess, i1)

  # distinguish regression trials from unmod and `plus` mod
  scaleStr = np.append(scaleStr, np.repeat('_plus', i1.size))

  for iRep, sStr in zip(iReprocess, scaleStr):
      # grab k-distribution for each band BEFORE g-point combining 
      # for this iteration
      fluxInputs = coObjMod.fluxInputsAll[iRep]
      kNC = kBandAll['band{:02d}'.format(fluxInputs['bandID']+1)].kInNC

      # 0 and 1 mods already have k-dist files on disk
      if sStr == '_regress':
        g1, g2 = gCombineAll[iRep]

        # generate modified k-dist netCDF file with new scale weight
        kObjMod = REDUX.gCombine_kDist(kNC, fluxInputs['bandID'], 
          coObjMod.doLW, coObjMod.iCombine, 
          fullBandKDir=kBand.fullBandKDir, 
          fullBandFluxDir=kBand.fullBandFluxDir)
        kObjMod.gPointCombineSglPair(
          'regress', [[g1, g2]], scales[iRep]*weight)
      # endif regress

      # replace flux computations i/o with modified files
      fields = ['kNC', 'fluxNC', 'fluxDir']
      for field in fields: coObjMod.fluxInputsAll[iRep][field] = \
        str(fluxInputs[field]).replace('_2plus', sStr)

      coObjMod.fluxInputsAll[iRep]['fluxDir'] = \
        PL.Path(coObjMod.fluxInputsAll[iRep]['fluxDir'])
  # end reprocessing loop

  return coObjMod, iReprocess
# end whereRecompute()

def recompute(coObj0, coObjMod, scales, iRedo):
  """
  Recompute fluxes for only trials that need to be preprocessed
  (i.e., modified g-point combination is initiated or a band 
  contained a winner)

  coObj0 -- gCombine_Cost object, should be original object before 
    modification was applied
  coObjMod -- gCombine_Cost object, should be output from costModInit
  scales -- output from whereRecompute()
  iRedo -- int array, trials for which computation is re-done
  """

  coObjRep = DCP(coObjMod)

  # need to account for the trials where perturbations only worsened 
  # dCost wrt initial modification
  iInit = np.where(scales == 0)[0]

  # TO DO: recomputation for init seems unnecessary, but i don't know 
  # how the 2plus dCost propogated when 0 was the lowest dCost in a trial
  iRedo = np.append(iRedo, iInit)

  for i0 in iInit: 
    # replace flux computations i/o with modified files
    fields = ['kNC', 'fluxNC', 'fluxDir']
    for field in fields: 
      coObjRep.fluxInputsAll[i0][field] = \
        coObj0.fluxInputsAll[i0][field]
    # end field loop
  # end iInit loop

  # reprocessing should only be done over trials with a zero crossing
  # convert to array so we can use iRedo, then convert 
  # back to list for rest of processing
  for tr in TRIALVARS: 
    setattr(
      coObjRep, tr, np.array(getattr(coObjRep, tr))[iRedo].tolist()
    )

  coObjRep.fluxComputePool()
  coObjRep.fluxCombine()

  # grab components up until current, modified iteration
  coObjRep.totalCost = []
  coObjRep.dCost = []
  coObjRep.winnerCost = float(coObj0.winnerCost)
  for comp, cost0 in coObj0.cost0.items():
      coObjRep.cost0[comp] = cost0[:-1]
      coObjRep.dCost0[comp] = coObj0.dCost0[comp][:-1]
      coObjRep.costComps[comp] = coObj0.costComps[comp][:-1]
      coObjRep.dCostComps[comp] = coObj0.dCostComps[comp][:-1]
      coObjRep.costComp0[comp] = coObj0.costComp0[comp]
      coObjRep.dCostComps0[comp] = coObj0.dCostComps0[comp]
  # end comp loop

  coObjRep.costFuncComp()  
  
  return coObjRep, iRedo
# end recompute()

def trialConsolidate(coObjMod, coObjRep, iRedo, winnerCost0, 
               diagnostics=False):
  """
  Consolidate reprocessed trials with the rest
  """

  coObjMod.winnerCost = float(winnerCost0)
  for iRep, iMod in enumerate(iRedo): 

    coObjMod.totalCost[iMod] = float(coObjRep.totalCost[iRep])
    for comp, cost0 in coObjRep.cost0.items():
      coObjMod.costComps[comp][iMod] = coObjRep.costComps[comp][iRep]
      coObjMod.dCostComps[comp][iMod] = coObjRep.dCostComps[comp][iRep]
      coObjMod.costComp0[comp] = coObjRep.costComp0[comp]
      coObjMod.dCostComps0[comp] = coObjRep.dCostComps0[comp]
    # end comp loop
  # end reprocess loop

  return coObjMod
# end trialConsolidate()

def doBandTrials(inObj, kFiles, bandCost0, dCostMod, 
                 weight=0.05):
  """
  Do the same thing as costModInit, scaleWeightRegress, 
  whereRecompute, and recompute, but only for a single 
  band where parallelization is not necessary. Previously, 
  all trials for all bands were recomputed.
  
  inObj -- gCombine_Cost object, likely generated with 
    coSetupNextIterMod()
  kFiles -- dictionary with 'plus' and `2plus' keys 
    whose values are lists of strings, new k-distribution netCDFs 
    generated for band after winner was selected
  bandCost0 -- numpy float array, total costs of trials in band that 
    contained winner in previous iteration (should correspond to 
    where NANs are in inObj.totalCost)
  """

  # find trials of current iteration where costs need to be recomputed
  # these trials should have been designated with a Nan in 
  # coSetupNextIterMod

  optBand = inObj.fluxInputsAll[inObj.iOpt-1]['bandID']
  bandIDs = np.array(
    [fia['bandID'] for fia in inObj.fluxInputsAll]).astype(int)
  iNAN = np.where(bandIDs == inObj.optBand)[0]
  nNAN = iNAN.size

  assert nNAN != 0, 'OBJECT WAS NOT RESET'

  dCostBand = {}
  dCostBand['init'] = bandCost0 - inObj.winnerCost

  for key in kFiles.keys():
    # single object for given weight scale factor
    bandObj = DCP(inObj)
    bandObj.fluxInputsAll = []
    bandObj.totalCost = []
    bandObj.dCost = []
    bandObj.trialNC = []
    bandObj.combinedNC = []
    bandObj.winnerCost = float(inObj.winnerCost)

    for inan, iTrial in enumerate(iNAN):
      # replace flux computations i/o with modified files
      fields = ['kNC', 'fluxNC', 'fluxDir']
      kFile = kFiles[key][inan]
      inputs = inObj.fluxInputsAll[iTrial]
      inputs['kNC'] = str(kFile)
      fluxDir = PL.Path(kFile).with_suffix('')
      fluxFile = os.path.basename(kFile).replace(
        'coefficients', 'flux')
      inputs['fluxNC'] = '{}/{}'.format(fluxDir, fluxFile)
      inputs['fluxDir'] = fluxDir

      bandObj.fluxInputsAll.append(inputs)
    # end reprocessing loop

    # flux computation for single-band trials
    bandObj.fluxComputePool()
    bandObj.fluxCombine()

    # cost calculation for band trials
    bandObj.costFuncComp()
    dCostBand[key] = np.array(bandObj.totalCost) - inObj.winnerCost
  # end key loop

  for scale in ['plus', '2plus']:
    inan = np.where(iNAN == iTrial)[0]
    dCostMod[scale][iTrial] = dCostBand[scale][inan]
  # end scale loop
      
  # for inan, iTrial in enumerate(iNAN):

  # band should be the same for each trial in bandObj
  iBand = bandObj.fluxInputsAll[0]['bandID']

  newScales, cross = scaleWeightRegress(dCostBand)
  bandObjMod = whereRecompute(bandObj.distBands, bandObj, cross, 
    newScales, weight=weight, doBand=iBand)
  bandObjRep = recompute(bandObj, bandObjMod, np.where(cross)[0])

  return iNAN, bandObj, dCostMod
# end doBandTrials()

def recalibrate(inObj):
  """
  After a winner is chosen, costs for other trials change because the 
  band corresponding to the winner has changed. Adjust the fluxes and 
  delta costs from regression accordingly and print out recalibrated 
  diagnostics

  inObj -- cost optimization object after trial consolidation 
    (trialConsolidate() was applied)
  """

  # print('Recalibrating')

  # flux recalucation with modified band
  inObj.fluxCombine()
  inObj.findOptimal()
  inObj.costDiagnostics()

  return inObj
# end recalibrate

def repeat_mod_redux(inObj, doLW=False, iniWgt=0.05, 
  fullBandFluxDir='.', fullBandkDir='.'):
  """
  Instead of only redoing the winner band, recalculate all trials 
  
  This will be much slower and potentially will consume more HD space, 
  but I was having no luck with refitParabola

  Also pretty slow
  """

  # everything we do for an iteration before modifying combinations
  inObj.kMap()
  inObj.fluxComputePool()
  inObj.fluxCombine()
  inObj.costFuncComp()

  kBand = inObj.distBands

  # basically, this is every step that i did after iteration 94
  # and before iteration 95
  kBandDict = kModInit(inObj.distBands, inObj.iCombine, 
    doLW=doLW,  weight=iniWgt, 
    kDir=fullBandkDir, fluxDir=fullBandFluxDir)

  dCostMod, coObjMod = costModInit(inObj, kBandDict)
  newScales, cross = scaleWeightRegress(dCostMod)

  coObjMod, iRecompute = whereRecompute(
    kBand, coObjMod, cross, newScales, weight=iniWgt)
  coObjRep, iRecompute = recompute(inObj, coObjMod, newScales, iRecompute)
  coObjMod = trialConsolidate(
    coObjMod, coObjRep, iRecompute, inObj.winnerCost)
  coObjMod.iCombine = inObj.iCombine

  # some trials have less than 3 dCost points to which to fit and 
  # are thus not valid
  # TO DO: linear regression instead? different weight scales in fit?
  dCostArr = [dCostMod['init'], dCostMod['plus'], dCostMod['2plus']]
  for iBad, dca in enumerate(np.array(dCostArr).T):
    if np.unique(dca).size < 3: coObjMod.totalCost[iBad] = np.nan

  coObjRC = recalibrate(coObjMod)
  parabolaDiag(coObjRC, inObj.winnerCost, dCostMod, newScales)

  # coObjMod and by extension coObjRC have distBands with all 2plus 
  # distributions in it the winner bands needs to change to unmod, 
  # 0.05, 0.1, or regressed for the current iteration, but all others
  # remain the same
  for iBand, k in enumerate(kBand):
    if iBand == coObjRC.optBand: continue
    coObjRC.distBands['band{:02d}'.format(iBand+1)] = inObj.distBands[k]
  # end band loop

  coObjRC.setupNextIter()

  return coObjRC
# end repeat_mod_redux()

def parabolaDiag(inObj, cost0, dCostMod, scales):
  import pandas as PD

  # TO DO: g-point labels, number of g-points (use kNC)
  metric = np.array(inObj.totalCost)-cost0
  iSort = np.argsort(np.abs(metric))

  dcmi = dCostMod['init']
  dcm1 = dCostMod['plus']
  dcm2 = dCostMod['2plus']

  temp = []
  for iisort, isort in enumerate(iSort): 
    temp.append([
      iisort, isort, 
      inObj.fluxInputsAll[isort]['bandID']+1, scales[isort], 
      dcmi[isort], dcm1[isort], dcm2[isort], metric[isort]
    ])
  # end isort loop
  PD.DataFrame(temp, columns=(
    'index', 'trial', 'band', 'weight scale', 
    'dCost 0', 'dCost 1', 'dCost 2', 'dCost')).to_csv(
    'dCost_mods_iter{:03d}.csv'.format(inObj.iCombine))
# end parabolaDiag()
