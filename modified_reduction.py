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
  scaleWeight = ['init', 'plus', '2plus']

  kBandDict = {}
  # kBandDict['init'] = dict(kBand)

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

def costModInit(coObj0, bandKObjMod, 
                scaleWeight=['init', 'plus', '2plus'], 
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
    # print(iTrial, roots)
    iReplace = np.where((
      roots >= 0) & (roots <= 2) & np.all(np.isreal(roots)))[0]
    nReplace = iReplace.size
    # if nReplace > 0: print(iTrial, roots, iReplace)

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

  """
  # for noCross, we keep whatever scale weight produced (positive) minimum
  # and do not need to re-run flux/cost calcs
  # new scale factors for weights will start as just the scale that 
  # minimized dCost, then we fill in where a zero crossing was found
  iMinDelta = np.argmin(ordinates, axis=0)
  yMin = np.nanmin(ordinates, axis=0)
  xMin = (abscissa[iMinDelta])
  y0 = ordinates[0, :]

  # was there a crossing at a given trial? minimum must be negative
  cross = yMin <= 0
  noCross = yMin > 0

  newScales = np.array(iMinDelta).astype(float)
  newScales[cross] = (xMin - xMin * yMin / (yMin - y0))[cross]
  iNan = np.where(np.isnan(newScales))
  y0[iNan] = ordinates[2, iNan]
  newScales[iNan] = (xMin - xMin * yMin / (yMin - y0))[iNan]
  """

  if returnCoeffs:
    return newScales, cross, coeffs

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
  # for gCombine, fia in zip(gCombineAll, coObjMod.fluxInputsAll): print(fia['kNC'], gCombine)
  # sys.exit()

  # full k and flux directories are the same for all bands
  kBand = kBandAll['band01']

  # loop over all trials with a zero crossing
  iReprocess = np.where(trialZero)[0]

  # for iRep in tqdm(iReprocess):
  for iRep in iReprocess:
      # grab k-distribution for each band BEFORE g-point combining 
      # for this iteration
      fluxInputs = coObjMod.fluxInputsAll[iRep]
      kNC = kBandAll['band{:02d}'.format(fluxInputs['bandID']+1)].kInNC

      g1, g2 = gCombineAll[iRep]

      # generate modified k-dist netCDF file with new scale weight
      kObjMod = REDUX.gCombine_kDist(kNC, fluxInputs['bandID'], 
        coObjMod.doLW, coObjMod.iCombine, 
        fullBandKDir=kBand.fullBandKDir, 
        fullBandFluxDir=kBand.fullBandFluxDir)
      kObjMod.gPointCombineSglPair(
        'regress', [[g1, g2]], scales[iRep]*weight)

      # replace flux computations i/o with modified files
      fields = ['kNC', 'fluxNC', 'fluxDir']
      for field in fields: coObjMod.fluxInputsAll[iRep][field] = \
        str(fluxInputs[field]).replace('2plus', 'regress')

      coObjMod.fluxInputsAll[iRep]['fluxDir'] = \
        PL.Path(coObjMod.fluxInputsAll[iRep]['fluxDir'])
  # end reprocessing loop

  return coObjMod
# end whereRecompute()

def noRecompute(coObjMod, scales, cross):
  """
  replace k dist netCDF for trials where no flux recalculation 
  is needed (i.e., where there are no valid zero crossings)
  
  coObjMod -- object modified in costModInit()
  scales, cross -- outputs from scaleWeightRegress()
    should be the same length, and should also be consistent 
    with 
  """

  fia = coObjMod.fluxInputsAll
  nFIA = len(fia)
  nScales = len(scales)
  nCross = len(cross)
  # TO DO: check for this pythonically
  # assert nFIA == nScale == nCross             
  for iFIA, fi in enumerate(fia):
    scale = scales[iFIA]

    if cross[iFIA]:
      # skip zero crossing
      continue
    elif scale in [0, 1]:
      optScale = 'init' if scale == 0 else 'plus'
      # replace k and flux computations i/o with non-modified (init)  
      # or different weight scale (1plus)
      fields = ['kNC', 'fluxNC', 'fluxDir']
      for field in fields: coObjMod.fluxInputsAll[iFIA][field] = \
        str(fi[field]).replace('2plus', optScale)
    # endif scale

# end noRecompute()

def recompute(coObj0, coObjMod, iRedo):
  """
  Recompute fluxes for only trials that need to be preprocessed
  (i.e., modified g-point combination is initiated or a band 
  contained a winner)

  coObj0 -- gCombine_Cost object, should be original object before 
    modification was applied
  coObjMod -- gCombine_Cost object, should be output from costModInit
  iRedo -- int array, trials for which computation is re-done
  """

  coObjRep = DCP(coObjMod)

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
  
  return coObjRep
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

  # this function used to be modOptimal(), but we need to run 
  # recalibrate() before finding an optimal solution
  # coObjMod.findOptimal()
  # if diagnostics: coObjMod.costDiagnostics()

  return coObjMod
# end trialConsolidate()

def kModSetupNextIter(inObj, weight0, scaleWeight='plus'):
    """
    setupNextIter upgrade to lessen computational footprint by preserving 
    information for bands that did not contain winner

    inObj -- g_point_reduction.gCombine_Cost object after modified 
        g-point combining applied
    """

    bandKey = list(inObj.distBands.keys())[inObj.optBand]
    bandObj = inObj.distBands[bandKey]

    # combine g-points for next iteration
    inObj.iCombine += 1
    newObj = REDUX.gCombine_kDist(inObj.optNC, inObj.optBand, 
        bandObj.doLW, inObj.iCombine, fullBandKDir=bandObj.fullBandKDir, 
        fullBandFluxDir=bandObj.fullBandFluxDir)

    # at this point, we're always doing the modified combining
    # how many combinations exist for band?
    with xa.open_dataset(inObj.optNC) as ds: nCombine = ds.dims['gpt']-1

    # g-point indices, and its combination "partner"
    gCombine = [[x, x+1] for x in range(nCombine)]

    # generate the k-dist netCDF for each combination in band
    kFiles = []
    for comb in gCombine:
        kFiles.append(
          newObj.gPointCombineSglPair(scaleWeight, [comb], weight0))
    # end gCombine loop

    return newObj, kFiles
# end kModSetupNextIter

def coModSetupNextIter(inObj, dCostMod):
    """
    setupNextIter upgrade to lessen computational footprint by preserving 
    information for bands that did not contain winner

    inObj -- g_point_reduction.gCombine_Cost object after modified 
        g-point combining applied
    iRemain -- int list, index of trials that remain after modified 
      combination

    Returns
      outObj -- like inObj, but with NANs in the trials of band 
        where winner was chosen
      bandCosts -- float array, total costs of trials in "winner band"
    """

    outObj = DCP(inObj)
    bandCosts = []

    # reset cost optimization attributes
    for weight, comp in zip(outObj.costWeights, outObj.compNameCF):
        scale = weight * 100 / outObj.cost0[comp][0]
        outObj.costComp0[comp] = outObj.costComps[comp][outObj.iOpt]
        outObj.dCostComps0[comp] = outObj.dCostComps[comp][outObj.iOpt]
        outObj.cost0[comp].append(
          outObj.costComp0[comp].sum().values * scale)
        outObj.dCost0[comp].append(
          outObj.dCostComps0[comp].sum().values * scale)
    # end comp loop

    outObj.fullBandFluxes[int(inObj.optBand)] = \
        outObj.fluxInputsAll[inObj.iOpt]['fluxNC']

    outObj.winnerCost = inObj.totalCost[outObj.iOpt]

    # new object attribute of what trials need to be recomputed in next trial
    bandIDs = np.array([fia['bandID'] for fia in inObj.fluxInputsAll])
    outObj.iRecompute = np.where(bandIDs == inObj.optBand)[0]

    # populate fill values into any trials associated with "winner" band
    for iTrial in outObj.iRecompute:
      if iTrial != inObj.iOpt:
        bandCosts.append(inObj.totalCost[iTrial])

      outObj.dCost[iTrial] = np.nan
      outObj.totalCost[iTrial] = np.nan
      # outObj.combinedNC[iTrial] = None
      # outObj.trialNC[iTrial] = None
      outObj.fluxInputsAll[iTrial]['kNC'] = None
      outObj.fluxInputsAll[iTrial]['fluxNC'] = None

      for comp in outObj.compNameCF:
        outObj.costComps[comp][iTrial][:] = np.nan
    # end invalid trial loop

    iRm = inObj.iOpt
    outObj.dCost.pop(iRm)
    outObj.totalCost.pop(iRm)
    outObj.fluxInputsAll.pop(iRm)
    outObj.combinedNC.pop(iRm)
    outObj.trialNC.pop(iRm)
    for comp in outObj.compNameCF:
      outObj.costComps[comp].pop(iRm)
      outObj.dCostComps[comp].pop(iRm)
    # end comp loop

    for scale in ['init', 'plus', '2plus']:
      dCostMod[scale] = np.delete(dCostMod[scale], iRm)

    # NOTE: indexing at iOpt at iteration 143 was crashing because 
    # iOpt is the last trial, then when remove that element with iRm
    # easy fix is to do iOpt-1 at this point, but we have not applied
    # it to the results from iteration 94-142, so their cost components
    # are slightly wrong
    # i actually think this still might be wrong, because i think  
    # decrementing depends on the placement of the trial in the 
    # band, but we're gonna go with this 
    for weight, comp in zip(outObj.costWeights, outObj.compNameCF):
      scale = weight * 100 / outObj.cost0[comp][0]
      outObj.costComp0[comp] = outObj.costComps[comp][outObj.iOpt-1]
      outObj.dCostComps0[comp] = outObj.dCostComps[comp][outObj.iOpt-1]
      outObj.cost0[comp].append(
        outObj.costComp0[comp].sum().values * scale)
      outObj.dCost0[comp].append(
        outObj.dCostComps0[comp].sum().values * scale)
    # end comp loop

    # need to do this eventually, but also need optBand for doBandTrials
    # outObj.optBand = None
    # outObj.optNC = None

    outObj.iCombine += 1
    return outObj, np.array(bandCosts), dCostMod
# end coSetupNextIterMod()

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

  # TO DO: method of NaN search is obsolete -- just need optBand
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

  # fill in dCostMod with new band dCosts
  """
  for iTrial in range(len(inObj.totalCost)):
    for scale in ['init', 'plus', '2plus']:
      if iTrial in iNAN:
        inan = np.where(iNAN == iTrial)[0]
        dCostMod[scale][iTrial] = dCostBand[scale][inan]
      else:
        dCostMod[scale][iTrial] = \
          inObj.totalCost[iTrial] - inObj.winnerCost
    # end scale loop
  # end trial loop
  """
  for scale in ['init', 'plus', '2plus']:
    inan = np.where(iNAN == iTrial)[0]
    dCostMod[scale][iTrial] = dCostBand[scale][inan]
  # end scale loop
      
  # for inan, iTrial in enumerate(iNAN):

  # band should be the same for each trial in bandObj
  iBand = bandObj.fluxInputsAll[0]['bandID']

  # for scale in ['init', 'plus', '2plus']: dCostMod0[scale] = np.nan

  # dCostMod, coObjNew = MODRED.costModInit(coObj0, kBandDict, diagnostics=True)
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

def rootUpdate(inObj, dCostMod, offset, iBandTrials, weight=0.05):
  """
  Fitted parabolas will have new roots after every iteration, since 
  they are dependent on the dCost, which in turn is dependent 
  on the totalCost, which changes every iteration. Find new roots, 
  then find a winner based on the new roots, and ONLY FOR THE WINNER
  update the k- and flux-files

  Also do what we do in recalibrate()

  Inputs
    inObj -- cost optimization object after recalibration
  dCostMod -- dictionary with 'init', 'plus', and '2plus' fields
    and associated arrays of trial dCost that will be adjusted; 
    this is the initial dictionary once modified combining began
  dCostMod -- dictionary with 'init', 'plus', and '2plus' fields
    and associated arrays of trial dCost that will be adjusted; 
    this is the initial dictionary once modified combining began
  offset -- float; totalCost that was used as baseline in 
    calculation of dCostMod
  iRemain -- int list; list of indices of remaining trials
  """

  # local module (part of repo)
  from rrtmgp_cost_compute import flux_cost_compute as FCC

  print('Refitting parabolas and recalibrating fluxes')
  print('Refit winnerCost: ', inObj.winnerCost)

  # only keep dCost for remaining trials, then adjust dCost with
  # costs from previous iteration
  # TO DO: maybe just recalc completely with totalCost and winnerCost?
  """
  for key, val in dCostMod.items():
    for iTrial in range(len(val)):
      if iTrial not in iBandTrials:
        dCostMod[key][iTrial] = val[iTrial] + offset - inObj.winnerCost
    # end trial loop
  # end key loop
  """

  # refit parabolas
  scales, cross, fits = scaleWeightRegress(dCostMod, returnCoeffs=True)
  fits = fits.T
  # print(dCostMod['init'][1], dCostMod['plus'][1], dCostMod['2plus'][1])
  # print(np.array(inObj.totalCost) - inObj.winnerCost)
  # sys.exit()

  # fitted delta-costs from which winner will be chosen
  dCostFit = []
  for fit, scale in zip(fits, scales):
    dCostPoly = np.poly1d(fit)  
    dCostFit.append(dCostPoly(scale))
  # end fit/scale loop

  # with multiple zero crossings, we take the one with the steepest
  # quadratic coefficient, because that has the least variability 
  # in its roots, and so RRTMGP calculations should stray less from 
  # zero than flatter parabolas
  iZero = np.where(np.abs(dCostFit) == 0)[0]
  for iz in tqdm(iZero):
    inObj.iOpt = int(iz)

    # generate new k and flux files for winner
    fia = inObj.fluxInputsAll[inObj.iOpt]
    kNC = fia['kNC']
    bandStr = 'band{:02d}'.format(fia['bandID']+1)

    split = PL.Path(kNC).name.split('_')
    gComb = split[2]
    g1, g2 = int(gComb[1:3])-1, int(gComb[4:])-1
    # if split[4] == '2plus': kNC = kNC.replace('2plus', 'regress')
    # if split[4] == 'plus': kNC = kNC.replace('plus', 'regress')

    # this will overwrite whatever has been written for this 
    # g-pt combo regressed solution at this iteration
    print('New k file')
    rootScale = scales[inObj.iOpt]
    kNC = inObj.distBands[bandStr].gPointCombineSglPair(
      'regress', [[g1, g2]], rootScale*weight)
    inObj.fluxInputsAll[inObj.iOpt]['kNC'] = str(kNC)

    # flux calculation for optimal trial
    print('Recalibrating flux')
    inObj.trialNC[inObj.iOpt] = FCC.fluxCompute(
      kNC, fia['profiles'], \
      fia['exe'], fia['fluxDir'], fia['fluxNC'])
    """
    if 'regress' in kNC:
      # '2plus' and 'plus' k-files are still valid because 
      # no roots were found and thus no new weight scale necessary
      # convention: coefficients_LW_g02-03_iter094_2plus.nc
      split = PL.Path(kNC).name.split('_')
      gComb = split[2]
      g1, g2 = int(gComb[1:3])-1, int(gComb[4:])-1

      # this will overwrite whatever has been written for this 
      # g-pt combo regressed solution at this iteration
      print('New k file')
      rootScale = scales[inObj.iOpt]
      inObj.distBands[bandStr].gPointCombineSglPair(
        'regress', [[g1, g2]], rootScale*weight)

      # flux calculation for optimal trial
      print('Recalibrating flux')
      inObj.trialNC[inObj.iOpt] = FCC.fluxCompute(
        fia['kNC'], fia['profiles'], \
        fia['exe'], fia['fluxDir'], fia['fluxNC'])
    # endif kNC
    """

    # combined new fluxes for trial with "accepted" full-band fluxes
    inObj.fluxCombine()

    # new cost calculation...wondering if this should only be done 
    # for the winner chosen in this method; the root finding is an 
    # approximation, so there is a chance that iOpt comes in a little 
    # higher than other trials
    inObj.costFuncCompSgl(inObj.combinedNC[inObj.iOpt])
    # print(np.abs(np.array(inObj.totalCost)-inObj.winnerCost))

    # new optimal solution
    inObj.findOptimal(fromFit=True, iOptFit=inObj.iOpt)

    # just grab dCosts from regression for winner
    diagCosts = [dCostMod[key][inObj.iOpt] for key in dCostMod.keys()]
    print('Regression delta-costs:')
    print('0 (0): {} 0.05 (1): {} 0.1 (2): {}'.format(*diagCosts))
    print('\tRoot at {:.6f} ({:.6f}): {:.6f}'.format(
      weight*rootScale, rootScale, inObj.totalCost[inObj.iOpt]-inObj.winnerCost))
    # diff = np.abs(np.array(inObj.totalCost) - inObj.winnerCost)
    # print(scales[inObj.iOpt]*weight, dCostFit[inObj.iOpt], diff[inObj.iOpt], diff.min(), diff.argmin())

    # because costFuncComp calculates total cost and components, 
    # that information can still be used and no new calculations 
    # are needed after parabola fit (since the fit is to dCost, 
    # which is dependent on total cost and the previous winner cost)
    inObj.costDiagnostics()
  # end iZero loop

  # TO DO: replace k- and flux-file attributes so new winner is propagated
  # kFiles = {}
  # inObj.distBands[bandStr], kFiles['fit'] = kModSetupNextIter(
  #   inObj, weight, scaleWeight=rootScale)

  return inObj, dCostMod
# end refitParabola

def repeat_mod_redux(inObj, doLW=False, iniWgt=0.05, 
  fullBandFluxDir='.', fullBandkDir='.'):
  """
  Instead of only redoing the winner band, recalculate all trials 
  
  This will be much slower and potentially will consume more HD space, 
  but I was having no luck with refitParabola
  
  NOPE. VERY disk heavy, slow, and SVD error in polyfit on iter 95
  """

  kBand = inObj.distBands

  # basically, this is every step that i did after iteration 94
  # and before iteration 95
  kBandDict = kModInit(inObj.distBands, inObj.iCombine, 
    doLW=doLW,  weight=iniWgt, 
    kDir=fullBandkDir, fluxDir=fullBandFluxDir)

  dCostMod, coObjMod = costModInit(inObj, kBandDict)
  newScales, cross = scaleWeightRegress(dCostMod)

  noRecompute(coObjMod, newScales, cross)
  coObjMod = whereRecompute(
    kBand, coObjMod, cross, newScales, weight=iniWgt)
  coObjRep = recompute(inObj, coObjMod, np.where(cross)[0])
  coObjMod = trialConsolidate(
    coObjMod, coObjRep, np.where(cross)[0], inObj.winnerCost)
  coObjMod.iCombine = inObj.iCombine

  coObjRC = recalibrate(coObjMod)

  # kFiles will likely just contain plus and 2plus (no init), 
  # because we're always doing the modified combination at this point
  kFiles = {}
  for sWgt in ['init', 'plus', '2plus']:
    objModCP = DCP(coObjMod)
    bandStr = 'band{:02d}'.format(coObjMod.optBand+1)
    kBandDict[sWgt][bandStr], kFiles[sWgt] = kModSetupNextIter(
      objModCP, iniWgt, scaleWeight=sWgt)
    coObjMod.distBands[bandStr] = kBandDict[sWgt][bandStr]
  # end scaleWeight loop

  coObjNew, bandCosts, dCostMod = coModSetupNextIter(
    coObjMod, dCostMod)

  iBandTrials, bandObj, dCostMod = doBandTrials(
    coObjNew, kFiles, bandCosts, dict(dCostMod))
  trialConsolidate(coObjNew, bandObj, iBandTrials, coObjNew.winnerCost)

  # used to do this in coModSetupNextIter()
  coObjNew.optBand = None
  coObjNew.optNC = None

  return coObjNew
# end repeat_mod_redux()