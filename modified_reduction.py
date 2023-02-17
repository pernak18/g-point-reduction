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
      print(sWgt)
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
      if diagnostics: coObjMod.costDiagnostics()
      dCostMod[sWgt] = np.array(coObjMod.totalCost) - coObj0.winnerCost
  # end scaleWeight loop

  return dCostMod, coObjMod
# end costModInit()

def scaleWeightRegress(dCostMod):
  """
  dCostMod -- dictionary from costModInit
  """
  # fit line for zero-crossing estimation
  abscissa = np.array([0, 1, 2])
  ordinates = np.vstack([dCost for dCost in dCostMod.values()])
  coeffs = np.polyfit(abscissa, ordinates, 1)

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

    # g-point indices, and its combination "partner"
    gCombineAll += [[x, x+1] for x in range(nCombine)]
  # end band loop

  # full k and flux directories are the same for all bands
  kBand = kBandAll['band01']

  # loop over all trials with a zero crossing
  iReprocess = np.where(trialZero)[0]
  for iRep in tqdm(iReprocess):
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

def modOptimal(coObjMod, coObjRep, iRedo, winnerCost0):
  """
  Find optimal solution after modifying g-point combination approach
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

  coObjMod.findOptimal()
  coObjMod.costDiagnostics()

  return coObjMod
# end modOptimal()

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

def coModSetupNextIter(inObj):
    """
    setupNextIter upgrade to lessen computational footprint by preserving 
    information for bands that did not contain winner

    inObj -- g_point_reduction.gCombine_Cost object after modified 
        g-point combining applied
    kFiles -- list of strings of files generated in kMod

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
      outObj.combinedNC[iTrial] = None
      outObj.trialNC[iTrial] = None
      outObj.fluxInputsAll[iTrial]['kNC'] = None
      outObj.fluxInputsAll[iTrial]['fluxNC'] = None

      for comp in outObj.compNameCF:
        outObj.costComps[comp][iTrial][:] = np.nan
    # end invalid trial loop

    # decrement number of trials in prep for next iter
    # print(inObj.totalCost[inObj.iOpt], outObj.totalCost[inObj.iOpt])
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
    
    outObj.optBand = None
    outObj.optNC = None

    for weight, comp in zip(outObj.costWeights, outObj.compNameCF):
      scale = weight * 100 / outObj.cost0[comp][0]
      outObj.costComp0[comp] = outObj.costComps[comp][outObj.iOpt]
      outObj.dCostComps0[comp] = outObj.dCostComps[comp][outObj.iOpt]
      outObj.cost0[comp].append(
        outObj.costComp0[comp].sum().values * scale)
      outObj.dCost0[comp].append(
        outObj.dCostComps0[comp].sum().values * scale)
    # end comp loop

    outObj.iCombine += 1
    return outObj, np.array(bandCosts)
# end coSetupNextIterMod()

def doBandTrials(inObj, kFiles, cost0, weight=0.05):
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
  cost0 -- numpy float array, total costs of trials in band that 
    contained winner in previous iteration (should correspond to 
    where NANs are in inObj.totalCost)
  """

  # find trials of current iteration where costs need to be recomputed
  # these trials should have been designated with a Nan in 
  # coSetupNextIterMod

  iNAN = np.where(np.isnan(inObj.totalCost))[0]
  nNAN = iNAN.size

  assert nNAN != 0, 'OBJECT WAS NOT RESET'

  dCostMod = {}
  dCostMod['init'] = cost0 - inObj.winnerCost

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
    dCostMod[key] = np.array(bandObj.totalCost) - inObj.winnerCost
  # end key loop

  # band should be the same for each trial in bandObj
  iBand = bandObj.fluxInputsAll[0]['bandID']

  # dCostMod, coObjNew = MODRED.costModInit(coObj0, kBandDict, diagnostics=True)
  newScales, cross = scaleWeightRegress(dCostMod)
  bandObjMod = whereRecompute(bandObj.distBands, bandObj, cross, 
    newScales, weight=weight, doBand=iBand)
  bandObjRep = recompute(bandObj, bandObjMod, np.where(cross)[0])
  # print(np.array(bandObj.totalCost) - inObj.winnerCost)

  return iNAN, bandObj
# end doBandTrials()
