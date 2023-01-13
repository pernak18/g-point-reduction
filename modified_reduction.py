# standard libraries
import os, sys
from copy import deepcopy as DCP

# pip installs
import numpy as np
import xarray as xa

# local module
import g_point_reduction as REDUX

def kModInit(kBand, coIter, iniWgt=0.05, doLW=False, 
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
          kObjMod.gPointCombineSglPair(sWgt, [comb], iniWgt)

        kBandDict[sWgt]['band{:02d}'.format(band)] = kObjMod
      # end kBand loop
  # end scaleWeight loop

  return kBandDict
# end kModInit()

def costModInit(coObj0, bandKObjMod, scaleWeight=['plus', '2plus']):
  """
  coObj0 -- gCombine_Cost object before modified g-point combining
  bandKObjMod -- dictionary containing gCombine_kDist objects for 
    each band (1 key per band)
  """

  dCostMod = {}
  dCostMod['init'] = coObj0.totalCost - coObj0.winnerCost

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
      coObjMod.costDiagnostics()
      dCostMod[sWgt] = coObjMod.totalCost - coObj0.winnerCost
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
  varsReduced = [
    'fluxInputsAll', 'trialNC', 'combinedNC', 'dCost', 'totalCost'
  ]

  # convert to array so we can use iRedo, then convert 
  # back to list for rest of processing
  for vr in varsReduced: 
    setattr(coObjRep, vr, np.array(getattr(coObjRep, vr))[iRedo].tolist())

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
    newObj = gCombine_kDist(inObj.optNC, inObj.optBand, 
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
    """

    # reset cost optimization attributes
    for weight, comp in zip(inObj.costWeights, inObj.compNameCF):
        scale = weight * 100 / inObj.cost0[comp][0]
        inObj.costComp0[comp] = inObj.costComps[comp][inObj.iOpt]
        inObj.dCostComps0[comp] = inObj.dCostComps[comp][inObj.iOpt]
        inObj.cost0[comp].append(
          inObj.costComp0[comp].sum().values * scale)
        inObj.dCost0[comp].append(
          inObj.dCostComps0[comp].sum().values * scale)
    # end comp loop

    inObj.fullBandFluxes[int(inObj.optBand)] = \
        inObj.fluxInputsAll[inObj.iOpt]['fluxNC']

    inObj.winnerCost = inObj.totalCost[inObj.iOpt]


    # new object attribute of what trials need to be recomputed in next trial
    bandIDs = np.array([fia['bandID'] for fia in inObj.fluxInputsAll])
    inObj.iRecompute = np.where(bandIDs == inObj.optBand)[0]

    # populate fill values into any trials associated with "winner" band
    for iTrial in inObj.iRecompute:
      inObj.dCost[iTrial] = np.nan
      inObj.totalCost[iTrial] = np.nan
      inObj.fluxInputsAll[iTrial]['kNC'] = None
      inObj.fluxInputsAll[iTrial]['fluxNC'] = None

      for comp in inObj.compNameCF:
        inObj.costComps[comp][iTrial][:] = np.nan
    # end invalid trial loop

    inObj.optBand = None
    inObj.optNC = None
    return
# end setupNextIterMod()
