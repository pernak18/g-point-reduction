import os, sys, shutil
import multiprocessing

# "standard" install
import numpy as np

# directory in which libraries installed with conda are saved
PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
PATHS = ['common', PIPPATH]
for path in PATHS: sys.path.append(path)

# user must do `pip install xarray` on cori (or other NERSC machines)
import xarray as xa

# local module
import g_point_reduction as REDUX

# GLOBAL VARIABLES (paths)
PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
GARAND = '{}/reference_netCDF/g-point-reduce/'.format(PROJECT) + \
  'multi_garand_template_single_band.nc'
CWD = os.getcwd()

# these paths are needed for all-band flux calculations
EXEFULL = PROJECT + \
  '/g-point-reduction/k-distribution-opt/rrtmgp_garand_atmos'
NCFULLPROF = PROJECT + \
    '/reference_netCDF/g-point-reduce/multi_garand_template_broadband.nc'

# default cost function components, level indices (surface, , weights
CFCOMPS = ['flux_net', 'band_flux_net']
BOUNDS = [0, 26, 42]
CFLEVS = {'flux_net': BOUNDS, 'band_flux_net': BOUNDS}
CFWGT = [0.5, 0.5]

# THIS NEEDS TO BE ADJUSTED FOR NERSC! cutting the total nCores in half
NCORES = multiprocessing.cpu_count()
CHUNK = int(NCORES)

def pathCheck(path, mkdir=False):
    """
    Determine if file exists. If not, throw an Assertion Exception
    """

    if mkdir:
        # mkdir -p -- create dir tree
        if not os.path.exists(path): os.makedirs(path)
    else:
        assert os.path.exists(path), 'Could not find {}'.format(path)
    # endif mkdir
# end pathCheck

def fluxCompute(inK, profiles, exe, fluxDir, outFile):
    """
    Compute fluxes for a given k-distribution and set of atmospheric
    conditions

    Inputs
        inK -- string, k-distribution file to use in flux calcs
        profiles -- string, netCDF with profile specifications
        exe -- string, RRTMGP flux calculation executable path
        fluxDir -- string, directory in which the executable is run
        outFile -- string, file to which RRTMGP output is renamed
    """

    # standard library
    import subprocess as sub

    pathCheck(fluxDir, mkdir=True)
    cwd = os.getcwd()
    os.chdir(fluxDir)

    # file staging for RRTMGP run
    # trying to keep this simple/compact so we don't end up with a
    # bunch of files in a directory
    aPaths = [exe, inK]
    rPaths = ['./run_rrtmgp', 'coefficients.nc']
    for aPath, rPath in zip(aPaths, rPaths):
        if os.path.islink(rPath): os.unlink(rPath)
        os.symlink(aPath, rPath)
    # end rPath loop

    # so we don't overwrite the LBL results
    inRRTMGP = 'rrtmgp-inputs-outputs.nc'
    shutil.copyfile(profiles, inRRTMGP)

    # assuming the RRTMGP call sequence is `exe inputs k-dist`
    rPaths.insert(1, inRRTMGP)

    # run the model with inputs
    sub.call(rPaths)

    # save outputs (inRRTMGP gets overwritten every run)
    os.rename(inRRTMGP, outFile)
    #print('Wrote {}'.format(outFile))

    os.chdir(cwd)

    return outFile
# end fluxCompute()

def costCalc(lblNC, testNC, doLW, compNameCF, pLevCF, costComp0, scale, init):
    """
    Calculate cost of test dataset with respect to reference dataset 
    at a given number of pressure levels and for a given set of 
    components (upwelling flux, net flux, heating rate, etc.). Also keep 
    other diagnostics normalized to cost of initial 256 g-point k-distribution

    Call
        costDict = costCalc(
            lblDS, testDS, doLW, compNameCF, pLevCF, costComp0, scale, init)

    Inputs
        lblDS -- xarray dataset, LBLRTM reference
        testDS -- xarray dataset, RRTMGP trial dataset (i.e., g-points combined)
        doLW -- boolean, LW or SW parameters used in cost
        compNameCF -- string list of cost function component names (e.g., "flux_net")
        pLevCF -- dictionary; keys for each compNameCF, float iterable (list) values 
            of pressure levels at which each CF component is evaluated
        costComp0 -- dictionary; keys for each compNameCF, float scalar values of 
            initial cost associated with RRTMGP full k-distribution
        scale -- dictionary; keys for each compNameCF, float scalar values
        init -- boolean, calculate initial cost i.e., with the full
            256 g-point k-distribution)

    Outputs
        dictionary with following keys:
            allComps -- float array, weighted cost for each 
                cost function component, normalized WRT cost of full 
                k-distribution (= 100)
            totalCost -- float, total of allComps array
            dCost -- float, change in cost WRT cost of full k-distribution
            costComps -- dictionary; keys are cost components names, 
                values are weighted cost at given component and pressure 
                level, summed over all profiles, normalized with initial cost
            dCostComps -- dictionary; keys are cost components names, 
                values are weighted changes in cost at given component and 
                pressure level, summed over all profiles, normalized with 
                initial cost

    Keywords
        None
    """

    lblDS = xa.open_dataset(lblNC)
    testDS = xa.open_dataset(testNC)
    allComps = []

    # add diffuse to SW dataset
    # should this be done outside of code?
    # TO DO: BAND DIRECT NO WORKING YET
    if not doLW:
        lblDS['flux_dif_net'] = lblDS['flux_dif_dn'] - \
            lblDS['flux_up']

        if init:
            testDS['flux_dif_dn'] = testDS['flux_dn'] - \
                testDS['flux_dir_dn']
            testDS['flux_dif_net'] = testDS['flux_dif_dn'] - \
                testDS['flux_up']
        # endif init
    # endif LW

    # for diagnostics, we keep the cost and delta-cost for 
    # each component; in the total cost, we average over profiles 
    # AND pLevCF, but for diagnostics we break down by pLevCF
    costComps = {}
    dCostComps = {}

    # first calculate weighted cost for each component
    for comp in compNameCF:
        # pressure dimension will depend on parameter
        # layer for HR, level for everything else
        pStr = 'lay' if 'heating_rate' in comp else 'lev'

        if 'forcing' in comp:
            # assuming comp is following '*_forcing_N' where 
            # * is the parameter (flux_net, heating_rate, etc.), 
            # N is the forcing record index
            iForce = int(comp.split('_')[-1])-1
           
            # extract baseline and forcing scenarios
            # baseline is record 0 (Present Day) or 
            # 1 (Preindustrial) -- see 
            # https://github.com/pernak18/g-point-reduction/wiki/LW-Forcing-Number-Convention
            if not doLW:
                # keeping minor-19 for Eli (see LW-Forcing-Number-Convention link)
                # but code iForce needs to be recalibrated
                if iForce == 18: iForce -= 11
            # endif doLW

            iBase = 1 if iForce < 7 else 0

            selDict = {'record': iBase, pStr: pLevCF[comp]}
            bTest = testDS.isel(selDict)
            bLBL = lblDS.isel(selDict)

            # calculate forcing
            selDict['record'] = int(iForce)
            fTest = testDS.isel(selDict)
            fLBL = lblDS.isel(selDict)
            testDSf = fTest - bTest
            lblDSf = fLBL - bLBL
            subsetErr = testDSf - lblDSf

            # what parameter are we extracting from dataset?
            if doLW:
                compDS = comp.replace('_forcing_{}'.format(iForce+1), '')
            else:
                if iForce == 7:
                    compDS = comp.replace('_forcing_19', '')
                else:
                    compDS = comp.replace('_forcing_{}'.format(iForce+1), '')
                # endif iForce
            # end doLW
        else:
            # Compute differences in all variables in datasets at 
            # levels closest to user-provided pressure levels
            # particularly important for heating rate since its
            # vertical dimension is layers and not levels
            # baseline is record 0 (Garand Present Day)
            try:
                # allow for different atmospheric specs 
                # (PI, PI 2xCH4) to be requested using the 
                # "param_N" convention with "N" being the forcing
                # scenario index
                iForce = int(comp.split('_')[-1])-1
                compDS = comp.replace('_{}'.format(iForce+1), '')
            except:
                # default to present day Garand atm specs
                iForce = 0
                compDS = str(comp)
            # stop trying

            selDict = {'record': iForce, pStr: pLevCF[comp]}
            subsetErr = (testDS-lblDS).isel(selDict)
        # endif forcing

        # get array for variable, then compute its test-ref RMS
        # over all columns at given pressure levels for a given
        # forcing scenario
        cfDA = getattr(np.fabs(subsetErr), compDS)

        # determine which dimensions over which to average
        dims = subsetErr[compDS].dims
        calcDims = ['col', pStr]
        if 'band' in dims: calcDims.append('band')

        # components will be scaled by their own initial cost
        costComps[comp] = cfDA.sum(dim=['col'])

        # total cost (sum of compCosts) will be scaled to 100
        # WRT initial cost to keep HR and Flux in same range
        compCost = cfDA.sum(dim=calcDims).values * scale[comp]
        allComps.append(np.sum(compCost))
        if not init:
            dCostComps[comp] = (costComps[comp] - costComp0[comp])
    # end CF component loop

    # now calculate total cost with all components and its 
    # relative difference from 256 g-point reference cost (scaled to 100)
    allComps = np.array(allComps)
    totalCost = allComps.sum()
    dCost = totalCost - 100

    lblDS.close()
    testDS.close()

    return {'allComps': allComps, 'totalCost': totalCost, 'dCost': dCost, 
            'costComps': costComps, 'dCostComps': dCostComps}
# end costCalc()

class gCombine_Cost:
    def __init__(self, bandDict, fullBandFluxes, 
                fluxesLBL, fluxesRRTMGP, iCombine, lw, 
                profilesNC=GARAND, exeRRTMGP=EXE, 
                cleanup=False, 
                costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
                costWeights=CFWGT, 
                optDir='{}/iter_optimizations'.format(os.getcwd()), 
                optFluxNC='optimized_fluxes.nc', 
                test=False):
        """
        flesh this doc out...

        - For a given band, loop over possible g-point combinations within
            each band, creating k-distribution and band-wise flux files for
            each possible combination
        - Run a RRTMGP executable that performs computations for a single band
        - Compute broadband fluxes and heating rates
        - Compute cost function from broadband parameters and determine
            optimal combination of g-points

        Input
            bandDict -- dictionary of `gCombine_kDist` 
                objects (one for each band)
            fullBandFluxes -- list of strings, paths to flux netCDFs for 
                each band without any g-point reduction
            fluxesLBL -- string, path to LBLRTM flux netCDF file
            fluxesRRTMGP -- string, path to RRTMGP flux netCDF file
            iCombine -- int, index for what iteration of g-point combining is
                underway

        Keywords
            lw -- boolean, do longwave domain (otherwise shortwave)
            profilesNC -- string, path to netCDF with atmospheric profiles
            topDir -- string, path to top level of git repository clone
            exeRRTMGP -- string, path to RRTMGP executable that is run
                in flux calculations
            cleanup -- boolean, delete files and subdirectories in working 
                directories after each iteration and after full optimization
            costFuncComp -- list of strings; netCDF variable names of the 
                arrays to include in the cost function
            costFuncLevs -- list of floats; pressure levels in Pa to be 
                used in the cost function
            costWeights -- list of weights for each cost function component
            optDir -- string, directory with optimized solution for 
                every iteration and associated diagnostics (if specified)
            test -- boolean, testing mode (only run 1 band)
            optFluxNC -- string, path for optimized flux netCDF after 
                g-point reduction
        """

        paths = [fluxesLBL, fluxesRRTMGP, profilesNC, exeRRTMGP]
        for path in paths: pathCheck(path)

        self.distBands = dict(bandDict)
        self.lblNC = str(fluxesLBL)
        self.rrtmgpNC = str(fluxesRRTMGP)
        self.iCombine = int(iCombine)
        self.doLW = bool(lw)
        self.profiles = str(profilesNC)
        self.exe = str(exeRRTMGP)
        self.fullBandFluxes = list(fullBandFluxes)
        self.testing = bool(test)
        self.optFluxNC = str(optFluxNC)
        self.cleanup = bool(cleanup)

        for fluxNC in self.fullBandFluxes: pathCheck(fluxNC)

        self.bands = list(self.distBands.keys())
        self.nBands = len(self.bands)

        errMsg = 'Inconsistent k-distributions and fluxes'
        assert self.nBands == len(self.fullBandFluxes), errMsg

        self.compNameCF = list(costFuncComp); nComp = len(costFuncComp)
        self.pLevCF = dict(costFuncLevs); nLev = len(costFuncLevs.keys())
        self.costWeights = list(costWeights); nWgt = len(costWeights)

        if not (nComp == nLev == nWgt):
            print('# of components: {}'.format(nComp))
            print('# of level fields: {}'.format(nLev))
            print('# of weights: {}'.format(nWgt))
            print('Number of cost components, level fields, ', end='')
            print('and weights must be equal. Please try again')
            sys.exit(1)
        # endif n

        self.optDir = str(optDir)
        pathCheck(self.optDir, mkdir=True)

        # ATTRIBUTES THAT WILL GET RE-ASSIGNED IN OBJECT

        # complete list of dictionaries with combinations of 
        # g-points for all bands and their associated flux working 
        # directories
        self.fluxInputsAll = []

        # metadata for keeping track of how g-points were
        # combined; we will keep appending after each iteration
        self.gCombine = {}

        # flux datasets with single g-point combinations, 
        # these are not combined with full-band fluxes
        self.trialNC = []

        # list of xarray datasets that combines g-point combination 
        # arrays (self.iBand) with full-band arrays (!= self.iBand)
        self.combinedNC = []

        self.iOpt = None
        self.dCost = []

        self.cost0 = {}
        self.dCost0 = {}
        for comp in self.compNameCF:
            self.cost0[comp] = []
            self.dCost0[comp] = []
        # end comp loop

        # cost components -- 1 key per cost variable, values are 
        # nTrial x nCostFuncLevs array; for diagnostics
        self.costComp0 = {}
        self.costComps = {}
        self.dCostComps0 = {}
        self.dCostComps = {}

        # total cost of combining given g-points; should be one 
        # element per combination
        self.totalCost = []

        # what band contained the optimal g-point combination?
        self.optBand = None
        self.optNC = None

        # have we arrived at our final optimization?
        self.optimized = False
    # end constructor

    def kMap(self):
        """
        Map every g-point combination to a corresponding flux file and 
        zero-offset ID number
        """

        import pathlib as PL
        
        # should be nBands * (nGpt-1) elements initially 
        # (e.g., 16*15=240 for LW),
        # then decreases by 1 for every iteration because of g-point combining
        for iBand, band in enumerate(self.bands):
            # only work with band(s) that needs modification
            #if iBand not in self.modBand: continue
            kObj = self.distBands[band]
            bandKey = 'Band{:02d}'.format(iBand+1)

            for nc in kObj.trialNC:
                # flux directory is same path as netCDF without .nc extension
                fluxDir = PL.Path(nc).with_suffix('')

                # flux files have the same naming convention as 
                # coefficients files
                baseNC = os.path.basename(nc).replace('coefficients', 'flux')
                fluxNC = '{}/{}'.format(fluxDir, baseNC)
                self.fluxInputsAll.append(
                    {'kNC': nc, 'fluxNC': fluxNC, 
                    'profiles': self.profiles, 'exe': self.exe, 
                    'fluxDir': fluxDir, 'bandID': iBand})
            # end nc loop
        # end band loop
    # end kMap
    
    def fluxComputePool(self):
        """
        Use for parallelization of fluxCompute() calls
        """

        argsMap = [(i['kNC'], i['profiles'], \
                    i['exe'], i['fluxDir'], i['fluxNC']) for i in \
                   self.fluxInputsAll]

        # using processes (slower, separate memory) instead of threads
        #print('Calculating fluxes')
        with multiprocessing.Pool(processes=NCORES, maxtasksperchild=1) as pool:
            result = pool.starmap_async(fluxCompute, argsMap, chunksize=CHUNK)
            # is order preserved?
            # https://stackoverflow.com/a/57725895 => yes
            self.trialNC = result.get()
        # endwith
    # end fluxComputePool()

    def fluxCombine(self):
        """
        Concatenate fluxes from separate files for each band into a single
        file with by-band and broadband fluxes and heating rates

        Heating rates and broadband fluxes are computed in this method
        rather than using the RRTMGP calculations
        """

        import pathlib as PL

        # corresponding band numbers (zero-offset)
        bandIDs = [inputs['bandID'] for inputs in self.fluxInputsAll]

        # open all of the full-band netCDFs as xarray datasets
        # will be combined accordingly with single-band g-point combinations
        """
        fullDS = []
        for bandNC in self.fullBandFluxes:
            with xa.open_dataset(bandNC) as bandDS:
                bandDS.load()
                fullDS.append(bandDS)
            # end with
        # end bandNC loop
        """

        #print('Combining trial fluxes with full-band fluxes')

        workdirCombine = PL.Path('./combinedBands_trials')
        if not workdirCombine.is_dir(): os.makedirs(workdirCombine)

        # trial = g-point combination
        argsMap = []
        iTrials = range(1, len(self.trialNC)+1)
        for iBand, trial, iTrial in zip(bandIDs, self.trialNC, iTrials):
            combNC = '{}/combinedBandsNC_trial{:03d}.nc'.format(
                workdirCombine, iTrial)
            argsMap.append(
                (iBand, self.fullBandFluxes, trial, self.doLW, combNC, False))
        # end iBand/trial/iTrial loop

        with multiprocessing.Pool(processes=NCORES, maxtasksperchild=1) as pool:
            result = pool.starmap_async(
              REDUX.combineBands, argsMap, chunksize=CHUNK)
            # is order preserved?
            # https://stackoverflow.com/a/57725895 => yes
            self.combinedNC = result.get()
        # endwith
    # end fluxCombine()

    def costFuncComp(self, init=False):
        """
        Calculate flexible cost function where RRTMGP-LBLRTM RMS error for
        any number of allowed parameters (usually just flux or HR) over many
        levels is computed.

        Input
            testDS -- xarray Dataset with RRTMGP fluxes

        Keywords
            init -- boolean, evalulate the cost function for the initial, 
                full g-point k-distribution
        """

        from itertools import repeat

        #print('Calculating cost for each trial')

        """
        if init:
            with xa.open_dataset(self.rrtmgpNC) as rrtmDS: allDS = [rrtmDS]
        else:
            allDS = [xa.open_dataset(combNC) for combNC in self.combinedNC]
        # endif init
        """

        # normalize to get HR an fluxes on same scale
        # so each cost component has its own scale to 100
        scale = {}
        for comp, weight in zip(self.compNameCF, self.costWeights):
            scale[comp] = 1 if init else weight * 100 / self.cost0[comp][0]

        #lblDS = xa.open_dataset(self.lblNC)
        #lblDS.load()

        for comp in self.compNameCF:
            # locally, we'll average over profiles AND pLevCF, but 
            # the object will break down by pLevCF for diagnostics
            self.costComps[comp] = []
            self.dCostComps[comp] = []
        # end comp loop

        if init:
            # initial cost calculation; used for scaling -- 
            # how did cost change relative to original 256 g-point cost
            costDict = costCalc(
                self.lblNC, self.rrtmgpNC, self.doLW, self.compNameCF, 
                self.pLevCF, self.costComp0, scale, True)

            self.costComps = dict(costDict['costComps'])
            self.dCostComps = dict(costDict['dCostComps'])

            # if we have an empty dictionary for the initial cost 
            # components, assign to it what should be the cost from the 
            # full k-distribution (for which 
            # there should only be 1 element in the list)
            for iComp, comp in enumerate(self.compNameCF):
                self.cost0[comp].append(costDict['allComps'][iComp])
                self.costComp0[comp] = self.costComps[comp]
            # end component loop
        else:
            # trial = g-point combination
            # set up arguments for parallelization
            argsMap = []
            for testNC in self.combinedNC: argsMap.append(
                (self.lblNC, testNC, self.doLW, self.compNameCF, 
                 self.pLevCF, self.costComp0, scale, False))

            # parallize cost calculation for trials and extract output
            with multiprocessing.Pool(processes=NCORES, maxtasksperchild=1) as pool:
                result = pool.starmap_async(costCalc, argsMap, chunksize=CHUNK)
                # is order preserved?
                # https://stackoverflow.com/a/57725895 => yes
                allCostDict = result.get()
            # endwith

            for iDict, costDict in enumerate(allCostDict):
                self.totalCost.append(costDict['totalCost'])
                self.dCost.append(costDict['dCost'])
                for comp in self.compNameCF:
                    self.costComps[comp].append(costDict['costComps'][comp])
                    self.dCostComps[comp].append(costDict['dCostComps'][comp])
                # end comp loop
        # endif init

        #lblDS.close()
    # end costFuncComp

    def costFuncCompSgl(self, inNC):
        """
        Calculate flexible cost function where RRTMGP-LBLRTM RMS error for
        any number of allowed parameters (usually just flux or HR) over many
        levels is computed.

        Input
            inNC-- NC file:  RRTMGP fluxes of a single trial
        """

        #print('Calculating cost for each trial')

        # normalize to get HR an fluxes on same scale
        # so each cost component has its own scale to 100
        scale = {}
        for comp, weight in zip(self.compNameCF, self.costWeights):
            scale[comp] = weight * 100 / self.cost0[comp][0]

        #lblDS = xa.open_dataset(self.lblNC)
        #lblDS.load()

        # trial = g-point combination
        costDict = costCalc(self.lblNC, inNC, self.doLW, 
            self.compNameCF, self.pLevCF, self.costComp0, scale, False)

        self.totalCost[self.iOpt] = costDict['totalCost']
        self.dCost[self.iOpt] = costDict['dCost']
        for comp in self.compNameCF:
            self.costComps[comp][self.iOpt] = costDict['costComps'][comp]
            self.dCostComps[comp][self.iOpt] = costDict['dCostComps'][comp]
        # end comp loop

        #lblDS.close()
    # end costFuncCompSgl

    def findOptimal(self):
        """
        Determine which g-point combination for a given iteration in a band
        optimized the cost function, save the associated k netCDF
        """

        while True:
            # find optimizal k-distribution
            self.iOpt = np.nanargmin(np.abs(self.dCost))
            optNC = self.fluxInputsAll[self.iOpt]['kNC']

            # if no more g-point combining is possible for associated band, 
            # find the optimization in a different band
            with xa.open_dataset(optNC) as optDS: nGpt = optDS.dims['gpt']

            if nGpt > 1: break

            # remove trial from consideration if no more g-point combining 
            # is possible
            self.fluxInputsAll.pop(self.iOpt)
            self.combinedNC.pop(self.iOpt)
            self.totalCost.pop(self.iOpt)
            self.dCost.pop(self.iOpt)
            for comp in self.compNameCF:
                self.costComps[comp].pop(self.iOpt)
                self.dCostComps[comp].pop(self.iOpt)
            # end comp loop

            # no more trial to consider
            if len(self.totalCost) == 0:
                self.optimized = True
                return
            # endif 0
        # end while

        self.optBand = self.fluxInputsAll[self.iOpt]['bandID']

        # keep a copy of the optimal k-distribution
        # assuming coefficients_LW_g??-??_iter???.nc convention
        base = os.path.basename(optNC)
        base = '{}{:03d}.nc'.format(base[:-6], self.iCombine)
        base = base.replace(
            'coefficients', 'band{:02d}_coefficients'.format(self.optBand+1))
        cpNC = '{}/{}'.format(self.optDir, base)
        shutil.copyfile(optNC, cpNC)
        self.optNC = str(cpNC)
        #print('Saved optimal combination to {}'.format(cpNC))
    # end findOptimal()

    def costDiagnostics(self,sglFlag=False):
        """
        Write cost components for the current iteration to a netCDF file
        """

        # offset: dCost from previous iteration; only needed for diagnostics
        if self.iCombine > 1:
            self.deltaCost0 = [self.dCost0[comp][-1] for comp in self.compNameCF]
            self.deltaCost0 = sum(self.deltaCost0)
        else:
            self.deltaCost0 = 0
        # endif iCombine

        print('{}, Trial: {:d}, Cost: {:4f}, Delta-Cost: {:.4f}'.format(
            os.path.basename(self.optNC), self.iOpt+1, 
            self.totalCost[self.iOpt], (self.dCost[self.iOpt] - self.deltaCost0)))

        diagDir = '{}/diagnostics'.format(self.optDir)
        pathCheck(diagDir, mkdir=True)

        # combine datasets for each cost component and generate a single 
        # netCDF for each component that contains the component's 
        # contributions at each level and band
        # need new lev' (components) dimension
        outDS = xa.Dataset()
        contribs = []
        for comp in self.compNameCF:
            # scaling factor for more meaningful metrics
            scale = 100 / self.cost0[comp][0]

            pStr = 'lay' if 'heat' in comp else 'lev'
            dims = (pStr, 'band') if 'band' in comp else (pStr)

            outDS['cost0_{}'.format(comp)] = xa.DataArray(
                self.costComp0[comp] * scale, dims=dims)

            contrib = self.costComps[comp][self.iOpt] * scale

            outDS['cost_{}'.format(comp)] = xa.DataArray(contrib, dims=dims)

            dCC0 = self.dCostComps0[comp] if self.iCombine > 1 else 0
            trialDC = xa.concat(self.dCostComps[comp], dim='trial')
            outDS['dCost_{}'.format(comp)] = (trialDC - dCC0) * scale

            contribs.append(contrib.sum().values)
        # end comp loop

        outComp, outContrib = [], []
        for iComp, comp in enumerate(self.compNameCF):
            outComp.append(comp)
            outContrib.append('{:.4f}'.format(contribs[iComp]))
        # end comp loop
        print('\t{} = {}'.format(', '.join(outComp), ', '.join(outContrib)))

        outDS['trial_total_cost'] = \
            xa.DataArray(self.totalCost, dims=('trial'))

        outNC = '{}/cost_components_iter{:03d}.nc'.format(
            diagDir, self.iCombine)

        outDS.attrs['optimal'] = os.path.basename(self.optNC)

        outDS.to_netcdf(outNC)
        #print('Wrote cost components to {}'.format(outNC))
    # end costDiagnostics()

    def setupNextIter(self):
        """
        Re-orient object for band that optimized cost function -- i.e., 
        prepare it for another set of g-point combinations. Also clean up 
        its working directory
        """

        bandKey = list(self.distBands.keys())[self.optBand]
        bandObj = self.distBands[bandKey]

        # clean up the optimal band's working directory
        if self.cleanup: shutil.rmtree(bandObj.workDir)

        # combine g-points for next iteration
        #print('Recombining')
        self.iCombine += 1
        newObj = REDUX.gCombine_kDist(self.optNC, self.optBand, 
            bandObj.doLW, self.iCombine, fullBandKDir=bandObj.fullBandKDir, 
            fullBandFluxDir=bandObj.fullBandFluxDir)
        newObj.gPointCombine()
        self.distBands['band{:02d}'.format(self.optBand+1)] = newObj
        
        # reset cost optimization attributes
        #self.modBand = [int(self.optBand)]
        for weight, comp in zip(self.costWeights, self.compNameCF):
            pStr = 'lay' if 'heating_rate' in comp else 'lev'

            scale = weight * 100 / self.cost0[comp][0]
            self.costComp0[comp] = self.costComps[comp][self.iOpt]
            self.dCostComps0[comp] = self.dCostComps[comp][self.iOpt]
            self.cost0[comp].append(self.costComp0[comp].sum().values * scale)
            self.dCost0[comp].append(self.dCostComps0[comp].sum().values * scale)
        # end comp loop

        self.fullBandFluxes[int(self.optBand)] = \
            self.fluxInputsAll[self.iOpt]['fluxNC']
        self.fluxInputsAll = []

        for combNC in self.combinedNC: os.remove(combNC)
        self.combinedNC = []
        self.dCost = []
        self.totalCost = []
        self.optBand = None
        self.optNC = None
        self.costComps = {}
    # end setupNextIter()

    def kDistOpt(self, kRefNC, kOutNC='rrtmgp-data-lw-g-red.nc'):
        """
        Combine the k-distributions from each of the bands after optimization
        has been performed

        TO DO: CLEAN THIS THE EFF UP!
        """

        ncFiles = [self.distBands[key].kInNC for key in self.distBands.keys()]

        # wanna do an xarray.merge(), but it's not that simple
        # initialize outputs
        fullDict = {}

        # what netCDF variables have a g-point dimension and will thus
        # need to be modified in the combination iterations?
        kMajor = list(self.distBands['band01'].kMajVars)
        kMajor.remove('gpt_weights')

        # kminor variables that are nontrivial to combine
        # for minor contributors
        regions = ['lower', 'upper']
        kMinor = ['kminor_{}'.format(reg) for reg in regions]

        strVars = ['identifier_minor', 'gas_minor', 'gas_names']
        minorVars = []
        for reg in regions:
            minorVars.append('minor_gases_{}'.format(reg))
            minorVars.append('scaling_gas_{}'.format(reg))
        # end region loop

        scalars = ['absorption_coefficient_ref_P',
                   'absorption_coefficient_ref_T', 
                   'press_ref_trop']

        # no contributions in a given band
        upNoBandMinor = [3, 11, 13, 14, 15] if self.doLW else \
            [1, 3, 4, 12, 13]
        loNoBandMinor = [13] if self.doLW else [12, 13]

        nBands = 16 if self.doLW else 14
        nGpt = []

        # number of minor contributors per band
        nPerBandUp = np.zeros(nBands).astype(int)
        nPerBandLo = np.zeros(nBands).astype(int)
        for iNC, ncFile in enumerate(ncFiles):
            with xa.open_dataset(ncFile) as kDS:
                sizeDS = kDS.sizes
                nGpt.append(sizeDS['gpt'])
                ncVars = list(kDS.keys())
                for ncVar in ncVars:
                    # for the full band file, we probably don't need it anymore
                    if ncVar == 'gpt_weights': continue

                    varDA = kDS[ncVar]
                    varArr = varDA.values
                    varDims = varDA.dims
                    varSizes = varDA.sizes

                    # is the variable empty (e.g., no minor contributors)?
                    if not varSizes:
                        if ncVar in scalars: fullDict[ncVar] = varDA
                        continue
                    # endif empty var

                    # minor contributors coming back to haunt me
                    minCond1 = ('minor_absorber_intervals_upper' in varDims or \
                        'contributors_upper' in varDims) and \
                        iNC in upNoBandMinor
                    minCond2 = ('minor_absorber_intervals_lower' in varDims or \
                        'contributors_lower' in varDims) and \
                        iNC in loNoBandMinor
                    if minCond1 or minCond2: continue

                    vDim = 'minor_absorber_intervals_upper'
                    if vDim in varDims:
                        nPerBandUp[iNC] = varSizes[vDim]

                    vDim = 'minor_absorber_intervals_lower'
                    if vDim in varDims:
                        nPerBandLo[iNC] = varSizes[vDim]

                    concatAx = -1 if ncVar in kMajor + kMinor else 0
                    if ncVar not in fullDict.keys():
                        if ncVar in minorVars:
                            fullDict[ncVar] = varArr.tolist()
                        else:
                            fullDict[ncVar] = varArr
                        # endif ncVar
                    elif '_ref' in ncVar or ncVar in strVars:
                        # these are the same for every band and 
                        # can be overwritten
                        fullDict[ncVar] = varArr
                    else:
                        if ncVar in minorVars:
                            fullDict[ncVar] += varArr.tolist()
                        elif ncVar in strVars:
                            fullDict[ncVar].append(varArr)
                        else:
                            fullDict[ncVar] = np.concatenate(
                                (fullDict[ncVar], varArr), axis=concatAx)
                        # endif ncVar
                    # endif ncVar
                # end ncVar loop
            # endwith
        # end band loop

        # transform string lists
        for sv in strVars: fullDict[sv] = np.array(fullDict[sv]).T
        for mv in minorVars: fullDict[mv] = np.array(fullDict[mv]).T

        # recalculate g-point related indices
        fullDict['bnd_limits_gpt'] = \
            (fullDict['bnd_limits_gpt']-[0,1]).flatten(
            ).cumsum().reshape(-1, 2)

        i1, iStartLo, iStartUp = 1, 1, 1
        lims_gpt_up, lims_gpt_lo = [], []
        startUp, startLo = [], []
        for nG, nUp, nLo in zip(nGpt, nPerBandUp, nPerBandLo):
            i2 = i1 + nG - 1
            for i in range(nUp):
                lims_gpt_up.append([i1, i2])
                startUp.append(iStartUp)
                iStartUp += nG
            # end i loop

            for i in range(nLo): 
                lims_gpt_lo.append([i1, i2])
                startLo.append(iStartLo)
                iStartLo += nG
            # end i loop

            i1 += nG
        # end minor contributor loop

        fullDict['minor_limits_gpt_lower'] = np.array(lims_gpt_lo)
        fullDict['minor_limits_gpt_upper'] = np.array(lims_gpt_up)
        fullDict['kminor_start_lower'] = np.array(startLo)
        fullDict['kminor_start_upper'] = np.array(startUp)

        #for key in fullDict.keys(): print(key, fullDict[key].shape)
        outDims = {
            'bnd': len(ncFiles), 
            'atmos_layer': 2, 
            'pair': 2, 
            'temperature': 14, 
            'pressure_interp': 60, 
            'mixing_fraction': 9, 
            'gpt': sum(nGpt), 
            'temperature_Planck': 196, 
            'minor_absorber': 21, 
            'absorber': 19, 
            'minor_absorber_intervals_lower': len(lims_gpt_lo), 
            'minor_absorber_intervals_upper': len(lims_gpt_up), 
            'pressure': 59, 
            'absorber_ext': 20, 
            'fit_coeffs': 2, 
            'contributors_lower': fullDict['kminor_lower'].shape[-1], 
            'contributors_upper': fullDict['kminor_upper'].shape[-1],
            'string_len': 32
        }

        # extract dimensions from reference LBL netCDF for consistency
        outDict = {}
        with xa.open_dataset(kRefNC) as refDS:
            for key in fullDict.keys():
                data = fullDict[key]
                dims = refDS[key].dims
                outDict[key] = {"dims": dims, "data": data}
            # end key loop
        # endwith

        # coordinates for xarray dataset
        outCoord = {}
        for key in outDims.keys(): 
            outCoord[key] = {'dims': (key), "data": range(outDims[key])}

        # make an acceptable dictionary for xarray.Dataset.from_dict()
        dsDict = {"coords": outCoord, "dims": outDims, "data_vars": outDict}

        outDS = xa.Dataset.from_dict(dsDict)
        outDS.to_netcdf(kOutNC)
    # end kDistOpt()
    
    def calcOptFlux(self, exeFull=EXEFULL, ncFullProf=NCFULLPROF, 
                    fluxOutNC='optimized_fluxes.nc'):
        """
        Once the optimized solution has been found, combine k-distribution
        from each band into single netCDF, then calculate flux for the 
        distribution across all bands
        """

        fullDS = []
        for bandNC in self.fullBandFluxes:
            with xa.open_dataset(bandNC) as bandDS: fullDS.append(bandDS)

        finalDS = REDUX.combineBands(
            0, self.fullBandFluxes, self.fullBandFluxes[0], self.doLW, 
            finalDS=True, outNC=fluxOutNC)
    # end calcOptFlux()
