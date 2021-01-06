import os, sys, shutil
import subprocess as sub

# "standard" install
import numpy as np

# directory in which libraries installed with conda are saved
PIPPATH = '{}/.local/'.format(os.path.expanduser('~')) + \
    'cori/3.7-anaconda-2019.10/lib/python3.7/site-packages'
PATHS = ['common', PIPPATH]
for path in PATHS: sys.path.append(path)

# GLOBAL VARIABLES (paths)
PROJECT = '/global/project/projectdirs/e3sm/pernak18/'
EXE = '{}/g-point-reduction/garand_atmos/rrtmgp_garand_atmos'.format(
    PROJECT)
GARAND = '{}/reference_netCDF/g-point-reduce/'.format(PROJECT) + \
  'multi_garand_template_single_band.nc'
CWD = os.getcwd()

# user must do `pip install xarray` on cori (or other NERSC machines)
import xarray as xa

# full k distribution weights for each g-point (same for every band)
WEIGHTS = [
    0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544,
    0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116,
    0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086,
    0.0022199750, 0.0014140010, 0.0005330000, 0.0000750000
]

# default cost function components, levels, weights
CFCOMPS = ['flux_net', 'heating_rate', 'band_flux_net']
CFLEVS = [0, 10000, 102000]
CFWGT = [1., 1., 1., .5]

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
# end fluxCompute()

def parallelFlux(inDict):
    # couldn't get pool.apply_async to work; i do it this way 
    # with pool.map
    fluxCompute(inDict['kNC'], inDict['profiles'], 
                inDict['exe'], inDict['fluxDir'], inDict['fluxNC'])
# end fluxComputePool

class gCombine_kDist:
    def __init__(self, kFile, band, lw, iCombine, 
                topDir=CWD, exeRRTMGP=EXE, profilesNC=GARAND, 
                fullBandKDir='band_k_dist', 
                fullBandFluxDir='flux_calculations', cleanup=True):
        """
        - For a given band, loop over possible g-point combinations within
            each band, creating k-distribution and band-wise flux files for
            each possible combination
        - Run a RRTMGP executable that performs computations for a single band
        - Compute broadband fluxes and heating rates
        - Compute cost function from broadband parameters and determine
            optimal combination of g-points

        Input
            kFile -- string, netCDF with full (iteration 0) or reduced 
                k-distribution
            band -- int, band number that is being processed with object
            lw -- boolean, do longwave domain (otherwise shortwave)
            iCombine -- int, combination iteration number

        Keywords
            profilesNC -- string, path to netCDF with atmospheric profiles
            topDir -- string, path to top level of git repository clone
            exeRRTMGP -- string, path to RRTMGP executable that is run
                in flux calculations
            fullBandKDir -- string, path to directory with single-band 
                k-distribution netCDF files
            fullBandFluxDir -- string, path to directory with single-band 
                flux RRTMGP netCDF files
        """

        # see constructor doc
        #print(kFile)
        paths = [kFile, topDir, exeRRTMGP, profilesNC, \
                 fullBandKDir, fullBandFluxDir]
        for path in paths: pathCheck(path)

        self.kInNC = str(kFile)
        self.iBand = int(band)
        self.band = self.iBand+1
        self.doLW = bool(lw)
        self.iCombine = int(iCombine)
        self.domainStr = 'LW' if lw else 'SW'
        self.profiles = str(profilesNC)
        self.topDir = str(topDir)
        self.exe = str(exeRRTMGP)
        self.fullBandKDir = str(fullBandKDir)
        self.fullBandFluxDir = str(fullBandFluxDir)

        # directory where model will be run for each g-point
        # combination
        self.workDir = '{}/workdir_band_{}'.format(self.topDir, self.band)

        paths = [self.workDir, self.profiles]
        for path in paths: pathCheck(path, mkdir=True)

        # what netCDF variables have a g-point dimension and will thus
        # need to be modified in the combination iterations?
        self.kMajVars = ['kmajor', 'gpt_weights']
        if self.doLW:
            self.kMajVars.append('plank_fraction')
        else:
            self.kMajVars += ['rayl_lower', 'rayl_upper',
                              'solar_source_facular' ,
                              'solar_source_sunspot', 'solar_source_quiet']
        # endif doLW

        # kminor variables that are nontrivial to combine
        # for minor contributors
        regions = ['lower', 'upper']
        self.kMinor = ['kminor_{}'.format(reg) for reg in regions]
        self.kMinorStart = ['kminor_start_{}'.format(reg) for reg in regions]
        self.kMinorLims = \
            ['minor_limits_gpt_{}'.format(reg) for reg in regions]
        self.kMinorInt = \
            ['minor_absorber_intervals_{}'.format(reg) for reg in regions]
        self.kMinorContrib = ['contributors_{}'.format(reg) for reg in regions]

        # xarray datasets for initial (full-k) RRTMGP and LBLRTM fluxes
        #self.rrtmgpDS = xa.open_dataset(self.rrtmgpNC)
        #self.lblDS = xa.open_dataset(self.lblNC)

        # ATTRIBUTES THAT WILL GET RE-ASSIGNED IN OBJECT

        # weights after a given combination iteration; start off with 
        # full set of weights
        self.iterWeights = list(WEIGHTS)
        self.nWeights = len(self.iterWeights)

        # netCDF with the band k-distribution (to which kDistBand 
        # writes its output if it is band splitting); corresponding flux
        # start off as full band parameters, then are overwritten with
        # files that combine g-points 
        self.kBandNC = '{}/{}/coefficients_{}_band{:02d}.nc'.format(
            self.topDir, self.fullBandKDir, self.domainStr, self.band)
        self.fluxBandNC = '{}/{}/flux_{}_band{:02d}.nc'.format(
            self.topDir, self.fullBandFluxDir, self.domainStr, self.band)

        # string identifiers for what g-points are combined at each trial
        self.gCombStr = []
        
        # list of netCDFs for each g-point combination in a given band
        # and combination iteration
        self.trialNC = []
        
        # file for this band that is used when combining g-points in
        # another band; starts off as full band k distribution file
        self.iterNC = str(self.kInNC)

        # the number of g-points in a given comb iter
        self.nGpt = 16

    # end constructor

    def kDistBand(self, combine=False):
        """
        Split a full k-distribution into separate files for a given band 
        (default) or combine g-points in a given band and generate 
        corresponding netCDF

        Keywords
            combine -- boolean, specifies whether g-points are being 
                combined; default is False and full k-distribution is 
                split into bands with an equal number of g-points in 
                each band
        """

        weightsDA = xa.DataArray(self.iterWeights, 
            dims={'gpt': range(self.nWeights)}, name='gpt_weights')

        zipMinor = zip(
            self.kMinorInt, self.kMinorContrib, self.kMinor, self.kMinorStart)

        with xa.open_dataset(self.kInNC) as kAllDS:
            gLims = kAllDS.bnd_limits_gpt.values
            ncVars = list(kAllDS.keys())

            # for minor absorbers, determine bands for contributions
            # based on initial, full-band k-distribution (i.e.,
            # before combining g-points)
            minorLims, iKeepAll = {}, {}
            for absIntDim, lim in zip(self.kMinorInt, self.kMinorLims):
                minorLims[absIntDim] = kAllDS[lim].values

            # Dataset that will be written to netCDF with new variables and
            # unedited global attribues
            outDS = xa.Dataset()

            # determine which variables need to be parsed
            for ncVar in ncVars:
                ncDat = kAllDS[ncVar]
                varDims = kAllDS[ncVar].dims

                if 'gpt' in varDims:
                    # grab only the g-point information for this band
                    # and convert to zero-offset
                    i1, i2 = gLims[self.iBand]-1
                    ncDat = ncDat.isel(gpt=slice(i1, i2+1))
                elif 'bnd' in varDims:
                    # list [iBand] to preserve `bnd` dim
                    # https://stackoverflow.com/a/52191682
                    ncDat = ncDat.isel(bnd=[self.iBand])
                # endif

                # have to process contributors vars *after* absorber intervals
                if self.kMinorContrib[0] in varDims: continue
                if self.kMinorContrib[1] in varDims: continue

                for absIntDim in self.kMinorInt:
                    if absIntDim in varDims:
                        # https://stackoverflow.com/a/25823710
                        # possibly less efficient, but more readable way to
                        # get index of band whose g-point limits match the
                        # limits from the contributor; not robust -- assumes
                        # contributions only happen in a single band
                        # limits for contributors must match BOTH band limits
                        iKeep = np.where(
                            (minorLims[absIntDim] == gLims[self.iBand]).all(
                            axis=1))[0]
                        iKeepAll[absIntDim] = np.array(iKeep)

                        if iKeep.size == 0:
                            # TO DO: also not robust -- make it more so;
                            # it assumes this conditional is met with only
                            # arrays of dimension minor x string_len (= 32) or
                            # minor x 2
                            # can't have 0-len dimension, so make dummy data
                            if 'pair' in varDims:
                                ncDat = xa.DataArray(np.zeros((1, 2)), 
                                    dims=(absIntDim, 'pair'))
                            else:
                                ncDat = ncDat.isel({absIntDim: 0})
                            # endif varDims
                        else:
                            ncDat = ncDat.isel({absIntDim: iKeep})
                        # endif iKeep
                    # endif absIntDim
                # end absIntDim loop

                # define "local" g-point limits for given band rather than
                # using limits from entire k-distribution
                if 'limits_gpt' in ncVar: ncDat[:] = [1, self.nWeights]

                # write variable to output dataset
                outDS[ncVar] = xa.DataArray(ncDat)
            # end ncVar loop

            # now process upper and lower contributors
            # this is where things get WACKY
            for absIntDim, contribDim, contribVar, startVar in zipMinor:
                contribDS = kAllDS[contribVar]

                # "You want kminor_lower[:,:,i:i+16]
                # (i being 1-based here, using  i = minor_start_lower(j) )
                # and the j are the intervals that fall in the band"
                startDS = kAllDS[startVar]
                iKeep = []
                for j in iKeepAll[absIntDim]:
                    iStart = int(startDS.isel({absIntDim: j}))
                    iEnd = iStart + self.nWeights
                    iKeep.append(np.arange(iStart, iEnd).astype(int)-1)
                # end j loop

                # need a vector to use with array indexing
                iKeep = np.array(iKeep).flatten()

                if iKeep.size == 0:
                    # TO DO: also not robust -- make it more so;
                    varShape = contribDS.shape
                    # can't actually have a zero-len dimension, so
                    # we will make fake data
                    newDim = 1
                    newShape = (varShape[0], varShape[1], newDim)

                    contribDS = xa.DataArray(
                        np.zeros(newShape), dims=contribDS.dims)
                    startDS = xa.DataArray(np.zeros((1)), dims=absIntDim)
                else:
                    startDS = startDS.isel({absIntDim: iKeepAll[absIntDim]})
                    contribDS = contribDS.isel({contribDim: iKeep})
                # endif iKeep

                # kminor_start_* needs to refer to local bands now, and 
                # needs to be unit-offset
                outDS[startVar] = startDS-startDS[0]+1
                outDS[contribVar] = xa.DataArray(contribDS)
            # end zipMinor loop

            # write weights to output file
            outDS['gpt_weights'] = weightsDA

            outDS.to_netcdf(self.kBandNC, mode='w')
            #print('Completed {}'.format(outNC))
        # endwith
    # end kDistBand()

    def gPointCombine(self):
        """
        Combine g-points in a given band with adjacent g-point and
        store into a netCDF for further processing

        TO DO: will probably have to modify other variables in
        self.kInNC like Ben does in combine_gpoints_fn.py
        """

        with xa.open_dataset(self.kInNC) as kDS:
            kVal = kDS.kmajor
            weights = kDS.gpt_weights
            ncVars = list(kDS.keys())

            # combine all nearest neighbor g-point indices
            # and associated weights for given band
            self.nGpt = kDS.dims['gpt']
            gCombine = [[x, x+1] for x in range(self.nGpt-1)]
            wCombine = [weights[np.array(gc)] for gc in gCombine]
            print(gCombine)
            sys.exit()

            for gc, wc in zip(gCombine, wCombine):
                g1, g2 = gc
                w1, w2 = wc

                # loop over each g-point combination and create
                # a k-distribution netCDF for each
                gCombStr = 'g{:02d}-{:02d}_iter{:03d}'.format(
                    g1+1, g2+1, self.iCombine)
                outNC='{}/coefficients_{}_{}.nc'.format(
                    self.workDir, self.domainStr, gCombStr)
                self.trialNC.append(outNC)
                self.gCombStr.append(gCombStr)

                outDS = xa.Dataset()

                # each trial netCDF has its own set of g-points
                # that we will save for metadata purposes --
                # the combination that optimizes the cost function
                # will have its `g_combine` attribute perpetuated
                # append g-point combinations metadata for given
                # band and iteration in given band
                outDS.attrs['g_combine'] = '{}+{}'.format(g1+1, g2+1)

                for ncVar in ncVars:
                    ncDat = xa.DataArray(kDS[ncVar])
                    varDims = ncDat.dims
                    if ncVar in self.kMajVars:
                        kg1, kg2 = ncDat.isel(gpt=g1), ncDat.isel(gpt=g2)

                        if ncVar == 'gpt_weights':
                            # replace g1' weight with integrated weight at
                            # g1 and g2
                            ncDat = xa.where(
                                ncDat.gpt == g1, w1 + w2, ncDat)
                        # define "local" g-point limits for given band rather than
                        # using limits from entire k-distribution
                        else:
                            # replace g1' slice with weighted average of
                            # g1 and g2; TO DO: make sure this is how
                            # other params in addition to k are treated
                            # dimensions get swapped for some reason
                            ncDat = xa.where(ncDat.gpt == g1,
                                (kg1*w1 + kg2*w2) / (w1 + w2), ncDat)
                            ncDat = ncDat.transpose(*varDims)
                        # endif ncVar

                        # remove the g2 slice; weird logic:
                        # http://xarray.pydata.org/en/stable/generated/
                        # xarray.DataArray.where.html#xarray.DataArray.where
                        ncDat = ncDat.where(ncDat.gpt != g2, drop=True)
                    elif ncVar in self.kMinorLims + ['bnd_limits_gpt']:
                        ncDat[:] = [1, self.nGpt-1]
                    elif ncVar in self.kMinorStart:
                        # 1 less g-point, but first starting index is unchanged
                        ncDat[1:] = ncDat[1:]-1
                    elif ncVar in self.kMinor:
                        # kminor_* has absorption of zero if there are no 
                        # minor contributors because RRTMGP does not 
                        # accept zero-length arrays, so we can just keep 
                        # this array since it is a dummy
                        if np.nansum(ncDat.values) != 0:
                            # upper (0) or lower (1) atmosphere variable?
                            iVar = self.kMinor.index(ncVar)

                            # number of minor contributor intervals for this 
                            # band and region
                            nInterval = outDS.sizes[self.kMinorInt[iVar]]

                            # upper or lower contributors dimension?
                            contribDim = self.kMinorContrib[iVar]

                            # combine g-points in each minor interval
                            intervals = self.nGpt * np.arange(nInterval)
                            iCon1 = g1 + intervals
                            iCon2 = g2 + intervals

                            kg1 = ncDat.isel({contribDim: iCon1})
                            kg2 = ncDat.isel({contribDim: iCon2})

                            # weighted average is easy...
                            ncDat[{contribDim: iCon1}] = \
                               (kg1*w1 + kg2*w2) / (w1 + w2)

                            # ...removing an index is clunky
                            ncDat[{contribDim: iCon2}] = np.nan
                            ncDat = ncDat.dropna(contribDim, how='all')
                        else:
                            pass
                        # endif nansum
                    else:
                        # retain any variables without a gpt dimension
                        pass
                    # endif ncVar

                    # stuff new dataset with combined or unaltered data
                    outDS[ncVar] = xa.DataArray(ncDat)
                # end ncVar loop

                outDS.to_netcdf(outNC, 'w')
            # end combination loop
        # endwith kDS
    # end gPointCombine()
# end gCombine_kDist

class gCombine_Cost:
    def __init__(self, bandDict, fullBandFluxes, 
                fluxesLBL, fluxesRRTMGP, 
                idxForce, iCombine,
                profilesNC=GARAND, exeRRTMGP=EXE, 
                cleanup=True, 
                costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
                costWeights=CFWGT, 
                optDir='{}/iter_optimizations'.format(os.getcwd())):
        """
        flesh this out...

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
            idxForce -- int, index of forcing scenario
            iCombine -- int, index for what iteration of g-point combining is
                underway

        Keywords
            profilesNC -- string, path to netCDF with atmospheric profiles
            topDir -- string, path to top level of git repository clone
            exeRRTMGP -- string, path to RRTMGP executable that is run
                in flux calculations
            costFuncComp -- list of strings; netCDF variable names of the 
                arrays to include in the cost function
            costFuncLevs -- list of floats; pressure levels in Pa to be 
                used in the cost function
            costWeights -- list of weights for each cost function component
            optDir -- string, directory with optimized solution for 
                every iteration
        """

        paths = [fluxesLBL, fluxesRRTMGP, profilesNC, exeRRTMGP]
        for path in paths: pathCheck(path)

        self.distBands = dict(bandDict)
        self.lblNC = str(fluxesLBL)
        self.rrtmgpNC = str(fluxesRRTMGP)
        self.iForce = int(idxForce)
        self.iCombine = int(iCombine)
        self.profiles = str(profilesNC)
        self.exe = str(exeRRTMGP)
        self.fullBandFluxes = list(fullBandFluxes)

        for fluxNC in self.fullBandFluxes: pathCheck(fluxNC)

        self.bands = list(self.distBands.keys())
        errMsg = 'Inconsistent k-distributions and fluxes'
        assert len(self.bands) == len(self.fullBandFluxes), errMsg

        self.compNameCF = list(costFuncComp)
        self.pLevCF = list(costFuncLevs)
        self.costWeights = list(costWeights)

        self.optDir = str(optDir)
        pathCheck(self.optDir, mkdir=True)

        # ATTRIBUTES THAT WILL GET RE-ASSIGNED IN OBJECT

        # complete list of dictionaries with combinations of 
        # g-points for all bands and their associated flux working 
        # directories
        self.fluxInputs = []

        # metadata for keeping track of how g-points were
        # combined; we will keep appending after each iteration
        self.gCombine = {}

        # list of xarray datasets that combines g-point combination 
        # arrays (self.iBand) with full-band arrays (!= self.iBand)
        self.trialDS = []

        # weights after a given combination iteration; start off with 
        # full set of weights
        self.iterWeights = list(WEIGHTS)
        self.nWeights = len(self.iterWeights)

        # normalization factors defined in normCost() method
        self.norm = []

        # total cost of combining given g-points; should be one 
        # element per combination
        self.totalCost = []

        # what band contained the optimal g-point combination?
        self.optBand = None
        self.optNC = None

        # original g-point IDs for a given band
        # TO DO: have not started trying to preserve these guys
        #self.gOrigID = range(1, self.nGpt+1)
    # end constructor

    def kCombine(self):
        """
        Map every g-point combination to a corresponding flux file and 
        zero-offset ID number
        """

        import pathlib as PL

        # should be nBands * (nGpt-1) elements initially 
        # (e.g., 16*15=240 for LW),
        # then decreases by 1 for every iteration because of g-point combining
        for iBand, band in enumerate(self.bands):
            kObj = self.distBands[band]
            for nc in kObj.trialNC:
                # flux directory is same path as netCDF without .nc extension
                fluxDir = PL.Path(nc).with_suffix('')

                # flux files have the same naming convention as 
                # coefficients files
                baseNC = os.path.basename(nc).replace('coefficients', 'flux')
                fluxNC = '{}/{}'.format(fluxDir, baseNC)
                self.fluxInputs.append({'kNC': nc, 'fluxNC': fluxNC, 
                    'profiles': self.profiles, 'exe': self.exe, 
                    'fluxDir': fluxDir, 'bandID': iBand})
            # end nc loop
        # end band loop
    # end kCombine
    
    def fluxComputePool(self):
        """
        Use for parallelization of fluxCompute() calls
        """

        import multiprocessing

        # THIS NEEDS TO BE ADJUSTED FOR NERSC! cutting the total nCores in half
        nCores = multiprocessing.cpu_count() // 2

        print('Calculating fluxes')
        with multiprocessing.Pool(nCores) as pool:
            pool.map(parallelFlux, self.fluxInputs)
    # end fluxComputePool()

    def comboFlux(self):
        """
        Determine all possible combinations of g-points for which the cost 
        function is calculated. This will include all g-point combinations 
        in a given band along with the broadband fluxes from the other bands
        """

        print('Combining netCDFs for flux computation')
        fluxNC = []

        # should be nBands * (nGpt-1) elements initially (240 for LW),
        # then decreases by 1 for every iteration because of g-point combining
        for iBand, band in enumerate(self.distBands.keys()):
            for combine in self.distBands[band].trialNC:
                temp = list(self.fullBandNC)
                #temp[iBand] = str(combine)
                #self.fluxPool.append(temp)
            # end loop over g-point combinations
        # end band loop
    # end comboFlux
    
    def fluxCombine(self):
        """
        Concatenate fluxes from separate files for each band into a single
        file with by-band and broadband fluxes and heating rates

        Heating rates and broadband fluxes are computed in this method
        rather than using the RRTMGP calculations
        
        Time consuming. Parallizeable?
        """

        # flux file for each g-point combination
        bandTrials = [inputs['fluxNC'] for inputs in self.fluxInputs]

        # corresponding band numbers (zero-offset)
        bandIDs = [inputs['bandID'] for inputs in self.fluxInputs]

        nBands = len(self.fullBandFluxes)

        # open all of the netCDFs as xarray datasets for future processing
        fullDS = []
        for bandNC in self.fullBandFluxes:
            with xa.open_dataset(bandNC) as bandDS: fullDS.append(bandDS)

        nForce = 7
        bandVars = ['flux_up', 'flux_dn', 'flux_net', 
                    'emis_sfc', 'band_lims_wvn']
        fluxVars = bandVars[:3]

        # for flux to HR conversion
        # flux needs to be in W/m2 and P in mbar
        HEATFAC = 8.4391        

        print('Combining trial fluxes with full-band fluxes')

        # trial = g-point combination
        for iBand, trial in zip(bandIDs, bandTrials):
            if iBand > 0: continue
            print(iBand)
            outDS = xa.Dataset()

            with xa.open_dataset(trial) as trialDS:
                # replace original fluxes for trial band with modified one
                fluxesMod = list(fullDS)
                fluxesMod[iBand] = trialDS

                # TO DO: consider xarray.merge()
                ncVars = list(trialDS.keys())
                for ncVar in ncVars:
                    if ncVar in bandVars:
                        # concat variables on the band dimension
                        modDS = [bandDS[ncVar] for bandDS in fluxesMod]
                        outDat = xa.concat(modDS, 'band')

                        # add record/forcing dimension
                        if ncVar == 'emis_sfc':
                            newDims = ('record', 'col', 'band')
                        elif ncVar == 'band_lims_wvn':
                            newDims = ('record', 'band', 'pair')
                            outDat = outDat.expand_dims(
                                dim={'record': nForce}, axis=0)
                        else:
                            newDims = ('record', 'lev', 'col', 'band')
                        # endif newDims

                        outDat = outDat.transpose(*newDims)
                    elif ncVar == 'band_lims_gpt':
                        # this is a tough one
                        # do we need to reassign, though?
                        # and probably can instead consider 
                        # for ds in fluxesMod: print(ds['band_lims_gpt'].values)
                        # and some integration at each band
                        #continue
                        gptLims = []
                        for iBand, bandDS in enumerate(fluxesMod):
                            bandLims = bandDS['band_lims_gpt'].values.squeeze()
                            if iBand == 0:
                                gptLims.append(bandLims)
                            else:
                                offset = gptLims[-1][1]
                                gptLims.append(bandLims+offset)
                            # endif iBand
                        # end band loop

                        # add record/forcing dimension
                        modDims = {'record': np.arange(nForce), 
                            'band': np.arange(nBands), 'pair': np.arange(2)}
                        outDat = xa.DataArray(
                            [gptLims] * nForce, dims=modDims)
                    elif 'heating_rate' in ncVar:
                        continue
                    else:
                        # retain any variables with no band dimension
                        outDat = trialDS[ncVar]
                    # endif ncVar
                    outDS[ncVar] = outDat
                # end ncVar loop

                # calculate broadband fluxes
                for fluxVar in fluxVars:
                    bandFlux = 'band_{}'.format(fluxVar)
                    broadband = outDS[fluxVar].sum(dim='band')

                    # first rename band flux (currently, the executable 
                    # calls flux for a single band flux_*)
                    outDS = outDS.rename_vars({fluxVar: bandFlux})

                    # then reassign flux_* true broadband fluxes
                    # TO DO: unfortunately, broadband variables still have 
                    # a `band` dimension; not sure how to fix
                    #outDS[fluxVar] = xa.DataArray(broadband, dims=['record', 'lev', 'col'])
                    outDS[fluxVar] = xa.DataArray(broadband)
                # end fluxVar loop

                # calculate heating rates
                # TO DO: doing it this way leaves HR with a lev dim, not lay
                dNetBand = outDS['band_flux_net'].diff('lev')
                dNetBB = outDS['flux_net'].diff('lev')
                dP = outDS['p_lev'].diff('lev') / 10
                outDS['band_heating_rate'] = HEATFAC * dNetBand / dP
                outDS['heating_rate'] = HEATFAC * dNetBB / dP
                #print(dNetBand.shape, dNetBB.shape, dP.shape)
                #print(outDS.band_heating_rate.dims, outDS.heating_rate.dims)
                self.trialDS.append(outDS)
            # endwith
        # end trial loop
    # end fluxCombine()

    def costFuncComp(self, normOption=0):
        """
        Calculate flexible cost function where RRTMGP-LBLRTM RMS error for
        any number of allowed parameters (usually just flux or HR) over many
        levels is computed. If self.norm is empty

        Input
            testDS -- xarray Dataset with RRTMGP fluxes

        Keywords
            normOption -- int; ID for different normalization techniques
                (not implemented)

        Time consuming. Parallizeable?
        """

        print('Calculating cost for each trial')

        with xa.open_dataset(self.lblNC) as lblDS:
            for testDS in self.trialDS:
                costComps = []

                for cfVar in self.compNameCF:
                    # pressure dimension will depend on parameter
                    # layer for HR, level for everything else
                    pStr = 'lay' if 'heating_rate' in cfVar else 'lev'

                    # Compute differences in all variables in datasets at 
                    # levels closest to user-provided pressure levels
                    # particularly important for heating rate since its
                    # vertical dimension is layers and not levels
                    # TO DO: confirm this is doing what we expect it to
                    subsetErr = (testDS-lblDS).sel(
                        {pStr:self.pLevCF}, method='nearest')

                    # determine which dimensions over which to average
                    dims = subsetErr.dims
                    calcDims = ['col', pStr]
                    if 'band' in dims: calcDims.append('band')

                    # get array for variable, then compute its test-ref RMS
                    # over all columns at given pressure levels for a given
                    # forcing scenario
                    cfVar = getattr(subsetErr, cfVar)
                    costComps.append(
                        (cfVar.isel(record=self.iForce)**2).mean(
                        dim=calcDims).values)
                # end ncVar loop

                if not self.norm: self.norm = list(costComps)

                normCosts = [np.sqrt((c).mean())/n for (c, n) in \
                             zip(costComps, self.norm)]
                totCost = np.sqrt(np.sum([w * c**2 for (w, c) in \
                    zip(self.costWeights, normCosts)])/np.sum(self.costWeights))
                self.totalCost.append(totCost)
            # end testDS loop
        # endwith LBLDS
    # end costFuncComp

    def findOptimal(self):
        """
        Determine which g-point combination for a given iteration in a band
        optimized the cost function, save the associated k netCDF
        """

        iOpt = np.nanargmin(self.totalCost)
        optNC = self.fluxInputs[iOpt]['kNC']
        self.optBand = self.fluxInputs[iOpt]['bandID']
        base = os.path.basename(optNC)
        base = base.replace(
            'coefficients', 'band{:02d}_coefficients'.format(self.optBand+1))
        cpNC = '{}/{}'.format(self.optDir, base)
        shutil.copyfile(optNC, cpNC)
        self.optNC = str(cpNC)
        print('Saved optimal combination to {}'.format(cpNC))

        # determine optimal combination and grab g-point combination attribute
        # TO DO: try to preserve more metadata (iteration, band, combined gs) 
        # from optimized solution
        """
        with xa.open_dataset(self.optNC) as optDS:
            self.gCombine['iter{:02d}'.format(iCombine)] = \
              optDS.attrs['g_combine']

        for i in self.gCombine.keys(): print(self.gCombine[i])
        """
    # end findOptimal()

    def setupNextIter(self):
        """
        Re-orient object for band that optimized cost function -- i.e., 
        prepare it for another set of g-point combinations. Also clean up 
        its working directory
        """

        import copy
        bandKey = list(self.distBands.keys())[self.optBand]
        bandObj = self.distBands[bandKey]

        # modify attributes of object to reflect next iteration
        newObj = copy.deepcopy(bandObj)
        newObj.kInNC = str(self.optNC)
        newObj.iCombine = self.iCombine + 1

        # clean up the optimal band's working directory
        shutil.rmtree(bandObj.workDir)

        # combine g-points for next iteration
        newObj.gPointCombine()
    # end setupNextIter()
# end gCombine_Cost
