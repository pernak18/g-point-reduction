import os, sys, shutil, glob
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

    #print('Computing flux for {}'.format(inK))

    """
    if combine:
        # extract "g??-??_iter??" and make a directory for it
        fluxDir = '{}/{}'.format(self.workDir, suffix)
        outFile = '{}/{}'.format(fluxDir, 
            os.path.basename(inK).replace('coefficients', 'flux'))
    else:
        fluxDir = str(self.fullBandFluxDir)
        outFile = str(self.fluxBandNC)
    # endif combine
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

class gCombine_kDist:
    def __init__(self, kFile, band, lw,
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

        # for minor contributors; this should all be in the constructor now
        """
        alts = ['lower', 'upper']
        absIntDims = ['minor_absorber_intervals_{}'.format(alt) for alt in alts]
        limDims = ['minor_limits_gpt_{}'.format(alt) for alt in alts]
        contribDims = ['contributors_{}'.format(alt) for alt in alts]
        contribVars = ['kminor_{}'.format(alt) for alt in alts]
        startVars = ['kminor_start_{}'.format(alt) for alt in alts]
        zipMinor = zip(absIntDims, contribDims, contribVars, startVars)
        """

        """
        self.kMinor = ['kminor_{}'.format(reg) for reg in regions]
        self.kMinorStart = ['kminor_start_{}'.format(reg) for alt in regions]
        self.kMinorLims = \
            ['minor_limits_gpt_{}'.format(reg) for alt in regions]
        self.kMinorInt = \
            ['minor_absorber_intervals_{}'.format(reg) in regions]
        self.kMinorContrib = ['contributors_{}'.format(reg) for reg in regions]
        """
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

            for gc, wc in zip(gCombine, wCombine):
                g1, g2 = gc
                w1, w2 = wc

                # loop over each g-point combination and create
                # a k-distribution netCDF for each
                gCombStr = 'g{:02d}-{:02d}_init'.format(g1+1, g2+1)
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
                        # upper (0) or lower (1) atmosphere variable?
                        iVar = self.kMinor.index(ncVar)

                        # number of minor contributor intervals for this band 
                        # and region
                        nInterval = outDS.sizes[self.kMinorInt[iVar]]

                        # upper or lower contributors dimension?
                        contribDim = self.kMinorContrib[iVar]

                        # combine g-points in each minor interval
                        intervals = self.nGpt * np.arange(nInterval)
                        if np.nansum(ncDat.values) != 0:
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
                            # kminor_* has absorption of zero if there are no 
                            # minor contributors because RRTMGP does not 
                            # accept zero-length arrays, so we can just keep 
                            # this array since it is a dummy
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
    def __init__(self, bandDict, fluxesLBL, fluxesRRTMGP, 
                idxForce, iCombine,
                profilesNC=GARAND, topDir=CWD, exeRRTMGP=EXE, 
                fullBandKDir='band_k_dist', 
                fullBandFluxDir='flux_calculations', cleanup=True, 
                costFuncComp=CFCOMPS, costFuncLevs=CFLEVS, 
                costWeights=CFWGT):
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
            fullBandKDir -- string, path to directory with single-band 
                k-distribution netCDF files
            fullBandFluxDir -- string, path to directory with single-band 
                flux RRTMGP netCDF files
            costFuncComp -- list of strings; netCDF variable names of the 
                arrays to include in the cost function
            costFuncLevs -- list of floats; pressure levels in mbar to be 
                used in the cost function
            costWeights -- list of weights for each cost function component
        """

        paths = [fluxesLBL, fluxesRRTMGP, profilesNC, topDir, exeRRTMGP, \
                 fullBandKDir, fullBandFluxDir]
        for path in paths: pathCheck(path)

        self.distBands = dict(bandDict)
        self.lblNC = str(fluxesLBL)
        self.rrtmgpNC = str(fluxesRRTMGP)
        self.iForce = int(idxForce)
        self.iCombine = int(iCombine)
        self.profiles = str(profilesNC)
        self.topDir = str(topDir)
        self.exe = str(exeRRTMGP)
        self.fullBandKDir = str(fullBandKDir)
        self.fullBandFluxDir = str(fullBandFluxDir)

        self.fullBandNC = sorted(
            glob.glob('{}/flux_*.nc'.format(self.fullBandFluxDir)))

        self.compNameCF = list(costFuncComp)
        self.levCF = list(costFuncLevs)
        self.costWeights = list(costWeights)

        # ATTRIBUTES THAT WILL GET RE-ASSIGNED IN OBJECT

        # complete list of dictionaries with combinations of 
        # g-points for all bands and their associated flux working 
        # directories
        self.ncCombine = []

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

        # list of dictionaries used for fluxComputePool() input
        self.fluxInputs = []

        # normalization factors defined in normCost() method
        self.norm = []

        # total cost of combining given g-points; should be one 
        # element per combination
        self.totalCost = []

        # original g-point IDs for a given band
        # TO DO: have not started trying to preserve these guys
        #self.gOrigID = range(1, self.nGpt+1)
    # end constructor

    def comboK(self):
        """
        Map every g-point combination to a corresponding flux file
        """

        import pathlib as PL

        # should be nBands * (nGpt-1) elements initially 
        # (e.g., 16*15=240 for LW),
        # then decreases by 1 for every iteration because of g-point combining
        for iBand, band in enumerate(self.distBands.keys()):
            kObj = self.distBands[band]
            for nc in kObj.trialNC:
                # flux directory is same path as netCDF without .nc extension
                self.ncCombine.append(
                    {'kNC': nc, 'fluxDir': PL.Path(nc).with_suffix('')})
            # end nc loop
        # end k object loop
    # end comboK

    def comboFlux(self):
        """
        Determine all possible combinations of g-points for which the cost 
        function is calculated. This will include all g-point combinations 
        in a given band along with the broadband fluxes from the other bands
        """

        print('Combining netCDFs for flux computation')

        # should be nBands * (nGpt-1) elements initially (240 for LW),
        # then decreases by 1 for every iteration because of g-point combining
        for iBand, band in enumerate(self.distBands.keys()):
            for combine in self.distBands[band].trialNC:
                temp = list(self.fullBandNC)
                #temp[iBand] = str(combine)
                #self.ncCombine.append(temp)
            # end loop over g-point combinations
        # end band loop
    # end comboFlux
    
    def configParallel(self):
        """
        Generate input dictionaries for fluxComputePool() function
        outside of class
        """

        for trial, combination in zip(self.trialNC, self.gCombStr):
            self.fluxInputs.append(
                {'inK': trial, 'combine': True, 'comb_iter': combination})
        # end trial loop
    # end configParallel()

    def fluxComputePool(self, inDict):
        """
        Use for parallelization of fluxCompute() calls
        """

        # extract "g??-??_iter??" and make a directory for it
        fluxDir = '{}/{}'.format(self.workDir, suffix)
        outFile = '{}/{}'.format(fluxDir, 
            os.path.basename(inK).replace('coefficients', 'flux'))

        # by the time we're multithreading, we are in the 
        # g-point combining process
        fluxCompute(kObj.kBandNC, kObj.profiles, kObj.exe, 
                    kObj.fullBandFluxDir, kObj.fluxBandNC)
        self.fluxCompute(
            inDict['inK'], combine=True, suffix=inDict['comb_iter'])
    # end fluxComputePool()

    def fluxCombine(self):
        """
        Concatenate fluxes from separate files for each band into a single
        file with by-band and broadband fluxes and heating rates

        Input
            fluxFiles -- list of strings, paths to RRTMGP flux files

        Keywords
            concatDim -- string, name of dimension on which to combine fluxFiles
            outNC -- string, path of netCDF with combined fluxes and HRs
        """

        # TO DO: want to save these in the object attributes
        # flux files for this band and g-point combination iteration
        # all possible g-point combinations
        # should only be n-combination files
        bandTrials = sorted(glob.glob('{}/g??-??_iter{:02d}/flux_*.nc'.format(
            self.workDir, self.iCombine)))

        # full-band flux file fetch; should only be nbands files
        fullNC = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
            self.fullBandFluxDir, self.domainStr)))
        nBands = len(fullNC)

        # open all of the netCDFs as xarray datasets for future processing
        fullDS = []
        for bandNC in fullNC:
            with xa.open_dataset(bandNC) as bandDS: fullDS.append(bandDS)

        # used for g-point reassignment (gPerBand not used with self.iBand)
        gPerBand = 16
        g1 = 1
        nForce = 7
        bandVars = ['flux_up', 'flux_dn', 'flux_net', 
                    'emis_sfc', 'band_lims_wvn']
        fluxVars = bandVars[:3]

        # for flux to HR conversion
        # flux needs to be in W/m2 and P in mbar
        HEATFAC = 8.4391        

        # trial = g-point combination
        for iTrial, trial in enumerate(bandTrials):
            outDS = xa.Dataset()

            with xa.open_dataset(trial) as trialDS:
                # replace original fluxes for band with modified one
                fluxesMod = list(fullDS)
                fluxesMod.remove(fluxesMod[self.iBand])
                fluxesMod.insert(self.iBand, trialDS)

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
                        gptLimsMod = []
                        for iBand, bandDS in enumerate(fluxesMod):
                            if iBand == self.iBand:
                                g2 = g1+self.nGpt-1
                                gptLimsMod.append([g1, g2])
                            else:
                                g2 = g1+gPerBand-1
                                gptLimsMod.append([g1, g2])
                            # endif iBand
                            g1 = g2+1
                            g2 = g1+gPerBand-1
                        # end band loop

                        # add record/forcing dimension
                        modDims = {'record': np.arange(nForce), 
                            'band': np.arange(nBands), 'pair': np.arange(2)}
                        outDat = xa.DataArray(
                            [gptLimsMod] * nForce, dims=modDims)
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
                    outDS = outDS.rename_vars({fluxVar: bandFlux})
                    outDS[fluxVar] = xa.DataArray(broadband)
                # end fluxVar loop

                # calculate heating rates
                dNetBand = outDS['band_flux_net'].diff('lev')
                dNetBB = outDS['flux_net'].diff('lev')
                dP = outDS['p_lev'].diff('lev') / 10
                outDS['band_heating_rate'] = HEATFAC * dNetBand / dP
                outDS['heating_rate'] = HEATFAC * dNetBB / dP
                
                self.trialDS.append(outDS)
            # endwith
        # end trial loop
    # end fluxCombine()

    def costFuncComp(self, testDS, normOption=0):
        """
        Calculate flexible cost function where RRTMGP-LBLRTM RMS error for
        any number of allowed parameters (usually just flux or HR) over many
        levels is computed. If self.norm is empty

        Input
            testDS -- xarray Dataset with RRTMGP fluxes

        Keywords
            normOption -- int; ID for different normalization techniques
                (not implemented)
        """

        costs = []

        # Compute differences in all variables in datasets at levels
        # closest to user-provided pressure levels
        # TO DO: confirm this is doing what we expect it to
        subsetErr = (testDS-self.lblDS).sel(lev=self.levCF, method='nearest')
        for cfVar in self.compNameCF:
            # pressure dimension will depend on parameter
            # layer for HR, level for everything else
            pStr = 'lay' if 'heating_rate' in cfVar else 'lev'

            # get array for variable, then compute its test-ref RMS
            # over all columns at given pressure levels for a given
            # forcing scenario
            cfVar = getattr(subsetErr, cfVar)
            costs.append(
                (cfVar.isel(record=self.iForce)**2).mean(
                dim=('col', pStr)).values)
        # end ncVar loop

        if not self.norm: self.norm = list(costs)

        normCosts = \
            [np.sqrt((c).mean())/n for (c, n) in zip(costs, self.norm)]
        totCost = np.sqrt(np.sum([w * c**2 for (w, c) in \
            zip(self.costWeights, normCosts)])/np.sum(self.costWeights))
        self.totalCost.append(totCost)
    # end costFuncComp

    def findOptimal(self, iCombine):
        """
        Determine which g-point combination for a given iteration in a band
        optimized the cost function

        Input
            iCombine -- int, iteration number for g-point combinations
                in a given band
        """

        # TO DO: loop through trial netCDFs, calculate their normalized
        # cost function components, then determine what is the optimal solution
        iOpt = np.nanargmin(self.totalCost)
        self.optNC = self.trialNC[iOpt]

        # determine optimal combination and grab g-point combination attribute
        with xa.open_dataset(self.optNC) as optDS:
            self.gCombine['iter{:02d}'.format(iCombine)] = \
              optDS.attrs['g_combine']

        for i in self.gCombine.keys(): print(self.gCombine[i])
    # end findOptimal()
# end gCombine_Cost
