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

def costFuncComp(tst_file, ref_file, levs=[0, 10000, 102000], iRecord=0,
                 ncVars=['net_flux', 'heating_rate', 'band_flux_net']):
    """
    Calculate flexible cost function where RRTMGP-LBLRTM RMS error for
    any number of allowed parameters (usually just flux or HR) over many
    levels is computed

    Inputs
        tst_file -- string, RRTMGP (test model) netCDF file with fluxes
        ref_file -- string, LBLRTM (reference model) netCDF file with fluxes

    Output
        outParams -- list of cost function arrays (RMS test-ref differences
          averaged over columns); 1 element per input variable (ncVars)

    Keywords
        levs -- list of floats; pressure levels of interest in Pa
        iRecord -- int; index for forcing scenario (default 0 is no forcing)
        ncVars -- list of strings; netCDF variable names of the arrays to
          include in the cost function
    """

    outParams = []
    with xr.open_dataset(tst_file) as tst, xr.open_dataset(ref_file) as ref:
        # Compute differences in all variables in datasets at levels
        # closest to user-provided pressure levels
        # TODO: confirm this is doing what we expect it to
        subsetErr = (tst-ref).sel(lev=levs, method='nearest')
        for ncVar in ncVars:
            # pressure dimension will depend on parameter
            # layer for HR, level for everything else
            pStr = 'lay' if 'heating_rate' in ncVar else 'lev'

            # get array for variable, then compute its test-ref RMS
            # over all columns and given pressure levels for a given
            # forcing scenario
            ncParam = getattr(subsetErr, ncVar)
            outParams.append(
                (ncParam.isel(record=iRecord)**2).mean(dim=('col', pStr)))
        # end ncVar loop
    # endwith

    return outParams
# end costFuncComp

def normCost(tst_file, ref_file, norm,
             ncVars=['net_flux', 'heating_rate', 'band_flux_net'],
             levs=[0, 10000, 102000], ):
    """
    Returns the summary terms in the cost function
      Each element in each term is normalized (normally by the error at i
      teration 0)

    Inputs
        tst_file -- string, RRTMGP (test model) netCDF file with fluxes
        ref_file -- string, LBLRTM (reference model) netCDF file with fluxes
        norm -- list of floats with RMS error for a given
          cost function component

    Output
        list of floats that are the RMS error (RRTMGP-LBLRTM)
        for each cost function component normalized by the input
        `norm` parameter

    Keywords
        levs -- list of floats; pressure levels of interest in Pa
        iRecord -- int; index for whatever the 'record' dimension is in
          the input netCDF files
        ncVars -- list of strings; netCDF variable names of the arrays to
          include in the cost function

    """

    tst_cost = costFuncComp(tst_file, ref_file, ncVars=ncVars, levs=levs)

    # Each scalar term in the cost function is the RMS across the
    #   normalized error in each component. cost_function_components() returns
    #   the squared error
    return [np.sqrt((c/n).mean()) for (c, n) in zip(tst_cost, norm)]
# end normCost

def bandOptimize(kBandFile, iBand, doLW, iForce, fluxFiles, cleanup=False):
    """
    needs a lot of work -- just spitballin

    single-band optimization, to be combined with broadband fluxes 
    from other bands before cost function optimization 
    """

    band = iBand + 1
    iterBand = 1
    while True:
        print('Starting Band {}, iteration {}'.format(band, iterBand))
        # start with `kFile` with no g-point combinations for a given band
        kObj = kDistOptBand(kBandFile, iBand, doLW, iForce, iterBand)

        # combine g-points in band and generate corresponding netCDF
        kObj.gPointCombine()

        # if there are not enough g-points to combine, stop iterating
        if kObj.nGpt == 1: break

        # run RRTMGP on all files self.trialNC (each g-point combination)
        # generate input dictionaries for fluxComputePool()
        kObj.configParallel()
        #print(kObj.fluxInputs)

        # replace `kFile` with netCDF that corresponds to g-point combination
        # that minimizes the cost function
        kBandFile = kObj.trialNC[0]
        break

        # next iteration
        iterBand += 1
        continue

        # calculate fluxes corresponding to every g-point combination
        # break out of kDistOptBand object for 
        # computation of band fluxes in parallel
        fluxComputePool(kObj.fluxInputs)

        # replace original fluxes for band with modified one
        fluxesMod = list(fluxFiles)
        fluxesMod.remove(fluxFiles[iBand])
        fluxesMod.insert(iBand, outFile)

        # combine fluxes from modified band with unmodified bands
        #fluxCombine(kFilesMod)

        # determine optimal combination
        kObj.findOptimal(kObj.iCombine)

        # keep a copy of the optimal netCDF
        shutil.copy2(kObj.optNC, '{}/{}'.format(
            kObj.optDir, os.path.basename(kObj.optNC)))
        
        # replace `kFile` with netCDF that corresponds to g-point combination
        # that minimizes the cost function
        kBandFile = kObj.optNC
        
        # REASSIGN self.kBandNC and self.optNC!

        # next iteration
        iterBand += 1
    # end while

    # cleanup
    if cleanup: shutil.rmtree(kObj.workDir)
# end bandOptimize

class kDistOptBand:
    def __init__(self, inFile, band, lw, idxForce, iCombine,
                profilesNC=GARAND, topDir=CWD, exeRRTMGP=EXE, 
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
            inFile -- string, netCDF with full k-distribution
                the starting point
            band -- int, band number that is being processed with object
            lw -- boolean, do longwave domain (otherwise shortwave)
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
        """

        # see constructor doc
        #print(inFile)
        paths = [inFile, profilesNC, topDir, exeRRTMGP]
        for path in paths: pathCheck(path)

        self.inNC = str(inFile)
        self.iBand = int(band)
        self.band = self.iBand+1
        self.doLW = bool(lw)
        self.domainStr = 'LW' if lw else 'SW'
        self.iForce = int(idxForce)
        self.iCombine = int(iCombine)
        self.profiles = str(profilesNC)
        self.topDir = str(topDir)
        self.exe = str(exeRRTMGP)
        self.fullBandKDir = str(fullBandKDir)
        self.fullBandFluxDir = str(fullBandFluxDir)
        self.initWeights = list(WEIGHTS)
        self.nWeights0 = len(self.initWeights)

        # directory where model will be run for each g-point
        # combination
        self.workDir = '{}/workdir_band_{}'.format(self.topDir, self.band)

        # directory to store optimal netCDFs for each iteration and band
        self.optDir = '{}/band_{}_opt'.format(self.topDir, self.band)

        paths = [self.workDir, self.optDir]
        for path in paths: pathCheck(path, mkdir=True)

        # metadata for keeping track of how g-points were
        # combined; we will keep appending after each iteration
        self.gCombine = {}

        # what netCDF variables have a g-point dimension and will thus
        # need to be modified in the combination iterations?
        self.gptVars = ['kmajor', 'gpt_weights']
        if self.doLW:
            self.gptVars.append('plank_fraction')
        else:
            self.gptVars += ['rayl_lower', 'rayl_upper',
                            'solar_source_facular' ,
                            'solar_source_sunspot', 'solar_source_quiet']
        # endif doLW

        # ATTRIBUTES THAT WILL GET RE-ASSIGNED IN CLASS

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
        
        # list of xarray datasets that combines g-point combination 
        # arrays (self.iBand) with full-band arrays (!= self.iBand)
        self.trialDS = []

        # the trialNC that optimizes cost function for given comb iter
        # starts off as input file
        self.optNC = str(self.inNC)

        # the number of g-points in a given comb iter
        self.nGpt = 16

        # list of dictionaries used for fluxComputePool() input
        self.fluxInputs = []

        # weights after a given combination iteration; start off with 
        # full set of weights
        self.iterWeights = list(WEIGHTS)
        self.nWeights = len(self.iterWeights)

        # original g-point IDs for a given band
        # TO DO: have not started trying to preserve these guys
        self.gOrigID = range(1, self.nGpt+1)
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

        # for minor contributors
        alts = ['lower', 'upper']
        absIntDims = ['minor_absorber_intervals_{}'.format(alt) for alt in alts]
        limDims = ['minor_limits_gpt_{}'.format(alt) for alt in alts]
        contribDims = ['contributors_{}'.format(alt) for alt in alts]
        contribVars = ['kminor_{}'.format(alt) for alt in alts]
        startVars = ['kminor_start_{}'.format(alt) for alt in alts]

        with xa.open_dataset(self.inNC) as kAllDS:
            gLims = kAllDS.bnd_limits_gpt.values
            ncVars = list(kAllDS.keys())

            # for minor absorbers, determine bands for contributions
            # based on initial, full-band k-distribution (i.e.,
            # before combining g-points)
            minorLims, iKeepAll = {}, {}
            for absIntDim, lim in zip(absIntDims, limDims):
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
                if contribDims[0] in varDims: continue
                if contribDims[1] in varDims: continue

                for absIntDim in absIntDims:
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
            zipMinor = zip(absIntDims, contribDims, contribVars, startVars)
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

    def fluxCompute(self, inK, combine=False, suffix=None):
        """
        Compute fluxes for a given k-distribution and set of atmospheric
        conditions

        Inputs
            inFile -- string, k-distribution file to use in flux calcs

        Keywords
            combine -- boolean, specifies whether g-points are being 
                combined; default is False and full k-distribution is 
                split into bands with an equal number of g-points in 
                each band
            suffix -- string, g-point combination and iteration identifier
        """

        #print('Computing flux for {}'.format(inK))

        if combine:
            # extract "g??-??_iter??" and make a directory for it
            fluxDir = '{}/{}'.format(self.workDir, suffix)
            outFile = '{}/{}'.format(fluxDir, 
                os.path.basename(inK).replace('coefficients', 'flux'))
        else:
            fluxDir = str(self.fullBandFluxDir)
            outFile = str(self.fluxBandNC)
        # endif combine

        pathCheck(fluxDir, mkdir=True)
        os.chdir(fluxDir)

        # file staging for RRTMGP run
        # trying to keep this simple/compact so we don't end up with a
        # bunch of files in a directory
        aPaths = [self.exe, inK]
        rPaths = ['./run_rrtmgp', 'coefficients.nc']
        for aPath, rPath in zip(aPaths, rPaths):
            if os.path.islink(rPath): os.unlink(rPath)
            os.symlink(aPath, rPath)
        # end rPath loop

        # so we don't overwrite the LBL results
        inRRTMGP = 'rrtmgp-inputs-outputs.nc'
        shutil.copyfile(self.profiles, inRRTMGP)

        # assuming the RRTMGP call sequence is `exe inputs k-dist`
        rPaths.insert(1, inRRTMGP)

        # run the model with inputs
        sub.call(rPaths)

        # save outputs (inRRTMGP gets overwritten every run)
        os.rename(inRRTMGP, outFile)
        #print('Wrote {}'.format(outFile))

        os.chdir(self.topDir)
    # end fluxCompute()

    def gPointCombine(self):
        """
        Combine g-points in a given band with adjacent g-point and
        store into a netCDF for further processing

        TODO: will probably have to modify other variables in
        self.inNC like Ben does in combine_gpoints_fn.py
        """

        with xa.open_dataset(self.inNC) as kDS:
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
                gCombStr = 'g{:02d}-{:02d}_iter{:02d}'.format(
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
                    ncDat = kDS[ncVar]
                    varDims = ncDat.dims
                    if ncVar in self.gptVars:
                        kg1, kg2 = ncDat.isel(gpt=g1), ncDat.isel(gpt=g2)

                        if ncVar == 'gpt_weights':
                            # replace g1' weight with integrated weight at
                            # g1 and g2
                            ncDat = xa.where(
                                ncDat.gpt == g1, w1 + w2, ncDat)
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
        self.nGpt -= 1
    # end gPointCombine()

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

        # by the time we're multithreading, we are in the 
        # g-point combining process
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

        import glob

        bandVars = ['flux_up', 'flux_dn', 'flux_net', 'emis_sfc', 'band_lims_wvn']

        # flux files for this band and g-point combination iteration
        # all possible g-point combinations
        # should only be n-combination files
        bandTrials = sorted(glob.glob('{}/g??-??_iter{:02d}/flux_*.nc'.format(
            self.workDir, self.iCombine)))

        # full-band flux file fetch; should only be nbands files
        fullNC = sorted(glob.glob('{}/flux_{}_band??.nc'.format(
            self.fullBandFluxDir, self.domainStr)))
        nBands = len(fullNC)
        fullDS = []
        for bandNC in fullNC:
            with xa.open_dataset(bandNC) as bandDS: fullDS.append(bandDS)

        # trial = g-point combination
        for iTrial, trial in enumerate(bandTrials):
            outDS = xa.Dataset()

            with xa.open_dataset(trial) as trialDS:
                # replace original fluxes for band with modified one
                fluxesMod = list(fullDS)
                fluxesMod.remove(fluxesMod[self.iBand])
                fluxesMod.insert(self.iBand, trialDS)

                # consider xarray.merge()
                ncVars = list(trialDS.keys())
                for ncVar in ncVars:
                    if ncVar in bandVars:
                        # concat variables in the band dimension
                        newFluxes = [bandDS[ncVar] for bandDS in fluxesMod]
                        outDat = xa.concat(newFluxes, 'band')
                    elif ncVar == 'band_lims_gpt':
                        # this is a tough one
                        wvnLimsMod = []
                        gPerBand = 16
                        g1 = 1
                        for iBand, bandDS in enumerate(fluxesMod):
                            if iBand == self.iBand:
                                g2 = g1+self.nGpt-1
                                wvnLimsMod.append([g1, g2])
                                g1 = g2+1
                                g2 = g1+gPerBand-1
                            else:
                                g2 = g1+gPerBand-1
                                wvnLimsMod.append([g1, g2])
                                g1 = g2+1
                                g2 = g1+gPerBand-1
                            # endif iBand
                        # end band loop
                        modDims = \
                            {'band': np.arange(nBands), 'pair': np.arange(2)}
                        outDat = xa.DataArray(wvnLimsMod, dims=modDims)
                    else:
                        # retain any variables with no band dimension
                        outDat = trialDS[ncVar]
                    # endif ncVar
                    outDS[ncVar] = outDat
                    outDS.to_netcdf('temp.nc')
                # end ncVar loop

                # calculate broadband fluxes

                # calculate heating rates
                
                self.trialDS.append(outDS)
            # endwith
        # end trial loop

    # end fluxCombine()

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
        iOpt = 0
        self.optNC = self.trialNC[iOpt]

        # determine optimal combination and grab g-point combination attribute
        with xa.open_dataset(self.optNC) as optDS:
            self.gCombine['iter{:02d}'.format(iCombine)] = \
              optDS.attrs['g_combine']

        for i in self.gCombine.keys(): print(self.gCombine[i])
    # end findOptimal()
# end kDistOptBand
