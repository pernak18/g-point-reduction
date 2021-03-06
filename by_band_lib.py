import os, sys, shutil
import subprocess as sub
import multiprocessing

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

# these paths are needed for all-band flux calculations
EXEFULL = PROJECT + \
  '/g-point-reduction/k-distribution-opt/rrtmgp_garand_atmos'
NCFULLPROF = PROJECT + \
    '/reference_netCDF/g-point-reduce/multi_garand_template_broadband.nc'

# user must do `pip install xarray` on cori (or other NERSC machines)
import xarray as xa

# full k distribution weights for each g-point (same for every band)
WEIGHTS = [
    0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544,
    0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116,
    0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086,
    0.0022199750, 0.0014140010, 0.0005330000, 0.0000750000
]

# default cost function components, level indices, weights
CFCOMPS = ['flux_net', 'band_flux_net']
CFLEVS = {}
CFLEVS['flux_net'] = [0, 26, 42]
CFLEVS['band_flux_net'] = [0, 26, 42]
CFWGT = [0.5, 0.5]

# THIS NEEDS TO BE ADJUSTED FOR NERSC! cutting the total nCores in half
NCORES = 24 #multiprocessing.cpu_count() // 2
CHUNK = 10

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

    with xa.open_dataset(outFile) as outDS:
        outDS.load()
        os.chdir(cwd)

        return outDS
    # endwith
# end fluxCompute()

def combineBands(iBand, fullDS, trialDS, lw, finalDS=False):
    """
    Combine a given trial fluxes dataset in a given band with the full-band 
    fluxes from the rest of the bands

    Call
        outDS = combineBands(iBand, fullDS, trialDS)
    
    Input
        iBand -- int, zero-offset band number that was modified (i.e., 
            band for which g-points were combined)
        fullDS -- list of xarray Datasets, full-band fluxes for each 
            of the bands that were not modified
        trialDS -- xarray Dataset, fluxes for band where g-points were 
            combined
        lw -- boolean, longwave instead of shortwave flux 
            parameters saved to output netCDF
    
    Keywords
        finalDS -- boolean, merge all bands together after full 
            optimization is complete
        
    Output
        outDS -- xarray Dataset, fluxes for all bands
    """

    nForce = fullDS[0].sizes['record']
    bandVars = ['flux_up', 'flux_dn', 'flux_net', 'heating_rate', 
                'emis_sfc', 'band_lims_wvn']
    fluxVars = bandVars[:4]
    if not lw:
        bandVars.append('flux_dir_dn')
        fluxVars.append('flux_dir_dn')
    # end shortWave

    inBand = int(iBand)
    outDS = xa.Dataset()

    # replace original fluxes for trial band with modified one
    fluxesMod = list(fullDS)
    if not finalDS: fluxesMod[iBand] = trialDS
    nBands = len(fullDS)

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
                if ncVar == 'heating_rate':
                    pDim = 'lay'
                else:
                    pDim = 'lev'
                # endif HR

                newDims = ('record', pDim, 'col', 'band')
            # endif newDims

            outDat = outDat.transpose(*newDims)
        elif ncVar == 'band_lims_gpt':
            gptLims = []
            for iBand, bandDS in enumerate(fluxesMod):
                bandLims = bandDS['band_lims_gpt'].squeeze()
                if iBand == 0:
                    gptLims.append(bandLims)
                else:
                    offset = gptLims[-1][1]
                    gptLims.append(bandLims+offset)
                # endif iBand
            # end band loop

            # add record/forcing dimension
            modDims = {'record': np.arange(nForce), 
                       'band': np.arange(nBands), 
                       'pair': np.arange(2)}
            outDat = xa.DataArray(
                [gptLims] * nForce, dims=modDims)
        else:
            # retain any variables with no band dimension
            outDat = trialDS[ncVar]
        # endif ncVar

        outDS[ncVar] = outDat
    # end ncVar loop

    if not lw:
        outDS['flux_dif_dn'] = outDS['flux_dn'] - outDS['flux_dir_dn']
        outDS['flux_dif_net'] = outDS['flux_dif_dn'] - outDS['flux_up']
        fluxVars.append('flux_dif_dn')
        fluxVars.append('flux_dif_net')
    # endif LW

    # calculate broadband fluxes
    for fluxVar in fluxVars:
        pDim = 'lay' if 'heating_rate' in fluxVar else 'lev'
        dimsBB = ('record', pDim, 'col')
        outDS = outDS.rename({fluxVar: 'band_{}'.format(fluxVar)})
        broadband = outDS['band_{}'.format(
            fluxVar)].sum(dim='band')
        outDS[fluxVar] = xa.DataArray(broadband, dims=dimsBB)
    # end fluxVar loop

    return outDS
# end combineBands()

def costCalc(lblDS, testDS, doLW, compNameCF, pLevCF, costComp0, scale, init):
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
        cfDA = getattr(subsetErr, compDS)**2

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

    return {'allComps': allComps, 'totalCost': totalCost, 'dCost': dCost, 
            'costComps': costComps, 'dCostComps': dCostComps}
# end costCalc()

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

    # end constructor

    def kDistBand(self):
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
                    newDim = self.nWeights
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
        """

        with xa.open_dataset(self.kInNC) as kDS:
            kVal = kDS.kmajor
            weights = kDS.gpt_weights
            ncVars = list(kDS.keys())

            # combine all nearest neighbor g-point indices
            # and associated weights for given band
            nNew = kDS.dims['gpt']-1
            gCombine = [[x, x+1] for x in range(nNew)]
            wCombine = [weights[np.array(gc)] for gc in gCombine]

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
                        if ncVar == 'gpt_weights':
                            # replace g1' weight with integrated weight at
                            # g1 and g2
                            ncDat = xa.where(
                                ncDat.gpt == g1, w1 + w2, ncDat)
                        elif ncVar in ['kmajor', 'rayl_upper', 'rayl_lower']:
                            # replace g1' slice with weighted average of
                            # g1 and g2;
                            # dimensions get swapped for some reason
                            kg1, kg2 = ncDat.isel(gpt=g1), ncDat.isel(gpt=g2)
                            ncDat = xa.where(ncDat.gpt == g1,
                                (kg1*w1 + kg2*w2) / (w1 + w2), ncDat)
                            ncDat = ncDat.transpose(*varDims)
                        else:
                            # replace g1' weight with integrated values at
                            # g1 and g2
                            pg1, pg2 = ncDat.isel(gpt=g1), ncDat.isel(gpt=g2)
                            ncDat = xa.where(ncDat.gpt == g1, pg1 + pg2, ncDat)
                            ncDat = ncDat.transpose(*varDims)
                        # endif ncVar

                        # remove the g2 slice; weird logic:
                        # http://xarray.pydata.org/en/stable/generated/
                        # xarray.DataArray.where.html#xarray.DataArray.where
                        ncDat = ncDat.where(ncDat.gpt != g2, drop=True)
                    elif ncVar in self.kMinorLims + ['bnd_limits_gpt']:
                        ncDat[:] = [1, nNew]
                    elif ncVar in self.kMinor:
                        continue
                    else:
                        # retain any variables without a gpt dimension
                        pass
                    # endif ncVar

                    # stuff new dataset with combined or unaltered data
                    outDS[ncVar] = xa.DataArray(ncDat)
                # end ncVar loop

                # some minor contributor variables need to be done after 
                # the ncVar loop because g-points need to be combined
                for iVar, minIntVar in enumerate(self.kMinorInt):
                    minCon = self.kMinorContrib[iVar]
                    kMinor = self.kMinor[iVar]
                    ncDat = kDS[kMinor]
                    varDims = ncDat.dims

                    # no minor absorption contributions
                    if self.doLW:
                        minCond1 = 'upper' in minCon and \
                            self.iBand in [3, 11, 13, 14, 15]
                        minCond2 = 'lower' in minCon and self.iBand == 13
                    else:
                        minCond1 = 'upper' in minCon and \
                            self.iBand in [1, 3, 4, 12, 13]
                        minCond2 = 'lower' in minCon and self.iBand in [12, 13]
                    # endif LW                        

                    if minCond1 or minCond2:
                        # grab a single, arbitrary slice of the kminor array
                        # and replace it with zeroes (cannot have 0-length 
                        # arrays in RRTMGP)
                        ncDat = ncDat.isel({minCon: slice(0, nNew)}) * 0
                    else:
                        for minInt in range(kDS.dims[minIntVar]):
                            dG = minInt*nNew
                            cg1, cg2 = g1 + dG, g2 + dG
                            kg1 = ncDat.isel({minCon: cg1})
                            kg2 = ncDat.isel({minCon: cg2})

                            ncDat = xa.where(ncDat[minCon] == cg1,
                                (kg1*w1 + kg2*w2) / (w1 + w2), ncDat)
                            ncDat = ncDat.where(ncDat[minCon] != cg2, 
                                drop=True)
                            ncDat = ncDat.transpose(*varDims)
                        # end interval loop
                    # endif no absorption
                    outDS[kMinor] = xa.DataArray(ncDat)
                # end interval variable loop

                for minStart, minLim in zip(self.kMinorStart, self.kMinorLims):
                    outDS[minStart] = outDS[minLim][:,1].cumsum()-nNew+1
                    
                outDS.to_netcdf(outNC, 'w')
            # end combination loop
        # endwith kDS
    # end gPointCombine()
# end gCombine_kDist

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
        self.trialDS = []

        # list of xarray datasets that combines g-point combination 
        # arrays (self.iBand) with full-band arrays (!= self.iBand)
        self.combinedDS = []

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
        with multiprocessing.Pool(NCORES) as pool:
            result = pool.starmap_async(fluxCompute, argsMap, chunksize=CHUNK)
            # is order preserved?
            # https://stackoverflow.com/a/57725895 => yes
            self.trialDS = result.get()
        # endwith
    # end fluxComputePool()

    def fluxCombine(self):
        """
        Concatenate fluxes from separate files for each band into a single
        file with by-band and broadband fluxes and heating rates

        Heating rates and broadband fluxes are computed in this method
        rather than using the RRTMGP calculations
        """

        # corresponding band numbers (zero-offset)
        bandIDs = [inputs['bandID'] for inputs in self.fluxInputsAll]

        # open all of the full-band netCDFs as xarray datasets
        # will be combined accordingly with single-band g-point combinations
        fullDS = []
        for bandNC in self.fullBandFluxes:
            with xa.open_dataset(bandNC) as bandDS:
                bandDS.load()
                fullDS.append(bandDS)
            # end with
        # end bandNC loop

        #print('Combining trial fluxes with full-band fluxes')

        # trial = g-point combination
        argsMap = []
        for iBand, trial in zip(bandIDs, self.trialDS):
            argsMap.append((iBand, fullDS, trial, self.doLW, False))

        with multiprocessing.Pool(NCORES) as pool:
            result = pool.starmap_async(combineBands, argsMap, chunksize=CHUNK)
            # is order preserved?
            # https://stackoverflow.com/a/57725895 => yes
            self.combinedDS = result.get()
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

        if init:
            with xa.open_dataset(self.rrtmgpNC) as rrtmDS: allDS = [rrtmDS]
        else:
            allDS = list(self.combinedDS)
        # endif init

        # normalize to get HR an fluxes on same scale
        # so each cost component has its own scale to 100
        scale = {}
        for comp, weight in zip(self.compNameCF, self.costWeights):
            scale[comp] = 1 if init else weight * 100 / self.cost0[comp][0]

        lblDS = xa.open_dataset(self.lblNC)
        lblDS.load()

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
                lblDS, allDS[0], self.doLW, self.compNameCF, 
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
            for testDS in allDS: argsMap.append(
                (lblDS, testDS, self.doLW, self.compNameCF, 
                 self.pLevCF, self.costComp0, scale, False))

            # parallize cost calculation for trials and extract output
            with multiprocessing.Pool(NCORES) as pool:
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

        lblDS.close()
    # end costFuncComp

    def findOptimal(self):
        """
        Determine which g-point combination for a given iteration in a band
        optimized the cost function, save the associated k netCDF
        """

        while True:
            # find optimizal k-distribution
            self.iOpt = np.nanargmin(self.dCost)
            optNC = self.fluxInputsAll[self.iOpt]['kNC']

            # if no more g-point combining is possible for associated band, 
            # find the optimization in a different band
            with xa.open_dataset(optNC) as optDS: nGpt = optDS.dims['gpt']

            if nGpt > 1: break

            # remove trial from consideration if no more g-point combining 
            # is possible
            self.fluxInputsAll.pop(self.iOpt)
            self.combinedDS.pop(self.iOpt)
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

    def costDiagnostics(self):
        """
        Write cost components for the current iteration to a netCDF file
        """

        # offset: dCost from previous iteration; only needed for diagnostics
        if self.iCombine > 1:
            dCost0 = [self.dCost0[comp][-1] for comp in self.compNameCF]
            dCost0 = sum(dCost0)
        else:
            dCost0 = 0
        # endif iCombine

        print('{}, Trial: {:d}, Cost: {:4f}, Delta-Cost: {:.4f}'.format(
            os.path.basename(self.optNC), self.iOpt+1, 
            self.totalCost[self.iOpt], (self.dCost[self.iOpt] - dCost0)))

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
        newObj = gCombine_kDist(self.optNC, self.optBand, bandObj.doLW, 
            self.iCombine, fullBandKDir=bandObj.fullBandKDir, 
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
        self.combinedDS = []
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

        # dimensions after g-point reduction
        """
        outDims['gpt'] = sum(nGpt)
        outDims['minor_absorber_intervals_lower'] = len(lims_gpt_lo)
        outDims['minor_absorber_intervals_upper'] = len(lims_gpt_up)
        outDims['contributors_lower'] = fullDict['kminor_lower'].shape[-1]
        outDims['contributors_upper'] = fullDict['kminor_upper'].shape[-1]
        """

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

        finalDS = combineBands(0, fullDS, fullDS[0], self.doLW, finalDS=True)
        finalDS.heating_rate.attrs['units'] = 'K/s'
        finalDS.band_heating_rate.attrs['units'] = 'K/s'
        finalDS.to_netcdf(fluxOutNC)
    # end calcOptFlux()
# end gCombine_Cost