import os, sys

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

# local module (part of repo)
import flux_cost_compute as FCC

# full k distribution weights for each g-point (same for every band)
WEIGHTS = [
    0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544,
    0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116,
    0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086,
    0.0022199750, 0.0014140010, 0.0005330000, 0.0000750000
]

def combineBands(iBand, fullNC, trialNC, lw, outNC='trial_band_combined.nc', 
                 finalDS=False):
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

    fullDS = [xa.open_dataset(fNC) for fNC in fullNC]
    trialDS = xa.open_dataset(trialNC)
    
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

    for fDS in fullDS: fDS.close()
    trialDS.close()

    outDS.heating_rate.attrs['units'] = 'K/s'
    outDS.band_heating_rate.attrs['units'] = 'K/s'

    outDS.to_netcdf(outNC)

    return outNC
# end combineBands()

def combineBandsSgl(iBand,fullNC,trialNC,lw,outNC='trial_band_combined.nc', finalDS=False):
    """
    Combine a given trial fluxes dataset in a given band with the full-band 
    fluxes from the rest of the bands

    Call
        outDS = combineBands(iBand, trialNC,fullBandFluxes)
    
    Input
        iBand -- int, zero-offset band number that was modified (i.e., 
            band for which g-points were combined)
        trialNC -- string, netCDF file with fluxes for band where 
            g-points were combined
        fullBandFLuxes - list of nectdf files with full-band fluxes 
            for each band that was not modified
        lw -- boolean, longwave instead of shortwave flux 
            parameters saved to output netCDF
    
    Keywords
        finalDS -- boolean, merge all bands together after full 
            optimization is complete
        
    Output
        outDS -- xarray Dataset, fluxes for all bands
    """

    bandVars = ['flux_up', 'flux_dn', 'flux_net', 'heating_rate', 
                'emis_sfc', 'band_lims_wvn']
    fluxVars = bandVars[:4]
    if not lw:
        bandVars.append('flux_dir_dn')
        fluxVars.append('flux_dir_dn')
    # end shortWave

    inBand = int(iBand)
    outDS = xa.Dataset()
     
    #print ("in CombineBandsSgl")
    #print (trialNC)

    # If trial data and flux data are  coming in as NC, store in xarray
    with xa.open_dataset(trialNC) as trialDS:
            trialDS.load()

    fullDS = []
    for bandNC in fullNC:
        with xa.open_dataset(bandNC) as bandDS:
            bandDS.load()
            fullDS.append(bandDS)
        # end with
    # end bandNC loop

    nForce = fullDS[0].sizes['record']

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

    for fDS in fullDS: fDS.close()
    trialDS.close()

    outDS.heating_rate.attrs['units'] = 'K/s'
    outDS.band_heating_rate.attrs['units'] = 'K/s'

    outDS.to_netcdf(outNC)

    return outNC
# end combineBandsSgl()

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
        for path in paths: FCC.pathCheck(path)

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
        for path in paths: FCC.pathCheck(path, mkdir=True)

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

    def gPointCombineSglPair(self,pmFlag,gCombine,xWeight):
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
            wCombine = [weights[np.array(gc)] for gc in gCombine]
            #print ("in gPointCombineSglPair")
            #print ("self.iCombine")
            #print (self.iCombine)

            for gc, wc in zip(gCombine, wCombine):
                g1, g2 = gc
                w1, w2 = wc

                # loop over each g-point combination and create
                # a k-distribution netCDF for each
                gCombStr = 'g{:02d}-{:02d}_iter{:03d}'.format(
                    g1+1, g2+1, self.iCombine)
                outNC='{}/coefficients_{}_{}_{}.nc'.format(
                    self.workDir, self.domainStr, gCombStr,pmFlag)
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

                if pmFlag == '2plus':
                    nscale = 2
                else:
                    nscale= 1
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
                            delta = xWeight*nscale
                            print(xWeight,nscale,pmFlag,delta)
                            kg1, kg2 = ncDat.isel(gpt=g1), ncDat.isel(gpt=g2)
                            ncDat = xa.where(ncDat.gpt == g1,
                                kg1*((w1/(w1+w2))+delta) + kg2*((w2/(w1+w2))-delta),ncDat)
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
    # end gPointCombineSgl()
# end gCombine_kDist
